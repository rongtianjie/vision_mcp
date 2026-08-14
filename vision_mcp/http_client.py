"""共享 HTTP 客户端与带重试的视觉 API 调用。

- 连接池化的 httpx.AsyncClient（由 FastMCP lifespan 创建/关闭）
- 瞬时错误（超时、429/500/502/503/504）自动重试：尊重 Retry-After 头，
  否则使用带全抖动（full jitter）的指数退避，避免多客户端同时重试放大压力
- 兼容 content 为字符串或分段 list 两种 OpenAI-compatible 响应结构
- 所有失败统一转换为 VisionAPIError（消息为面向用户的中文描述）
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

import httpx

from vision_mcp.config import ProviderConfig, VisionConfig

logger = logging.getLogger("vision-mcp")

RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})
_MAX_RETRY_WAIT = 60.0  # 单次重试等待上限（秒）
_BACKOFF_BASE = 2.0  # 退避基数：base * 2^attempt


class VisionAPIError(Exception):
    """视觉 API 调用失败，消息为面向用户的中文描述。"""


class VisionClient:
    """管理连接池化 HTTP client，封装带重试的 chat/completions 调用。"""

    def __init__(self, config: VisionConfig) -> None:
        self.config = config
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        """惰性创建连接池化 client（首次使用时）。"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.timeout),
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            )
            logger.info(
                "HTTP client created (pool max_connections=10, timeout=%.0fs)", self.config.timeout
            )
        return self._client

    async def aclose(self) -> None:
        """关闭底层连接池（供 lifespan 退出时调用）。"""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            logger.info("HTTP client closed")

    async def download_image(self, url: str, max_bytes: int) -> tuple[bytes, str | None]:
        """流式下载网络图片，超过 max_bytes 立即中止。返回 (data, content_type)。"""
        async with self.client.stream("GET", url) as resp:
            resp.raise_for_status()
            mime = resp.headers.get("content-type")
            chunks: list[bytes] = []
            total = 0
            async for chunk in resp.aiter_bytes():
                total += len(chunk)
                if total > max_bytes:
                    raise VisionAPIError(
                        f"错误：图片下载超过大小限制（{max_bytes // (1024 * 1024)}MB）：{url}"
                    )
                chunks.append(chunk)
        return b"".join(chunks), mime

    async def chat(
        self,
        provider: ProviderConfig,
        messages: list[dict[str, Any]],
        max_tokens: int,
    ) -> str:
        """调用 /chat/completions 并解析出文本内容；失败抛 VisionAPIError。"""
        url = f"{provider.api_base.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {provider.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": provider.model, "messages": messages, "max_tokens": max_tokens}
        max_retries = self.config.max_retries

        for attempt in range(max_retries + 1):
            try:
                resp = await self.client.post(url, headers=headers, json=payload)
            except httpx.TimeoutException:
                if attempt < max_retries:
                    wait = _backoff_seconds(attempt)
                    logger.warning(
                        "请求超时（第 %d/%d 次），%.1fs 后重试",
                        attempt + 1,
                        max_retries,
                        wait,
                    )
                    await asyncio.sleep(wait)
                    continue
                raise VisionAPIError(
                    f"错误：视觉模型 API 请求超时（{provider.api_base}），"
                    "请检查网络连接或增大 VISION_TIMEOUT。"
                ) from None

            if resp.status_code in RETRYABLE_STATUSES and attempt < max_retries:
                wait = _retry_after_seconds(resp.headers) or _backoff_seconds(attempt)
                logger.warning(
                    "API %d（第 %d/%d 次），%.1fs 后重试",
                    resp.status_code,
                    attempt + 1,
                    max_retries,
                    wait,
                )
                await asyncio.sleep(wait)
                continue

            if resp.status_code >= 400:
                self._raise_status_error(provider, resp)

            try:
                data = resp.json()
            except ValueError:
                logger.error("API 返回非 JSON：%s", resp.text[:300])
                raise VisionAPIError("错误：视觉模型 API 返回了非 JSON 内容。") from None

            content = extract_content(data)
            if content is None:
                raise VisionAPIError("错误：视觉模型 API 返回了异常响应结构，请检查模型配置。")
            if not content.strip():
                logger.warning("API 返回空内容")
                raise VisionAPIError("视觉模型返回了空内容，图片可能无法被识别。")
            return content

        # 理论不可达（重试耗尽时会抛出异常）
        raise VisionAPIError(f"错误：请求失败（已重试 {max_retries} 次）。")

    def _raise_status_error(self, provider: ProviderConfig, resp: httpx.Response) -> None:
        """将 HTTP 错误转换为带用户提示的 VisionAPIError。"""
        status = resp.status_code
        detail = resp.text[:500]
        logger.error("API HTTP %d: %s", status, detail[:200])
        if status == 429:
            raise VisionAPIError("错误：API 请求过于频繁，请稍后重试。")
        if status == 401:
            raise VisionAPIError("错误：API 认证失败，请检查 VISION_API_KEY 配置。")
        if status == 400:
            raise VisionAPIError("错误：API 请求参数错误 (400)，请检查图片格式或模型配置。")
        raise VisionAPIError(f"错误：视觉模型 API 返回错误 ({status})：{detail}")


class ServerRuntime:
    """lifespan 提供的运行时：配置 + HTTP 客户端。"""

    def __init__(self, config: VisionConfig) -> None:
        self.config = config
        self.client = VisionClient(config)


# ── 内部工具函数（独立定义便于单测）───────────────────────────


def _backoff_seconds(attempt: int) -> float:
    """指数退避 + 全抖动：U(0, min(60, base * 2^attempt))。"""
    ceiling = min(_BACKOFF_BASE * (2**attempt), _MAX_RETRY_WAIT)
    return random.uniform(0, ceiling)


def _retry_after_seconds(headers: Any) -> float | None:
    """解析 Retry-After 头（秒数），超出上限或非法时返回 None。"""
    raw = headers.get("retry-after")
    if raw is None:
        return None
    try:
        secs = float(raw)
    except (TypeError, ValueError):
        return None
    return min(max(secs, 0.0), _MAX_RETRY_WAIT)


def extract_content(data: dict[str, Any]) -> str | None:
    """从 /chat/completions 响应中提取文本内容。

    兼容两种结构：
    - content: "纯文本"
    - content: [{"type": "text", "text": "..."}, ...]（分段多模态）
    结构异常返回 None（由调用方转为用户错误）。
    """
    choices = data.get("choices")
    if not choices or not isinstance(choices, list) or len(choices) == 0:
        logger.error("Unexpected API response: choices missing or empty — %s", str(data)[:300])
        return None

    message = choices[0].get("message")
    if not message or not isinstance(message, dict):
        logger.error("Unexpected choice structure: %s", str(choices[0])[:300])
        return None

    content = message.get("content")
    if content is None:
        logger.error("message 缺少 content 字段: %s", str(message)[:300])
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif (
                isinstance(item, dict)
                and item.get("type") == "text"
                and isinstance(item.get("text"), str)
            ):
                parts.append(item["text"])
        return "\n".join(parts)
    logger.error("Unexpected content type: %s", type(content).__name__)
    return None
