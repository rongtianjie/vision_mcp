#!/usr/bin/env python3
"""
Vision MCP Server — expose describe_image tool so LLMs without vision can read images.

Environment variables:
  VISION_API_BASE   OpenAI-compatible base URL (default: http://localhost:8000/v1)
  VISION_API_KEY    API key (default: "not-needed")
  VISION_MODEL      Model name (default: qwen-vl-plus)
  VISION_MAX_TOKENS Max tokens in response (default: 2000)
"""

import asyncio
import base64
import logging
import mimetypes
import os
import traceback
from pathlib import Path

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# ── Logging ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vision-mcp")

# ── Load .env from project root (before config) ───────────────
_dotenv_path = Path(__file__).resolve().parent.parent / ".env"
if _dotenv_path.exists():
    load_dotenv(_dotenv_path, override=False)
    logger.info("Loaded env from %s", _dotenv_path)
else:
    logger.debug("No .env found at %s", _dotenv_path)

# ── Configuration ──────────────────────────────────────────────
API_BASE = os.environ.get("VISION_API_BASE", "http://localhost:8000/v1")
API_KEY = os.environ.get("VISION_API_KEY", "not-needed")
MODEL = os.environ.get("VISION_MODEL", "qwen-vl-plus")
MAX_TOKENS = int(os.environ.get("VISION_MAX_TOKENS", "2000"))

mcp = FastMCP("vision-mcp")

# ── Shared HTTP client (connection-pooled) ─────────────────────
_http_client: httpx.AsyncClient | None = None


RETRYABLE_STATUSES = frozenset({429, 502, 503, 504})


def _get_http_client() -> httpx.AsyncClient:
    """Get or create a connection-pooled HTTP client."""
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(120.0),
            limits=httpx.Limits(
                max_connections=10,
                max_keepalive_connections=5,
            ),
        )
        logger.info(
            "HTTP client created (pool max_connections=10, timeout=120s)"
        )
    return _http_client


async def _call_with_retry(
    client: httpx.AsyncClient,
    url: str,
    headers: dict,
    json_data: dict,
    max_retries: int = 2,
) -> httpx.Response:
    """POST with retry on transient errors (timeout, 429, 502, 503, 504)."""
    for attempt in range(max_retries + 1):
        try:
            resp = await client.post(url, headers=headers, json=json_data)
        except httpx.TimeoutException:
            if attempt < max_retries:
                wait = 2 ** (attempt + 1)
                logger.warning(
                    "Request timeout (attempt %d/%d), retrying in %ds",
                    attempt + 1,
                    max_retries,
                    wait,
                )
                await asyncio.sleep(wait)
                continue
            logger.error("Request timeout after %d retries", max_retries)
            raise

        if resp.status_code in RETRYABLE_STATUSES and attempt < max_retries:
            wait = 2 ** (attempt + 1)
            logger.warning(
                "API %d (attempt %d/%d), retrying in %ds",
                resp.status_code,
                attempt + 1,
                max_retries,
                wait,
            )
            await asyncio.sleep(wait)
            continue

        resp.raise_for_status()
        return resp

    # Should not reach here
    raise httpx.RequestError(f"Request failed after {max_retries + 1} attempts")


# ── Diagnostic tool ──────────────────────────────────────────────
@mcp.tool(
    name="vision_ping",
    description="Diagnostic: return a test string to verify MCP communication.",
)
async def vision_ping(msg: str = "ping") -> str:
    logger.info("vision_ping called: msg=%s", msg)
    return f"pong: {msg} (server alive, httpx={httpx.__version__})"


# ── Tool ────────────────────────────────────────────────────────
@mcp.tool(
    name="describe_image",
    description=(
        "理解并描述图片内容。传入本地图片文件绝对路径，返回对该图片的详细文字描述。"
        "当用户让你查看、理解、分析或描述任何图片时，你必须调用此工具。"
        "支持 PNG、JPG、JPEG、GIF、WebP、BMP 等常见图片格式。"
    ),
)
async def describe_image(
    image_path: str,
    prompt: str = "",
    max_tokens: int | None = None,
) -> str:
    """Describe image content via multimodal vision API."""
    path = Path(image_path)
    if not path.exists():
        return f"错误：图片文件不存在：{image_path}"
    if not path.is_file():
        return f"错误：路径不是文件：{image_path}"

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > 20:
        return f"错误：图片文件过大（{size_mb:.1f}MB），请使用 20MB 以内的图片。"

    mime_type, _ = mimetypes.guess_type(str(path))
    if not mime_type or not mime_type.startswith("image/"):
        return f"错误：无法识别图片格式：{image_path}"

    try:
        image_data = path.read_bytes()
        image_b64 = base64.b64encode(image_data).decode("utf-8")
    except OSError as e:
        return f"错误：读取图片失败：{e}"

    if not prompt:
        prompt = "请详细描述这张图片的内容。"

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    payload_max_tokens = max_tokens if max_tokens is not None else MAX_TOKENS
    logger.info(
        "describe_image: path=%s, prompt_len=%d, max_tokens=%s, size=%.1fMB",
        image_path,
        len(prompt),
        payload_max_tokens,
        size_mb,
    )

    try:
        client = _get_http_client()
        resp = await _call_with_retry(
            client,
            f"{API_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json_data={
                "model": MODEL,
                "messages": messages,
                "max_tokens": payload_max_tokens,
            },
        )

        data = resp.json()

        # ── Defensive validation of API response ──
        choices = data.get("choices")
        if not choices or not isinstance(choices, list) or len(choices) == 0:
            logger.error(
                "Unexpected API response: choices missing or empty — %s",
                str(data)[:300],
            )
            return "错误：视觉模型 API 返回了异常响应结构，请检查模型配置。"

        message = choices[0].get("message")
        if not message or not isinstance(message, dict):
            logger.error(
                "Unexpected choice structure: %s",
                str(choices[0])[:300],
            )
            return "错误：视觉模型 API 返回了异常 message 结构。"

        content = message.get("content")
        if not content:
            logger.warning("API returned empty content for %s", image_path)
            return "视觉模型返回了空内容，图片可能无法被识别。"

        return content

    except httpx.TimeoutException:
        logger.error("Request timeout after retries to %s", API_BASE)
        return (
            f"错误：视觉模型 API 请求超时（{API_BASE}），请检查网络连接或增大超时时间。"
        )
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        detail = e.response.text[:500]
        logger.error("API HTTP %d: %s", status, detail[:200])
        if status == 429:
            return "错误：API 请求过于频繁，请稍后重试。"
        if status == 401:
            return "错误：API 认证失败，请检查 VISION_API_KEY 配置。"
        if status == 400:
            return "错误：API 请求参数错误 (400)，请检查图片格式或模型配置。"
        return f"错误：视觉模型 API 返回错误 ({status})：{detail}"
    except httpx.RequestError as e:
        logger.error("Connection error: %s: %s", type(e).__name__, e)
        return (
            f"错误：无法连接到视觉模型 API（{API_BASE}）：{type(e).__name__}: {e}"
        )
    except Exception as e:
        logger.exception("Unexpected error describing image")
        return f"错误：未知异常 {type(e).__name__}: {e}\n{traceback.format_exc()}"


# ── Entry point ────────────────────────────────────────────────
def main():
    asyncio.run(mcp.run_stdio_async())


if __name__ == "__main__":
    main()
