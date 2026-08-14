"""MCP 工具定义。

工具函数为模块级定义（便于直接调用与单测），由 `register_tools()` 注册到
FastMCP 实例。工具通过 `ctx: Context` 获取 lifespan 提供的 ServerRuntime
（含配置与 HTTP 客户端）；直接调用（单元测试等无请求上下文的场景）时回退到
模块级 runtime，可由 `set_runtime()` 注入。
"""

from __future__ import annotations

import base64
import logging
import traceback
from typing import Any

import httpx
from mcp.server.fastmcp import Context, FastMCP

from vision_mcp.http_client import ServerRuntime, VisionAPIError
from vision_mcp.image_utils import VisionImageError, load_image_source

logger = logging.getLogger("vision-mcp")

DEFAULT_PROMPT = "请详细描述这张图片的内容。"
SUPPORTED_FORMATS = "PNG、JPG、JPEG、GIF、WebP、BMP、HEIC、AVIF、TIFF 等常见图片格式"

_runtime: ServerRuntime | None = None


def set_runtime(runtime: ServerRuntime | None) -> None:
    """设置/清除模块级运行时（lifespan 启动/退出时调用，测试中也可注入）。"""
    global _runtime
    _runtime = runtime


def _get_runtime(ctx: Context | None) -> ServerRuntime:
    """优先从请求上下文取 lifespan 提供的 runtime，否则用模块级 runtime。"""
    if ctx is not None:
        try:
            return ctx.request_context.lifespan_context
        except ValueError:
            pass
    if _runtime is None:
        raise RuntimeError("vision-mcp runtime 未初始化（server 尚未启动或已被关闭）")
    return _runtime


async def describe_image(
    image_path: str | None = None,
    image_url: str | None = None,
    images: list[str] | None = None,
    prompt: str = "",
    max_tokens: int | None = None,
    provider: str | None = None,
    ctx: Context | None = None,
) -> str:
    """理解并描述一张或多张图片（本地路径 / URL / data URI）。"""
    runtime = _get_runtime(ctx)
    config = runtime.config

    # 汇总图片来源：三种参数合并，保持向后兼容
    sources: list[str] = []
    for src in (image_path, image_url):
        if src and src.strip():
            sources.append(src.strip())
    if images:
        sources.extend(s.strip() for s in images if s and s.strip())
    if not sources:
        return (
            "错误：请提供 image_path（本地图片路径）、image_url（网络图片 URL）"
            "或 images（多张图片列表）。"
        )

    # 选择后端
    if provider:
        p = config.providers.get(provider.strip().lower())
        if p is None:
            return (
                f"错误：未知 provider {provider!r}。可用 provider："
                f"{', '.join(sorted(config.providers))}。"
            )
    else:
        p = config.provider

    # 解析并加载全部图片（串行，任一失败即返回对应错误）
    try:
        loaded: list[tuple[bytes, str]] = []
        for src in sources:
            data, mime = await load_image_source(
                src,
                client=runtime.client,
                max_image_mb=config.max_image_mb,
                max_upload_mb=config.max_upload_mb,
            )
            loaded.append((data, mime))
    except VisionImageError as exc:
        return str(exc)

    payload_max_tokens = max_tokens if max_tokens and max_tokens > 0 else config.max_tokens
    total_mb = sum(len(d) for d, _ in loaded) / (1024 * 1024)
    logger.info(
        "describe_image: sources=%d total=%.1fMB prompt_len=%d max_tokens=%d provider=%s",
        len(loaded),
        total_mb,
        len(prompt),
        payload_max_tokens,
        p.name,
    )

    content_parts: list[dict[str, Any]] = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{base64.b64encode(data).decode('utf-8')}"},
        }
        for data, mime in loaded
    ]
    content_parts.append({"type": "text", "text": prompt or DEFAULT_PROMPT})
    messages = [{"role": "user", "content": content_parts}]

    try:
        return await runtime.client.chat(p, messages, payload_max_tokens)
    except VisionAPIError as exc:
        return str(exc)
    except Exception as exc:  # 兜底：任何未预期异常都转成可读消息
        logger.exception("Unexpected error describing image")
        return f"错误：未知异常 {type(exc).__name__}: {exc}\n{traceback.format_exc()}"


async def vision_ping(
    msg: str = "ping",
    probe_api: bool = False,
    ctx: Context | None = None,
) -> str:
    """返回服务器状态；probe_api=True 时探测视觉 API 连通性。"""
    runtime = _get_runtime(ctx)
    p = runtime.config.provider
    base = (
        f"pong: {msg} (server alive, httpx={httpx.__version__}, provider={p.name}, model={p.model})"
    )
    if not probe_api:
        return base

    models_url = f"{p.api_base.rstrip('/')}/models"
    try:
        resp = await runtime.client.client.get(
            models_url,
            headers={"Authorization": f"Bearer {p.api_key}"},
            timeout=10.0,
        )
    except httpx.RequestError as exc:
        return f"{base}\nAPI 探测失败：无法连接 {models_url}（{type(exc).__name__}：{exc}）"

    if resp.status_code == 200:
        try:
            data = resp.json()
            models = data.get("data", [])
            names = [m.get("id") for m in models if isinstance(m, dict)][:5]
        except (ValueError, AttributeError):
            names = []
        return f"{base}\nAPI 连通正常（{models_url}），可用模型示例：{names or '无'}"
    return f"{base}\nAPI 探测失败 HTTP {resp.status_code}：{resp.text[:200]}"


_DESCRIBE_DESCRIPTION = (
    "理解并描述图片内容。支持本地文件绝对/相对路径、http(s) 网络图片 URL，"
    "也可一次传入多张图片进行对比或总结。"
    f"支持 {SUPPORTED_FORMATS}，单张不超过 20MB（超限可用 images 配合自动压缩）。"
    "当用户让你查看、理解、分析或描述任何图片时，你必须调用此工具。"
    "优先使用 image_path（本地）或 image_url（网络）；多张图片时使用 images 列表。"
)

_PING_DESCRIPTION = (
    "诊断工具：验证 MCP 通信与视觉 API 连通性。"
    "默认仅返回服务器存活信息；probe_api=True 时额外请求 GET /models "
    "检查默认 provider 的 API 是否可用。"
)


def register_tools(mcp: FastMCP) -> None:
    """向 FastMCP 实例注册全部工具。"""
    mcp.tool(name="describe_image", description=_DESCRIBE_DESCRIPTION)(describe_image)
    mcp.tool(name="vision_ping", description=_PING_DESCRIPTION)(vision_ping)
