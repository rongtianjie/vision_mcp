#!/usr/bin/env python3
"""
Vision MCP Server — expose describe_image tool so LLMs without vision can read images.

Environment variables:
  VISION_API_BASE   OpenAI-compatible base URL (default: http://localhost:8000/v1)
  VISION_API_KEY    API key (default: "not-needed")
  VISION_MODEL      Model name (default: qwen-vl-plus)
  VISION_MAX_TOKENS Max tokens in response (default: 2000)
"""

import base64
import mimetypes
import os
from pathlib import Path

import httpx
from mcp.server.fastmcp import FastMCP

# ── Configuration ──────────────────────────────────────────────
API_BASE = os.environ.get("VISION_API_BASE", "http://localhost:8000/v1")
API_KEY = os.environ.get("VISION_API_KEY", "not-needed")
MODEL = os.environ.get("VISION_MODEL", "qwen-vl-plus")
MAX_TOKENS = int(os.environ.get("VISION_MAX_TOKENS", "2000"))

mcp = FastMCP("vision-mcp")


# ── Diagnostic tool ──────────────────────────────────────────────
@mcp.tool(name="vision_ping", description="Diagnostic: return a test string to verify MCP communication.")
async def vision_ping(msg: str = "ping") -> str:
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

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{API_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MODEL,
                    "messages": messages,
                    "max_tokens": MAX_TOKENS,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    except httpx.HTTPStatusError as e:
        detail = e.response.text[:500] if e.response is not None else str(e)
        status = e.response.status_code if e.response is not None else "N/A"
        return f"错误：视觉模型 API 返回错误 ({status})：{detail}"
    except httpx.RequestError as e:
        return f"错误：无法连接到视觉模型 API（{API_BASE}）：{type(e).__name__}: {e}"
    except Exception as e:
        import traceback
        return f"错误：未知异常 {type(e).__name__}: {e}\n{traceback.format_exc()}"


# ── Entry point ────────────────────────────────────────────────
def main():
    import asyncio

    asyncio.run(mcp.run_stdio_async())


if __name__ == "__main__":
    main()
