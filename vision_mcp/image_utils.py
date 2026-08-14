"""图片来源解析与预处理。

支持三种图片来源（按传入字符串自动识别）：
- 本地文件路径（绝对或相对，相对路径基于 server 进程的工作目录）
- http(s):// 网络图片 URL（流式下载，限制大小）
- data: URI（base64 或 URL 编码的图片数据）

格式识别：内置常见扩展名映射表（含 HEIC/AVIF/TIFF/JFIF），扩展名不可靠时
用文件头 magic number 兜底。

预处理：超过 VISION_MAX_UPLOAD_MB 的图片用 Pillow 自动缩放/转 JPEG 压缩
（最长边 2048px，质量从 85 递减至 50），以降低上传体积与 API 拒绝率；
未安装 Pillow 或压缩失败时原样上传。
"""

from __future__ import annotations

import base64
import io
import logging
import mimetypes
import re
import urllib.parse
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from vision_mcp.http_client import VisionClient

logger = logging.getLogger("vision-mcp")

_MB = 1024 * 1024

# 内置扩展名 → MIME 映射（mimetypes 对 jfif/heic/avif 等识别不可靠）
IMAGE_EXT_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".jfif": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".heic": "image/heic",
    ".heif": "image/heif",
    ".avif": "image/avif",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
}

# 文件头 magic number → MIME（兜底识别，不依赖扩展名）
_MAGIC_MIME = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
)

_DATA_URI_RE = re.compile(r"^data:([^;,]+)?(;base64)?,(.*)$", re.S)

# 自动压缩参数
MAX_COMPRESS_EDGE = 2048  # 压缩时最长边像素
_MIN_QUALITY = 50  # JPEG 最低质量

try:  # Pillow 为可选运行时依赖：缺失时跳过压缩
    from PIL import Image, ImageOps

    _PIL_AVAILABLE = True
except ImportError:  # pragma: no cover - 依赖缺失路径
    _PIL_AVAILABLE = False
    Image = None  # type: ignore[assignment]
    ImageOps = None  # type: ignore[assignment]


class VisionImageError(Exception):
    """图片来源解析或预处理失败，消息为面向用户的中文描述。"""


def guess_image_mime(path: str | Path) -> str | None:
    """按扩展名识别图片 MIME；内置映射表优先，mimetypes 兜底。"""
    ext = Path(str(path)).suffix.lower()
    if ext in IMAGE_EXT_MIME:
        return IMAGE_EXT_MIME[ext]
    mime, _ = mimetypes.guess_type(str(path))
    if mime and mime.startswith("image/"):
        return mime
    return None


def sniff_image_mime(data: bytes) -> str | None:
    """按文件头 magic number 识别图片 MIME（扩展名缺失/错误时兜底）。"""
    for magic, mime in _MAGIC_MIME:
        if data.startswith(magic):
            return mime
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def parse_data_uri(uri: str) -> tuple[bytes, str]:
    """解析 data:image/png;base64,xxx 形式的数据 URI。返回 (data, mime)。"""
    match = _DATA_URI_RE.match(uri)
    if not match:
        raise ValueError("无效的 data URI")
    mime = (match.group(1) or "application/octet-stream").strip().lower()
    payload = match.group(3)
    try:
        if match.group(2):
            data = base64.b64decode(payload, validate=True)
        else:
            data = urllib.parse.unquote_to_bytes(payload)
    except Exception as exc:  # b64decode 对非法字符抛 binascii.Error
        raise ValueError("data URI 的内容编码无效") from exc
    if not data:
        raise ValueError("data URI 内容为空")
    return data, mime


def _load_local(path_str: str, max_image_mb: float) -> tuple[bytes, str]:
    """读取本地图片文件并校验大小/格式。"""
    path = Path(path_str)
    if not path.exists():
        raise VisionImageError(f"错误：图片文件不存在：{path_str}")
    if not path.is_file():
        raise VisionImageError(f"错误：路径不是文件：{path_str}")

    size = path.stat().st_size
    if size > max_image_mb * _MB:
        raise VisionImageError(
            f"错误：图片文件过大（{size / _MB:.1f}MB），请使用 {max_image_mb:.0f}MB 以内的图片。"
        )

    mime = guess_image_mime(path)
    if not mime:
        raise VisionImageError(
            f"错误：无法识别图片格式：{path_str}（支持 {', '.join(IMAGE_EXT_MIME)}）"
        )

    try:
        data = path.read_bytes()
    except OSError as exc:
        raise VisionImageError(f"错误：读取图片失败：{exc}") from exc

    # 扩展名不可靠时用文件头兜底纠正
    sniffed = sniff_image_mime(data)
    if sniffed and sniffed != mime:
        logger.info("图片 %s 实际格式为 %s（扩展名识别为 %s），已纠正", path_str, sniffed, mime)
        mime = sniffed
    return data, mime


async def load_image_source(
    src: str,
    *,
    client: "VisionClient",
    max_image_mb: float,
    max_upload_mb: float,
) -> tuple[bytes, str]:
    """解析单个图片来源（本地路径 / URL / data URI），返回 (data, mime)。

    client 需提供 stream()（用于下载 URL），即 VisionClient。
    """
    source = src.strip()
    if source.lower().startswith("data:"):
        try:
            data, mime = parse_data_uri(source)
        except ValueError as exc:
            raise VisionImageError(f"错误：{exc}") from exc
    elif source.lower().startswith(("http://", "https://")):
        data, mime = await _load_url(source, client=client, max_image_mb=max_image_mb)
    else:
        data, mime = _load_local(source, max_image_mb)

    return maybe_compress(data, mime, max_upload_mb)


async def _load_url(url: str, *, client: "VisionClient", max_image_mb: float) -> tuple[bytes, str]:
    """下载网络图片并确定 MIME（Content-Type → URL 后缀 → 文件头）。"""
    max_bytes = int(max_image_mb * _MB)
    try:
        data, content_type = await client.download_image(url, max_bytes)
    except httpx.HTTPStatusError as exc:
        raise VisionImageError(
            f"错误：下载图片失败 HTTP {exc.response.status_code}：{url}"
        ) from exc
    except httpx.RequestError as exc:
        raise VisionImageError(f"错误：无法下载图片（{type(exc).__name__}）：{url}") from exc

    mime = None
    if content_type:
        mime = content_type.split(";")[0].strip().lower()
        if not mime.startswith("image/"):
            mime = None
    if mime is None:
        mime = guess_image_mime(urllib.parse.urlparse(url).path)
    if mime is None:
        mime = sniff_image_mime(data)
    if mime is None:
        raise VisionImageError(f"错误：无法识别图片格式：{url}")
    return data, mime


def maybe_compress(data: bytes, mime: str, max_upload_mb: float) -> tuple[bytes, str]:
    """图片体积超过 max_upload_mb 时压缩；未超限、无 Pillow 或压缩失败则原样返回。"""
    limit = int(max_upload_mb * _MB)
    if len(data) <= limit:
        return data, mime
    if not _PIL_AVAILABLE:
        logger.warning(
            "图片 %.1fMB 超过压缩阈值 %.0fMB，但未安装 Pillow，将原样上传",
            len(data) / _MB,
            max_upload_mb,
        )
        return data, mime

    try:
        compressed, out_mime = _compress_with_pillow(data, limit)
    except Exception as exc:
        logger.warning("图片压缩失败（%s），将原样上传", exc)
        return data, mime

    ratio = len(compressed) / len(data) if data else 1.0
    logger.info(
        "图片已自动压缩：%.1fMB -> %.1fMB（%s，压缩率 %.0f%%）",
        len(data) / _MB,
        len(compressed) / _MB,
        out_mime,
        100 * (1 - ratio),
    )
    return compressed, out_mime


def _compress_with_pillow(data: bytes, limit: int) -> tuple[bytes, str]:
    """用 Pillow 缩放并转 JPEG 压缩到 limit 以内。"""
    assert Image is not None and ImageOps is not None  # _PIL_AVAILABLE 保证
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)  # 按 EXIF 方向摆正
    img.load()

    # 动图取第一帧
    if getattr(img, "is_animated", False):
        img.seek(0)

    # 透明通道垫白底（JPEG 不支持 alpha）
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    img.thumbnail((MAX_COMPRESS_EDGE, MAX_COMPRESS_EDGE))

    quality = 85
    while True:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        out = buf.getvalue()
        if len(out) <= limit or quality <= _MIN_QUALITY:
            return out, "image/jpeg"
        quality -= 10
