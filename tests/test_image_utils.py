"""Tests for vision_mcp.image_utils."""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from vision_mcp import image_utils
from vision_mcp.image_utils import (
    VisionImageError,
    guess_image_mime,
    load_image_source,
    maybe_compress,
    parse_data_uri,
    sniff_image_mime,
)

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
JPEG_BYTES = b"\xff\xd8\xff" + b"\x00" * 32


class TestGuessImageMime:
    def test_known_extensions(self):
        assert guess_image_mime("a.png") == "image/png"
        assert guess_image_mime("a.jpg") == "image/jpeg"
        assert guess_image_mime("a.jfif") == "image/jpeg"
        assert guess_image_mime("a.heic") == "image/heic"
        assert guess_image_mime("a.avif") == "image/avif"
        assert guess_image_mime("a.tiff") == "image/tiff"

    def test_uppercase_extension(self):
        assert guess_image_mime("A.PNG") == "image/png"

    def test_unknown_extension(self):
        assert guess_image_mime("a.xyz") is None


class TestSniffImageMime:
    def test_png(self):
        assert sniff_image_mime(PNG_BYTES) == "image/png"

    def test_jpeg(self):
        assert sniff_image_mime(JPEG_BYTES) == "image/jpeg"

    def test_webp(self):
        assert sniff_image_mime(b"RIFF\x10\x00\x00\x00WEBPVP8 ") == "image/webp"

    def test_unknown(self):
        assert sniff_image_mime(b"\x00\x01\x02") is None


class TestParseDataUri:
    def test_base64(self):
        raw = b"fake-image"
        uri = f"data:image/png;base64,{base64.b64encode(raw).decode()}"
        data, mime = parse_data_uri(uri)
        assert data == raw
        assert mime == "image/png"

    def test_urlencoded(self):
        uri = "data:image/jpeg,%00%01"
        data, mime = parse_data_uri(uri)
        assert data == b"\x00\x01"
        assert mime == "image/jpeg"

    def test_invalid_base64(self):
        with pytest.raises(ValueError):
            parse_data_uri("data:image/png;base64,!!!not-base64!!!")

    def test_empty(self):
        with pytest.raises(ValueError):
            parse_data_uri("data:image/png;base64,")

    def test_not_data_uri(self):
        with pytest.raises(ValueError):
            parse_data_uri("http://example.com/a.png")


class TestLoadLocalPath:
    """本地路径：不存在 / 目录 / 过大 / 未知格式 / 成功 / magic 纠正。"""

    async def test_not_found(self):
        with pytest.raises(VisionImageError, match="不存在"):
            await load_image_source(
                "/nonexistent.png", client=AsyncMock(), max_image_mb=20, max_upload_mb=8
            )

    async def test_is_directory(self, tmp_path):
        with pytest.raises(VisionImageError, match="不是文件"):
            await load_image_source(
                str(tmp_path), client=AsyncMock(), max_image_mb=20, max_upload_mb=8
            )

    async def test_too_large(self, tmp_path):
        img = tmp_path / "big.png"
        img.write_bytes(b"x" * (21 * 1024 * 1024))
        with pytest.raises(VisionImageError, match="过大"):
            await load_image_source(str(img), client=AsyncMock(), max_image_mb=20, max_upload_mb=8)

    async def test_unknown_format(self, tmp_path):
        img = tmp_path / "file.xyz"
        img.write_bytes(b"data")
        with pytest.raises(VisionImageError, match="无法识别图片格式"):
            await load_image_source(str(img), client=AsyncMock(), max_image_mb=20, max_upload_mb=8)

    async def test_success(self, tmp_path):
        img = tmp_path / "photo.png"
        img.write_bytes(PNG_BYTES)
        data, mime = await load_image_source(
            str(img), client=AsyncMock(), max_image_mb=20, max_upload_mb=8
        )
        assert data == PNG_BYTES
        assert mime == "image/png"

    async def test_magic_corrects_wrong_extension(self, tmp_path):
        """扩展名与内容不符时按文件头纠正（如 .png 实际是 JPEG）。"""
        img = tmp_path / "photo.png"
        img.write_bytes(JPEG_BYTES)
        data, mime = await load_image_source(
            str(img), client=AsyncMock(), max_image_mb=20, max_upload_mb=8
        )
        assert mime == "image/jpeg"
        assert data == JPEG_BYTES


class TestLoadUrl:
    """网络图片：Content-Type → URL 后缀 → 文件头兜底。"""

    def _client_with(self, download_retval=None, download_side_effect=None):
        client = AsyncMock()
        if download_retval is not None:
            client.download_image.return_value = download_retval
        if download_side_effect is not None:
            client.download_image.side_effect = download_side_effect
        return client

    async def test_content_type_wins(self):
        client = self._client_with(download_retval=(b"data", "image/png; charset=utf-8"))
        data, mime = await load_image_source(
            "https://example.com/img", client=client, max_image_mb=20, max_upload_mb=8
        )
        assert (data, mime) == (b"data", "image/png")

    async def test_fallback_to_url_extension(self):
        client = self._client_with(download_retval=(b"data", "text/html"))
        data, mime = await load_image_source(
            "https://example.com/a.jpeg", client=client, max_image_mb=20, max_upload_mb=8
        )
        assert mime == "image/jpeg"

    async def test_fallback_to_magic(self):
        client = self._client_with(download_retval=(PNG_BYTES, None))
        data, mime = await load_image_source(
            "https://example.com/img", client=client, max_image_mb=20, max_upload_mb=8
        )
        assert mime == "image/png"

    async def test_unrecognized_rejected(self):
        client = self._client_with(download_retval=(b"data", "application/octet-stream"))
        with pytest.raises(VisionImageError, match="无法识别图片格式"):
            await load_image_source(
                "https://example.com/img", client=client, max_image_mb=20, max_upload_mb=8
            )

    async def test_http_error(self):
        request = httpx.Request("GET", "https://example.com/img")
        response = httpx.Response(404, request=request)
        client = self._client_with(
            download_side_effect=httpx.HTTPStatusError("404", request=request, response=response)
        )
        with pytest.raises(VisionImageError, match="HTTP 404"):
            await load_image_source(
                "https://example.com/img", client=client, max_image_mb=20, max_upload_mb=8
            )

    async def test_connection_error(self):
        client = self._client_with(download_side_effect=httpx.ConnectError("refused"))
        with pytest.raises(VisionImageError, match="无法下载图片"):
            await load_image_source(
                "https://example.com/img", client=client, max_image_mb=20, max_upload_mb=8
            )

    async def test_oversize(self):
        client = self._client_with(
            download_side_effect=VisionImageError("错误：图片下载超过大小限制")
        )
        with pytest.raises(VisionImageError, match="大小限制"):
            await load_image_source(
                "https://example.com/img", client=client, max_image_mb=20, max_upload_mb=8
            )


class TestLoadDataUri:
    async def test_valid(self):
        raw = b"uri-image"
        uri = f"data:image/png;base64,{base64.b64encode(raw).decode()}"
        data, mime = await load_image_source(
            uri, client=AsyncMock(), max_image_mb=20, max_upload_mb=8
        )
        assert data == raw
        assert mime == "image/png"

    async def test_invalid(self):
        with pytest.raises(VisionImageError, match="无效|编码无效"):
            await load_image_source(
                "data:image/png;base64,!!!",
                client=AsyncMock(),
                max_image_mb=20,
                max_upload_mb=8,
            )


class TestMaybeCompress:
    def test_small_image_untouched(self):
        data, mime = maybe_compress(b"tiny", "image/png", max_upload_mb=8)
        assert (data, mime) == (b"tiny", "image/png")

    @patch("vision_mcp.image_utils._PIL_AVAILABLE", False)
    def test_large_without_pillow_untouched(self):
        big = b"x" * (10 * 1024 * 1024)
        data, mime = maybe_compress(big, "image/png", max_upload_mb=8)
        assert data == big
        assert mime == "image/png"

    def test_large_compressed_to_jpeg(self):
        """真实 Pillow 压缩：噪声图应被压成 JPEG 且体积显著减小。"""
        if not image_utils._PIL_AVAILABLE:  # pragma: no cover
            pytest.skip("Pillow 未安装")
        from PIL import Image

        img = Image.effect_noise((1000, 1000), 100).convert("RGB")
        buf = __import__("io").BytesIO()
        img.save(buf, format="PNG")
        big = buf.getvalue()
        assert len(big) > 200 * 1024  # 噪声 PNG 一定超过阈值

        data, mime = maybe_compress(big, "image/png", max_upload_mb=0.2)
        assert mime == "image/jpeg"
        assert len(data) < len(big)
