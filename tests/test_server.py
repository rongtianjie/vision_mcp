"""Tests for vision_mcp server."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from vision_mcp.server import (
    MAX_TOKENS,
    _call_with_retry,
    _get_http_client,
    describe_image,
    vision_ping,
)


# ── vision_ping ────────────────────────────────────────────────


class TestVisionPing:
    """Diagnostic tool should always reply with a pong."""

    async def test_default_msg(self):
        result = await vision_ping()
        assert result.startswith("pong: ping")
        assert "httpx" in result

    async def test_custom_msg(self):
        result = await vision_ping("hello")
        assert result.startswith("pong: hello")


# ── _get_http_client ───────────────────────────────────────────


class TestGetHttpClient:
    """HTTP client factory should create once and reuse."""

    def _reset(self):
        import vision_mcp.server as server

        server._http_client = None

    async def test_creates_and_reuses(self):
        self._reset()
        try:
            c1 = _get_http_client()
            c2 = _get_http_client()
            assert c1 is c2  # same instance
            assert hasattr(c1, "post")
        finally:
            self._reset()

    async def test_returns_async_client(self):
        self._reset()
        try:
            client = _get_http_client()
            assert isinstance(client, httpx.AsyncClient)
        finally:
            self._reset()


# ── _call_with_retry ───────────────────────────────────────────


class TestCallWithRetry:
    """Internal retry logic for transient HTTP failures."""

    @pytest.fixture
    def mock_client(self):
        return AsyncMock(spec=httpx.AsyncClient)

    @pytest.fixture
    def ok_resp(self):
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        return resp

    @pytest.fixture
    def http_request(self):
        return httpx.Request("POST", "http://localhost:8000/v1/chat/completions")

    async def test_success_on_first_attempt(self, mock_client, ok_resp):
        mock_client.post.return_value = ok_resp

        resp = await _call_with_retry(mock_client, "http://test/", {}, {})
        assert resp is ok_resp
        mock_client.post.assert_called_once()

    async def test_retry_on_429(self, mock_client, ok_resp):
        resp_429 = MagicMock(spec=httpx.Response)
        resp_429.status_code = 429
        mock_client.post.side_effect = [resp_429, ok_resp]

        resp = await _call_with_retry(mock_client, "http://test/", {}, {})
        assert resp is ok_resp
        assert mock_client.post.call_count == 2

    async def test_retry_on_503(self, mock_client, ok_resp):
        resp_503 = MagicMock(spec=httpx.Response)
        resp_503.status_code = 503
        mock_client.post.side_effect = [resp_503, ok_resp]

        resp = await _call_with_retry(mock_client, "http://test/", {}, {})
        assert resp is ok_resp
        assert mock_client.post.call_count == 2

    async def test_retry_on_timeout(self, mock_client, ok_resp):
        mock_client.post.side_effect = [httpx.TimeoutException("timeout"), ok_resp]

        resp = await _call_with_retry(mock_client, "http://test/", {}, {})
        assert resp is ok_resp
        assert mock_client.post.call_count == 2

    async def test_gives_up_after_retries(self, mock_client, http_request):
        resp_503 = MagicMock(spec=httpx.Response)
        resp_503.status_code = 503
        resp_503.raise_for_status.side_effect = httpx.HTTPStatusError(
            "503 Service Unavailable", request=http_request, response=resp_503
        )
        mock_client.post.return_value = resp_503

        with pytest.raises(httpx.HTTPStatusError):
            await _call_with_retry(mock_client, "http://test/", {}, {})
        assert mock_client.post.call_count == 3  # 1 original + 2 retries

    async def test_timeout_gives_up(self, mock_client):
        mock_client.post.side_effect = httpx.TimeoutException("timeout")

        with pytest.raises(httpx.TimeoutException):
            await _call_with_retry(mock_client, "http://test/", {}, {})
        assert mock_client.post.call_count == 3

    async def test_non_retryable_error_does_not_retry(self, mock_client, http_request):
        """400 errors should NOT be retried."""
        resp_400 = MagicMock(spec=httpx.Response)
        resp_400.status_code = 400
        resp_400.raise_for_status.side_effect = httpx.HTTPStatusError(
            "400 Bad Request", request=http_request, response=resp_400
        )
        mock_client.post.return_value = resp_400

        with pytest.raises(httpx.HTTPStatusError):
            await _call_with_retry(mock_client, "http://test/", {}, {})
        mock_client.post.assert_called_once()


# ── describe_image ─────────────────────────────────────────────


def _make_mock_path(
    *,
    exists: bool = True,
    is_file: bool = True,
    size_bytes: int = 500 * 1024,
    read_bytes: bytes = b"fake_image_data",
) -> MagicMock:
    """Helper to build a mocked Path with file metadata."""
    path = MagicMock()
    path.exists.return_value = exists
    path.is_file.return_value = is_file
    path.stat.return_value.st_size = size_bytes
    path.read_bytes.return_value = read_bytes
    return path


def _ok_response(content: str = "描述内容"):
    """Return a mock HTTP response with a successful vision API payload."""
    resp = AsyncMock(spec=httpx.Response)
    resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    return resp


class TestDescribeImage:
    """Main image-description tool: file validation & API integration."""

    # ── File validation (return early, no HTTP) ──

    @patch("vision_mcp.server.Path")
    async def test_file_not_found(self, mock_path_cls):
        mock_path_cls.return_value = _make_mock_path(exists=False)

        result = await describe_image("/nonexistent.png")
        assert "图片文件不存在" in result

    @patch("vision_mcp.server.Path")
    async def test_not_a_file(self, mock_path_cls):
        mock_path_cls.return_value = _make_mock_path(is_file=False)

        result = await describe_image("/dir/path")
        assert "路径不是文件" in result

    @patch("vision_mcp.server.Path")
    async def test_file_too_large(self, mock_path_cls):
        mock_path_cls.return_value = _make_mock_path(size_bytes=21 * 1024 * 1024)

        result = await describe_image("/large.png")
        assert "图片文件过大" in result

    @patch("vision_mcp.server.Path")
    @patch("vision_mcp.server.mimetypes.guess_type")
    async def test_unrecognized_format(self, mock_guess_type, mock_path_cls):
        mock_path_cls.return_value = _make_mock_path()
        mock_guess_type.return_value = (None, None)

        result = await describe_image("/file.xyz")
        assert "无法识别图片格式" in result

    @patch("vision_mcp.server.Path")
    @patch("vision_mcp.server.mimetypes.guess_type")
    async def test_read_error(self, mock_guess_type, mock_path_cls):
        mock_guess_type.return_value = ("image/png", None)
        path = _make_mock_path()
        path.read_bytes.side_effect = OSError("Permission denied")
        mock_path_cls.return_value = path

        result = await describe_image("/protected.png")
        assert "读取图片失败" in result

    @patch("vision_mcp.server.Path")
    @patch("vision_mcp.server.mimetypes.guess_type")
    async def test_mime_not_image(self, mock_guess_type, mock_path_cls):
        mock_path_cls.return_value = _make_mock_path()
        mock_guess_type.return_value = ("application/octet-stream", None)

        result = await describe_image("/file.bin")
        assert "无法识别图片格式" in result

    # ── API success ──

    @patch("vision_mcp.server._call_with_retry")
    @patch("vision_mcp.server._get_http_client")
    @patch("vision_mcp.server.mimetypes.guess_type")
    @patch("vision_mcp.server.Path")
    async def test_success(
        self, mock_path_cls, mock_guess_type, mock_get_client, mock_retry
    ):
        mock_path_cls.return_value = _make_mock_path()
        mock_guess_type.return_value = ("image/jpeg", None)
        mock_retry.return_value = _ok_response("这是一张风景图片")

        result = await describe_image("/photo.jpg", "描述这张图")
        assert result == "这是一张风景图片"

    @patch("vision_mcp.server._call_with_retry")
    @patch("vision_mcp.server._get_http_client")
    @patch("vision_mcp.server.mimetypes.guess_type")
    @patch("vision_mcp.server.Path")
    async def test_empty_prompt_uses_default(
        self, mock_path_cls, mock_guess_type, mock_get_client, mock_retry
    ):
        mock_path_cls.return_value = _make_mock_path()
        mock_guess_type.return_value = ("image/png", None)
        mock_retry.return_value = _ok_response("desc")

        await describe_image("/test.png")
        call_kwargs = mock_retry.call_args[1]
        prompt_text = call_kwargs["json_data"]["messages"][0]["content"][1]["text"]
        assert prompt_text == "请详细描述这张图片的内容。"

    @patch("vision_mcp.server._call_with_retry")
    @patch("vision_mcp.server._get_http_client")
    @patch("vision_mcp.server.mimetypes.guess_type")
    @patch("vision_mcp.server.Path")
    async def test_custom_prompt(
        self, mock_path_cls, mock_guess_type, mock_get_client, mock_retry
    ):
        mock_path_cls.return_value = _make_mock_path()
        mock_guess_type.return_value = ("image/png", None)
        mock_retry.return_value = _ok_response("desc")

        await describe_image("/test.png", "提取所有文字")
        call_kwargs = mock_retry.call_args[1]
        prompt_text = call_kwargs["json_data"]["messages"][0]["content"][1]["text"]
        assert prompt_text == "提取所有文字"

    # ── max_tokens ──

    @patch("vision_mcp.server._call_with_retry")
    @patch("vision_mcp.server._get_http_client")
    @patch("vision_mcp.server.mimetypes.guess_type")
    @patch("vision_mcp.server.Path")
    async def test_max_tokens_override(
        self, mock_path_cls, mock_guess_type, mock_get_client, mock_retry
    ):
        mock_path_cls.return_value = _make_mock_path()
        mock_guess_type.return_value = ("image/png", None)
        mock_retry.return_value = _ok_response()

        await describe_image("/test.png", max_tokens=4096)
        assert mock_retry.call_args[1]["json_data"]["max_tokens"] == 4096

    @patch("vision_mcp.server._call_with_retry")
    @patch("vision_mcp.server._get_http_client")
    @patch("vision_mcp.server.mimetypes.guess_type")
    @patch("vision_mcp.server.Path")
    async def test_max_tokens_default(
        self, mock_path_cls, mock_guess_type, mock_get_client, mock_retry
    ):
        mock_path_cls.return_value = _make_mock_path()
        mock_guess_type.return_value = ("image/png", None)
        mock_retry.return_value = _ok_response()

        await describe_image("/test.png")
        assert mock_retry.call_args[1]["json_data"]["max_tokens"] == MAX_TOKENS

    # ── API error handling ──

    @patch("vision_mcp.server._call_with_retry")
    @patch("vision_mcp.server._get_http_client")
    @patch("vision_mcp.server.mimetypes.guess_type")
    @patch("vision_mcp.server.Path")
    async def test_api_timeout(
        self, mock_path_cls, mock_guess_type, mock_get_client, mock_retry
    ):
        mock_path_cls.return_value = _make_mock_path()
        mock_guess_type.return_value = ("image/png", None)
        mock_retry.side_effect = httpx.TimeoutException("timeout")

        result = await describe_image("/test.png")
        assert "请求超时" in result

    @patch("vision_mcp.server._call_with_retry")
    @patch("vision_mcp.server._get_http_client")
    @patch("vision_mcp.server.mimetypes.guess_type")
    @patch("vision_mcp.server.Path")
    async def test_api_429(
        self, mock_path_cls, mock_guess_type, mock_get_client, mock_retry
    ):
        mock_path_cls.return_value = _make_mock_path()
        mock_guess_type.return_value = ("image/png", None)

        request = httpx.Request("POST", "http://test/")
        response = httpx.Response(429, request=request, text="Too Many Requests")
        mock_retry.side_effect = httpx.HTTPStatusError(
            "429", request=request, response=response
        )

        result = await describe_image("/test.png")
        assert "请求过于频繁" in result

    @patch("vision_mcp.server._call_with_retry")
    @patch("vision_mcp.server._get_http_client")
    @patch("vision_mcp.server.mimetypes.guess_type")
    @patch("vision_mcp.server.Path")
    async def test_api_401(
        self, mock_path_cls, mock_guess_type, mock_get_client, mock_retry
    ):
        mock_path_cls.return_value = _make_mock_path()
        mock_guess_type.return_value = ("image/png", None)

        request = httpx.Request("POST", "http://test/")
        response = httpx.Response(401, request=request, text="Unauthorized")
        mock_retry.side_effect = httpx.HTTPStatusError(
            "401", request=request, response=response
        )

        result = await describe_image("/test.png")
        assert "VISION_API_KEY" in result

    @patch("vision_mcp.server._call_with_retry")
    @patch("vision_mcp.server._get_http_client")
    @patch("vision_mcp.server.mimetypes.guess_type")
    @patch("vision_mcp.server.Path")
    async def test_api_400(
        self, mock_path_cls, mock_guess_type, mock_get_client, mock_retry
    ):
        mock_path_cls.return_value = _make_mock_path()
        mock_guess_type.return_value = ("image/png", None)

        request = httpx.Request("POST", "http://test/")
        response = httpx.Response(400, request=request, text="Bad Request")
        mock_retry.side_effect = httpx.HTTPStatusError(
            "400", request=request, response=response
        )

        result = await describe_image("/test.png")
        assert "请求参数错误" in result

    @patch("vision_mcp.server._call_with_retry")
    @patch("vision_mcp.server._get_http_client")
    @patch("vision_mcp.server.mimetypes.guess_type")
    @patch("vision_mcp.server.Path")
    async def test_api_connection_error(
        self, mock_path_cls, mock_guess_type, mock_get_client, mock_retry
    ):
        mock_path_cls.return_value = _make_mock_path()
        mock_guess_type.return_value = ("image/png", None)
        mock_retry.side_effect = httpx.RequestError("Connection refused")

        result = await describe_image("/test.png")
        assert "无法连接到" in result

    # ── API response validation ──

    @patch("vision_mcp.server._call_with_retry")
    @patch("vision_mcp.server._get_http_client")
    @patch("vision_mcp.server.mimetypes.guess_type")
    @patch("vision_mcp.server.Path")
    async def test_empty_choices(
        self, mock_path_cls, mock_guess_type, mock_get_client, mock_retry
    ):
        mock_path_cls.return_value = _make_mock_path()
        mock_guess_type.return_value = ("image/png", None)
        resp = AsyncMock(spec=httpx.Response)
        resp.json.return_value = {"choices": []}
        mock_retry.return_value = resp

        result = await describe_image("/test.png")
        assert "异常响应结构" in result

    @patch("vision_mcp.server._call_with_retry")
    @patch("vision_mcp.server._get_http_client")
    @patch("vision_mcp.server.mimetypes.guess_type")
    @patch("vision_mcp.server.Path")
    async def test_missing_message(
        self, mock_path_cls, mock_guess_type, mock_get_client, mock_retry
    ):
        mock_path_cls.return_value = _make_mock_path()
        mock_guess_type.return_value = ("image/png", None)
        resp = AsyncMock(spec=httpx.Response)
        resp.json.return_value = {"choices": [{"foo": "bar"}]}
        mock_retry.return_value = resp

        result = await describe_image("/test.png")
        assert "异常 message 结构" in result

    @patch("vision_mcp.server._call_with_retry")
    @patch("vision_mcp.server._get_http_client")
    @patch("vision_mcp.server.mimetypes.guess_type")
    @patch("vision_mcp.server.Path")
    async def test_empty_content(
        self, mock_path_cls, mock_guess_type, mock_get_client, mock_retry
    ):
        mock_path_cls.return_value = _make_mock_path()
        mock_guess_type.return_value = ("image/png", None)
        resp = AsyncMock(spec=httpx.Response)
        resp.json.return_value = {"choices": [{"message": {"content": ""}}]}
        mock_retry.return_value = resp

        result = await describe_image("/test.png")
        assert "空内容" in result

    @patch("vision_mcp.server._call_with_retry")
    @patch("vision_mcp.server._get_http_client")
    @patch("vision_mcp.server.mimetypes.guess_type")
    @patch("vision_mcp.server.Path")
    async def test_unexpected_exception(
        self, mock_path_cls, mock_guess_type, mock_get_client, mock_retry
    ):
        mock_path_cls.return_value = _make_mock_path()
        mock_guess_type.return_value = ("image/png", None)
        mock_retry.side_effect = RuntimeError("unexpected")

        result = await describe_image("/test.png")
        assert "未知异常" in result
