"""Tests for vision_mcp.http_client."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from vision_mcp.config import ProviderConfig, VisionConfig
from vision_mcp.http_client import (
    VisionAPIError,
    VisionClient,
    _backoff_seconds,
    _retry_after_seconds,
    extract_content,
)

PROVIDER = ProviderConfig(name="default", api_base="http://test/v1", api_key="key", model="model-a")


def _cfg(**overrides) -> VisionConfig:
    values = dict(
        providers={"default": PROVIDER},
        max_tokens=2000,
        timeout=30.0,
        max_retries=2,
        max_image_mb=20.0,
        max_upload_mb=8.0,
    )
    values.update(overrides)
    return VisionConfig(**values)


def _client_with_mock_post(
    mock_client: AsyncMock, config: VisionConfig | None = None
) -> VisionClient:
    client = VisionClient(config or _cfg())
    client._client = mock_client  # 直接注入 mock，跳过惰性创建
    return client


def _http_response(status: int, json_data: dict | None = None, text: str = "") -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status
    resp.json.return_value = json_data if json_data is not None else {}
    resp.text = text
    resp.headers = {}
    return resp


def _ok_resp(content: str = "描述内容") -> MagicMock:
    return _http_response(200, {"choices": [{"message": {"content": content}}]})


# ── 退避与 Retry-After ────────────────────────────────────────


class TestBackoff:
    @patch("vision_mcp.http_client.random.uniform", side_effect=lambda a, b: b)
    def test_exponential(self, mock_uniform):
        """uniform 取上限时：2s → 4s → 8s 指数增长。"""
        assert _backoff_seconds(0) == 2.0
        assert _backoff_seconds(1) == 4.0
        assert _backoff_seconds(2) == 8.0

    @patch("vision_mcp.http_client.random.uniform", side_effect=lambda a, b: b)
    def test_capped(self, mock_uniform):
        assert _backoff_seconds(10) == 60.0

    @patch("vision_mcp.http_client.random.uniform", side_effect=lambda a, b: a)
    def test_jitter_can_be_zero(self, mock_uniform):
        """full jitter：等待时间可为 [0, ceiling) 内任意值，包括 0。"""
        assert _backoff_seconds(0) == 0.0


class TestRetryAfter:
    def test_seconds(self):
        assert _retry_after_seconds({"retry-after": "3"}) == 3.0

    def test_capped(self):
        assert _retry_after_seconds({"retry-after": "300"}) == 60.0

    def test_invalid(self):
        assert _retry_after_seconds({"retry-after": "abc"}) is None

    def test_missing(self):
        assert _retry_after_seconds({}) is None


# ── 响应解析 ───────────────────────────────────────────────────


class TestExtractContent:
    def test_plain_string(self):
        assert extract_content({"choices": [{"message": {"content": "ok"}}]}) == "ok"

    def test_content_list(self):
        data = {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "第一段"},
                            {"type": "image_url", "image_url": {"url": "..."}},
                            {"type": "text", "text": "第二段"},
                        ]
                    }
                }
            ]
        }
        assert extract_content(data) == "第一段\n第二段"

    def test_content_list_of_strings(self):
        data = {"choices": [{"message": {"content": ["a", "b"]}}]}
        assert extract_content(data) == "a\nb"

    def test_empty_choices(self):
        assert extract_content({"choices": []}) is None

    def test_missing_choices(self):
        assert extract_content({}) is None

    def test_missing_message(self):
        assert extract_content({"choices": [{"foo": "bar"}]}) is None

    def test_missing_content(self):
        assert extract_content({"choices": [{"message": {}}]}) is None

    def test_bad_content_type(self):
        assert extract_content({"choices": [{"message": {"content": 123}}]}) is None


# ── chat 调用与重试 ────────────────────────────────────────────


class TestChat:
    async def test_success(self):
        client = _client_with_mock_post(AsyncMock(post=AsyncMock(return_value=_ok_resp("结果"))))
        result = await client.chat(PROVIDER, [{"role": "user", "content": "x"}], 100)
        assert result == "结果"
        client._client.post.assert_called_once()
        _, kwargs = client._client.post.call_args
        assert kwargs["json"]["model"] == "model-a"
        assert kwargs["json"]["max_tokens"] == 100
        assert kwargs["headers"]["Authorization"] == "Bearer key"

    async def test_retry_on_429_then_success(self):
        resp_429 = _http_response(429)
        mock_post = AsyncMock(side_effect=[resp_429, _ok_resp("ok")])
        client = _client_with_mock_post(AsyncMock(post=mock_post))
        with patch("asyncio.sleep", new=AsyncMock()) as mock_sleep:
            result = await client.chat(PROVIDER, [], 100)
        assert result == "ok"
        assert mock_post.call_count == 2
        mock_sleep.assert_awaited_once()

    async def test_retry_uses_retry_after_header(self):
        resp_503 = _http_response(503)
        resp_503.headers = {"retry-after": "5"}
        mock_post = AsyncMock(side_effect=[resp_503, _ok_resp("ok")])
        client = _client_with_mock_post(AsyncMock(post=mock_post))
        with patch("asyncio.sleep", new=AsyncMock()) as mock_sleep:
            await client.chat(PROVIDER, [], 100)
        mock_sleep.assert_awaited_once_with(5.0)

    async def test_retry_on_timeout_then_success(self):
        mock_post = AsyncMock(side_effect=[httpx.TimeoutException("slow"), _ok_resp("ok")])
        client = _client_with_mock_post(AsyncMock(post=mock_post))
        with patch("asyncio.sleep", new=AsyncMock()):
            result = await client.chat(PROVIDER, [], 100)
        assert result == "ok"
        assert mock_post.call_count == 2

    async def test_gives_up_after_retries(self):
        mock_post = AsyncMock(return_value=_http_response(503))
        client = _client_with_mock_post(AsyncMock(post=mock_post), _cfg(max_retries=2))
        with patch("asyncio.sleep", new=AsyncMock()):
            with pytest.raises(VisionAPIError, match="500|503"):
                await client.chat(PROVIDER, [], 100)
        assert mock_post.call_count == 3  # 1 次原始 + 2 次重试

    async def test_timeout_gives_up(self):
        mock_post = AsyncMock(side_effect=httpx.TimeoutException("slow"))
        client = _client_with_mock_post(AsyncMock(post=mock_post))
        with patch("asyncio.sleep", new=AsyncMock()):
            with pytest.raises(VisionAPIError, match="超时"):
                await client.chat(PROVIDER, [], 100)
        assert mock_post.call_count == 3

    async def test_400_not_retried(self):
        mock_post = AsyncMock(return_value=_http_response(400, text="bad"))
        client = _client_with_mock_post(AsyncMock(post=mock_post))
        with pytest.raises(VisionAPIError, match="400"):
            await client.chat(PROVIDER, [], 100)
        mock_post.assert_called_once()

    async def test_401_message(self):
        mock_post = AsyncMock(return_value=_http_response(401, text="unauth"))
        client = _client_with_mock_post(AsyncMock(post=mock_post))
        with pytest.raises(VisionAPIError, match="VISION_API_KEY"):
            await client.chat(PROVIDER, [], 100)

    async def test_429_exhausted_message(self):
        mock_post = AsyncMock(return_value=_http_response(429))
        client = _client_with_mock_post(AsyncMock(post=mock_post), _cfg(max_retries=0))
        with pytest.raises(VisionAPIError, match="过于频繁"):
            await client.chat(PROVIDER, [], 100)

    async def test_non_json_response(self):
        resp = _http_response(200)
        resp.json.side_effect = ValueError("not json")
        client = _client_with_mock_post(AsyncMock(post=AsyncMock(return_value=resp)))
        with pytest.raises(VisionAPIError, match="非 JSON"):
            await client.chat(PROVIDER, [], 100)

    async def test_empty_content(self):
        client = _client_with_mock_post(AsyncMock(post=AsyncMock(return_value=_ok_resp("  "))))
        with pytest.raises(VisionAPIError, match="空内容"):
            await client.chat(PROVIDER, [], 100)

    async def test_content_list_joined(self):
        data = {
            "choices": [
                {
                    "message": {
                        "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
                    }
                }
            ]
        }
        client = _client_with_mock_post(
            AsyncMock(post=AsyncMock(return_value=_http_response(200, data)))
        )
        result = await client.chat(PROVIDER, [], 100)
        assert result == "a\nb"


# ── 生命周期与下载 ─────────────────────────────────────────────


class TestClientLifecycle:
    async def test_lazy_create_and_close(self):
        client = VisionClient(_cfg())
        assert client._client is None
        _ = client.client  # 触发创建
        assert client._client is not None
        await client.aclose()
        assert client._client is None

    async def test_aclose_idempotent(self):
        client = VisionClient(_cfg())
        await client.aclose()
        await client.aclose()  # 不抛异常


class TestDownloadImage:
    def _mock_stream(self, chunks: list[bytes], headers: dict | None = None) -> MagicMock:
        async def _iter():
            for c in chunks:
                yield c

        resp = MagicMock()
        resp.status_code = 200
        resp.headers = headers or {}
        resp.raise_for_status.return_value = None
        resp.aiter_bytes.return_value = _iter()
        stream_cm = AsyncMock()  # async context manager（__aenter__ awaitable）
        stream_cm.__aenter__.return_value = resp
        return stream_cm

    def _client_with_stream(self, stream_cm) -> VisionClient:
        # 注意：stream() 不 await，需用 MagicMock（AsyncMock 的方法会返回 coroutine）
        mock_post = MagicMock()
        mock_post.stream.return_value = stream_cm
        return _client_with_mock_post(mock_post)

    async def test_download_success(self):
        client = self._client_with_stream(
            self._mock_stream([b"ab", b"cd"], {"content-type": "image/png"})
        )
        data, mime = await client.download_image("https://x/a.png", max_bytes=1024)
        assert data == b"abcd"
        assert mime == "image/png"

    async def test_download_oversize_aborts(self):
        client = self._client_with_stream(self._mock_stream([b"x" * 100] * 5))
        with pytest.raises(VisionAPIError, match="大小限制"):
            await client.download_image("https://x/big.png", max_bytes=250)
