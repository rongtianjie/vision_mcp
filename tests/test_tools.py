"""Tests for vision_mcp.tools."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from vision_mcp.config import ProviderConfig, VisionConfig
from vision_mcp.http_client import ServerRuntime, VisionAPIError
from vision_mcp.image_utils import VisionImageError
from vision_mcp.tools import describe_image, set_runtime, vision_ping

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


def _make_config() -> VisionConfig:
    return VisionConfig(
        providers={
            "default": ProviderConfig("default", "http://test/v1", "key", "model-a"),
            "local": ProviderConfig("local", "http://local/v1", "k2", "llava:13b"),
        },
        max_tokens=2000,
        timeout=30.0,
        max_retries=2,
        max_image_mb=20.0,
        max_upload_mb=8.0,
    )


@pytest.fixture(autouse=True)
def runtime():
    """注入模块级 runtime：chat/download_image 为 mock，避免真实网络请求。"""
    rt = ServerRuntime(_make_config())
    rt.client = AsyncMock()
    set_runtime(rt)
    yield rt
    set_runtime(None)


def _source_ok(data: bytes = PNG_BYTES, mime: str = "image/png"):
    return (data, mime)


def _chat_messages(call) -> list[dict]:
    """从 chat 调用中取出 content 列表（chat(provider, messages, max_tokens)）。"""
    return call.args[1][0]["content"]


class TestDescribeImageSources:
    """图片来源参数：image_path / image_url / images 三种入口。"""

    @patch("vision_mcp.tools.load_image_source")
    async def test_no_source_returns_error(self, mock_load):
        result = await describe_image()
        assert "请提供" in result
        mock_load.assert_not_called()

    @patch("vision_mcp.tools.load_image_source")
    async def test_single_local_path(self, mock_load, runtime):
        mock_load.return_value = _source_ok()
        runtime.client.chat.return_value = "描述内容"

        result = await describe_image(image_path="/tmp/a.png")
        assert result == "描述内容"
        mock_load.assert_awaited_once()
        src_kwargs = mock_load.await_args.kwargs
        assert src_kwargs["max_image_mb"] == 20.0
        runtime.client.chat.assert_awaited_once()

    @patch("vision_mcp.tools.load_image_source")
    async def test_url_entry(self, mock_load, runtime):
        mock_load.return_value = _source_ok()
        runtime.client.chat.return_value = "ok"
        await describe_image(image_url="https://example.com/a.png")
        assert mock_load.await_args.args[0] == "https://example.com/a.png"

    @patch("vision_mcp.tools.load_image_source")
    async def test_multiple_images(self, mock_load, runtime):
        mock_load.side_effect = [
            _source_ok(b"aaa", "image/png"),
            _source_ok(b"bbb", "image/jpeg"),
        ]
        runtime.client.chat.return_value = "对比结果"

        result = await describe_image(images=["/tmp/a.png", "/tmp/b.jpg"], prompt="对比结果")
        assert result == "对比结果"
        assert mock_load.await_count == 2

        content = _chat_messages(runtime.client.chat.await_args)
        assert len(content) == 3  # 2 图 + 1 文本
        assert content[0]["type"] == "image_url"
        assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
        assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        assert content[2]["type"] == "text"
        assert content[2]["text"] == "对比结果"

    @patch("vision_mcp.tools.load_image_source")
    async def test_merged_sources(self, mock_load, runtime):
        """image_path + image_url + images 应全部合并。"""
        mock_load.return_value = _source_ok()
        runtime.client.chat.return_value = "ok"
        await describe_image(
            image_path="/tmp/a.png",
            image_url="https://example.com/b.png",
            images=["/tmp/c.png"],
        )
        assert mock_load.await_count == 3
        sources = [c.args[0] for c in mock_load.await_args_list]
        assert sources == ["/tmp/a.png", "https://example.com/b.png", "/tmp/c.png"]

    @patch("vision_mcp.tools.load_image_source")
    async def test_load_error_returns_message(self, mock_load, runtime):
        mock_load.side_effect = VisionImageError("错误：图片文件不存在：/x.png")
        result = await describe_image(image_path="/x.png")
        assert "图片文件不存在" in result
        runtime.client.chat.assert_not_called()

    @patch("vision_mcp.tools.load_image_source")
    async def test_default_prompt(self, mock_load, runtime):
        mock_load.return_value = _source_ok()
        runtime.client.chat.return_value = "ok"
        await describe_image(image_path="/tmp/a.png")
        content = _chat_messages(runtime.client.chat.await_args)
        assert content[-1]["text"] == "请详细描述这张图片的内容。"


class TestDescribeImagePayload:
    """max_tokens / provider 参数与 chat 调用。"""

    @patch("vision_mcp.tools.load_image_source")
    async def test_max_tokens_override(self, mock_load, runtime):
        mock_load.return_value = _source_ok()
        runtime.client.chat.return_value = "ok"
        await describe_image(image_path="/tmp/a.png", max_tokens=4096)
        assert runtime.client.chat.await_args.args[2] == 4096

    @patch("vision_mcp.tools.load_image_source")
    async def test_max_tokens_zero_falls_back(self, mock_load, runtime):
        mock_load.return_value = _source_ok()
        runtime.client.chat.return_value = "ok"
        await describe_image(image_path="/tmp/a.png", max_tokens=0)
        assert runtime.client.chat.await_args.args[2] == 2000

    @patch("vision_mcp.tools.load_image_source")
    async def test_named_provider(self, mock_load, runtime):
        mock_load.return_value = _source_ok()
        runtime.client.chat.return_value = "ok"
        await describe_image(image_path="/tmp/a.png", provider="local")
        provider = runtime.client.chat.await_args.args[0]
        assert provider.name == "local"
        assert provider.model == "llava:13b"

    @patch("vision_mcp.tools.load_image_source")
    async def test_unknown_provider(self, mock_load, runtime):
        mock_load.return_value = _source_ok()
        result = await describe_image(image_path="/tmp/a.png", provider="nope")
        assert "未知 provider" in result
        runtime.client.chat.assert_not_called()

    @patch("vision_mcp.tools.load_image_source")
    async def test_default_provider_used(self, mock_load, runtime):
        mock_load.return_value = _source_ok()
        runtime.client.chat.return_value = "ok"
        await describe_image(image_path="/tmp/a.png")
        provider = runtime.client.chat.await_args.args[0]
        assert provider.name == "default"


class TestDescribeImageErrors:
    """API 失败应转成用户可读消息而非异常。"""

    @patch("vision_mcp.tools.load_image_source")
    async def test_api_error_message(self, mock_load, runtime):
        mock_load.return_value = _source_ok()
        runtime.client.chat.side_effect = VisionAPIError(
            "错误：API 认证失败，请检查 VISION_API_KEY 配置。"
        )
        result = await describe_image(image_path="/tmp/a.png")
        assert "VISION_API_KEY" in result

    @patch("vision_mcp.tools.load_image_source")
    async def test_unexpected_exception(self, mock_load, runtime):
        mock_load.return_value = _source_ok()
        runtime.client.chat.side_effect = RuntimeError("boom")
        result = await describe_image(image_path="/tmp/a.png")
        assert "未知异常" in result
        assert "RuntimeError" in result


class TestVisionPing:
    async def test_default(self, runtime):
        result = await vision_ping()
        assert result.startswith("pong: ping")
        assert "provider=default" in result
        assert "model=model-a" in result

    async def test_custom_msg(self, runtime):
        assert (await vision_ping("hello")).startswith("pong: hello")

    async def test_probe_api_success(self, runtime):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"data": [{"id": "model-1"}, {"id": "model-2"}]}
        runtime.client.client.get = AsyncMock(return_value=resp)

        result = await vision_ping(probe_api=True)
        assert "API 连通正常" in result
        assert "model-1" in result
        runtime.client.client.get.assert_awaited_once()

    async def test_probe_api_http_error(self, runtime):
        resp = MagicMock()
        resp.status_code = 500
        resp.text = "server error"
        runtime.client.client.get = AsyncMock(return_value=resp)

        result = await vision_ping(probe_api=True)
        assert "HTTP 500" in result

    async def test_probe_api_connection_error(self, runtime):
        runtime.client.client.get = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
        result = await vision_ping(probe_api=True)
        assert "无法连接" in result
