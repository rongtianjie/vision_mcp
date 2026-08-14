"""Tests for vision_mcp.server (组装与 CLI 入口)。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vision_mcp.config import ProviderConfig, VisionConfig
from vision_mcp.server import build_mcp, main

TEST_CONFIG = VisionConfig(
    providers={
        "default": ProviderConfig("default", "http://test/v1", "key", "model-a"),
        "local": ProviderConfig("local", "http://local/v1", "k2", "llava:13b"),
    },
)


class TestBuildMcp:
    async def test_registers_tools(self):
        mcp = build_mcp(TEST_CONFIG)
        tool_names = {t.name for t in await mcp.list_tools()}
        assert tool_names == {"describe_image", "vision_ping"}

    def test_has_lifespan(self):
        mcp = build_mcp(TEST_CONFIG)
        assert mcp.settings.lifespan is not None


class TestMain:
    def test_parses_cli_args(self, monkeypatch):
        """--provider / --env-file 应传入 load_config 并用于构建 server。"""
        monkeypatch.setenv("VISION_PROVIDER_LOCAL_MODEL", "llava:13b")
        with (
            patch("vision_mcp.server.build_mcp") as mock_build,
            patch("vision_mcp.server.asyncio.run"),
        ):
            mock_build.return_value = MagicMock()
            main(["--provider", "local", "--env-file", "/tmp/nonexistent.env"])

        mock_build.assert_called_once()
        config = mock_build.call_args.args[0]
        assert config.default_provider == "local"
        assert config.dotenv_path is None  # 文件不存在时忽略

    def test_version_exits(self):
        with pytest.raises(SystemExit):
            main(["--version"])

    def test_default_args(self):
        with (
            patch("vision_mcp.server.build_mcp") as mock_build,
            patch("vision_mcp.server.asyncio.run"),
        ):
            mock_build.return_value = MagicMock()
            main([])
        config = mock_build.call_args.args[0]
        assert config.default_provider == "default"
