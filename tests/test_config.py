"""Tests for vision_mcp.config."""

from __future__ import annotations

import os

import pytest

from vision_mcp import config
from vision_mcp.config import (
    DEFAULT_API_BASE,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    load_config,
)

# 真实的 _load_env_files（autouse fixture 会临时替换为 no-op）
_REAL_LOAD_ENV_FILES = config._load_env_files


@pytest.fixture(autouse=True)
def _isolated_env():
    """每个测试前后清空 VISION_* 环境变量，并禁用自动 .env 发现。"""
    for key in list(os.environ):
        if key.startswith("VISION_"):
            del os.environ[key]
    config._load_env_files = lambda env_file=None: None
    yield
    for key in list(os.environ):
        if key.startswith("VISION_"):
            del os.environ[key]
    config._load_env_files = _REAL_LOAD_ENV_FILES


class TestDefaults:
    """无任何配置时应使用内置默认值。"""

    def test_defaults(self):
        cfg = load_config()
        assert cfg.default_provider == DEFAULT_PROVIDER
        assert cfg.provider.api_base == DEFAULT_API_BASE
        assert cfg.provider.api_key == "not-needed"
        assert cfg.provider.model == DEFAULT_MODEL
        assert cfg.max_tokens == DEFAULT_MAX_TOKENS
        assert cfg.timeout == 120.0
        assert cfg.max_retries == DEFAULT_MAX_RETRIES
        assert cfg.dotenv_path is None

    def test_returns_vision_config(self):
        assert isinstance(load_config(), config.VisionConfig)


class TestEnvParsing:
    """环境变量应覆盖默认值；非法值应容错回退而不是崩溃。"""

    def test_env_override(self):
        os.environ["VISION_API_BASE"] = "https://example.com/v1"
        os.environ["VISION_API_KEY"] = "sk-test"
        os.environ["VISION_MODEL"] = "model-x"
        cfg = load_config()
        assert cfg.provider.api_base == "https://example.com/v1"
        assert cfg.provider.api_key == "sk-test"
        assert cfg.provider.model == "model-x"

    def test_invalid_max_tokens_falls_back(self):
        """非法整数不应导致 server 启动崩溃（回归：旧版 import 时 int() 直接抛 ValueError）。"""
        os.environ["VISION_MAX_TOKENS"] = "abc"
        assert load_config().max_tokens == DEFAULT_MAX_TOKENS

    def test_invalid_timeout_falls_back(self):
        os.environ["VISION_TIMEOUT"] = "fast"
        assert load_config().timeout == 120.0

    def test_invalid_retries_falls_back(self):
        os.environ["VISION_MAX_RETRIES"] = "-1x"
        assert load_config().max_retries == DEFAULT_MAX_RETRIES

    def test_empty_value_uses_default(self):
        os.environ["VISION_MAX_TOKENS"] = ""
        assert load_config().max_tokens == DEFAULT_MAX_TOKENS


class TestProviders:
    """命名 provider（VISION_PROVIDER_<NAME>_*）解析与选择。"""

    def test_named_provider(self):
        os.environ["VISION_PROVIDER_LOCAL_API_BASE"] = "http://localhost:11434/v1"
        os.environ["VISION_PROVIDER_LOCAL_API_KEY"] = "k"
        os.environ["VISION_PROVIDER_LOCAL_MODEL"] = "llava:13b"
        cfg = load_config()
        assert set(cfg.providers) == {"default", "local"}
        assert cfg.providers["local"].api_base == "http://localhost:11434/v1"
        assert cfg.providers["local"].model == "llava:13b"

    def test_default_provider_selection(self):
        os.environ["VISION_PROVIDER_LOCAL_MODEL"] = "llava:13b"
        os.environ["VISION_PROVIDER"] = "local"
        assert load_config().default_provider == "local"

    def test_unknown_provider_falls_back(self):
        os.environ["VISION_PROVIDER"] = "nope"
        assert load_config().default_provider == DEFAULT_PROVIDER

    def test_cli_provider_wins_over_env(self):
        os.environ["VISION_PROVIDER_LOCAL_MODEL"] = "llava:13b"
        os.environ["VISION_PROVIDER"] = "local"
        assert load_config(provider="default").default_provider == "default"

    def test_invalid_provider_name_ignored(self):
        os.environ["VISION_PROVIDER_BAD NAME_API_BASE"] = "http://x/v1"
        assert "bad name" not in load_config().providers

    def test_unknown_provider_var_ignored(self):
        os.environ["VISION_PROVIDER_FOO_OTHER"] = "whatever"
        assert "foo" not in load_config().providers


class TestEnvFileLoading:
    """--env-file 显式指定时应优先加载且覆盖环境变量。"""

    def test_cli_env_file(self, tmp_path, monkeypatch):
        env_file = tmp_path / "custom.env"
        env_file.write_text("VISION_MODEL=from-file\nVISION_MAX_TOKENS=1234\n", encoding="utf-8")
        monkeypatch.setattr(config, "_load_env_files", _REAL_LOAD_ENV_FILES)

        cfg = load_config(env_file=str(env_file))
        assert cfg.dotenv_path == env_file.resolve()
        assert cfg.provider.model == "from-file"
        assert cfg.max_tokens == 1234

    def test_cli_env_file_overrides_process_env(self, tmp_path, monkeypatch):
        os.environ["VISION_MODEL"] = "from-env"
        env_file = tmp_path / "custom.env"
        env_file.write_text("VISION_MODEL=from-file\n", encoding="utf-8")
        monkeypatch.setattr(config, "_load_env_files", _REAL_LOAD_ENV_FILES)

        cfg = load_config(env_file=str(env_file))
        assert cfg.provider.model == "from-file"

    def test_missing_cli_env_file_ignored(self, tmp_path):
        cfg = load_config(env_file=str(tmp_path / "nope.env"))
        assert cfg.dotenv_path is None
        assert cfg.provider.model == DEFAULT_MODEL


class TestAutoDiscovery:
    """自动发现 .env：cwd 优先，项目根兜底，且不覆盖已有环境变量。"""

    def test_cwd_preferred_over_project_root(self, tmp_path, monkeypatch):
        cwd_env = tmp_path / "cwd"
        cwd_env.mkdir()
        (cwd_env / ".env").write_text("VISION_MODEL=from-cwd\n", encoding="utf-8")
        root_env = tmp_path / "root"
        root_env.mkdir()
        (root_env / ".env").write_text("VISION_MODEL=from-root\n", encoding="utf-8")
        monkeypatch.setattr(config, "PROJECT_ROOT", root_env)
        monkeypatch.chdir(cwd_env)
        monkeypatch.setattr(config, "_load_env_files", _REAL_LOAD_ENV_FILES)

        cfg = load_config()
        assert cfg.provider.model == "from-cwd"
        assert cfg.dotenv_path == (cwd_env / ".env").resolve()

    def test_project_root_fallback(self, tmp_path, monkeypatch):
        root_env = tmp_path / "root"
        root_env.mkdir()
        (root_env / ".env").write_text("VISION_MODEL=from-root\n", encoding="utf-8")
        monkeypatch.setattr(config, "PROJECT_ROOT", root_env)
        monkeypatch.chdir(tmp_path)  # cwd 下没有 .env
        monkeypatch.setattr(config, "_load_env_files", _REAL_LOAD_ENV_FILES)

        cfg = load_config()
        assert cfg.provider.model == "from-root"
        assert cfg.dotenv_path == (root_env / ".env").resolve()

    def test_process_env_wins_over_auto_dotenv(self, tmp_path, monkeypatch):
        """自动发现的 .env 使用 override=False：已有环境变量优先。"""
        os.environ["VISION_MODEL"] = "from-env"
        root_env = tmp_path / "root"
        root_env.mkdir()
        (root_env / ".env").write_text("VISION_MODEL=from-root\n", encoding="utf-8")
        monkeypatch.setattr(config, "PROJECT_ROOT", root_env)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(config, "_load_env_files", _REAL_LOAD_ENV_FILES)

        assert load_config().provider.model == "from-env"


class TestLogSummary:
    """配置摘要不应泄露 API Key。"""

    def test_summary_hides_api_key(self, caplog):
        os.environ["VISION_API_KEY"] = "sk-super-secret"
        os.environ["VISION_API_BASE"] = "https://secret-host/v1"
        with caplog.at_level("INFO"):
            load_config()
        logs = "\n".join(r.getMessage() for r in caplog.records)
        assert "sk-super-secret" not in logs
        assert "配置摘要" in logs
