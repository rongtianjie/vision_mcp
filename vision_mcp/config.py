"""配置加载与校验。

优先级（从高到低）：
1. 进程环境变量（已存在的 VISION_* 不会被覆盖）
2. CLI 显式指定的 --env-file 文件（override=True）
3. 自动发现的 .env：当前工作目录优先，项目根目录兜底
4. 内置默认值

支持多后端（provider）：默认 provider 由 VISION_API_BASE / VISION_API_KEY /
VISION_MODEL 配置（向后兼容）；命名 provider 通过
VISION_PROVIDER_<NAME>_API_BASE / _API_KEY / _MODEL 配置。
默认使用哪个 provider 由 VISION_PROVIDER 或 CLI --provider 指定。
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger("vision-mcp")

# ── 默认值 ────────────────────────────────────────────────────
DEFAULT_API_BASE = "http://localhost:8000/v1"
DEFAULT_API_KEY = "not-needed"
DEFAULT_MODEL = "qwen-vl-plus"
DEFAULT_MAX_TOKENS = 2000
DEFAULT_TIMEOUT = 120.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_MAX_IMAGE_MB = 20.0  # 单张图片最大体积，超过拒绝
DEFAULT_MAX_UPLOAD_MB = 8.0  # 超过该体积自动压缩后再上传（需 Pillow）
DEFAULT_PROVIDER = "default"

_PROVIDER_PREFIX = "VISION_PROVIDER_"
_PROVIDER_SUFFIXES = ("_API_BASE", "_API_KEY", "_MODEL")
_PROVIDER_NAME_RE = re.compile(r"^[a-z0-9_-]+$")

# 包所在的项目根目录（用于兜底查找 .env）
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ProviderConfig:
    """单个视觉后端（provider）的配置。"""

    name: str
    api_base: str
    api_key: str
    model: str


@dataclass(frozen=True)
class VisionConfig:
    """Server 全部配置，加载完成后不可变。"""

    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    default_provider: str = DEFAULT_PROVIDER
    max_tokens: int = DEFAULT_MAX_TOKENS
    timeout: float = DEFAULT_TIMEOUT
    max_retries: int = DEFAULT_MAX_RETRIES
    max_image_mb: float = DEFAULT_MAX_IMAGE_MB
    max_upload_mb: float = DEFAULT_MAX_UPLOAD_MB
    dotenv_path: Path | None = None

    @property
    def provider(self) -> ProviderConfig:
        """当前默认使用的 provider。"""
        return self.providers[self.default_provider]


def _load_env_files(env_file: str | None) -> Path | None:
    """加载 .env 文件并返回实际使用的路径（未找到返回 None）。"""
    if env_file:
        path = Path(env_file).expanduser().resolve()
        if not path.is_file():
            logger.warning("指定的 --env-file 不存在，忽略：%s", path)
        else:
            load_dotenv(path, override=True)
            logger.info("Loaded env from %s (CLI --env-file)", path)
            return path
        return None

    for candidate in (Path.cwd() / ".env", PROJECT_ROOT / ".env"):
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            logger.info("Loaded env from %s", candidate)
            return candidate

    logger.debug("未找到 .env（已查找 cwd 与项目根目录），将使用环境变量或默认值")
    return None


def _get_int(name: str, default: int) -> int:
    """读取整数环境变量，非法值告警并回退默认值（避免启动崩溃）。"""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("环境变量 %s=%r 不是合法整数，使用默认值 %d", name, raw, default)
        return default


def _get_float(name: str, default: float) -> float:
    """读取浮点环境变量，非法值告警并回退默认值。"""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("环境变量 %s=%r 不是合法数字，使用默认值 %s", name, raw, default)
        return default


def _collect_providers() -> dict[str, ProviderConfig]:
    """收集全部 provider：default（VISION_*）+ 命名 provider（VISION_PROVIDER_<NAME>_*）。"""
    providers: dict[str, ProviderConfig] = {
        DEFAULT_PROVIDER: ProviderConfig(
            name=DEFAULT_PROVIDER,
            api_base=os.environ.get("VISION_API_BASE", DEFAULT_API_BASE),
            api_key=os.environ.get("VISION_API_KEY", DEFAULT_API_KEY),
            model=os.environ.get("VISION_MODEL", DEFAULT_MODEL),
        )
    }

    # VISION_PROVIDER_<NAME>_{API_BASE,API_KEY,MODEL}
    raw: dict[str, dict[str, str]] = {}
    for key, value in os.environ.items():
        if not key.startswith(_PROVIDER_PREFIX):
            continue
        suffix_part = key[len(_PROVIDER_PREFIX) :]
        matched = None
        for field_name, suffix in (
            ("api_base", "_API_BASE"),
            ("api_key", "_API_KEY"),
            ("model", "_MODEL"),
        ):
            if suffix_part.endswith(suffix):
                matched = (field_name, suffix)
                break
        if matched is None:
            logger.warning("忽略无法识别的 provider 变量：%s", key)
            continue
        field_name, suffix = matched
        name = suffix_part[: -len(suffix)].lower()
        if not _PROVIDER_NAME_RE.fullmatch(name):
            logger.warning("跳过非法 provider 名 %r（仅允许字母/数字/-/_）：%s", name, key)
            continue
        raw.setdefault(name, {})[field_name] = value

    for name, fields in sorted(raw.items()):
        providers[name] = ProviderConfig(
            name=name,
            api_base=fields.get("api_base", DEFAULT_API_BASE),
            api_key=fields.get("api_key", DEFAULT_API_KEY),
            model=fields.get("model", DEFAULT_MODEL),
        )
    return providers


def _resolve_default_provider(
    providers: dict[str, ProviderConfig], cli_provider: str | None
) -> str:
    """确定默认 provider 名：CLI --provider > VISION_PROVIDER > default。"""
    wanted = (cli_provider or os.environ.get("VISION_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
    if wanted in providers:
        return wanted
    # 大小写不敏感匹配
    matched = next((name for name in providers if name.lower() == wanted), None)
    if matched:
        return matched
    logger.warning(
        "provider %r 未配置（可用：%s），回退到 %r", wanted, ", ".join(providers), DEFAULT_PROVIDER
    )
    return DEFAULT_PROVIDER


def load_config(env_file: str | None = None, provider: str | None = None) -> VisionConfig:
    """加载全部配置（.env + 环境变量 + 默认值），任何非法值均容错回退。"""
    dotenv_path = _load_env_files(env_file)
    providers = _collect_providers()

    config = VisionConfig(
        providers=providers,
        default_provider=_resolve_default_provider(providers, provider),
        max_tokens=_get_int("VISION_MAX_TOKENS", DEFAULT_MAX_TOKENS),
        timeout=_get_float("VISION_TIMEOUT", DEFAULT_TIMEOUT),
        max_retries=_get_int("VISION_MAX_RETRIES", DEFAULT_MAX_RETRIES),
        max_image_mb=_get_float("VISION_MAX_IMAGE_MB", DEFAULT_MAX_IMAGE_MB),
        max_upload_mb=_get_float("VISION_MAX_UPLOAD_MB", DEFAULT_MAX_UPLOAD_MB),
        dotenv_path=dotenv_path,
    )
    log_config_summary(config)
    return config


def log_config_summary(config: VisionConfig) -> None:
    """启动时打印配置摘要（不含 API Key），便于排障。"""
    p = config.provider
    logger.info(
        "配置摘要: provider=%s model=%s api_base=%s max_tokens=%d timeout=%.0fs "
        "max_retries=%d 单图上限=%.0fMB 自动压缩阈值=%.0fMB",
        p.name,
        p.model,
        p.api_base,
        config.max_tokens,
        config.timeout,
        config.max_retries,
        config.max_image_mb,
        config.max_upload_mb,
    )
    extra = [n for n in config.providers if n != config.default_provider]
    if extra:
        logger.info("其他可用 provider: %s", ", ".join(sorted(extra)))
    if config.dotenv_path is None:
        logger.warning(
            "未找到 .env 配置文件，使用默认配置。"
            "可在项目根目录创建 .env（参考 .env.example），或启动时用 --env-file 指定。"
        )
