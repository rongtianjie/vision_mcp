#!/usr/bin/env python3
"""
Vision MCP Server — expose describe_image tool so LLMs without vision can read images.

CLI 用法:
    vision-mcp [--env-file PATH] [--provider NAME] [--version]

配置通过项目根目录 .env / 环境变量管理（见 vision_mcp.config 与 .env.example），
无需在 MCP 注册时携带 env 字段。
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version

from mcp.server.fastmcp import FastMCP

import vision_mcp.tools as tools
from vision_mcp.config import VisionConfig, load_config
from vision_mcp.http_client import ServerRuntime

# ── Logging ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vision-mcp")

try:
    __version__ = version("vision-mcp")
except PackageNotFoundError:  # 未安装（如源码直接运行）时兜底
    __version__ = "0.0.0"


def build_mcp(config: VisionConfig) -> FastMCP:
    """组装 FastMCP 实例：lifespan 负责 HTTP 客户端的创建与关闭。"""

    @asynccontextmanager
    async def lifespan(server: FastMCP):
        runtime = ServerRuntime(config)
        tools.set_runtime(runtime)
        try:
            yield runtime
        finally:
            await runtime.client.aclose()
            tools.set_runtime(None)
            logger.info("vision-mcp 已关闭，HTTP 连接池已释放")

    mcp = FastMCP("vision-mcp", lifespan=lifespan)
    tools.register_tools(mcp)
    return mcp


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="vision-mcp",
        description="为缺少多模态能力的 LLM 提供图片理解能力的 MCP Server。",
    )
    parser.add_argument(
        "--env-file",
        metavar="PATH",
        help="从指定 .env 文件加载配置（优先级最高，可替代自动发现的 .env）",
    )
    parser.add_argument(
        "--provider",
        metavar="NAME",
        help="默认使用的 provider 名称（覆盖 VISION_PROVIDER 环境变量）",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    args = parser.parse_args(argv)

    config = load_config(env_file=args.env_file, provider=args.provider)
    mcp = build_mcp(config)
    asyncio.run(mcp.run_stdio_async())


if __name__ == "__main__":
    main()
