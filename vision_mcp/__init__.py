"""
Vision MCP Server — expose describe_image tool so LLMs without vision
can understand image content via external multimodal APIs.

支持本地图片路径、网络 URL、多图对比与自动压缩，详见 README。
"""

from vision_mcp.server import main

__all__ = ["main"]
