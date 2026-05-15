"""
Vision MCP Server — expose describe_image tool so LLMs without vision
can understand image content via external multimodal APIs.
"""

from vision_mcp.server import main

__all__ = ["main"]
