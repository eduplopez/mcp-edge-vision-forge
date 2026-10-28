"""mcp-edge-vision-forge public package.

The package exposes an MCP server and a Typer CLI on top of a reusable core that
exports PyTorch models to ONNX, quantizes them, measures numerical drift and
benchmarks latency on the current host.
"""

from __future__ import annotations

from edge_forge.config import ForgeSettings, get_settings
from edge_forge.errors import EdgeForgeError

__version__ = "0.1.0"
__all__ = ["EdgeForgeError", "ForgeSettings", "__version__", "get_settings"]
