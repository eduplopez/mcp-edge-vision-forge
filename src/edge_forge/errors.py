"""Exception hierarchy for mcp-edge-vision-forge.

All errors raised by the public API derive from :class:`EdgeForgeError` so that
callers (CLI, MCP server, embedding applications) can handle failures with a
single catch clause when appropriate.
"""

from __future__ import annotations


class EdgeForgeError(Exception):
    """Base class for every error produced by edge_forge."""


class ConfigurationError(EdgeForgeError):
    """Raised when the runtime configuration is invalid or unusable."""


class ModelNotFoundError(EdgeForgeError):
    """Raised when a model identifier or artifact path cannot be resolved."""


class ModelLoadError(EdgeForgeError):
    """Raised when a PyTorch checkpoint cannot be loaded safely."""


class PathAccessError(EdgeForgeError):
    """Raised when a path escapes the roots permitted for agent driven access."""


class ExportError(EdgeForgeError):
    """Raised when PyTorch to ONNX export fails."""


class UnsupportedOpsetError(ExportError):
    """Raised when the requested ONNX opset is outside the supported range."""


class OnnxValidationError(EdgeForgeError):
    """Raised when an ONNX artifact fails structural or numerical validation."""


class QuantizationError(EdgeForgeError):
    """Raised when a quantization strategy cannot be applied."""


class CalibrationError(QuantizationError):
    """Raised when calibration data cannot be produced or read."""


class DriftAnalysisError(EdgeForgeError):
    """Raised when reference and candidate outputs cannot be compared."""


class BenchmarkError(EdgeForgeError):
    """Raised when a benchmark cannot be executed."""


class TuningError(EdgeForgeError):
    """Raised when the autonomous search cannot produce a usable result."""


class ReportError(EdgeForgeError):
    """Raised when a report cannot be rendered or persisted."""


class RunNotFoundError(EdgeForgeError):
    """Raised when a persisted tuning run cannot be found."""
