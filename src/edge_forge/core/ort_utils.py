"""Centralized ONNX Runtime session creation.

Every component that opens an ``InferenceSession`` goes through this module so
that logging severity, thread counts and graph optimization are consistent. The
native runtime writes its diagnostics to stderr; raising the severity to ERROR
keeps the MCP server's protocol stream quiet while still surfacing real failures
as exceptions.
"""

from __future__ import annotations

import onnxruntime as ort

LOG_SEVERITY_ERROR: int = 3


def create_session(
    model_path: str,
    *,
    provider: str = "CPUExecutionProvider",
    intra_op_threads: int = 0,
    inter_op_threads: int = 1,
    enable_mem_pattern: bool = True,
    graph_optimization_level: ort.GraphOptimizationLevel | None = None,
    log_severity_level: int = LOG_SEVERITY_ERROR,
) -> ort.InferenceSession:
    """Create an ``InferenceSession`` with explicit, uniform options.

    Args:
        model_path: Path to the ONNX artifact.
        provider: Execution provider to request.
        intra_op_threads: Intra-op thread count; zero keeps the runtime default.
        inter_op_threads: Inter-op thread count.
        enable_mem_pattern: Whether to enable the memory pattern optimization.
        graph_optimization_level: Optional graph optimization level.
        log_severity_level: Runtime log severity; defaults to ERROR only.

    Returns:
        A ready to use inference session.

    Raises:
        Exception: Any error raised by ONNX Runtime while loading the artifact.
    """
    options = ort.SessionOptions()
    if intra_op_threads > 0:
        options.intra_op_num_threads = intra_op_threads
    options.inter_op_num_threads = inter_op_threads
    options.enable_mem_pattern = enable_mem_pattern
    options.log_severity_level = log_severity_level
    if graph_optimization_level is not None:
        options.graph_optimization_level = graph_optimization_level
    return ort.InferenceSession(
        model_path,
        sess_options=options,
        providers=[provider],
    )
