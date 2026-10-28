"""Hard limits that keep local and agent driven workloads bounded.

The MCP server is a long running process reachable by an autonomous agent. Every
parameter that can multiply CPU time, memory or disk usage is validated against
one of these constants so that a single call cannot exhaust the host.
"""

from __future__ import annotations

import math

MAX_BENCHMARK_ITERATIONS: int = 1000
MAX_WARMUP_RUNS: int = 1000
MAX_INPUT_ELEMENTS: int = 25_000_000
MAX_TUNING_CANDIDATES: int = 64
MAX_CALIBRATION_SAMPLES: int = 4096
MAX_DRIFT_SAMPLES: int = 512
MAX_MODEL_WIDTH: int = 512
MAX_MODEL_DEPTH: int = 32
MAX_MODEL_CLASSES: int = 4096


def validate_input_shape(shape: tuple[int, ...], *, label: str = "input_shape") -> tuple[int, ...]:
    """Validate a tensor shape and bound its element count.

    Args:
        shape: Candidate dimensions.
        label: Name used in error messages.

    Returns:
        The validated shape.

    Raises:
        ValueError: If the shape is empty, has non-positive dimensions or
            exceeds :data:`MAX_INPUT_ELEMENTS`.
    """
    if not shape:
        raise ValueError(f"{label} must contain at least one dimension")
    if any(dimension <= 0 for dimension in shape):
        raise ValueError(f"{label} dimensions must be strictly positive")
    elements = math.prod(shape)
    if elements > MAX_INPUT_ELEMENTS:
        limit_mb = MAX_INPUT_ELEMENTS * 4 / (1024 * 1024)
        raise ValueError(
            f"{label} contains {elements:,} elements, exceeding the limit of "
            f"{MAX_INPUT_ELEMENTS:,} (about {limit_mb:.0f} MiB in float32)"
        )
    return tuple(shape)


def validate_iterations(iterations: int, *, label: str = "iterations") -> int:
    """Validate a benchmark iteration count against the hard upper bound."""
    if iterations < 1:
        raise ValueError(f"{label} must be at least 1")
    if iterations > MAX_BENCHMARK_ITERATIONS:
        raise ValueError(
            f"{label} ({iterations}) exceeds the maximum of {MAX_BENCHMARK_ITERATIONS}"
        )
    return iterations


def validate_warmup_runs(warmup_runs: int, *, label: str = "warmup_runs") -> int:
    """Validate a warmup count against the hard upper bound."""
    if warmup_runs < 0:
        raise ValueError(f"{label} cannot be negative")
    if warmup_runs > MAX_WARMUP_RUNS:
        raise ValueError(f"{label} ({warmup_runs}) exceeds the maximum of {MAX_WARMUP_RUNS}")
    return warmup_runs


def validate_num_samples(num_samples: int, *, label: str = "num_samples") -> int:
    """Validate a drift sample count against the hard upper bound."""
    if num_samples < 1:
        raise ValueError(f"{label} must be at least 1")
    if num_samples > MAX_DRIFT_SAMPLES:
        raise ValueError(f"{label} ({num_samples}) exceeds the maximum of {MAX_DRIFT_SAMPLES}")
    return num_samples


def validate_model_parameters(
    *,
    width: int,
    depth: int,
    num_classes: int,
) -> tuple[int, int, int]:
    """Bound the reference architectures built on behalf of an MCP client.

    Returns:
        The validated ``(width, depth, num_classes)`` triple.

    Raises:
        ValueError: If any value is outside its permitted range.
    """
    if not 1 <= width <= MAX_MODEL_WIDTH:
        raise ValueError(f"width must be between 1 and {MAX_MODEL_WIDTH}")
    if not 1 <= depth <= MAX_MODEL_DEPTH:
        raise ValueError(f"depth must be between 1 and {MAX_MODEL_DEPTH}")
    if not 1 <= num_classes <= MAX_MODEL_CLASSES:
        raise ValueError(f"num_classes must be between 1 and {MAX_MODEL_CLASSES}")
    return width, depth, num_classes
