"""ONNX Runtime benchmarking with latency percentiles and resource sampling."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import psutil

from edge_forge.config import ForgeSettings, get_settings
from edge_forge.core.exporter import inspect_onnx
from edge_forge.errors import BenchmarkError, ModelNotFoundError
from edge_forge.models.schemas import (
    BenchmarkReport,
    HardwareProfile,
    LatencyStats,
    Precision,
    ResourceUsage,
    get_hardware_profile,
)
from edge_forge.utils import file_size_mb

_LOGGER = logging.getLogger(__name__)


class ResourceMonitor:
    """Sample process RSS and CPU utilization on a background thread."""

    def __init__(self, interval_s: float = 0.005) -> None:
        if interval_s <= 0:
            raise ValueError("sampling interval must be positive")
        self._interval_s = interval_s
        self._process = psutil.Process()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._rss_samples: list[float] = []
        self._cpu_samples: list[float] = []
        self._baseline_rss_mb = 0.0

    def __enter__(self) -> ResourceMonitor:
        self._process.cpu_percent(interval=None)
        self._baseline_rss_mb = self._process.memory_info().rss / (1024 * 1024)
        self._rss_samples.append(self._baseline_rss_mb)
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._record_sample()

    def _sample_loop(self) -> None:
        while not self._stop.wait(self._interval_s):
            self._record_sample()

    def _record_sample(self) -> None:
        try:
            self._rss_samples.append(self._process.memory_info().rss / (1024 * 1024))
            self._cpu_samples.append(float(self._process.cpu_percent(interval=None)))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return

    @property
    def peak_rss_mb(self) -> float:
        """Highest resident set size observed in mebibytes."""
        return round(max(self._rss_samples, default=0.0), 3)

    @property
    def rss_delta_mb(self) -> float:
        """Difference between peak and baseline resident memory."""
        return round(self.peak_rss_mb - self._baseline_rss_mb, 3)

    @property
    def cpu_percent_mean(self) -> float:
        """Mean process CPU utilization across samples."""
        if not self._cpu_samples:
            return 0.0
        return round(float(np.mean(self._cpu_samples)), 3)

    @property
    def cpu_percent_peak(self) -> float:
        """Peak process CPU utilization across samples."""
        if not self._cpu_samples:
            return 0.0
        return round(float(np.max(self._cpu_samples)), 3)


def _input_arrays(
    session: ort.InferenceSession,
    input_shape: tuple[int, ...] | None,
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    arrays: dict[str, np.ndarray] = {}
    for index, model_input in enumerate(session.get_inputs()):
        if input_shape is not None and index == 0:
            shape = tuple(input_shape)
        else:
            shape = tuple(
                dimension if isinstance(dimension, int) and dimension > 0 else 1
                for dimension in model_input.shape
            )
        if "float16" in model_input.type:
            dtype: Any = np.float16
        elif "float64" in model_input.type:
            dtype = np.float64
        elif "int64" in model_input.type:
            dtype = np.int64
        elif "int32" in model_input.type:
            dtype = np.int32
        else:
            dtype = np.float32
        if np.issubdtype(dtype, np.floating):
            arrays[model_input.name] = rng.standard_normal(size=shape).astype(dtype)
        else:
            arrays[model_input.name] = np.zeros(shape, dtype=dtype)
    return arrays


def _latency_stats(latencies_ms: list[float], duration_s: float) -> LatencyStats:
    values = np.asarray(latencies_ms, dtype=np.float64)
    mean_ms = float(np.mean(values))
    return LatencyStats(
        count=int(values.size),
        mean_ms=round(mean_ms, 4),
        min_ms=round(float(np.min(values)), 4),
        max_ms=round(float(np.max(values)), 4),
        std_ms=round(float(np.std(values, ddof=1)) if values.size > 1 else 0.0, 4),
        p50_ms=round(float(np.percentile(values, 50)), 4),
        p90_ms=round(float(np.percentile(values, 90)), 4),
        p99_ms=round(float(np.percentile(values, 99)), 4),
        fps=round(1000.0 / mean_ms, 2) if mean_ms > 0 else 0.0,
        throughput_per_second=round(values.size / duration_s, 2) if duration_s > 0 else 0.0,
    )


class ModelProfiler:
    """Benchmark ONNX artifacts on the local host with warmup and percentiles."""

    def __init__(
        self,
        settings: ForgeSettings | None = None,
        *,
        provider: str = "CPUExecutionProvider",
        logger: logging.Logger | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._provider = provider
        self._logger = logger or _LOGGER

    @property
    def settings(self) -> ForgeSettings:
        """Runtime settings used by the profiler."""
        return self._settings

    def benchmark(
        self,
        model_path: str | Path,
        *,
        input_shape: tuple[int, ...] | None = None,
        warmup_runs: int = 5,
        iterations: int = 30,
        hardware_profile: str | HardwareProfile | None = None,
        precision: Precision | None = None,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BenchmarkReport:
        """Run a benchmark and return latency and resource statistics."""
        if iterations < 1:
            raise BenchmarkError("iterations must be at least 1")
        if warmup_runs < 0:
            raise BenchmarkError("warmup_runs cannot be negative")
        resolved = Path(model_path).expanduser().resolve()
        if not resolved.exists():
            raise ModelNotFoundError(f"ONNX artifact does not exist: {resolved}")
        available = ort.get_available_providers()
        if self._provider not in available:
            raise BenchmarkError(
                f"provider '{self._provider}' is not available; available: {available}"
            )
        info = inspect_onnx(resolved, name=name or resolved.stem)
        profile = (
            get_hardware_profile(hardware_profile)
            if isinstance(hardware_profile, str)
            else hardware_profile
        )
        options = ort.SessionOptions()
        if self._settings.intra_op_threads > 0:
            options.intra_op_num_threads = self._settings.intra_op_threads
        options.inter_op_num_threads = self._settings.inter_op_threads
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.enable_mem_pattern = True
        options.log_severity_level = 3
        try:
            session = ort.InferenceSession(
                str(resolved),
                sess_options=options,
                providers=[self._provider],
            )
        except Exception as exc:
            raise BenchmarkError(f"unable to create ONNX Runtime session: {exc}") from exc

        feeds = _input_arrays(session, input_shape, seed=self._settings.seed)
        try:
            for _ in range(warmup_runs):
                session.run(None, feeds)
        except Exception as exc:
            raise BenchmarkError(f"warmup inference failed: {exc}") from exc

        latencies_ms: list[float] = []
        started = time.perf_counter()
        try:
            with ResourceMonitor() as monitor:
                for _ in range(iterations):
                    sample_start = time.perf_counter_ns()
                    session.run(None, feeds)
                    latencies_ms.append((time.perf_counter_ns() - sample_start) / 1_000_000.0)
        except Exception as exc:
            raise BenchmarkError(f"benchmark inference failed: {exc}") from exc
        duration_s = time.perf_counter() - started
        latency = _latency_stats(latencies_ms, duration_s)
        process = psutil.Process()
        resources = ResourceUsage(
            peak_rss_mb=monitor.peak_rss_mb,
            rss_delta_mb=monitor.rss_delta_mb,
            cpu_percent_mean=monitor.cpu_percent_mean,
            cpu_percent_peak=monitor.cpu_percent_peak,
            thread_count=max(1, process.num_threads()),
            model_size_mb=file_size_mb(resolved),
        )
        meets_budget: bool | None = None
        if profile is not None:
            meets_budget = (
                latency.p99_ms <= profile.typical_latency_budget_ms
                and resources.model_size_mb <= profile.max_model_size_mb
            )
        report_metadata: dict[str, Any] = {
            "onnxruntime_version": ort.__version__,
            "available_providers": ",".join(available),
            "graph_optimization": "ORT_ENABLE_ALL",
            "input_shape": list(input_shape) if input_shape is not None else None,
        }
        if metadata:
            report_metadata.update(metadata)
        report = BenchmarkReport(
            model_name=name or resolved.stem,
            model_path=str(resolved),
            precision=precision or info.precision,
            opset=info.opset,
            parameters=info.parameters,
            provider=session.get_providers()[0],
            hardware_profile=profile.name if profile is not None else "current-host",
            latency_budget_ms=(profile.typical_latency_budget_ms if profile is not None else None),
            meets_budget=meets_budget,
            input_specs=info.input_specs,
            latency=latency,
            resources=resources,
            warmup_runs=warmup_runs,
            iterations=iterations,
            metadata=report_metadata,
        )
        self._logger.info(
            "benchmark %s: p50=%.3f ms p99=%.3f ms fps=%.1f",
            report.model_name,
            latency.p50_ms,
            latency.p99_ms,
            latency.fps,
        )
        return report

    def profile(self, model_path: str | Path, **kwargs: Any) -> BenchmarkReport:
        """Alias for :meth:`benchmark` kept for ergonomic service composition."""
        return self.benchmark(model_path, **kwargs)


def available_providers() -> list[str]:
    """Return the ONNX Runtime execution providers installed on this host."""
    return list(ort.get_available_providers())


def runtime_version() -> str:
    """Return the ONNX Runtime version used for benchmarks."""
    return str(ort.__version__)
