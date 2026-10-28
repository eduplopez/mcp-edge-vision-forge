"""Tests for ONNX Runtime benchmarking and resource accounting."""

from __future__ import annotations

from pathlib import Path

import pytest

from edge_forge.core.profiler import ModelProfiler, ResourceMonitor, available_providers
from edge_forge.errors import BenchmarkError, ModelNotFoundError
from edge_forge.models.schemas import Precision, get_hardware_profile


def test_benchmark_reports_latency_and_resources(
    settings,
    fp32_onnx: Path,
    input_shape: tuple[int, ...],
) -> None:
    report = ModelProfiler(settings).benchmark(
        fp32_onnx,
        input_shape=input_shape,
        warmup_runs=2,
        iterations=8,
        hardware_profile="generic-x86",
    )
    assert report.iterations == 8
    assert report.warmup_runs == 2
    assert report.latency.count == 8
    assert report.latency.mean_ms > 0
    assert report.latency.p50_ms <= report.latency.p90_ms <= report.latency.p99_ms
    assert report.latency.fps > 0
    assert report.latency.throughput_per_second > 0
    assert report.resources.peak_rss_mb > 0
    assert report.resources.model_size_mb > 0
    assert report.resources.thread_count >= 1
    assert report.provider == "CPUExecutionProvider"
    assert report.hardware_profile == "generic-x86"
    assert report.latency_budget_ms == 50.0
    assert isinstance(report.meets_budget, bool)
    assert report.metadata["onnxruntime_version"]


def test_benchmark_detects_precision(
    settings,
    int8_onnx: Path,
    input_shape: tuple[int, ...],
) -> None:
    report = ModelProfiler(settings).benchmark(
        int8_onnx,
        input_shape=input_shape,
        warmup_runs=1,
        iterations=3,
    )
    assert report.precision is Precision.INT8_DYNAMIC
    assert report.hardware_profile == "current-host"
    assert report.latency_budget_ms is None
    assert report.meets_budget is None


def test_benchmark_respects_explicit_precision(
    settings,
    fp32_onnx: Path,
    input_shape: tuple[int, ...],
) -> None:
    report = ModelProfiler(settings).benchmark(
        fp32_onnx,
        input_shape=input_shape,
        iterations=2,
        precision=Precision.INT8_STATIC,
    )
    assert report.precision is Precision.INT8_STATIC


def test_zero_warmup_is_allowed(settings, fp32_onnx: Path) -> None:
    report = ModelProfiler(settings).benchmark(
        fp32_onnx,
        input_shape=(1, 3, 32, 32),
        warmup_runs=0,
        iterations=1,
    )
    assert report.latency.count == 1


def test_invalid_iterations_raise(settings, fp32_onnx: Path) -> None:
    with pytest.raises(BenchmarkError):
        ModelProfiler(settings).benchmark(fp32_onnx, iterations=0)


def test_missing_model_raises(settings, tmp_path: Path) -> None:
    with pytest.raises(ModelNotFoundError):
        ModelProfiler(settings).benchmark(tmp_path / "missing.onnx")


def test_unknown_provider_raises(settings, fp32_onnx: Path) -> None:
    with pytest.raises(BenchmarkError, match="not available"):
        ModelProfiler(settings, provider="UnsupportedExecutionProvider").benchmark(fp32_onnx)


def test_hardware_budget_can_fail(settings, fp32_onnx: Path) -> None:
    profile = get_hardware_profile("coral-edge-tpu").model_copy(
        update={"typical_latency_budget_ms": 0.0001}
    )
    report = ModelProfiler(settings).benchmark(
        fp32_onnx,
        input_shape=(1, 3, 32, 32),
        iterations=2,
        hardware_profile=profile,
    )
    assert report.hardware_profile == "coral-edge-tpu"
    assert report.latency_budget_ms == profile.typical_latency_budget_ms
    assert report.meets_budget is False


def test_resource_monitor_collects_samples() -> None:
    with ResourceMonitor(interval_s=0.001) as monitor:
        sum(range(10000))
    assert monitor.peak_rss_mb > 0
    assert monitor.cpu_percent_peak >= 0.0


def test_cpu_provider_is_available() -> None:
    assert "CPUExecutionProvider" in available_providers()
