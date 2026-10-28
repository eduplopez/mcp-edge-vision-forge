"""Tests for the hardening layer added after the external audit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest
from pydantic import ValidationError
from torch import nn

import edge_forge.core.quantizer as quantizer_module
from edge_forge.core.auto_tuner import AutoTuner
from edge_forge.core.drift_analyzer import _aggregate_metrics
from edge_forge.core.profiler import ModelProfiler, _numpy_dtype
from edge_forge.core.quantizer import QuantizationEngine, RealCalibrationReader
from edge_forge.errors import (
    BenchmarkError,
    CalibrationError,
    ModelNotFoundError,
    PathAccessError,
    ReportError,
)
from edge_forge.limits import (
    MAX_BENCHMARK_ITERATIONS,
    MAX_DRIFT_SAMPLES,
    MAX_INPUT_ELEMENTS,
    MAX_MODEL_WIDTH,
    validate_file_size,
    validate_input_shape,
    validate_model_parameters,
    validate_num_samples,
)
from edge_forge.models.registry import resolve_or_build_model, save_checkpoint
from edge_forge.models.schemas import (
    CalibrationMethod,
    DriftMetrics,
    DriftThresholds,
    OutputDrift,
    Precision,
    QuantizationOptions,
    TensorSpec,
    TuningConstraints,
    get_hardware_profile,
)
from edge_forge.reporting.report_gen import _cell, validate_report_id
from edge_forge.utils import resolve_within_roots


def test_validate_input_shape_accepts_reasonable_shapes() -> None:
    assert validate_input_shape((1, 3, 224, 224)) == (1, 3, 224, 224)


@pytest.mark.parametrize("shape", [(), (1, 0, 8), (1, 3, -4, 4)])
def test_validate_input_shape_rejects_invalid_dimensions(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        validate_input_shape(shape)


def test_validate_input_shape_rejects_oversized_tensors() -> None:
    side = int(MAX_INPUT_ELEMENTS**0.5) + 2
    with pytest.raises(ValueError, match="exceeding the limit"):
        validate_input_shape((1, 3, side, side))


def test_validate_model_parameters_bounds_reference_architectures() -> None:
    assert validate_model_parameters(width=16, depth=3, num_classes=10) == (16, 3, 10)
    with pytest.raises(ValueError, match="width"):
        validate_model_parameters(width=MAX_MODEL_WIDTH + 1, depth=1, num_classes=1)
    with pytest.raises(ValueError, match="depth"):
        validate_model_parameters(width=1, depth=0, num_classes=1)
    with pytest.raises(ValueError, match="num_classes"):
        validate_model_parameters(width=1, depth=1, num_classes=0)


def test_validate_num_samples_bounds_drift_runs() -> None:
    assert validate_num_samples(MAX_DRIFT_SAMPLES) == MAX_DRIFT_SAMPLES
    with pytest.raises(ValueError, match="at least 1"):
        validate_num_samples(0)
    with pytest.raises(ValueError, match="exceeds the maximum"):
        validate_num_samples(MAX_DRIFT_SAMPLES + 1)


def test_benchmark_rejects_iterations_above_cap(settings, fp32_onnx: Path) -> None:
    with pytest.raises(BenchmarkError, match="exceeds the maximum"):
        ModelProfiler(settings).benchmark(fp32_onnx, iterations=MAX_BENCHMARK_ITERATIONS + 1)


def test_benchmark_rejects_oversized_input_shape(settings, fp32_onnx: Path) -> None:
    with pytest.raises(BenchmarkError, match="exceeding the limit"):
        ModelProfiler(settings).benchmark(fp32_onnx, input_shape=(1, 3, 4096, 4096))


def test_tuning_rejects_oversized_search_space() -> None:
    with pytest.raises(ValidationError, match="exceeding the limit"):
        TuningConstraints(
            max_size_mb=10.0,
            allowed_precisions=[
                Precision.FP32,
                Precision.FP16,
                Precision.INT8_DYNAMIC,
                Precision.INT8_STATIC,
            ],
            opset_candidates=list(range(11, 24)),
            simplify_candidates=[True, False],
        )


def test_tuning_rejects_zero_score_weights() -> None:
    with pytest.raises(ValidationError, match="score weight"):
        TuningConstraints(
            max_size_mb=10.0,
            score_weight_latency=0.0,
            score_weight_size=0.0,
            score_weight_drift=0.0,
        )


def test_nodes_to_exclude_reach_onnxruntime(
    monkeypatch: pytest.MonkeyPatch,
    settings,
    fp32_onnx: Path,
) -> None:
    captured: dict[str, object] = {}
    original = quantizer_module.quantize_dynamic

    def spy(**kwargs: object) -> object:
        captured.update(kwargs)
        return original(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(quantizer_module, "quantize_dynamic", spy)
    result = QuantizationEngine(settings).quantize(
        fp32_onnx,
        settings.quantized_dir / "excluded.int8.onnx",
        options=QuantizationOptions(
            precision=Precision.INT8_DYNAMIC,
            calibration_samples=4,
            nodes_to_exclude=["classifier"],
        ),
    )
    assert captured["nodes_to_exclude"] == ["classifier"]
    assert result.metadata["nodes_to_exclude"] == "classifier"


def test_dynamic_int8_quantizes_convolution_layers(int8_onnx: Path) -> None:
    graph = onnx.load(str(int8_onnx)).graph
    operators = {node.op_type for node in graph.node}
    assert "ConvInteger" in operators
    assert "DynamicQuantizeLinear" in operators


def test_tuning_emits_progress_updates(
    settings,
    tiny_model,
    input_shape: tuple[int, ...],
) -> None:
    events: list[tuple[int, int, str]] = []
    constraints = TuningConstraints(
        max_size_mb=10.0,
        allowed_precisions=[Precision.FP32, Precision.FP16],
        warmup_runs=1,
        iterations=2,
    )
    result = AutoTuner(settings).tune(
        tiny_model,
        constraints=constraints,
        input_shape=input_shape,
        model_name="edge_cnn",
        progress=lambda completed, total, message: events.append((completed, total, message)),
    )
    assert len(events) == len(result.candidates)
    assert [event[0] for event in events] == sorted(event[0] for event in events)
    assert all(event[1] >= event[0] for event in events)


def test_custom_score_weights_are_honoured(
    settings,
    tiny_model,
    input_shape: tuple[int, ...],
) -> None:
    constraints = TuningConstraints(
        max_size_mb=10.0,
        allowed_precisions=[Precision.FP32, Precision.FP16],
        warmup_runs=1,
        iterations=2,
        score_weight_latency=0.0,
        score_weight_size=0.0,
        score_weight_drift=1.0,
    )
    result = AutoTuner(settings).tune(
        tiny_model,
        constraints=constraints,
        input_shape=input_shape,
        model_name="edge_cnn",
    )
    assert result.candidates
    for candidate in result.candidates:
        assert 0.0 <= candidate.score <= 1.0


def test_resolve_within_roots_allows_children(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    target = root / "artifacts" / "model.onnx"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"onnx")
    assert resolve_within_roots(target, [root], must_exist=True) == target.resolve()


def test_resolve_within_roots_rejects_escapes(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    with pytest.raises(PathAccessError, match="outside the permitted roots"):
        resolve_within_roots(tmp_path / "outside.onnx", [root], label="model_path")


def test_resolve_within_roots_can_be_disabled(tmp_path: Path) -> None:
    outside = tmp_path / "outside.onnx"
    outside.write_bytes(b"onnx")
    resolved = resolve_within_roots(
        outside,
        [tmp_path / "workspace"],
        must_exist=True,
        enabled=False,
    )
    assert resolved == outside.resolve()


def test_resolve_within_roots_reports_missing_files(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    with pytest.raises(ModelNotFoundError, match="does not exist"):
        resolve_within_roots(root / "missing.onnx", [root], must_exist=True)


def test_validate_file_size_enforces_a_hard_budget(tmp_path: Path) -> None:
    target = tmp_path / "payload.bin"
    target.write_bytes(b"x" * 4096)
    with pytest.raises(ValueError, match="exceeding the limit"):
        validate_file_size(target, 0.001)
    assert validate_file_size(target, 1.0) == target


def test_edge_profiles_accept_static_int8() -> None:
    for name in ("raspberry-pi-5", "smartphone-midrange", "generic-x86"):
        assert Precision.INT8_STATIC in get_hardware_profile(name).supported_precisions


def test_real_calibration_requires_a_data_path() -> None:
    with pytest.raises(ValidationError, match="calibration_data_path"):
        TuningConstraints(calibration_method=CalibrationMethod.REAL)


def test_calibration_reader_rejects_traversal_names(tmp_path: Path) -> None:
    spec = TensorSpec(name="../escape", shape=(1, 3, 4, 4), dtype="float32")
    with pytest.raises(CalibrationError, match="unsafe tensor name"):
        RealCalibrationReader([spec], tmp_path)


def test_calibration_reader_rejects_pickled_arrays(tmp_path: Path) -> None:
    source = tmp_path / "objects.npy"
    np.save(source, np.array([{"payload": 1}], dtype=object))
    spec = TensorSpec(name="input", shape=(1, 1), dtype="float32")
    with pytest.raises(CalibrationError, match="unable to read"):
        RealCalibrationReader([spec], source)


def test_resolve_or_build_model_loads_checkpoints(tmp_path: Path, tiny_model: nn.Module) -> None:
    checkpoint = save_checkpoint(
        tiny_model,
        tmp_path / "tiny.pt",
        architecture="edge_cnn",
        model_kwargs={"width": 8, "depth": 1, "num_classes": 10},
    )
    model, name = resolve_or_build_model(checkpoint, input_shape=(1, 3, 32, 32))
    assert name == "edge_cnn"
    assert isinstance(model, nn.Module)


def test_resolve_or_build_model_falls_back_to_the_registry() -> None:
    model, name = resolve_or_build_model("edge_cnn", width=8, depth=1, num_classes=10)
    assert name == "edge_cnn"
    assert isinstance(model, nn.Module)


def _write_int8_probe(path: Path, operator: str) -> Path:
    inputs = [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 4])]
    outputs = [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 4])]
    scale = onnx.helper.make_tensor("scale", onnx.TensorProto.FLOAT, [], [0.1])
    zero = onnx.helper.make_tensor("zero", onnx.TensorProto.INT8, [], [0])
    node = onnx.helper.make_node(operator, ["x", "scale", "zero"], ["y"])
    graph = onnx.helper.make_graph([node], "probe", inputs, outputs, [scale, zero])
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])
    onnx.save(model, str(path))
    return path


def test_inspect_onnx_detects_static_int8_without_metadata(tmp_path: Path) -> None:
    from edge_forge.core.exporter import inspect_onnx

    static = inspect_onnx(_write_int8_probe(tmp_path / "static.onnx", "DequantizeLinear"))
    dynamic = inspect_onnx(_write_int8_probe(tmp_path / "dynamic.onnx", "ConvInteger"))
    assert static.precision is Precision.INT8_STATIC
    assert dynamic.precision is Precision.INT8_DYNAMIC


def test_numpy_dtype_supports_integer_and_boolean_inputs() -> None:
    assert _numpy_dtype("tensor(bool)") is np.bool_
    assert _numpy_dtype("tensor(uint8)") is np.uint8
    with pytest.raises(BenchmarkError, match="unsupported model input type"):
        _numpy_dtype("tensor(string)")


def test_report_identifiers_cannot_escape_the_report_directory() -> None:
    assert validate_report_id("run-2026-01") == "run-2026-01"
    for candidate in ("../escape", "..", "a/b", "a\\b", ""):
        with pytest.raises(ReportError, match="invalid report identifier"):
            validate_report_id(candidate)


def test_markdown_cells_escape_html_and_pipes() -> None:
    assert _cell("<script>alert(1)</script>") == "&lt;script&gt;alert(1)&lt;/script&gt;"
    assert _cell("a|b") == "a\\|b"


def test_score_ranks_on_tail_latency() -> None:
    constraints = TuningConstraints(
        score_weight_latency=1.0,
        score_weight_size=0.0,
        score_weight_drift=0.0,
    )
    baseline = TuningCandidateShim(20.0)
    fast_tail = AutoTuner._score(
        TuningCandidateShim(10.0),  # type: ignore[arg-type]
        baseline,  # type: ignore[arg-type]
        constraints,
    )
    slow_tail = AutoTuner._score(
        TuningCandidateShim(18.0),  # type: ignore[arg-type]
        baseline,  # type: ignore[arg-type]
        constraints,
    )
    assert fast_tail > slow_tail


class TuningCandidateShim:
    """Minimal candidate exposing only the attributes used by ``_score``."""

    def __init__(self, p99_ms: float) -> None:
        self.size_mb = 1.0
        self.benchmark = _BenchmarkShim(p99_ms)
        self.drift = _DriftShim()


class _BenchmarkShim:
    def __init__(self, p99_ms: float) -> None:
        self.latency = _LatencyShim(p99_ms)


class _LatencyShim:
    def __init__(self, p99_ms: float) -> None:
        self.p50_ms = 1.0
        self.p99_ms = p99_ms


class _DriftShim:
    def __init__(self) -> None:
        self.metrics = _MetricsShim()


class _MetricsShim:
    def __init__(self) -> None:
        self.cosine_similarity = 1.0


def test_multi_output_drift_averages_sample_metrics() -> None:
    def metrics(kl: float, agreement: float) -> DriftMetrics:
        return DriftMetrics(
            cosine_similarity=1.0,
            mse=0.0,
            mae=0.0,
            max_absolute_error=0.0,
            kl_divergence=kl,
            snr_db=120.0,
            relative_l1_error=0.0,
            top1_agreement=agreement,
            reference_norm=1.0,
            candidate_norm=1.0,
        )

    per_output = [
        OutputDrift(output_name="logits", shape=(2, 3), metrics=metrics(0.2, 0.5)),
        OutputDrift(output_name="aux", shape=(2, 3), metrics=metrics(0.4, 1.0)),
    ]
    reference = np.zeros((2, 6))
    candidate = np.zeros((2, 6))
    overall = _aggregate_metrics(reference, candidate, per_output, DriftThresholds())
    assert overall.kl_divergence == pytest.approx(0.3)
    assert overall.top1_agreement == pytest.approx(0.75)
