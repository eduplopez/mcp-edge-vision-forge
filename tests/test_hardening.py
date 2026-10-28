"""Tests for the hardening layer added after the external audit."""

from __future__ import annotations

from pathlib import Path

import onnx
import pytest
from pydantic import ValidationError

import edge_forge.core.quantizer as quantizer_module
from edge_forge.core.auto_tuner import AutoTuner
from edge_forge.core.profiler import ModelProfiler
from edge_forge.core.quantizer import QuantizationEngine
from edge_forge.errors import BenchmarkError, ModelNotFoundError, PathAccessError
from edge_forge.limits import (
    MAX_BENCHMARK_ITERATIONS,
    MAX_DRIFT_SAMPLES,
    MAX_INPUT_ELEMENTS,
    MAX_MODEL_WIDTH,
    validate_input_shape,
    validate_model_parameters,
    validate_num_samples,
)
from edge_forge.models.schemas import Precision, QuantizationOptions, TuningConstraints
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
