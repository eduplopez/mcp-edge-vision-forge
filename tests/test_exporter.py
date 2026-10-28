"""Tests for ONNX export, simplification and validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from torch import nn

from edge_forge.core.exporter import GraphSimplifier, OnnxExporter, inspect_onnx, validate_opset
from edge_forge.errors import ExportError, UnsupportedOpsetError
from edge_forge.models.schemas import ExportOptions


def test_export_creates_valid_artifact(
    settings,
    tiny_model: nn.Module,
    input_shape: tuple[int, ...],
    tmp_path: Path,
) -> None:
    destination = tmp_path / "model.onnx"
    result = OnnxExporter(settings).export(
        tiny_model,
        destination,
        input_shape=input_shape,
        options=ExportOptions(opset=17),
        name="edge_cnn",
    )
    assert destination.exists()
    assert result.opset == 17
    assert result.size_mb > 0
    assert result.parameters > 0
    assert result.validation is not None
    assert result.validation.passed is True
    assert result.validation.cosine_similarity > 0.9999
    assert result.checksum
    assert result.dynamic_axes["input"][0] == "batch"
    assert result.input_specs[0].shape == input_shape
    assert result.output_specs[0].shape[0] == 1
    onnx.checker.check_model(onnx.load(str(destination)))


def test_dynamic_batch_is_symbolic_on_disk(
    settings,
    tiny_model: nn.Module,
    input_shape: tuple[int, ...],
    tmp_path: Path,
) -> None:
    destination = tmp_path / "dynamic.onnx"
    OnnxExporter(settings).export(
        tiny_model,
        destination,
        input_shape=input_shape,
        options=ExportOptions(opset=17, dynamic_batch=True, simplify=False),
    )
    model = onnx.load(str(destination))
    batch_dimension = model.graph.input[0].type.tensor_type.shape.dim[0]
    assert batch_dimension.dim_param == "batch"


def test_simplification_preserves_outputs(
    settings,
    tiny_model: nn.Module,
    input_shape: tuple[int, ...],
    tmp_path: Path,
) -> None:
    plain = OnnxExporter(settings).export(
        tiny_model,
        tmp_path / "plain.onnx",
        input_shape=input_shape,
        options=ExportOptions(opset=17, simplify=False),
    )
    simplified = OnnxExporter(settings).export(
        tiny_model,
        tmp_path / "simplified.onnx",
        input_shape=input_shape,
        options=ExportOptions(opset=17, simplify=True),
    )
    assert simplified.validation is not None
    assert simplified.validation.passed is True
    plain_nodes = len(onnx.load(plain.model_path).graph.node)
    simplified_nodes = len(onnx.load(simplified.model_path).graph.node)
    assert simplified_nodes <= plain_nodes


def test_unsupported_opset_is_rejected() -> None:
    with pytest.raises(UnsupportedOpsetError):
        validate_opset(99)
    with pytest.raises(UnsupportedOpsetError):
        validate_opset(5)


def test_output_name_count_must_match(
    settings,
    tiny_model: nn.Module,
    input_shape: tuple[int, ...],
    tmp_path: Path,
) -> None:
    with pytest.raises(ExportError, match="output names"):
        OnnxExporter(settings).export(
            tiny_model,
            tmp_path / "bad.onnx",
            input_shape=input_shape,
            options=ExportOptions(opset=17, output_names=["first", "second"]),
        )


def test_inspect_onnx_reports_metadata(
    settings,
    fp32_onnx: Path,
) -> None:
    info = inspect_onnx(fp32_onnx)
    assert info.opset == 17
    assert info.architecture == "edge_cnn"
    assert info.parameters > 0
    assert info.input_specs[0].name == "input"
    assert info.output_specs[0].name == "output"
    assert info.checksum is not None


def test_graph_simplifier_removes_identity_nodes(tmp_path: Path) -> None:
    input_info = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 3])
    output_info = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 3])
    identity = onnx.helper.make_node("Identity", ["input"], ["middle"], name="identity")
    relu = onnx.helper.make_node("Relu", ["middle"], ["output"], name="relu")
    graph = onnx.helper.make_graph([identity, relu], "identity_graph", [input_info], [output_info])
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])
    source = tmp_path / "identity.onnx"
    onnx.save(model, str(source))
    destination = tmp_path / "simplified.onnx"
    GraphSimplifier().simplify(source, destination)
    simplified = onnx.load(str(destination))
    assert [node.op_type for node in simplified.graph.node] == ["Relu"]


def test_export_rejects_invalid_input_shape(
    settings,
    tiny_model: nn.Module,
    tmp_path: Path,
) -> None:
    with pytest.raises(ExportError, match="positive"):
        OnnxExporter(settings).export(
            tiny_model,
            tmp_path / "bad_shape.onnx",
            input_shape=(0, 3, 32, 32),
        )


def test_validate_detects_different_model(
    settings,
    tiny_model: nn.Module,
    fp32_onnx: Path,
    input_shape: tuple[int, ...],
) -> None:
    other = torch.nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
    validation = OnnxExporter(settings).validate(
        other,
        fp32_onnx,
        input_shape=input_shape,
        atol=1e-6,
        rtol=1e-6,
    )
    assert validation.passed is False
    assert validation.max_absolute_error > 0
    assert np.isfinite(validation.cosine_similarity)
