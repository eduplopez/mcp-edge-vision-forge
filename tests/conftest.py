"""Shared pytest fixtures for deterministic, fast pipeline tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from edge_forge.config import ForgeSettings
from edge_forge.core.exporter import OnnxExporter
from edge_forge.core.quantizer import QuantizationEngine
from edge_forge.models.registry import build_model
from edge_forge.models.schemas import ExportOptions, Precision, QuantizationOptions


@pytest.fixture(autouse=True)
def _deterministic_torch() -> None:
    torch.manual_seed(7)
    torch.set_num_threads(1)


@pytest.fixture
def settings(tmp_path: Path) -> ForgeSettings:
    return ForgeSettings(
        workspace=tmp_path / "artifacts",
        cache_dir=tmp_path / "cache",
        reports_dir=tmp_path / "reports",
        intra_op_threads=1,
        inter_op_threads=1,
        calibration_samples=4,
        seed=11,
    ).ensure_directories()


@pytest.fixture
def input_shape() -> tuple[int, ...]:
    return (1, 3, 32, 32)


@pytest.fixture
def tiny_model() -> nn.Module:
    return build_model("edge_cnn", width=8, depth=1, num_classes=10)


@pytest.fixture
def fp32_onnx(
    settings: ForgeSettings,
    tiny_model: nn.Module,
    input_shape: tuple[int, ...],
) -> Path:
    result = OnnxExporter(settings).export(
        tiny_model,
        settings.exports_dir / "tiny.onnx",
        input_shape=input_shape,
        options=ExportOptions(opset=17, simplify=True),
        name="edge_cnn",
    )
    return Path(result.model_path)


@pytest.fixture
def int8_onnx(
    settings: ForgeSettings,
    fp32_onnx: Path,
) -> Path:
    result = QuantizationEngine(settings).quantize(
        fp32_onnx,
        settings.quantized_dir / "tiny.int8_dynamic.onnx",
        options=QuantizationOptions(
            precision=Precision.INT8_DYNAMIC,
            calibration_samples=4,
        ),
    )
    return Path(result.output_path)
