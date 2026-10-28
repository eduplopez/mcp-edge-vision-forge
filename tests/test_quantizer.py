"""Tests for FP16 conversion, INT8 quantization and calibration readers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

from edge_forge.core.exporter import inspect_onnx
from edge_forge.core.quantizer import (
    CalibrationReaderFactory,
    QuantizationEngine,
    RealCalibrationReader,
    SyntheticCalibrationReader,
)
from edge_forge.errors import CalibrationError, ModelNotFoundError
from edge_forge.models.schemas import (
    CalibrationMethod,
    Precision,
    QuantizationOptions,
    TensorSpec,
)


def test_fp16_conversion(settings, fp32_onnx: Path, tmp_path: Path) -> None:
    result = QuantizationEngine(settings).quantize(
        fp32_onnx,
        tmp_path / "model.fp16.onnx",
        options=QuantizationOptions(precision=Precision.FP16),
    )
    assert Path(result.output_path).exists()
    assert result.precision is Precision.FP16
    assert result.size_mb < result.source_size_mb
    assert result.compression_ratio > 1.0
    assert inspect_onnx(result.output_path).precision is Precision.FP16
    session = ort.InferenceSession(result.output_path, providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].name == "input"


def test_dynamic_int8_quantization(settings, fp32_onnx: Path, tmp_path: Path) -> None:
    result = QuantizationEngine(settings).quantize(
        fp32_onnx,
        tmp_path / "model.int8.onnx",
        options=QuantizationOptions(
            precision=Precision.INT8_DYNAMIC,
            per_channel=True,
            calibration_samples=4,
        ),
    )
    assert result.precision is Precision.INT8_DYNAMIC
    assert result.calibration_samples == 0
    assert inspect_onnx(result.output_path).precision is Precision.INT8_DYNAMIC
    session = ort.InferenceSession(result.output_path, providers=["CPUExecutionProvider"])
    outputs = session.run(None, {"input": np.zeros((1, 3, 32, 32), dtype=np.float32)})
    assert outputs[0].shape == (1, 10)


def test_static_int8_quantization(settings, fp32_onnx: Path, tmp_path: Path) -> None:
    result = QuantizationEngine(settings).quantize(
        fp32_onnx,
        tmp_path / "model.static.onnx",
        options=QuantizationOptions(
            precision=Precision.INT8_STATIC,
            calibration_samples=4,
            calibration_method=CalibrationMethod.GAUSSIAN,
            per_channel=True,
        ),
    )
    assert result.precision is Precision.INT8_STATIC
    assert result.calibration_samples == 4
    assert result.metadata["calibration_method"] == "gaussian"
    assert inspect_onnx(result.output_path).precision is Precision.INT8_STATIC
    session = ort.InferenceSession(result.output_path, providers=["CPUExecutionProvider"])
    outputs = session.run(None, {"input": np.zeros((1, 3, 32, 32), dtype=np.float32)})
    assert outputs[0].shape == (1, 10)


def test_static_quantization_with_real_data(
    settings,
    fp32_onnx: Path,
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "calibration.npy"
    rng = np.random.default_rng(3)
    np.save(data_path, rng.standard_normal((5, 3, 32, 32)).astype(np.float32))
    result = QuantizationEngine(settings).quantize(
        fp32_onnx,
        tmp_path / "model.real.onnx",
        options=QuantizationOptions(
            precision=Precision.INT8_STATIC,
            calibration_samples=5,
            calibration_method=CalibrationMethod.REAL,
            calibration_data_path=str(data_path),
        ),
    )
    assert result.calibration_samples == 5
    assert Path(result.output_path).exists()


def test_real_calibration_requires_path() -> None:
    with pytest.raises(ValueError):
        QuantizationOptions(
            precision=Precision.INT8_STATIC,
            calibration_method=CalibrationMethod.REAL,
        )


def test_missing_calibration_file_raises(settings, fp32_onnx: Path, tmp_path: Path) -> None:
    with pytest.raises(CalibrationError):
        QuantizationEngine(settings).quantize(
            fp32_onnx,
            tmp_path / "missing.onnx",
            options=QuantizationOptions(
                precision=Precision.INT8_STATIC,
                calibration_method=CalibrationMethod.REAL,
                calibration_data_path=str(tmp_path / "does-not-exist.npy"),
            ),
        )


def test_synthetic_reader_exhausts() -> None:
    spec = TensorSpec(name="input", shape=(1, 3, 8, 8))
    reader = SyntheticCalibrationReader([spec], samples=2, method=CalibrationMethod.UNIFORM)
    first = reader.get_next()
    second = reader.get_next()
    assert first is not None and first["input"].shape == (1, 3, 8, 8)
    assert second is not None
    assert reader.get_next() is None
    reader.rewind()
    assert reader.get_next() is not None


def test_real_reader_validates_shape(tmp_path: Path) -> None:
    data_path = tmp_path / "bad.npy"
    np.save(data_path, np.zeros((4, 3, 16, 16), dtype=np.float32))
    spec = TensorSpec(name="input", shape=(1, 3, 32, 32))
    with pytest.raises(CalibrationError):
        RealCalibrationReader([spec], data_path, samples=4)


def test_factory_returns_reader_only_for_static() -> None:
    spec = TensorSpec(name="input", shape=(1, 3, 8, 8))
    assert (
        CalibrationReaderFactory.create(QuantizationOptions(precision=Precision.FP16), [spec])
        is None
    )
    reader = CalibrationReaderFactory.create(
        QuantizationOptions(precision=Precision.INT8_STATIC, calibration_samples=2),
        [spec],
    )
    assert isinstance(reader, SyntheticCalibrationReader)


def test_missing_source_raises(settings, tmp_path: Path) -> None:
    with pytest.raises(ModelNotFoundError):
        QuantizationEngine(settings).quantize(tmp_path / "missing.onnx")


def test_source_and_destination_must_differ(settings, fp32_onnx: Path) -> None:
    with pytest.raises(Exception, match="different"):
        QuantizationEngine(settings).quantize(
            fp32_onnx,
            fp32_onnx,
            options=QuantizationOptions(precision=Precision.FP16),
        )
