"""Tests for numerical drift analysis and metric primitives."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from edge_forge.core.drift_analyzer import DriftAnalyzer
from edge_forge.core.numerics import (
    cosine_similarity,
    kl_divergence,
    max_absolute_error,
    mean_absolute_error,
    mean_squared_error,
    signal_to_noise_db,
    top1_agreement,
)
from edge_forge.errors import DriftAnalysisError, ModelNotFoundError
from edge_forge.models.schemas import DriftThresholds


def test_identical_export_has_no_drift(
    settings,
    tiny_model: nn.Module,
    fp32_onnx: Path,
    input_shape: tuple[int, ...],
) -> None:
    report = DriftAnalyzer(settings).compare(
        tiny_model,
        fp32_onnx,
        input_shape=input_shape,
        num_samples=3,
        thresholds=DriftThresholds(),
        reference_name="edge_cnn",
    )
    assert report.passed is True
    assert report.metrics.cosine_similarity > 0.999999
    assert report.metrics.mse < 1e-10
    assert report.metrics.max_absolute_error < 1e-5
    assert report.per_output[0].output_name == "output"
    assert report.samples == 3


def test_quantized_model_reports_fidelity(
    settings,
    tiny_model: nn.Module,
    int8_onnx: Path,
    input_shape: tuple[int, ...],
) -> None:
    report = DriftAnalyzer(settings).compare(
        tiny_model,
        int8_onnx,
        input_shape=input_shape,
        num_samples=4,
    )
    assert report.passed is True
    assert 0.9 < report.metrics.cosine_similarity <= 1.0
    assert report.metrics.mse >= 0.0
    assert report.metrics.kl_divergence >= 0.0
    assert report.metrics.snr_db > 20.0
    assert 0.0 <= report.metrics.top1_agreement <= 1.0


def test_strict_thresholds_detect_drift(
    settings,
    tiny_model: nn.Module,
    int8_onnx: Path,
    input_shape: tuple[int, ...],
) -> None:
    thresholds = DriftThresholds(
        min_cosine_similarity=1.0,
        max_mse=0.0,
        max_mae=0.0,
        max_absolute_error=0.0,
        max_kl_divergence=0.0,
        min_snr_db=200.0,
    )
    report = DriftAnalyzer(settings).compare(
        tiny_model,
        int8_onnx,
        input_shape=input_shape,
        num_samples=2,
        thresholds=thresholds,
    )
    assert report.passed is False
    assert report.metrics.violations
    assert len(report.metrics.violations) >= 3


def test_compare_two_onnx_models(
    settings,
    fp32_onnx: Path,
    int8_onnx: Path,
    input_shape: tuple[int, ...],
) -> None:
    report = DriftAnalyzer(settings).compare_onnx(
        fp32_onnx,
        int8_onnx,
        input_shape=input_shape,
        num_samples=3,
    )
    assert report.reference_model == fp32_onnx.stem
    assert report.candidate_model == int8_onnx.stem
    assert report.metrics.cosine_similarity > 0.99
    assert report.passed is True


def test_missing_candidate_raises(settings, tiny_model: nn.Module, tmp_path: Path) -> None:
    with pytest.raises(ModelNotFoundError):
        DriftAnalyzer(settings).compare(
            tiny_model,
            tmp_path / "missing.onnx",
            input_shape=(1, 3, 32, 32),
        )


def test_num_samples_must_be_positive(
    settings,
    tiny_model: nn.Module,
    fp32_onnx: Path,
) -> None:
    with pytest.raises(DriftAnalysisError):
        DriftAnalyzer(settings).compare(
            tiny_model,
            fp32_onnx,
            num_samples=0,
        )


def test_numeric_primitives_are_exact() -> None:
    reference = np.array([1.0, 2.0, 3.0, 4.0])
    candidate = np.array([1.0, 2.0, 3.0, 4.0])
    assert cosine_similarity(reference, candidate) == pytest.approx(1.0)
    assert mean_squared_error(reference, candidate) == pytest.approx(0.0)
    assert mean_absolute_error(reference, candidate) == pytest.approx(0.0)
    assert max_absolute_error(reference, candidate) == pytest.approx(0.0)
    assert kl_divergence(reference, candidate) == pytest.approx(0.0, abs=1e-12)
    assert signal_to_noise_db(reference, candidate) == pytest.approx(120.0)
    assert top1_agreement(reference, candidate) == pytest.approx(1.0)


def test_numeric_primitives_detect_error() -> None:
    reference = np.array([[1.0, 0.0], [0.0, 1.0]])
    candidate = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert cosine_similarity(reference, candidate) == pytest.approx(0.0)
    assert mean_squared_error(reference, candidate) == pytest.approx(1.0)
    assert mean_absolute_error(reference, candidate) == pytest.approx(1.0)
    assert max_absolute_error(reference, candidate) == pytest.approx(1.0)
    assert kl_divergence(reference, candidate) > 0.0
    assert top1_agreement(reference, candidate) == pytest.approx(0.0)


def test_zero_vector_cosine_is_safe() -> None:
    assert cosine_similarity(np.zeros(3), np.zeros(3)) == pytest.approx(1.0)
    assert cosine_similarity(np.zeros(3), np.ones(3)) == pytest.approx(0.0)


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        cosine_similarity(np.ones(3), np.ones(4))


def test_compare_accepts_sequential_model(settings, fp32_onnx: Path) -> None:
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
    with torch.no_grad():
        model[1].weight.zero_()
        model[1].bias.zero_()
    report = DriftAnalyzer(settings).compare(
        model,
        fp32_onnx,
        input_shape=(1, 3, 32, 32),
        num_samples=2,
    )
    assert report.metrics.cosine_similarity == pytest.approx(0.0, abs=1e-6)
