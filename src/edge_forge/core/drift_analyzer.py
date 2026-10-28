"""Mathematical drift analysis between reference and optimized model outputs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from edge_forge.config import ForgeSettings, get_settings
from edge_forge.core.exporter import inspect_onnx
from edge_forge.core.numerics import (
    assert_finite,
    cosine_similarity,
    kl_divergence,
    max_absolute_error,
    mean_absolute_error,
    mean_squared_error,
    relative_l1_error,
    signal_to_noise_db,
    top1_agreement,
)
from edge_forge.core.ort_utils import create_session
from edge_forge.errors import DriftAnalysisError, ModelNotFoundError
from edge_forge.limits import validate_input_shape, validate_num_samples
from edge_forge.models.registry import extract_output_tensors
from edge_forge.models.schemas import (
    DriftMetrics,
    DriftReport,
    DriftThresholds,
    OutputDrift,
)

_LOGGER = logging.getLogger(__name__)


def _build_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    thresholds: DriftThresholds,
) -> DriftMetrics:
    reference_flat = np.asarray(reference, dtype=np.float64).reshape(-1)
    candidate_flat = np.asarray(candidate, dtype=np.float64).reshape(-1)
    assert_finite(reference_flat, label="reference output")
    assert_finite(candidate_flat, label="candidate output")
    metrics = DriftMetrics(
        cosine_similarity=cosine_similarity(reference_flat, candidate_flat),
        mse=mean_squared_error(reference_flat, candidate_flat),
        mae=mean_absolute_error(reference_flat, candidate_flat),
        max_absolute_error=max_absolute_error(reference_flat, candidate_flat),
        kl_divergence=kl_divergence(reference_flat, candidate_flat),
        snr_db=signal_to_noise_db(reference_flat, candidate_flat),
        relative_l1_error=relative_l1_error(reference_flat, candidate_flat),
        top1_agreement=top1_agreement(np.asarray(reference), np.asarray(candidate)),
        reference_norm=float(np.linalg.norm(reference_flat)),
        candidate_norm=float(np.linalg.norm(candidate_flat)),
    )
    violations = thresholds.violations(metrics)
    return metrics.model_copy(update={"passed": not violations, "violations": violations})


def _resolve_shape(
    input_shape: tuple[int, ...] | None,
    specs: list[Any],
    fallback: tuple[int, ...],
) -> tuple[int, ...]:
    if input_shape is not None:
        return tuple(input_shape)
    if specs:
        return tuple(dimension if dimension > 0 else 1 for dimension in specs[0].shape)
    return fallback


class DriftAnalyzer:
    """Compare a PyTorch reference against an ONNX candidate or two ONNX graphs."""

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

    def compare(
        self,
        reference_model: nn.Module,
        candidate_path: str | Path,
        *,
        input_shape: tuple[int, ...] | None = None,
        num_samples: int = 8,
        thresholds: DriftThresholds | None = None,
        reference_name: str | None = None,
    ) -> DriftReport:
        """Compare a PyTorch module with an ONNX artifact."""
        try:
            validate_num_samples(num_samples)
        except ValueError as exc:
            raise DriftAnalysisError(str(exc)) from exc
        candidate = Path(candidate_path).expanduser().resolve()
        if not candidate.exists():
            raise ModelNotFoundError(f"candidate ONNX artifact does not exist: {candidate}")
        info = inspect_onnx(candidate)
        if len(info.input_specs) != 1:
            raise DriftAnalysisError(
                "drift analysis currently supports models with exactly one input"
            )
        shape = _resolve_shape(input_shape, info.input_specs, self._settings.default_input_shape)
        try:
            validate_input_shape(shape)
        except ValueError as exc:
            raise DriftAnalysisError(str(exc)) from exc
        session = self._create_session(candidate)
        input_name = session.get_inputs()[0].name
        rng = np.random.default_rng(self._settings.seed)
        reference_batches: list[list[np.ndarray]] = []
        candidate_batches: list[list[np.ndarray]] = []
        reference_model.eval()
        for _ in range(num_samples):
            sample = rng.standard_normal(size=shape).astype(np.float32)
            try:
                with torch.inference_mode():
                    torch_outputs = [
                        tensor.detach().cpu().numpy()
                        for tensor in extract_output_tensors(
                            reference_model(torch.from_numpy(sample))
                        )
                    ]
                onnx_outputs = session.run(None, {input_name: sample})
            except Exception as exc:
                raise DriftAnalysisError(f"unable to execute models: {exc}") from exc
            if len(torch_outputs) != len(onnx_outputs):
                raise DriftAnalysisError(
                    f"output count mismatch: torch={len(torch_outputs)} onnx={len(onnx_outputs)}"
                )
            reference_batches.append(torch_outputs)
            candidate_batches.append(onnx_outputs)

        output_names = [output.name for output in session.get_outputs()]
        per_output: list[OutputDrift] = []
        aggregated_reference: list[np.ndarray] = []
        aggregated_candidate: list[np.ndarray] = []
        for index, output_name in enumerate(output_names):
            reference = np.concatenate(
                [np.asarray(batch[index]) for batch in reference_batches], axis=0
            )
            candidate_output = np.concatenate(
                [np.asarray(batch[index]) for batch in candidate_batches], axis=0
            )
            if reference.shape != candidate_output.shape:
                raise DriftAnalysisError(
                    f"shape mismatch for output '{output_name}': "
                    f"{reference.shape} vs {candidate_output.shape}"
                )
            output_thresholds = thresholds or DriftThresholds()
            metrics = _build_metrics(reference, candidate_output, output_thresholds)
            per_output.append(
                OutputDrift(
                    output_name=output_name,
                    shape=tuple(int(dimension) for dimension in reference.shape),
                    metrics=metrics,
                )
            )
            aggregated_reference.append(reference.reshape(-1))
            aggregated_candidate.append(candidate_output.reshape(-1))

        effective_thresholds = thresholds or DriftThresholds()
        overall = _build_metrics(
            np.concatenate(aggregated_reference),
            np.concatenate(aggregated_candidate),
            effective_thresholds,
        )
        report = DriftReport(
            reference_model=reference_name or type(reference_model).__name__,
            candidate_model=candidate.stem,
            input_shape=shape,
            samples=num_samples,
            metrics=overall,
            per_output=per_output,
            thresholds=effective_thresholds,
            passed=overall.passed,
        )
        self._logger.info(
            "drift cosine=%.6f mse=%.6g max_abs=%.6g passed=%s",
            report.metrics.cosine_similarity,
            report.metrics.mse,
            report.metrics.max_absolute_error,
            report.passed,
        )
        return report

    def compare_onnx(
        self,
        reference_path: str | Path,
        candidate_path: str | Path,
        *,
        input_shape: tuple[int, ...] | None = None,
        num_samples: int = 8,
        thresholds: DriftThresholds | None = None,
    ) -> DriftReport:
        """Compare two ONNX artifacts, useful for auditing quantized variants."""
        try:
            validate_num_samples(num_samples)
        except ValueError as exc:
            raise DriftAnalysisError(str(exc)) from exc
        reference = Path(reference_path).expanduser().resolve()
        candidate = Path(candidate_path).expanduser().resolve()
        for path in (reference, candidate):
            if not path.exists():
                raise ModelNotFoundError(f"ONNX artifact does not exist: {path}")
        reference_info = inspect_onnx(reference)
        shape = _resolve_shape(
            input_shape, reference_info.input_specs, self._settings.default_input_shape
        )
        try:
            validate_input_shape(shape)
        except ValueError as exc:
            raise DriftAnalysisError(str(exc)) from exc
        reference_session = self._create_session(reference)
        candidate_session = self._create_session(candidate)
        reference_input = reference_session.get_inputs()[0].name
        candidate_input = candidate_session.get_inputs()[0].name
        rng = np.random.default_rng(self._settings.seed)
        reference_batches: list[list[np.ndarray]] = []
        candidate_batches: list[list[np.ndarray]] = []
        for _ in range(num_samples):
            sample = rng.standard_normal(size=shape).astype(np.float32)
            try:
                reference_batches.append(reference_session.run(None, {reference_input: sample}))
                candidate_batches.append(candidate_session.run(None, {candidate_input: sample}))
            except Exception as exc:
                raise DriftAnalysisError(f"unable to execute ONNX models: {exc}") from exc
        output_names = [output.name for output in candidate_session.get_outputs()]
        if len(reference_batches[0]) != len(candidate_batches[0]):
            raise DriftAnalysisError("ONNX models expose a different number of outputs")
        per_output: list[OutputDrift] = []
        aggregated_reference: list[np.ndarray] = []
        aggregated_candidate: list[np.ndarray] = []
        for index, output_name in enumerate(output_names):
            reference_values = np.concatenate(
                [np.asarray(batch[index]) for batch in reference_batches], axis=0
            )
            candidate_values = np.concatenate(
                [np.asarray(batch[index]) for batch in candidate_batches], axis=0
            )
            if reference_values.shape != candidate_values.shape:
                raise DriftAnalysisError(
                    f"shape mismatch for output '{output_name}': "
                    f"{reference_values.shape} vs {candidate_values.shape}"
                )
            metrics = _build_metrics(
                reference_values,
                candidate_values,
                thresholds or DriftThresholds(),
            )
            per_output.append(
                OutputDrift(
                    output_name=output_name,
                    shape=tuple(int(dimension) for dimension in reference_values.shape),
                    metrics=metrics,
                )
            )
            aggregated_reference.append(reference_values.reshape(-1))
            aggregated_candidate.append(candidate_values.reshape(-1))
        overall = _build_metrics(
            np.concatenate(aggregated_reference),
            np.concatenate(aggregated_candidate),
            thresholds or DriftThresholds(),
        )
        return DriftReport(
            reference_model=reference.stem,
            candidate_model=candidate.stem,
            input_shape=shape,
            samples=num_samples,
            metrics=overall,
            per_output=per_output,
            thresholds=thresholds or DriftThresholds(),
            passed=overall.passed,
        )

    def _create_session(self, path: Path) -> ort.InferenceSession:
        try:
            return create_session(
                str(path),
                provider=self._provider,
                intra_op_threads=self._settings.intra_op_threads,
                inter_op_threads=self._settings.inter_op_threads,
            )
        except Exception as exc:
            raise DriftAnalysisError(f"unable to load ONNX artifact {path}: {exc}") from exc
