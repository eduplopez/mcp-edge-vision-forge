"""Autonomous search over export and quantization configurations."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from torch import nn

from edge_forge.config import ForgeSettings, get_settings
from edge_forge.core.drift_analyzer import DriftAnalyzer
from edge_forge.core.exporter import OnnxExporter
from edge_forge.core.profiler import ModelProfiler
from edge_forge.core.quantizer import QuantizationEngine
from edge_forge.errors import EdgeForgeError, ExportError, TuningError
from edge_forge.models.registry import inspect_torch_model
from edge_forge.models.schemas import (
    DriftReport,
    DriftThresholds,
    ExportOptions,
    HardwareProfile,
    ModelInfo,
    Precision,
    QuantizationOptions,
    TuningCandidate,
    TuningConstraints,
    TuningResult,
    get_hardware_profile,
)
from edge_forge.utils import ensure_directory, iso_timestamp, short_id, slugify

_LOGGER = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]


class AutoTuner:
    """Explore opset, simplification and precision combinations under constraints."""

    def __init__(
        self,
        settings: ForgeSettings | None = None,
        *,
        exporter: OnnxExporter | None = None,
        quantizer: QuantizationEngine | None = None,
        analyzer: DriftAnalyzer | None = None,
        profiler: ModelProfiler | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._exporter = exporter or OnnxExporter(self._settings)
        self._quantizer = quantizer or QuantizationEngine(self._settings)
        self._analyzer = analyzer or DriftAnalyzer(self._settings)
        self._profiler = profiler or ModelProfiler(self._settings)
        self._logger = logger or _LOGGER

    @property
    def settings(self) -> ForgeSettings:
        """Runtime settings used by the tuner."""
        return self._settings

    def tune(
        self,
        model: nn.Module,
        *,
        constraints: TuningConstraints | None = None,
        input_shape: tuple[int, ...] | None = None,
        model_name: str = "model",
        workdir: str | Path | None = None,
        run_id: str | None = None,
        drift_samples: int = 8,
        progress: ProgressCallback | None = None,
    ) -> TuningResult:
        """Search the configured space and return every evaluated candidate.

        Args:
            model: PyTorch module to optimize.
            constraints: Deployment constraints and search space.
            input_shape: Shape used for export, drift and benchmarking.
            model_name: Human readable model name.
            workdir: Directory that receives the generated artifacts.
            run_id: Explicit run identifier.
            drift_samples: Number of random samples used by drift analysis.
            progress: Optional callback invoked as ``(completed, total, message)``
                after every evaluated candidate so long searches can report status.
        """
        started = time.perf_counter()
        active_constraints = constraints or TuningConstraints()
        shape = tuple(input_shape or self._settings.default_input_shape)
        profile = get_hardware_profile(active_constraints.hardware_profile or "generic-x86")
        identifier = run_id or short_id("run-")
        slug = slugify(model_name)
        run_directory = (
            Path(workdir).expanduser().resolve()
            if workdir is not None
            else self._settings.runs_dir / f"{slug}-{iso_timestamp().replace(':', '')}-{identifier}"
        )
        ensure_directory(run_directory)
        prepared = model.eval()
        source_info = inspect_torch_model(
            prepared,
            shape,
            name=model_name,
            architecture=type(model).__name__,
            seed=self._settings.seed,
        )
        thresholds = self._thresholds(active_constraints)
        candidates: list[TuningCandidate] = []
        notes: list[str] = []
        counter = 0
        extra_precisions = sum(
            1
            for precision in active_constraints.allowed_precisions
            if precision is not Precision.FP32
        )
        planned = (
            len(active_constraints.opset_candidates)
            * len(active_constraints.simplify_candidates)
            * (1 + extra_precisions)
        )

        for opset in active_constraints.opset_candidates:
            for simplified in active_constraints.simplify_candidates:
                export_options = ExportOptions(
                    opset=opset,
                    dynamic_batch=active_constraints.dynamic_batch,
                    simplify=simplified,
                )
                stem = f"{slug}-opset{opset}-simplified{int(simplified)}"
                onnx_path = run_directory / f"{stem}.onnx"
                try:
                    export_result = self._exporter.export(
                        prepared,
                        onnx_path,
                        input_shape=shape,
                        options=export_options,
                        name=model_name,
                    )
                except ExportError as exc:
                    message = f"opset={opset} simplified={simplified}: export failed: {exc}"
                    self._logger.warning(message)
                    notes.append(message)
                    continue

                counter += 1
                fp32_candidate = self._evaluate_candidate(
                    candidate_id=f"c{counter:03d}",
                    model_path=Path(export_result.model_path),
                    precision=Precision.FP32,
                    opset=opset,
                    simplified=simplified,
                    dynamic_batch=active_constraints.dynamic_batch,
                    reference_model=prepared,
                    input_shape=shape,
                    thresholds=thresholds,
                    constraints=active_constraints,
                    profile=profile,
                    drift_samples=drift_samples,
                )
                candidates.append(fp32_candidate)
                self._report(
                    progress,
                    counter,
                    planned,
                    f"evaluated fp32 candidate opset={opset} simplified={simplified}",
                )

                for precision in active_constraints.allowed_precisions:
                    if precision is Precision.FP32:
                        continue
                    counter += 1
                    candidate_id = f"c{counter:03d}"
                    quantized_path = run_directory / f"{stem}.{precision.value}.onnx"
                    try:
                        quantization = self._quantizer.quantize(
                            onnx_path,
                            quantized_path,
                            options=QuantizationOptions(
                                precision=precision,
                                calibration_samples=self._settings.calibration_samples,
                            ),
                        )
                    except EdgeForgeError as exc:
                        message = f"{candidate_id} {precision.value}: quantization failed: {exc}"
                        self._logger.warning(message)
                        notes.append(message)
                        candidates.append(
                            TuningCandidate(
                                candidate_id=candidate_id,
                                precision=precision,
                                opset=opset,
                                simplified=simplified,
                                dynamic_batch=active_constraints.dynamic_batch,
                                model_path=str(quantized_path),
                                size_mb=0.0,
                                feasible=False,
                                score=0.0,
                                rejection_reasons=[f"quantization failed: {exc}"],
                                metadata={"export_path": str(onnx_path)},
                            )
                        )
                        self._report(
                            progress,
                            counter,
                            planned,
                            f"{candidate_id} {precision.value} failed during quantization",
                        )
                        continue
                    candidate = self._evaluate_candidate(
                        candidate_id=candidate_id,
                        model_path=Path(quantization.output_path),
                        precision=precision,
                        opset=opset,
                        simplified=simplified,
                        dynamic_batch=active_constraints.dynamic_batch,
                        reference_model=prepared,
                        input_shape=shape,
                        thresholds=thresholds,
                        constraints=active_constraints,
                        profile=profile,
                        drift_samples=drift_samples,
                    )
                    candidate.metadata["compression_ratio"] = quantization.compression_ratio
                    candidate.metadata["calibration_samples"] = quantization.calibration_samples
                    candidates.append(candidate)
                    self._report(
                        progress,
                        counter,
                        planned,
                        f"evaluated {precision.value} candidate {candidate_id}",
                    )

        if not candidates:
            raise TuningError(
                "no candidate could be exported; check the model and opset configuration"
            )

        baseline = next(
            (
                candidate
                for candidate in candidates
                if candidate.precision is Precision.FP32 and candidate.benchmark is not None
            ),
            None,
        )
        for candidate in candidates:
            candidate.score = self._score(candidate, baseline, active_constraints)
        candidates.sort(key=self._sort_key)
        feasible = [candidate for candidate in candidates if candidate.feasible]
        best = max(feasible, key=lambda candidate: candidate.score) if feasible else None
        if best is None:
            notes.append("no candidate satisfied every deployment constraint")
        duration = round(time.perf_counter() - started, 4)
        result = TuningResult(
            run_id=identifier,
            source_model=source_info,
            hardware_profile=profile,
            constraints=active_constraints,
            candidates=candidates,
            best_candidate_id=best.candidate_id if best is not None else None,
            duration_s=duration,
            workdir=str(run_directory),
            notes=notes,
        )
        self._logger.info(
            "tuning finished in %.2fs with %d candidates (best=%s)",
            duration,
            len(candidates),
            result.best_candidate_id,
        )
        return result

    def _evaluate_candidate(
        self,
        *,
        candidate_id: str,
        model_path: Path,
        precision: Precision,
        opset: int,
        simplified: bool,
        dynamic_batch: bool,
        reference_model: nn.Module,
        input_shape: tuple[int, ...],
        thresholds: DriftThresholds,
        constraints: TuningConstraints,
        profile: HardwareProfile,
        drift_samples: int,
    ) -> TuningCandidate:
        drift: DriftReport | None = None
        rejection_reasons: list[str] = []
        try:
            drift = self._analyzer.compare(
                reference_model,
                model_path,
                input_shape=input_shape,
                num_samples=drift_samples,
                thresholds=thresholds,
                reference_name=type(reference_model).__name__,
            )
        except EdgeForgeError as exc:
            rejection_reasons.append(f"drift analysis failed: {exc}")
        benchmark = None
        try:
            benchmark = self._profiler.benchmark(
                model_path,
                input_shape=input_shape,
                warmup_runs=constraints.warmup_runs,
                iterations=constraints.iterations,
                hardware_profile=profile,
                precision=precision,
            )
        except EdgeForgeError as exc:
            rejection_reasons.append(f"benchmark failed: {exc}")

        size_mb = benchmark.resources.model_size_mb if benchmark is not None else 0.0
        if drift is None:
            rejection_reasons.append("no drift report available")
        else:
            rejection_reasons.extend(drift.metrics.violations)
        if constraints.max_latency_ms is not None:
            if benchmark is None:
                rejection_reasons.append("latency constraint cannot be verified")
            elif benchmark.latency.p99_ms > constraints.max_latency_ms:
                rejection_reasons.append(
                    f"p99 latency {benchmark.latency.p99_ms:.3f} ms > "
                    f"{constraints.max_latency_ms:.3f} ms"
                )
        if constraints.max_size_mb is not None and size_mb > constraints.max_size_mb:
            rejection_reasons.append(f"size {size_mb:.3f} MiB > {constraints.max_size_mb:.3f} MiB")
        if size_mb > profile.max_model_size_mb:
            rejection_reasons.append(
                f"size {size_mb:.3f} MiB exceeds hardware limit "
                f"{profile.max_model_size_mb:.3f} MiB for {profile.name}"
            )
        if precision not in profile.supported_precisions:
            rejection_reasons.append(
                f"precision {precision.value} is not supported by {profile.name}"
            )
        if precision not in constraints.allowed_precisions:
            rejection_reasons.append(f"precision {precision.value} is not allowed")
        return TuningCandidate(
            candidate_id=candidate_id,
            precision=precision,
            opset=opset,
            simplified=simplified,
            dynamic_batch=dynamic_batch,
            model_path=str(model_path),
            size_mb=size_mb,
            drift=drift,
            benchmark=benchmark,
            feasible=not rejection_reasons,
            score=0.0,
            rejection_reasons=list(dict.fromkeys(rejection_reasons)),
            metadata={"hardware_profile": profile.name},
        )

    @staticmethod
    def _score(
        candidate: TuningCandidate,
        baseline: TuningCandidate | None,
        constraints: TuningConstraints,
    ) -> float:
        if candidate.benchmark is None or candidate.drift is None:
            return 0.0
        latency = candidate.benchmark.latency.p50_ms
        size = candidate.size_mb
        if baseline is not None and baseline.benchmark is not None:
            baseline_latency = max(baseline.benchmark.latency.p50_ms, 1e-6)
            baseline_size = max(baseline.size_mb, 1e-6)
        else:
            baseline_latency = max(latency, 1e-6)
            baseline_size = max(size, 1e-6)
        latency_gain = baseline_latency / max(latency, 1e-6)
        size_gain = baseline_size / max(size, 1e-6)
        fidelity = max(0.0, min(1.0, candidate.drift.metrics.cosine_similarity))
        return round(
            constraints.score_weight_latency * latency_gain
            + constraints.score_weight_size * size_gain
            + constraints.score_weight_drift * fidelity,
            6,
        )

    @staticmethod
    def _report(
        progress: ProgressCallback | None,
        completed: int,
        total: int,
        message: str,
    ) -> None:
        """Invoke the optional progress callback without breaking the search."""
        if progress is None:
            return
        try:
            progress(completed, total, message)
        except Exception:
            _LOGGER.debug("progress callback raised", exc_info=True)

    @staticmethod
    def _sort_key(candidate: TuningCandidate) -> tuple[int, float, int]:
        return (0 if candidate.feasible else 1, -candidate.score, candidate.precision.rank)

    @staticmethod
    def _thresholds(constraints: TuningConstraints) -> DriftThresholds:
        return DriftThresholds(
            min_cosine_similarity=constraints.min_cosine_similarity,
            max_mse=constraints.max_mse if constraints.max_mse is not None else 1e-2,
            max_absolute_error=(
                constraints.max_absolute_error
                if constraints.max_absolute_error is not None
                else 1.0
            ),
            max_kl_divergence=(
                constraints.max_kl_divergence if constraints.max_kl_divergence is not None else 0.1
            ),
        )


def summarize_tuning(result: TuningResult) -> dict[str, Any]:
    """Return a compact summary of a tuning run for logs and MCP clients."""
    best = result.best
    return {
        "run_id": result.run_id,
        "candidates": len(result.candidates),
        "feasible": sum(1 for candidate in result.candidates if candidate.feasible),
        "best_candidate_id": result.best_candidate_id,
        "best_precision": best.precision.value if best is not None else None,
        "best_latency_p99_ms": (
            best.benchmark.latency.p99_ms
            if best is not None and best.benchmark is not None
            else None
        ),
        "best_size_mb": best.size_mb if best is not None else None,
        "best_cosine_similarity": (
            best.drift.metrics.cosine_similarity
            if best is not None and best.drift is not None
            else None
        ),
        "duration_s": result.duration_s,
        "workdir": result.workdir,
    }


def model_info_for(model: nn.Module, input_shape: tuple[int, ...], name: str) -> ModelInfo:
    """Build model metadata without exporting it."""
    return inspect_torch_model(model, input_shape, name=name, architecture=type(model).__name__)
