"""Pydantic v2 data models shared by every subsystem.

The models in this module are the contract between the CLI, the MCP server, the
core optimization pipeline and the reporting layer. They are intentionally
serialization friendly: every instance can be dumped to JSON without custom
encoders beyond ``model_dump(mode="json")``.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from edge_forge.config import MAX_OPSET, MIN_OPSET
from edge_forge.limits import (
    MAX_BENCHMARK_ITERATIONS,
    MAX_CALIBRATION_SAMPLES,
    MAX_TUNING_CANDIDATES,
    MAX_WARMUP_RUNS,
)
from edge_forge.utils import utc_now


class Precision(StrEnum):
    """Supported numerical precisions for exported ONNX artifacts."""

    FP32 = "fp32"
    FP16 = "fp16"
    INT8_DYNAMIC = "int8_dynamic"
    INT8_STATIC = "int8_static"

    @property
    def rank(self) -> int:
        """Return a monotonic rank from the least to the most compressed format."""
        return {
            Precision.FP32: 0,
            Precision.FP16: 1,
            Precision.INT8_DYNAMIC: 2,
            Precision.INT8_STATIC: 3,
        }[self]

    @property
    def is_quantized(self) -> bool:
        """Whether the precision stores integer weights."""
        return self in (Precision.INT8_DYNAMIC, Precision.INT8_STATIC)


class TensorSpec(BaseModel):
    """Description of a single model tensor."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    shape: tuple[int, ...] = Field(min_length=1)
    dtype: str = Field(default="float32", min_length=1)

    @field_validator("shape")
    @classmethod
    def _validate_shape(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(dimension <= 0 for dimension in value):
            raise ValueError("tensor dimensions must be strictly positive")
        return value

    @property
    def numel(self) -> int:
        """Number of elements described by the tensor."""
        total = 1
        for dimension in self.shape:
            total *= dimension
        return total


class ModelInfo(BaseModel):
    """Static metadata describing a model artifact."""

    name: str = Field(min_length=1)
    architecture: str = Field(default="unknown", min_length=1)
    parameters: int = Field(default=0, ge=0)
    size_mb: float = Field(default=0.0, ge=0.0)
    input_specs: list[TensorSpec] = Field(default_factory=list)
    output_specs: list[TensorSpec] = Field(default_factory=list)
    precision: Precision = Precision.FP32
    opset: int | None = Field(default=None, ge=MIN_OPSET, le=MAX_OPSET)
    path: str | None = None
    checksum: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class HardwareProfile(BaseModel):
    """Target device constraints used to judge whether a model is deployable."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    cpu_cores: int = Field(ge=1)
    ram_mb: int = Field(ge=128)
    accelerator: Literal["cpu", "gpu", "npu", "dsp"] = "cpu"
    typical_latency_budget_ms: float = Field(gt=0.0)
    max_model_size_mb: float = Field(gt=0.0)
    supported_precisions: tuple[Precision, ...] = Field(default_factory=tuple)

    @field_validator("supported_precisions")
    @classmethod
    def _validate_precisions(cls, value: tuple[Precision, ...]) -> tuple[Precision, ...]:
        if not value:
            raise ValueError("supported_precisions cannot be empty")
        return tuple(dict.fromkeys(value))


HARDWARE_PROFILES: dict[str, HardwareProfile] = {
    "raspberry-pi-5": HardwareProfile(
        name="raspberry-pi-5",
        description="Raspberry Pi 5 with a quad-core Cortex-A76 CPU and 8 GiB of RAM.",
        cpu_cores=4,
        ram_mb=8192,
        accelerator="cpu",
        typical_latency_budget_ms=100.0,
        max_model_size_mb=128.0,
        supported_precisions=(Precision.FP32, Precision.FP16, Precision.INT8_DYNAMIC),
    ),
    "jetson-orin-nano": HardwareProfile(
        name="jetson-orin-nano",
        description="NVIDIA Jetson Orin Nano 8 GiB with a 1024-core Ampere GPU.",
        cpu_cores=6,
        ram_mb=8192,
        accelerator="gpu",
        typical_latency_budget_ms=30.0,
        max_model_size_mb=512.0,
        supported_precisions=(
            Precision.FP32,
            Precision.FP16,
            Precision.INT8_DYNAMIC,
            Precision.INT8_STATIC,
        ),
    ),
    "coral-edge-tpu": HardwareProfile(
        name="coral-edge-tpu",
        description="Google Coral Edge TPU accelerator with 4 GiB of host memory.",
        cpu_cores=4,
        ram_mb=4096,
        accelerator="npu",
        typical_latency_budget_ms=20.0,
        max_model_size_mb=64.0,
        supported_precisions=(Precision.INT8_STATIC,),
    ),
    "smartphone-midrange": HardwareProfile(
        name="smartphone-midrange",
        description="Mid-range ARM smartphone with 6 GiB of RAM and no dedicated NPU.",
        cpu_cores=8,
        ram_mb=6144,
        accelerator="cpu",
        typical_latency_budget_ms=40.0,
        max_model_size_mb=200.0,
        supported_precisions=(Precision.FP32, Precision.FP16, Precision.INT8_DYNAMIC),
    ),
    "generic-x86": HardwareProfile(
        name="generic-x86",
        description="Generic x86-64 development workstation used as a reference host.",
        cpu_cores=8,
        ram_mb=16384,
        accelerator="cpu",
        typical_latency_budget_ms=50.0,
        max_model_size_mb=1024.0,
        supported_precisions=(
            Precision.FP32,
            Precision.FP16,
            Precision.INT8_DYNAMIC,
            Precision.INT8_STATIC,
        ),
    ),
}


def get_hardware_profile(name: str) -> HardwareProfile:
    """Return a built-in hardware profile by name."""
    try:
        return HARDWARE_PROFILES[name]
    except KeyError as exc:
        available = ", ".join(sorted(HARDWARE_PROFILES))
        raise ValueError(f"unknown hardware profile '{name}'; available: {available}") from exc


def list_hardware_profiles() -> list[HardwareProfile]:
    """Return every built-in hardware profile sorted by name."""
    return [HARDWARE_PROFILES[key] for key in sorted(HARDWARE_PROFILES)]


class DriftThresholds(BaseModel):
    """Acceptance thresholds used to decide whether a candidate is faithful."""

    min_cosine_similarity: float = Field(default=0.99, ge=0.0, le=1.0)
    max_mse: float = Field(default=1e-2, ge=0.0)
    max_mae: float = Field(default=1e-1, ge=0.0)
    max_absolute_error: float = Field(default=1.0, ge=0.0)
    max_kl_divergence: float = Field(default=0.1, ge=0.0)
    min_snr_db: float = Field(default=20.0)

    def violations(self, metrics: DriftMetrics) -> list[str]:
        """Return a human readable list of threshold violations."""
        failures: list[str] = []
        if metrics.cosine_similarity < self.min_cosine_similarity:
            failures.append(
                f"cosine_similarity {metrics.cosine_similarity:.6f} < "
                f"{self.min_cosine_similarity:.6f}"
            )
        if metrics.mse > self.max_mse:
            failures.append(f"mse {metrics.mse:.6g} > {self.max_mse:.6g}")
        if metrics.mae > self.max_mae:
            failures.append(f"mae {metrics.mae:.6g} > {self.max_mae:.6g}")
        if metrics.max_absolute_error > self.max_absolute_error:
            failures.append(
                f"max_absolute_error {metrics.max_absolute_error:.6g} > "
                f"{self.max_absolute_error:.6g}"
            )
        if metrics.kl_divergence > self.max_kl_divergence:
            failures.append(
                f"kl_divergence {metrics.kl_divergence:.6g} > {self.max_kl_divergence:.6g}"
            )
        if metrics.snr_db < self.min_snr_db:
            failures.append(f"snr_db {metrics.snr_db:.2f} < {self.min_snr_db:.2f}")
        return failures


class DriftMetrics(BaseModel):
    """Numerical distance between a reference and a candidate output tensor."""

    cosine_similarity: float = Field(ge=-1.0, le=1.0)
    mse: float = Field(ge=0.0)
    mae: float = Field(ge=0.0)
    max_absolute_error: float = Field(ge=0.0)
    kl_divergence: float = Field(ge=0.0)
    snr_db: float
    relative_l1_error: float = Field(ge=0.0)
    top1_agreement: float = Field(ge=0.0, le=1.0)
    reference_norm: float = Field(ge=0.0)
    candidate_norm: float = Field(ge=0.0)
    passed: bool = True
    violations: list[str] = Field(default_factory=list)


class OutputDrift(BaseModel):
    """Drift metrics for one named model output."""

    output_name: str = Field(min_length=1)
    shape: tuple[int, ...] = Field(min_length=1)
    metrics: DriftMetrics


class DriftReport(BaseModel):
    """Complete fidelity report comparing two model executions."""

    reference_model: str = Field(min_length=1)
    candidate_model: str = Field(min_length=1)
    input_shape: tuple[int, ...] = Field(min_length=1)
    samples: int = Field(ge=1)
    metrics: DriftMetrics
    per_output: list[OutputDrift] = Field(default_factory=list)
    thresholds: DriftThresholds = Field(default_factory=DriftThresholds)
    passed: bool = True
    created_at: datetime = Field(default_factory=utc_now)


class ValidationResult(BaseModel):
    """Result of checking an exported ONNX graph against the PyTorch reference."""

    passed: bool
    max_absolute_error: float = Field(ge=0.0)
    mse: float = Field(ge=0.0)
    cosine_similarity: float = Field(ge=-1.0, le=1.0)
    message: str
    provider: str = "CPUExecutionProvider"


class ExportOptions(BaseModel):
    """Options controlling the PyTorch to ONNX export."""

    opset: int = Field(default=17, ge=MIN_OPSET, le=MAX_OPSET)
    dynamic_batch: bool = True
    dynamic_spatial: bool = False
    simplify: bool = True
    constant_folding: bool = True
    validate_output: bool = True
    input_names: list[str] = Field(default_factory=lambda: ["input"])
    output_names: list[str] = Field(default_factory=lambda: ["output"])
    atol: float = Field(default=1e-4, gt=0.0)
    rtol: float = Field(default=1e-3, gt=0.0)
    dynamo: bool = False

    @field_validator("input_names", "output_names")
    @classmethod
    def _validate_names(cls, value: list[str]) -> list[str]:
        if not value or any(not name.strip() for name in value):
            raise ValueError("tensor names must be non-empty strings")
        if len(set(value)) != len(value):
            raise ValueError("tensor names must be unique")
        return value


class ExportResult(BaseModel):
    """Outcome of exporting and validating an ONNX artifact."""

    model_path: str
    opset: int
    size_mb: float = Field(ge=0.0)
    parameters: int = Field(ge=0)
    input_specs: list[TensorSpec]
    output_specs: list[TensorSpec]
    dynamic_axes: dict[str, dict[int, str]] = Field(default_factory=dict)
    simplified: bool = False
    validation: ValidationResult | None = None
    checksum: str
    duration_s: float = Field(ge=0.0)


class CalibrationMethod(StrEnum):
    """Synthetic distribution used to build calibration batches."""

    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"
    ZEROS = "zeros"
    REAL = "real"


class QuantizationOptions(BaseModel):
    """Options controlling FP16 and INT8 conversion."""

    precision: Precision = Precision.INT8_DYNAMIC
    per_channel: bool = True
    reduce_range: bool = False
    calibration_samples: int = Field(default=16, ge=1, le=MAX_CALIBRATION_SAMPLES)
    calibration_method: CalibrationMethod = CalibrationMethod.GAUSSIAN
    calibration_data_path: str | None = None
    weight_type: Literal["qint8", "quint8"] = "qint8"
    activation_type: Literal["quint8", "qint8"] = "quint8"
    quant_format: Literal["qdq", "qoperator"] = "qdq"
    op_types_to_quantize: list[str] | None = None
    nodes_to_exclude: list[str] = Field(
        default_factory=list,
        description=(
            "Node names that must stay in floating point. Useful for operators "
            "whose quantized kernels are slower or numerically unstable on edge hardware."
        ),
    )
    extra_options: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_calibration_source(self) -> QuantizationOptions:
        if (
            self.precision is Precision.INT8_STATIC
            and self.calibration_method is CalibrationMethod.REAL
            and not self.calibration_data_path
        ):
            raise ValueError("real calibration requires calibration_data_path")
        return self


class QuantizationResult(BaseModel):
    """Outcome of converting a model to a compressed precision."""

    source_path: str
    output_path: str
    precision: Precision
    size_mb: float = Field(ge=0.0)
    source_size_mb: float = Field(ge=0.0)
    compression_ratio: float = Field(gt=0.0)
    opset: int | None = None
    calibration_samples: int = Field(default=0, ge=0)
    checksum: str
    duration_s: float = Field(ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LatencyStats(BaseModel):
    """Latency distribution collected during a benchmark."""

    count: int = Field(ge=1)
    mean_ms: float = Field(ge=0.0)
    min_ms: float = Field(ge=0.0)
    max_ms: float = Field(ge=0.0)
    std_ms: float = Field(ge=0.0)
    p50_ms: float = Field(ge=0.0)
    p90_ms: float = Field(ge=0.0)
    p99_ms: float = Field(ge=0.0)
    fps: float = Field(ge=0.0)
    throughput_per_second: float = Field(ge=0.0)


class ResourceUsage(BaseModel):
    """Process resources observed while running a benchmark."""

    peak_rss_mb: float = Field(ge=0.0)
    rss_delta_mb: float
    cpu_percent_mean: float = Field(ge=0.0)
    cpu_percent_peak: float = Field(ge=0.0)
    thread_count: int = Field(ge=1)
    model_size_mb: float = Field(ge=0.0)


class BenchmarkReport(BaseModel):
    """Benchmark of one ONNX artifact on the current host."""

    model_name: str = Field(min_length=1)
    model_path: str
    precision: Precision
    opset: int | None = None
    parameters: int = Field(default=0, ge=0)
    provider: str = Field(min_length=1)
    hardware_profile: str = Field(min_length=1)
    latency_budget_ms: float | None = Field(default=None, gt=0.0)
    meets_budget: bool | None = None
    input_specs: list[TensorSpec] = Field(default_factory=list)
    latency: LatencyStats
    resources: ResourceUsage
    warmup_runs: int = Field(ge=0)
    iterations: int = Field(ge=1)
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TuningConstraints(BaseModel):
    """Deployment constraints used by the autonomous search."""

    max_latency_ms: float | None = Field(default=None, gt=0.0)
    min_cosine_similarity: float = Field(default=0.99, ge=0.0, le=1.0)
    max_size_mb: float | None = Field(default=None, gt=0.0)
    max_mse: float | None = Field(default=None, ge=0.0)
    max_absolute_error: float | None = Field(default=None, ge=0.0)
    max_kl_divergence: float | None = Field(default=None, ge=0.0)
    allowed_precisions: list[Precision] = Field(
        default_factory=lambda: [
            Precision.FP32,
            Precision.FP16,
            Precision.INT8_DYNAMIC,
            Precision.INT8_STATIC,
        ]
    )
    opset_candidates: list[int] = Field(default_factory=lambda: [17])
    simplify_candidates: list[bool] = Field(default_factory=lambda: [True])
    dynamic_batch: bool = True
    hardware_profile: str | None = None
    warmup_runs: int = Field(default=5, ge=0, le=MAX_WARMUP_RUNS)
    iterations: int = Field(default=30, ge=1, le=MAX_BENCHMARK_ITERATIONS)
    score_weight_latency: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Relative weight of the latency term in the candidate score.",
    )
    score_weight_size: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Relative weight of the model size term in the candidate score.",
    )
    score_weight_drift: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Relative weight of the numerical fidelity term in the candidate score.",
    )

    @field_validator("allowed_precisions")
    @classmethod
    def _validate_precisions(cls, value: list[Precision]) -> list[Precision]:
        if not value:
            raise ValueError("allowed_precisions cannot be empty")
        return list(dict.fromkeys(value))

    @field_validator("opset_candidates")
    @classmethod
    def _validate_opsets(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("opset_candidates cannot be empty")
        for opset in value:
            if opset < MIN_OPSET or opset > MAX_OPSET:
                raise ValueError(f"opset {opset} is outside the supported range")
        return list(dict.fromkeys(value))

    @field_validator("simplify_candidates")
    @classmethod
    def _validate_simplify(cls, value: list[bool]) -> list[bool]:
        if not value:
            raise ValueError("simplify_candidates cannot be empty")
        return list(dict.fromkeys(value))

    @model_validator(mode="after")
    def _validate_budget(self) -> TuningConstraints:
        if (
            self.max_latency_ms is None
            and self.max_size_mb is None
            and len(self.allowed_precisions) == 1
            and self.allowed_precisions[0] is Precision.FP32
        ):
            raise ValueError(
                "constraints must bound latency or size, or allow a compressed precision"
            )
        if self.hardware_profile is not None:
            get_hardware_profile(self.hardware_profile)
        weight_sum = self.score_weight_latency + self.score_weight_size + self.score_weight_drift
        if weight_sum <= 0.0:
            raise ValueError("at least one score weight must be strictly positive")
        combinations = (
            len(self.allowed_precisions)
            * len(self.opset_candidates)
            * len(self.simplify_candidates)
        )
        if combinations > MAX_TUNING_CANDIDATES:
            raise ValueError(
                f"the search space has {combinations} candidates, exceeding the limit of "
                f"{MAX_TUNING_CANDIDATES}"
            )
        return self


class TuningCandidate(BaseModel):
    """One fully evaluated point of the search space."""

    candidate_id: str = Field(min_length=1)
    precision: Precision
    opset: int = Field(ge=MIN_OPSET, le=MAX_OPSET)
    simplified: bool
    dynamic_batch: bool
    model_path: str
    size_mb: float = Field(ge=0.0)
    drift: DriftReport | None = None
    benchmark: BenchmarkReport | None = None
    feasible: bool = False
    score: float = 0.0
    rejection_reasons: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TuningResult(BaseModel):
    """Complete output of an autonomous tuning run."""

    run_id: str = Field(min_length=1)
    source_model: ModelInfo
    hardware_profile: HardwareProfile
    constraints: TuningConstraints
    candidates: list[TuningCandidate] = Field(default_factory=list)
    best_candidate_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    duration_s: float = Field(ge=0.0)
    workdir: str
    notes: list[str] = Field(default_factory=list)

    @property
    def best(self) -> TuningCandidate | None:
        """Return the winning candidate, if the search found a feasible one."""
        if self.best_candidate_id is None:
            return None
        for candidate in self.candidates:
            if candidate.candidate_id == self.best_candidate_id:
                return candidate
        return None


class ForgeReport(BaseModel):
    """Aggregated, presentation-ready view of a tuning run."""

    report_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    generated_at: datetime = Field(default_factory=utc_now)
    source_model: ModelInfo
    hardware_profile: HardwareProfile
    constraints: TuningConstraints
    candidates: list[TuningCandidate] = Field(default_factory=list)
    best_candidate_id: str | None = None
    environment: dict[str, str] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    @property
    def best(self) -> TuningCandidate | None:
        """Return the winning candidate, if any."""
        if self.best_candidate_id is None:
            return None
        for candidate in self.candidates:
            if candidate.candidate_id == self.best_candidate_id:
                return candidate
        return None
