"""Dynamic and static INT8 quantization plus FP16 conversion for ONNX models."""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import helper
from onnxconverter_common import float16
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from onnxruntime.quantization import (
    CalibrationMethod as OrtCalibrationMethod,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from edge_forge.config import ForgeSettings, get_settings
from edge_forge.core.exporter import inspect_onnx
from edge_forge.core.ort_utils import create_session
from edge_forge.errors import CalibrationError, ModelNotFoundError, QuantizationError
from edge_forge.models.schemas import (
    CalibrationMethod,
    Precision,
    QuantizationOptions,
    QuantizationResult,
    TensorSpec,
)
from edge_forge.utils import file_size_mb, sha256_file, short_id

_LOGGER = logging.getLogger(__name__)
_WEIGHT_TYPES: dict[str, QuantType] = {"qint8": QuantType.QInt8, "quint8": QuantType.QUInt8}
_ACTIVATION_TYPES: dict[str, QuantType] = {
    "quint8": QuantType.QUInt8,
    "qint8": QuantType.QInt8,
}
_QUANT_FORMATS: dict[str, QuantFormat] = {
    "qdq": QuantFormat.QDQ,
    "qoperator": QuantFormat.QOperator,
}


def _normalized_shape(spec: TensorSpec) -> tuple[int, ...]:
    return tuple(dimension if dimension > 0 else 1 for dimension in spec.shape)


class SyntheticCalibrationReader(CalibrationDataReader):  # type: ignore[misc]
    """Generate deterministic calibration batches from synthetic distributions."""

    def __init__(
        self,
        input_specs: list[TensorSpec],
        *,
        samples: int = 16,
        method: CalibrationMethod = CalibrationMethod.GAUSSIAN,
        seed: int = 42,
    ) -> None:
        if not input_specs:
            raise CalibrationError("synthetic calibration requires at least one input spec")
        if samples < 1:
            raise CalibrationError("calibration samples must be positive")
        self._specs = input_specs
        self._samples = samples
        self._method = method
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._index = 0

    @property
    def samples(self) -> int:
        """Number of batches this reader will emit."""
        return self._samples

    def get_next(self) -> dict[str, np.ndarray] | None:
        """Return the next calibration batch or ``None`` when exhausted."""
        if self._index >= self._samples:
            return None
        batch: dict[str, np.ndarray] = {}
        for spec in self._specs:
            shape = _normalized_shape(spec)
            if self._method is CalibrationMethod.ZEROS:
                values = np.zeros(shape, dtype=np.float32)
            elif self._method is CalibrationMethod.UNIFORM:
                values = self._rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
            else:
                values = self._rng.standard_normal(size=shape).astype(np.float32)
            batch[spec.name] = values
        self._index += 1
        return batch

    def rewind(self) -> None:
        """Restart the deterministic sequence from the beginning."""
        self._rng = np.random.default_rng(self._seed)
        self._index = 0


class RealCalibrationReader(CalibrationDataReader):  # type: ignore[misc]
    """Read calibration batches from ``.npy``, ``.npz`` files or a directory."""

    def __init__(
        self,
        input_specs: list[TensorSpec],
        source: str | Path,
        *,
        samples: int = 16,
    ) -> None:
        if not input_specs:
            raise CalibrationError("real calibration requires at least one input spec")
        self._specs = input_specs
        self._source = Path(source).expanduser().resolve()
        self._arrays = self._load_arrays()
        available = min(len(array) for array in self._arrays.values())
        self._samples = min(samples, available)
        if self._samples < 1:
            raise CalibrationError("calibration source does not contain any sample")
        self._validate_shapes()
        self._index = 0

    @property
    def samples(self) -> int:
        """Number of batches available for calibration."""
        return self._samples

    def _load_arrays(self) -> dict[str, np.ndarray]:
        if not self._source.exists():
            raise CalibrationError(f"calibration data path does not exist: {self._source}")
        expected = {spec.name for spec in self._specs}
        if self._source.is_dir():
            arrays: dict[str, np.ndarray] = {}
            for spec in self._specs:
                candidate = self._source / f"{spec.name}.npy"
                if not candidate.exists():
                    raise CalibrationError(
                        f"missing calibration array '{candidate.name}' in {self._source}"
                    )
                arrays[spec.name] = np.load(candidate)
            return arrays
        if self._source.suffix.lower() == ".npz":
            with np.load(self._source) as archive:
                keys = set(archive.files)
                missing = expected - keys
                if missing:
                    raise CalibrationError(
                        f"calibration archive is missing arrays for: {sorted(missing)}"
                    )
                return {name: archive[name] for name in sorted(expected)}
        if self._source.suffix.lower() == ".npy":
            array = np.load(self._source)
            name = self._specs[0].name
            return {name: array}
        raise CalibrationError(
            f"unsupported calibration format '{self._source.suffix}'; use .npy, .npz or a directory"
        )

    def _validate_shapes(self) -> None:
        for spec in self._specs:
            array = self._arrays[spec.name]
            expected = _normalized_shape(spec)[1:]
            if array.ndim == 0:
                raise CalibrationError(f"calibration array for '{spec.name}' is scalar")
            if tuple(array.shape[1:]) != expected:
                raise CalibrationError(
                    f"calibration array for '{spec.name}' has per-sample shape "
                    f"{tuple(array.shape[1:])}, expected {expected}"
                )

    def get_next(self) -> dict[str, np.ndarray] | None:
        """Return the next real calibration batch or ``None`` when exhausted."""
        if self._index >= self._samples:
            return None
        batch = {
            spec.name: np.expand_dims(
                np.asarray(self._arrays[spec.name][self._index], dtype=np.float32), axis=0
            )
            for spec in self._specs
        }
        self._index += 1
        return batch

    def rewind(self) -> None:
        """Restart the dataset from the first sample."""
        self._index = 0


class CalibrationReaderFactory:
    """Build the calibration reader required by a quantization strategy."""

    @staticmethod
    def create(
        options: QuantizationOptions,
        input_specs: list[TensorSpec],
        *,
        seed: int = 42,
    ) -> CalibrationDataReader | None:
        """Return a reader for static quantization, otherwise ``None``."""
        if options.precision is not Precision.INT8_STATIC:
            return None
        if options.calibration_method is CalibrationMethod.REAL:
            if not options.calibration_data_path:
                raise CalibrationError("real calibration requires calibration_data_path")
            return RealCalibrationReader(
                input_specs,
                options.calibration_data_path,
                samples=options.calibration_samples,
            )
        return SyntheticCalibrationReader(
            input_specs,
            samples=options.calibration_samples,
            method=options.calibration_method,
            seed=seed,
        )


def _reader_sample_count(reader: CalibrationDataReader | None, fallback: int) -> int:
    samples = getattr(reader, "samples", None)
    if isinstance(samples, int):
        return samples
    return fallback


class QuantizationEngine:
    """Convert FP32 ONNX graphs to FP16, dynamic INT8 or static INT8."""

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

    @property
    def settings(self) -> ForgeSettings:
        """Runtime settings used by the engine."""
        return self._settings

    def quantize(
        self,
        source: str | Path,
        output_path: str | Path | None = None,
        *,
        options: QuantizationOptions | None = None,
    ) -> QuantizationResult:
        """Quantize a model and validate that the resulting graph can be loaded."""
        started = time.perf_counter()
        source_path = Path(source).expanduser().resolve()
        if not source_path.exists():
            raise ModelNotFoundError(f"source ONNX model does not exist: {source_path}")
        quant_options = options or QuantizationOptions(
            precision=Precision(self._settings.default_precision),
            calibration_samples=self._settings.calibration_samples,
        )
        destination = (
            Path(output_path).expanduser().resolve()
            if output_path is not None
            else source_path.with_name(f"{source_path.stem}.{quant_options.precision.value}.onnx")
        )
        if destination == source_path:
            raise QuantizationError("source and destination paths must be different")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.unlink(missing_ok=True)

        input_specs = inspect_onnx(source_path).input_specs
        reader = CalibrationReaderFactory.create(
            quant_options,
            input_specs,
            seed=self._settings.seed,
        )
        try:
            if quant_options.precision is Precision.FP32:
                shutil.copy2(source_path, destination)
            elif quant_options.precision is Precision.FP16:
                self._convert_fp16(source_path, destination)
            elif quant_options.precision is Precision.INT8_DYNAMIC:
                self._quantize_dynamic(source_path, destination, quant_options)
            else:
                self._quantize_static(source_path, destination, quant_options, reader)
            self._validate_artifact(destination)
            self._stamp_precision(destination, quant_options.precision)
        except (ModelNotFoundError, QuantizationError, CalibrationError):
            destination.unlink(missing_ok=True)
            raise
        except Exception as exc:
            destination.unlink(missing_ok=True)
            raise QuantizationError(
                f"quantization to {quant_options.precision.value} failed: {exc}"
            ) from exc

        source_size = file_size_mb(source_path)
        output_size = file_size_mb(destination)
        compression = source_size / output_size if output_size > 0 else 1.0
        metadata = {
            "strategy": quant_options.precision.value,
            "per_channel": str(quant_options.per_channel),
            "reduce_range": str(quant_options.reduce_range),
        }
        if quant_options.precision is Precision.INT8_STATIC:
            metadata["calibration_method"] = quant_options.calibration_method.value
        if quant_options.nodes_to_exclude:
            metadata["nodes_to_exclude"] = ",".join(quant_options.nodes_to_exclude)
        self._logger.info(
            "quantized %s to %s (%.4f MiB -> %.4f MiB)",
            source_path.name,
            quant_options.precision.value,
            source_size,
            output_size,
        )
        return QuantizationResult(
            source_path=str(source_path),
            output_path=str(destination),
            precision=quant_options.precision,
            size_mb=output_size,
            source_size_mb=source_size,
            compression_ratio=round(compression, 4),
            opset=inspect_onnx(destination).opset,
            calibration_samples=_reader_sample_count(reader, 0),
            checksum=sha256_file(destination),
            duration_s=round(time.perf_counter() - started, 4),
            metadata=metadata,
        )

    def _convert_fp16(self, source: Path, destination: Path) -> None:
        model = onnx.load(str(source))
        converted = float16.convert_float_to_float16(
            model,
            keep_io_types=True,
            disable_shape_infer=False,
        )
        onnx.checker.check_model(converted)
        onnx.save(converted, str(destination))

    def _quantize_dynamic(
        self,
        source: Path,
        destination: Path,
        options: QuantizationOptions,
    ) -> None:
        extra_options = dict(options.extra_options)
        extra_options.setdefault("MatMulConstBOnly", True)
        quantize_dynamic(
            model_input=str(source),
            model_output=str(destination),
            per_channel=options.per_channel,
            reduce_range=options.reduce_range,
            weight_type=_WEIGHT_TYPES[options.weight_type],
            op_types_to_quantize=options.op_types_to_quantize,
            nodes_to_exclude=options.nodes_to_exclude or None,
            extra_options=extra_options,
        )

    def _quantize_static(
        self,
        source: Path,
        destination: Path,
        options: QuantizationOptions,
        reader: CalibrationDataReader | None,
    ) -> None:
        if reader is None:
            raise CalibrationError("static quantization requires a calibration reader")
        preprocessed = self._settings.cache_dir / f"{source.stem}.preprocessed.{short_id()}.onnx"
        preprocessed.parent.mkdir(parents=True, exist_ok=True)
        try:
            quant_pre_process(
                input_model=str(source),
                output_model_path=str(preprocessed),
                skip_optimization=False,
                skip_onnx_shape=False,
                skip_symbolic_shape=False,
                auto_merge=True,
                int_max=2**31 - 1,
                guess_output_rank=False,
                verbose=0,
            )
            quantize_static(
                model_input=str(preprocessed),
                model_output=str(destination),
                calibration_data_reader=reader,
                quant_format=_QUANT_FORMATS[options.quant_format],
                per_channel=options.per_channel,
                reduce_range=options.reduce_range,
                activation_type=_ACTIVATION_TYPES[options.activation_type],
                weight_type=_WEIGHT_TYPES[options.weight_type],
                op_types_to_quantize=options.op_types_to_quantize,
                calibrate_method=OrtCalibrationMethod.MinMax,
                nodes_to_exclude=options.nodes_to_exclude or None,
                extra_options=options.extra_options,
            )
        finally:
            preprocessed.unlink(missing_ok=True)

    def _validate_artifact(self, path: Path) -> None:
        try:
            session = create_session(
                str(path),
                provider=self._provider,
                intra_op_threads=self._settings.intra_op_threads,
                inter_op_threads=self._settings.inter_op_threads,
            )
        except Exception as exc:
            raise QuantizationError(f"quantized artifact is not loadable: {exc}") from exc
        if not session.get_inputs():
            raise QuantizationError("quantized artifact exposes no inputs")
        if not session.get_outputs():
            raise QuantizationError("quantized artifact exposes no outputs")

    @staticmethod
    def _stamp_precision(path: Path, precision: Precision) -> None:
        model = onnx.load(str(path))
        metadata = {entry.key: entry.value for entry in model.metadata_props}
        metadata["precision"] = precision.value
        metadata["quantized_by"] = "mcp-edge-vision-forge"
        helper.set_model_props(model, metadata)
        onnx.save(model, str(path))


def quantization_summary(result: QuantizationResult) -> dict[str, Any]:
    """Return a compact dictionary suitable for logs and MCP responses."""
    return {
        "precision": result.precision.value,
        "output_path": result.output_path,
        "size_mb": result.size_mb,
        "compression_ratio": result.compression_ratio,
        "calibration_samples": result.calibration_samples,
    }
