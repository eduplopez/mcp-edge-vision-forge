"""PyTorch to ONNX export, graph simplification and numerical validation."""

from __future__ import annotations

import logging
import math
import sys
import time
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper
from torch import nn

from edge_forge.config import MAX_OPSET, MIN_OPSET, ForgeSettings, get_settings
from edge_forge.core.numerics import (
    assert_finite,
    cosine_similarity,
    max_absolute_error,
    mean_squared_error,
)
from edge_forge.core.onnx_io import load_onnx, validate_external_data
from edge_forge.core.ort_utils import create_session
from edge_forge.errors import ExportError, ModelNotFoundError, UnsupportedOpsetError
from edge_forge.limits import MAX_MODEL_FILE_MB, validate_file_size, validate_input_shape
from edge_forge.models.registry import extract_output_tensors, make_dummy_input
from edge_forge.models.schemas import (
    ExportOptions,
    ExportResult,
    ModelInfo,
    Precision,
    TensorSpec,
    ValidationResult,
)
from edge_forge.utils import file_size_mb, sha256_file

_LOGGER = logging.getLogger(__name__)
_DTYPE_NAMES: dict[int, str] = {
    TensorProto.FLOAT: "float32",
    TensorProto.FLOAT16: "float16",
    TensorProto.DOUBLE: "float64",
    TensorProto.INT8: "int8",
    TensorProto.UINT8: "uint8",
    TensorProto.INT32: "int32",
    TensorProto.INT64: "int64",
    TensorProto.BOOL: "bool",
}
_STATIC_INT8_OPERATORS = frozenset(
    {"QuantizeLinear", "DequantizeLinear", "QLinearConv", "QLinearMatMul", "QLinearAdd"}
)


def validate_opset(opset: int) -> int:
    """Validate an ONNX opset against the local ONNX and project support ranges."""
    if opset < MIN_OPSET or opset > MAX_OPSET:
        raise UnsupportedOpsetError(
            f"opset {opset} is outside the supported range [{MIN_OPSET}, {MAX_OPSET}]"
        )
    installed_max = int(onnx.defs.onnx_opset_version())
    if opset > installed_max:
        raise UnsupportedOpsetError(
            f"opset {opset} is not supported by the installed ONNX runtime (max {installed_max})"
        )
    return opset


def _shape_from_value_info(value_info: onnx.ValueInfoProto) -> tuple[int, ...]:
    dimensions: list[int] = []
    for dimension in value_info.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value") and dimension.dim_value > 0:
            dimensions.append(int(dimension.dim_value))
        else:
            dimensions.append(1)
    return tuple(dimensions) if dimensions else (1,)


def _spec_from_value_info(value_info: onnx.ValueInfoProto) -> TensorSpec:
    data_type = value_info.type.tensor_type.elem_type
    return TensorSpec(
        name=value_info.name,
        shape=_shape_from_value_info(value_info),
        dtype=_DTYPE_NAMES.get(data_type, "float32"),
    )


def _graph_inputs(model: onnx.ModelProto) -> list[onnx.ValueInfoProto]:
    initializers = {initializer.name for initializer in model.graph.initializer}
    return [value for value in model.graph.input if value.name not in initializers]


def _validate_external_data(model: onnx.ModelProto, path: Path) -> None:
    """Reject external tensors whose location escapes the model directory."""
    validate_external_data(model, path)


def inspect_onnx(path: str | Path, *, name: str | None = None) -> ModelInfo:
    """Read static metadata from an ONNX artifact."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise ModelNotFoundError(f"ONNX artifact does not exist: {resolved}")
    try:
        validate_file_size(resolved, MAX_MODEL_FILE_MB, label="ONNX artifact")
    except ValueError as exc:
        raise ExportError(str(exc)) from exc
    model = load_onnx(resolved, allow_external_data=False)
    _validate_external_data(model, resolved)

    metadata = {entry.key: entry.value for entry in model.metadata_props}
    parameter_count = sum(
        math.prod(int(dimension) for dimension in item.dims) for item in model.graph.initializer
    )
    data_types = {initializer.data_type for initializer in model.graph.initializer}
    node_operators = {node.op_type for node in model.graph.node}
    declared_precision = metadata.get("precision")
    precision = Precision.FP32
    if declared_precision:
        try:
            precision = Precision(declared_precision)
        except ValueError:
            precision = Precision.FP32
    elif TensorProto.FLOAT16 in data_types:
        precision = Precision.FP16
    elif TensorProto.INT8 in data_types or TensorProto.UINT8 in data_types:
        if _STATIC_INT8_OPERATORS & node_operators:
            precision = Precision.INT8_STATIC
        else:
            precision = Precision.INT8_DYNAMIC
    opset = next(
        (int(entry.version) for entry in model.opset_import if entry.domain in ("", "ai.onnx")),
        None,
    )
    return ModelInfo(
        name=name or resolved.stem,
        architecture=metadata.get("architecture", resolved.stem),
        parameters=parameter_count,
        size_mb=file_size_mb(resolved),
        input_specs=[_spec_from_value_info(value) for value in _graph_inputs(model)],
        output_specs=[_spec_from_value_info(value) for value in model.graph.output],
        precision=precision,
        opset=opset,
        path=str(resolved),
        checksum=sha256_file(resolved),
        metadata=metadata,
    )


class GraphSimplifier:
    """Deterministic ONNX graph cleanup used when onnxsim is unavailable."""

    def simplify(self, source: Path, destination: Path) -> Path:
        """Remove identity nodes, dead branches and unused initializers."""
        model = load_onnx(source)
        self._remove_identities(model)
        self._prune_dead_nodes(model)
        self._prune_initializers(model)
        self._prune_value_info(model)
        try:
            model = onnx.shape_inference.infer_shapes(model)
        except Exception as exc:
            _LOGGER.debug("shape inference skipped during simplification: %s", exc)
        onnx.checker.check_model(model)
        destination.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, str(destination))
        return destination

    @staticmethod
    def _remove_identities(model: onnx.ModelProto) -> None:
        replacements: dict[str, str] = {}
        for node in model.graph.node:
            if node.op_type == "Identity" and len(node.input) == 1 and len(node.output) == 1:
                source = replacements.get(node.input[0], node.input[0])
                replacements[node.output[0]] = source
        if not replacements:
            return
        for node in model.graph.node:
            for index, input_name in enumerate(node.input):
                node.input[index] = replacements.get(input_name, input_name)
        for output in model.graph.output:
            output.name = replacements.get(output.name, output.name)
        remaining = [node for node in model.graph.node if node.op_type != "Identity"]
        del model.graph.node[:]
        model.graph.node.extend(remaining)

    @staticmethod
    def _prune_dead_nodes(model: onnx.ModelProto) -> None:
        required: set[str] = {output.name for output in model.graph.output}
        kept: list[onnx.NodeProto] = []
        for node in reversed(list(model.graph.node)):
            if any(name in required for name in node.output):
                kept.append(node)
                required.update(name for name in node.input if name)
        kept.reverse()
        del model.graph.node[:]
        model.graph.node.extend(kept)

    @staticmethod
    def _prune_initializers(model: onnx.ModelProto) -> None:
        used: set[str] = set()
        for node in model.graph.node:
            used.update(name for name in node.input if name)
        kept = [initializer for initializer in model.graph.initializer if initializer.name in used]
        del model.graph.initializer[:]
        model.graph.initializer.extend(kept)

    @staticmethod
    def _prune_value_info(model: onnx.ModelProto) -> None:
        available: set[str] = {value.name for value in model.graph.input}
        available.update(output.name for output in model.graph.output)
        for node in model.graph.node:
            available.update(name for name in node.output if name)
        kept = [value for value in model.graph.value_info if value.name in available]
        del model.graph.value_info[:]
        model.graph.value_info.extend(kept)


class OnnxExporter:
    """Export PyTorch modules to validated, optionally simplified ONNX graphs."""

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
        self._simplifier = GraphSimplifier()

    @property
    def settings(self) -> ForgeSettings:
        """Runtime settings used by the exporter."""
        return self._settings

    def export(
        self,
        model: nn.Module,
        output_path: str | Path,
        *,
        input_shape: tuple[int, ...] | None = None,
        options: ExportOptions | None = None,
        name: str | None = None,
    ) -> ExportResult:
        """Export a model, simplify the graph and validate it against PyTorch."""
        started = time.perf_counter()
        export_options = options or ExportOptions(opset=self._settings.default_opset)
        validate_opset(export_options.opset)
        shape = tuple(input_shape or self._settings.default_input_shape)
        try:
            validate_input_shape(shape)
        except ValueError as exc:
            raise ExportError(str(exc)) from exc
        destination = Path(output_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)

        prepared = model.eval()
        dummy = make_dummy_input(shape, seed=self._settings.seed)
        with torch.inference_mode():
            reference_outputs = extract_output_tensors(prepared(dummy))
        if len(export_options.output_names) != len(reference_outputs):
            raise ExportError(
                f"model returns {len(reference_outputs)} tensors but "
                f"{len(export_options.output_names)} output names were supplied"
            )
        if len(export_options.input_names) != 1:
            raise ExportError("exporter currently supports exactly one model input")

        dynamic_axes = self._build_dynamic_axes(shape, export_options, reference_outputs)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                torch.onnx.export(
                    prepared,
                    (dummy,),
                    str(destination),
                    input_names=export_options.input_names,
                    output_names=export_options.output_names,
                    opset_version=export_options.opset,
                    dynamic_axes=dynamic_axes or None,
                    do_constant_folding=export_options.constant_folding,
                    export_params=True,
                    training=torch.onnx.TrainingMode.EVAL,
                    dynamo=export_options.dynamo,
                    verbose=False,
                )
        except Exception as exc:
            destination.unlink(missing_ok=True)
            raise ExportError(f"PyTorch to ONNX export failed: {exc}") from exc

        simplified = False
        if export_options.simplify:
            simplified = self._simplify_in_place(destination, export_options.opset)
        self._stamp_metadata(
            destination,
            architecture=name or type(model).__name__,
            precision=Precision.FP32,
            input_shape=shape,
        )
        info = inspect_onnx(destination, name=name or destination.stem)
        validation: ValidationResult | None = None
        if export_options.validate_output:
            validation = self.validate(
                prepared,
                destination,
                input_shape=shape,
                atol=export_options.atol,
                rtol=export_options.rtol,
            )
            if not validation.passed:
                destination.unlink(missing_ok=True)
                raise ExportError(
                    f"exported graph failed numerical validation: {validation.message}"
                )

        return ExportResult(
            model_path=str(destination),
            opset=export_options.opset,
            size_mb=info.size_mb,
            parameters=info.parameters,
            input_specs=info.input_specs,
            output_specs=info.output_specs,
            dynamic_axes=dynamic_axes,
            simplified=simplified,
            validation=validation,
            checksum=info.checksum or sha256_file(destination),
            duration_s=round(time.perf_counter() - started, 4),
        )

    def validate(
        self,
        model: nn.Module,
        onnx_path: str | Path,
        *,
        input_shape: tuple[int, ...],
        atol: float = 1e-4,
        rtol: float = 1e-3,
    ) -> ValidationResult:
        """Compare PyTorch and ONNX Runtime outputs on a deterministic input."""
        resolved = Path(onnx_path).expanduser().resolve()
        dummy = make_dummy_input(tuple(input_shape), seed=self._settings.seed)
        try:
            session = create_session(
                str(resolved),
                provider=self._provider,
                intra_op_threads=self._settings.intra_op_threads,
                inter_op_threads=self._settings.inter_op_threads,
            )
            input_name = session.get_inputs()[0].name
            onnx_outputs = session.run(None, {input_name: dummy.numpy()})
        except Exception as exc:
            raise ExportError(f"ONNX Runtime could not load {resolved}: {exc}") from exc
        with torch.inference_mode():
            torch_outputs = [
                tensor.detach().cpu().numpy() for tensor in extract_output_tensors(model(dummy))
            ]
        if len(onnx_outputs) != len(torch_outputs):
            return ValidationResult(
                passed=False,
                max_absolute_error=float("inf"),
                mse=float("inf"),
                cosine_similarity=0.0,
                message=(
                    f"output count mismatch: torch={len(torch_outputs)} onnx={len(onnx_outputs)}"
                ),
                provider=session.get_providers()[0],
            )
        reference = np.concatenate([np.asarray(item).reshape(-1) for item in torch_outputs])
        candidate = np.concatenate([np.asarray(item).reshape(-1) for item in onnx_outputs])
        assert_finite(reference, label="torch output")
        assert_finite(candidate, label="onnx output")
        difference = max_absolute_error(reference, candidate)
        similarity = cosine_similarity(reference, candidate)
        tolerance = atol + rtol * float(np.max(np.abs(reference)))
        passed = bool(np.allclose(reference, candidate, atol=atol, rtol=rtol))
        message = (
            "outputs match within tolerance"
            if passed
            else f"max absolute error {difference:.6g} exceeds tolerance {tolerance:.6g}"
        )
        return ValidationResult(
            passed=passed,
            max_absolute_error=difference,
            mse=mean_squared_error(reference, candidate),
            cosine_similarity=similarity,
            message=message,
            provider=session.get_providers()[0],
        )

    def _build_dynamic_axes(
        self,
        input_shape: tuple[int, ...],
        options: ExportOptions,
        reference_outputs: list[torch.Tensor],
    ) -> dict[str, dict[int, str]]:
        dynamic_axes: dict[str, dict[int, str]] = {}
        if options.dynamic_batch or options.dynamic_spatial:
            input_axes: dict[int, str] = {}
            if options.dynamic_batch:
                input_axes[0] = "batch"
            if options.dynamic_spatial and len(input_shape) == 4:
                input_axes[2] = "height"
                input_axes[3] = "width"
            if input_axes:
                dynamic_axes[options.input_names[0]] = input_axes
        for name, tensor in zip(options.output_names, reference_outputs, strict=True):
            output_axes: dict[int, str] = {}
            if options.dynamic_batch and tensor.dim() >= 1:
                output_axes[0] = "batch"
            if options.dynamic_spatial and tensor.dim() == 4:
                output_axes[2] = "height"
                output_axes[3] = "width"
            if output_axes:
                dynamic_axes[name] = output_axes
        return dynamic_axes

    def _simplify_in_place(self, path: Path, opset: int) -> bool:
        temporary = path.with_name(f"{path.stem}.simplified{path.suffix}")
        try:
            from onnxsim import simplify as onnxsim_simplify

            model = onnx.load(str(path))
            with redirect_stdout(sys.stderr):
                simplified_model, check = onnxsim_simplify(model, overwrite_input_shapes=False)
            if not check:
                raise ExportError("onnxsim reported an invalid simplified graph")
            for entry in simplified_model.opset_import:
                if entry.domain in ("", "ai.onnx"):
                    entry.version = opset
            onnx.checker.check_model(simplified_model)
            onnx.save(simplified_model, str(temporary))
            temporary.replace(path)
            return True
        except Exception as exc:
            _LOGGER.debug("onnxsim unavailable or failed, using internal simplifier: %s", exc)
            temporary.unlink(missing_ok=True)
            try:
                self._simplifier.simplify(path, temporary)
                temporary.replace(path)
                return True
            except Exception as fallback_exc:
                _LOGGER.debug("internal graph simplification skipped: %s", fallback_exc)
                temporary.unlink(missing_ok=True)
                return False

    @staticmethod
    def _stamp_metadata(
        path: Path,
        *,
        architecture: str,
        precision: Precision,
        input_shape: tuple[int, ...],
    ) -> None:
        model = onnx.load(str(path))
        metadata = {
            "architecture": architecture,
            "precision": precision.value,
            "input_shape": ",".join(str(dimension) for dimension in input_shape),
            "producer": "mcp-edge-vision-forge",
        }
        helper.set_model_props(model, metadata)
        onnx.save(model, str(path))

    def simplify(self, source: str | Path, destination: str | Path | None = None) -> Path:
        """Simplify an existing ONNX graph into a destination path."""
        source_path = Path(source).expanduser().resolve()
        if not source_path.exists():
            raise ModelNotFoundError(f"ONNX artifact does not exist: {source_path}")
        target = (
            Path(destination).expanduser().resolve()
            if destination is not None
            else source_path.with_name(f"{source_path.stem}.simplified{source_path.suffix}")
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        self._simplifier.simplify(source_path, target)
        return target

    def inspect(self, path: str | Path, *, name: str | None = None) -> ModelInfo:
        """Inspect an ONNX artifact."""
        return inspect_onnx(path, name=name)


def model_metadata(model: nn.Module, input_shape: tuple[int, ...]) -> dict[str, Any]:
    """Return lightweight metadata for logging and report headers."""
    dummy = make_dummy_input(tuple(input_shape), seed=0)
    with torch.inference_mode():
        outputs = extract_output_tensors(model(dummy))
    return {
        "class": type(model).__name__,
        "input_shape": list(input_shape),
        "output_shapes": [list(tensor.shape) for tensor in outputs],
    }
