"""Safe ONNX loading shared by every component.

ONNX models can reference tensors stored outside the ``.onnx`` file. Those
references are just text, so a malicious artifact can point at any path the
process can read. This module centralizes two rules:

* external tensor locations must resolve inside the model directory, following
  symbolic links, and their declared total size is bounded;
* callers that only need metadata load with ``load_external_data=False`` so the
  payload is never read.

The helpers are intentionally dependency-light and are imported by the exporter,
the quantizer and the ONNX Runtime session factory.
"""

from __future__ import annotations

import logging
from pathlib import Path

import onnx
from onnx import TensorProto

from edge_forge.errors import ExportError
from edge_forge.limits import MAX_EXTERNAL_DATA_MB

_LOGGER = logging.getLogger(__name__)


def validate_external_data(model: onnx.ModelProto, path: Path) -> None:
    """Reject external tensors whose location escapes the model directory.

    Resolution happens before the containment check, so a symbolic link inside
    the model directory cannot point at a file outside it. The declared byte
    sizes are also bounded to keep a single artifact from exhausting memory.

    Args:
        model: Parsed model, normally loaded with ``load_external_data=False``.
        path: ONNX file the model was read from.

    Raises:
        ExportError: If a location is absolute, contains ``..``, resolves outside
            the model directory or the declared payload is too large.
    """
    root = path.parent.expanduser().resolve()
    declared_bytes = 0
    for initializer in model.graph.initializer:
        if initializer.data_location != TensorProto.EXTERNAL:
            continue
        for entry in initializer.external_data:
            if entry.key == "length":
                try:
                    declared_bytes += int(entry.value)
                except ValueError:
                    continue
                continue
            if entry.key != "location":
                continue
            location = Path(entry.value)
            if location.is_absolute() or ".." in location.parts:
                raise ExportError(
                    f"ONNX external data for '{initializer.name}' points outside {root}: "
                    f"{entry.value}"
                )
            target = (root / location).resolve()
            if not target.is_relative_to(root):
                raise ExportError(
                    f"ONNX external data for '{initializer.name}' escapes {root} through a "
                    f"symbolic link: {entry.value}"
                )
    if declared_bytes > MAX_EXTERNAL_DATA_MB * 1024 * 1024:
        raise ExportError(
            f"ONNX external data declares {declared_bytes / (1024 * 1024):.0f} MiB, exceeding "
            f"the limit of {MAX_EXTERNAL_DATA_MB:.0f} MiB"
        )


def load_onnx(path: Path, *, allow_external_data: bool = True) -> onnx.ModelProto:
    """Parse an ONNX model after validating its external data references.

    The artifact is first parsed without external data so that the locations can
    be validated before any external byte is read. When ``allow_external_data``
    is false the returned model keeps the references but never touches them.

    Args:
        path: ONNX artifact to parse.
        allow_external_data: Whether to read validated external tensors.

    Returns:
        The parsed model.

    Raises:
        ExportError: If the artifact cannot be parsed or its external data is
            unsafe.
    """
    try:
        probe = onnx.load(str(path), load_external_data=False)
    except Exception as exc:
        raise ExportError(f"unable to parse ONNX artifact {path}: {exc}") from exc
    validate_external_data(probe, path)
    has_external = any(
        initializer.data_location == TensorProto.EXTERNAL for initializer in probe.graph.initializer
    )
    if not allow_external_data or not has_external:
        return probe
    try:
        return onnx.load(str(path), load_external_data=True)
    except Exception as exc:
        raise ExportError(f"unable to read external data for {path}: {exc}") from exc


def validate_onnx_file(path: str | Path) -> Path:
    """Validate an ONNX file before ONNX Runtime is allowed to open it.

    Args:
        path: Artifact about to be loaded by ONNX Runtime.

    Returns:
        The resolved path.

    Raises:
        ExportError: If the artifact references unsafe external data.
    """
    resolved = Path(path).expanduser().resolve()
    if resolved.suffix.lower() != ".onnx" or not resolved.exists():
        return resolved
    try:
        probe = onnx.load(str(resolved), load_external_data=False)
    except Exception as exc:
        raise ExportError(f"unable to parse ONNX artifact {resolved}: {exc}") from exc
    validate_external_data(probe, resolved)
    return resolved
