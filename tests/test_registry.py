"""Tests for reference model determinism and checkpoint loading safety."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import torch

from edge_forge.errors import ModelLoadError, ModelNotFoundError
from edge_forge.models.registry import (
    build_model,
    load_model,
    save_checkpoint,
)


def test_reference_models_are_deterministic() -> None:
    first = build_model("edge_cnn", width=8, depth=1)
    second = build_model("edge_cnn", width=8, depth=1)
    for left, right in zip(first.state_dict().values(), second.state_dict().values(), strict=True):
        assert torch.equal(left, right)


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    model = build_model("edge_cnn", width=8, depth=1)
    checkpoint = save_checkpoint(
        model,
        tmp_path / "model.pt",
        architecture="edge_cnn",
        model_kwargs={"width": 8, "depth": 1},
    )
    loaded = load_model(checkpoint)
    assert loaded.architecture == "edge_cnn"
    assert loaded.input_shape == (1, 3, 32, 32)
    for left, right in zip(
        model.state_dict().values(), loaded.model.state_dict().values(), strict=True
    ):
        assert torch.equal(left, right)


def test_full_module_pickle_requires_opt_in(tmp_path: Path) -> None:
    model = build_model("edge_cnn", width=8, depth=1)
    checkpoint = tmp_path / "unsafe.pt"
    torch.save(model, checkpoint)
    with pytest.raises(ModelLoadError, match="allow_unsafe"):
        load_model(checkpoint)
    loaded = load_model(checkpoint, allow_unsafe=True)
    assert isinstance(loaded.model, torch.nn.Module)


def test_torchscript_requires_opt_in(tmp_path: Path) -> None:
    model = build_model("edge_cnn", width=8, depth=1).eval()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        scripted = torch.jit.trace(model, torch.randn(1, 3, 32, 32))
        archive = tmp_path / "scripted.pt"
        torch.jit.save(scripted, str(archive))
    with pytest.raises(ModelLoadError, match="allow_unsafe"):
        load_model(archive)
    loaded = load_model(archive, allow_unsafe=True)
    assert loaded.architecture == "torchscript"
    assert loaded.metadata["format"] == "torchscript"


def test_missing_checkpoint_raises(tmp_path: Path) -> None:
    with pytest.raises(ModelNotFoundError):
        load_model(tmp_path / "missing.pt")


def test_unknown_architecture_raises(tmp_path: Path) -> None:
    with pytest.raises(ModelNotFoundError):
        build_model("does_not_exist")
