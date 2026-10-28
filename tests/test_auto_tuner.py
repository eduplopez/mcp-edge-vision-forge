"""Tests for the autonomous tuning loop and report persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from edge_forge.core.auto_tuner import AutoTuner, summarize_tuning
from edge_forge.errors import RunNotFoundError
from edge_forge.models.schemas import Precision, TuningConstraints
from edge_forge.reporting.report_gen import ReportGenerator, RunStore, render_html


def test_tune_returns_ranked_candidates(
    settings,
    tiny_model,
    input_shape: tuple[int, ...],
) -> None:
    constraints = TuningConstraints(
        max_latency_ms=1000.0,
        max_size_mb=10.0,
        allowed_precisions=[
            Precision.FP32,
            Precision.FP16,
            Precision.INT8_DYNAMIC,
            Precision.INT8_STATIC,
        ],
        opset_candidates=[17],
        simplify_candidates=[True],
        warmup_runs=1,
        iterations=3,
    )
    result = AutoTuner(settings).tune(
        tiny_model,
        constraints=constraints,
        input_shape=input_shape,
        model_name="edge_cnn",
    )
    assert result.best_candidate_id is not None
    assert len(result.candidates) == 4
    assert result.duration_s >= 0
    assert result.source_model.architecture == "EdgeCNN"
    assert result.best is not None
    assert result.best.feasible is True
    assert result.best.score > 0
    assert result.best.drift is not None
    assert result.best.benchmark is not None
    for candidate in result.candidates:
        assert Path(candidate.model_path).exists()
        assert candidate.benchmark is not None
    assert Path(result.workdir).exists()
    summary = summarize_tuning(result)
    assert summary["run_id"] == result.run_id
    assert summary["candidates"] == 4


def test_tune_without_feasible_candidate(
    settings,
    tiny_model,
    input_shape: tuple[int, ...],
) -> None:
    constraints = TuningConstraints(
        max_latency_ms=1e-6,
        allowed_precisions=[Precision.FP32, Precision.FP16],
        opset_candidates=[17],
        simplify_candidates=[True],
        warmup_runs=1,
        iterations=2,
    )
    result = AutoTuner(settings).tune(
        tiny_model,
        constraints=constraints,
        input_shape=input_shape,
        model_name="edge_cnn",
    )
    assert result.best_candidate_id is None
    assert all(candidate.feasible is False for candidate in result.candidates)
    assert any("no candidate satisfied" in note for note in result.notes)


def test_precision_filter_changes_winner(
    settings,
    tiny_model,
    input_shape: tuple[int, ...],
) -> None:
    constraints = TuningConstraints(
        max_size_mb=10.0,
        allowed_precisions=[Precision.FP16, Precision.INT8_DYNAMIC],
        opset_candidates=[17],
        simplify_candidates=[True],
        warmup_runs=1,
        iterations=3,
    )
    result = AutoTuner(settings).tune(
        tiny_model,
        constraints=constraints,
        input_shape=input_shape,
        model_name="edge_cnn",
    )
    assert result.best is not None
    assert result.best.precision in (Precision.FP16, Precision.INT8_DYNAMIC)
    fp32_candidates = [
        candidate for candidate in result.candidates if candidate.precision is Precision.FP32
    ]
    assert fp32_candidates
    assert fp32_candidates[0].feasible is False


def test_multiple_opsets_are_explored(
    settings,
    tiny_model,
    input_shape: tuple[int, ...],
) -> None:
    constraints = TuningConstraints(
        max_size_mb=10.0,
        allowed_precisions=[Precision.FP32, Precision.FP16],
        opset_candidates=[16, 17],
        simplify_candidates=[True],
        warmup_runs=1,
        iterations=2,
    )
    result = AutoTuner(settings).tune(
        tiny_model,
        constraints=constraints,
        input_shape=input_shape,
        model_name="edge_cnn",
    )
    assert {candidate.opset for candidate in result.candidates} == {16, 17}
    assert len(result.candidates) == 4


def test_report_generation_and_store_roundtrip(
    settings,
    tiny_model,
    input_shape: tuple[int, ...],
) -> None:
    constraints = TuningConstraints(
        max_size_mb=10.0,
        allowed_precisions=[Precision.FP32, Precision.FP16],
        warmup_runs=1,
        iterations=2,
    )
    result = AutoTuner(settings).tune(
        tiny_model,
        constraints=constraints,
        input_shape=input_shape,
        model_name="edge_cnn",
    )
    store = RunStore(settings)
    stored_path = store.save(result)
    assert stored_path.exists()
    loaded = store.load(result.run_id)
    assert loaded.run_id == result.run_id
    assert loaded.best_candidate_id == result.best_candidate_id
    index = store.list_runs()
    assert any(entry["run_id"] == result.run_id for entry in index)
    assert store.exists(result.run_id) is True

    artifacts = ReportGenerator(settings).generate(result)
    assert set(artifacts.files) == {"json", "markdown", "html"}
    payload = json.loads(Path(artifacts.files["json"]).read_text(encoding="utf-8"))
    assert payload["report_id"] == result.run_id
    markdown = Path(artifacts.files["markdown"]).read_text(encoding="utf-8")
    assert "Edge Vision Forge report" in markdown
    assert "Candidates" in markdown
    html = Path(artifacts.files["html"]).read_text(encoding="utf-8")
    assert html.startswith("<!DOCTYPE html>")
    assert "<style>" in html
    assert "<script" not in html
    assert "http://" not in html and "https://" not in html
    assert "edge_cnn" in html

    regenerated = ReportGenerator(settings).generate_from_run(result.run_id)
    assert regenerated.report_id == result.run_id


def test_store_rejects_unknown_run(settings) -> None:
    with pytest.raises(RunNotFoundError):
        RunStore(settings).load("run-does-not-exist")


def test_store_rejects_path_traversal(settings) -> None:
    with pytest.raises(RunNotFoundError):
        RunStore(settings).load("../escape")


def test_render_html_escapes_content(
    settings,
    tiny_model,
    input_shape: tuple[int, ...],
) -> None:
    constraints = TuningConstraints(
        max_size_mb=10.0,
        allowed_precisions=[Precision.FP32],
        warmup_runs=1,
        iterations=1,
    )
    result = AutoTuner(settings).tune(
        tiny_model,
        constraints=constraints,
        input_shape=input_shape,
        model_name="<script>alert(1)</script>",
    )
    from edge_forge.reporting.report_gen import build_forge_report

    html = render_html(build_forge_report(result))
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
