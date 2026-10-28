"""Tests for the Typer CLI using an isolated workspace."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from edge_forge.cli.main import app

runner = CliRunner()


def _invoke(workspace: Path, *arguments: str):
    return runner.invoke(
        app,
        [
            "--workspace",
            str(workspace / "artifacts"),
            "--reports-dir",
            str(workspace / "reports"),
            *arguments,
        ],
    )


def _json_output(result) -> dict:
    assert result.exit_code == 0, result.stdout
    return json.loads(result.stdout)


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.stdout


def test_models_and_hardware_json(tmp_path: Path) -> None:
    models = _json_output(_invoke(tmp_path, "models", "--json"))
    assert {entry["name"] for entry in models["models"]} >= {"edge_cnn", "tiny_vit"}
    hardware = _json_output(_invoke(tmp_path, "hardware", "--json"))
    assert {entry["name"] for entry in hardware["profiles"]} >= {"raspberry-pi-5"}


def test_system_command(tmp_path: Path) -> None:
    payload = _json_output(_invoke(tmp_path, "system", "--json"))
    assert payload["version"] == "0.1.0"
    assert "CPUExecutionProvider" in payload["providers"]


def test_export_quantize_benchmark_and_drift(tmp_path: Path) -> None:
    exported = _json_output(
        _invoke(
            tmp_path,
            "export",
            "edge_cnn",
            "--width",
            "8",
            "--depth",
            "1",
            "--output",
            str(tmp_path / "tiny.onnx"),
            "--json",
        )
    )
    model_path = exported["model_path"]
    assert Path(model_path).exists()
    assert exported["validation"]["passed"] is True

    inspected = _json_output(_invoke(tmp_path, "inspect", model_path, "--json"))
    assert inspected["opset"] == 17

    quantized = _json_output(
        _invoke(
            tmp_path,
            "quantize",
            model_path,
            "--precision",
            "int8_dynamic",
            "--json",
        )
    )
    assert quantized["precision"] == "int8_dynamic"

    benchmarked = _json_output(
        _invoke(
            tmp_path,
            "benchmark",
            quantized["output_path"],
            "--iterations",
            "3",
            "--warmup",
            "1",
            "--json",
        )
    )
    assert benchmarked["latency"]["count"] == 3

    drift = _json_output(
        _invoke(
            tmp_path,
            "drift",
            quantized["output_path"],
            "--reference-model",
            "edge_cnn",
            "--width",
            "8",
            "--depth",
            "1",
            "--samples",
            "3",
            "--json",
        )
    )
    assert drift["passed"] is True
    assert drift["metrics"]["cosine_similarity"] > 0.99


def test_tune_and_report_commands(tmp_path: Path) -> None:
    tuned = _json_output(
        _invoke(
            tmp_path,
            "tune",
            "edge_cnn",
            "--width",
            "8",
            "--depth",
            "1",
            "--precisions",
            "fp32,fp16",
            "--iterations",
            "2",
            "--warmup",
            "1",
            "--max-size-mb",
            "10",
            "--json",
        )
    )
    run_id = tuned["run_id"]
    assert tuned["best_candidate_id"] is not None
    assert tuned["feasible"] >= 1
    assert set(tuned["reports"]) == {"json", "markdown", "html"}

    rendered = _json_output(_invoke(tmp_path, "report", run_id, "--formats", "markdown", "--json"))
    assert set(rendered["files"]) == {"markdown"}
    report_path = Path(rendered["files"]["markdown"])
    assert report_path.exists()
    assert "Edge Vision Forge report" in report_path.read_text(encoding="utf-8")


def test_invalid_input_shape_fails_cleanly(tmp_path: Path) -> None:
    result = _invoke(
        tmp_path,
        "export",
        "edge_cnn",
        "--input-shape",
        "1,3,0,32",
    )
    assert result.exit_code == 1
    assert "error" in result.stdout.lower() or "error" in result.stderr.lower()


def test_unknown_report_fails_cleanly(tmp_path: Path) -> None:
    result = _invoke(tmp_path, "report", "run-missing")
    assert result.exit_code == 1
    assert "does not exist" in result.stdout or "does not exist" in result.stderr


def test_human_readable_catalog_commands(tmp_path: Path) -> None:
    version = runner.invoke(app, ["version"])
    assert version.exit_code == 0
    assert "mcp-edge-vision-forge" in version.stdout

    system = _invoke(tmp_path, "system")
    assert system.exit_code == 0
    assert "CPUExecutionProvider" in system.stdout

    models = _invoke(tmp_path, "models")
    assert models.exit_code == 0
    assert "Reference models" in models.stdout
    assert "edge_cnn" in models.stdout

    hardware = _invoke(tmp_path, "hardware")
    assert hardware.exit_code == 0
    assert "Hardware profiles" in hardware.stdout
    assert "Accelerator" in hardware.stdout
    assert "MiB" in hardware.stdout


def test_human_readable_pipeline_commands(tmp_path: Path) -> None:
    exported = _invoke(
        tmp_path,
        "export",
        "edge_cnn",
        "--width",
        "8",
        "--depth",
        "1",
        "--output",
        str(tmp_path / "render.onnx"),
    )
    assert exported.exit_code == 0, exported.stdout
    assert "Export result" in exported.stdout
    assert "Validated" in exported.stdout
    model_path = str(tmp_path / "render.onnx")

    inspected = _invoke(tmp_path, "inspect", model_path)
    assert inspected.exit_code == 0
    assert "Architecture" in inspected.stdout
    assert "Opset" in inspected.stdout

    quantized = _invoke(
        tmp_path,
        "quantize",
        model_path,
        "--precision",
        "fp16",
        "--output",
        str(tmp_path / "render.fp16.onnx"),
    )
    assert quantized.exit_code == 0, quantized.stdout
    assert "Quantization result" in quantized.stdout
    assert "Compression" in quantized.stdout

    drifted = _invoke(
        tmp_path,
        "drift",
        str(tmp_path / "render.fp16.onnx"),
        "--reference-onnx",
        model_path,
        "--samples",
        "2",
    )
    assert drifted.exit_code == 0, drifted.stdout
    assert "Cosine similarity" in drifted.stdout
    assert "Top-1 agreement" in drifted.stdout

    benchmarked = _invoke(
        tmp_path,
        "benchmark",
        model_path,
        "--iterations",
        "2",
        "--warmup",
        "1",
        "--hardware",
        "coral-edge-tpu",
    )
    assert benchmarked.exit_code == 0, benchmarked.stdout
    assert "Benchmark:" in benchmarked.stdout
    assert "Meets budget" in benchmarked.stdout


def test_human_readable_tune_without_reports(tmp_path: Path) -> None:
    tuned = _invoke(
        tmp_path,
        "tune",
        "edge_cnn",
        "--width",
        "8",
        "--depth",
        "1",
        "--precisions",
        "fp32,fp16",
        "--iterations",
        "1",
        "--warmup",
        "1",
        "--max-size-mb",
        "10",
        "--no-report",
    )
    assert tuned.exit_code == 0, tuned.stdout
    assert "Tuning candidates" in tuned.stdout
    assert "Tuning run run-" in tuned.stdout

    run_id = tuned.stdout.split("Tuning run run-", 1)[1].split()[0].rstrip("\n")
    rendered = _invoke(tmp_path, "report", f"run-{run_id}", "--formats", "markdown")
    assert rendered.exit_code == 0, rendered.stdout
    assert "Report run-" in rendered.stdout


def test_inspect_torch_checkpoint(tmp_path: Path) -> None:
    from edge_forge.models.registry import build_model, save_checkpoint

    model = build_model("edge_cnn", width=8, depth=1)
    checkpoint = save_checkpoint(
        model,
        tmp_path / "tiny.pt",
        architecture="edge_cnn",
        model_kwargs={"width": 8, "depth": 1},
    )
    result = _invoke(tmp_path, "inspect", str(checkpoint), "--input-shape", "1,3,32,32")
    assert result.exit_code == 0, result.stdout
    assert "Model: tiny" in result.stdout
    assert "edge_cnn" in result.stdout
