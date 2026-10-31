"""Tests for the MCP server tools and resources."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from edge_forge.errors import ConfigurationError
from edge_forge.mcp_server.server import create_server, main


def _run(coroutine: Coroutine[Any, Any, Any]) -> Any:
    return asyncio.run(coroutine)


def _payload(result: Any) -> dict[str, Any]:
    assert result.is_error is False
    assert result.content
    return json.loads(result.content[0].text)


def test_export_rejects_overwrite_without_flag(settings) -> None:
    server = create_server(settings)
    path = str(settings.exports_dir / "twice.onnx")
    first = _payload(
        _run(
            server.call_tool(
                "export_model",
                {
                    "model_name": "edge_cnn",
                    "output_path": path,
                    "opset": 17,
                    "input_shape": [1, 3, 32, 32],
                    "width": 8,
                    "depth": 1,
                    "num_classes": 10,
                },
            )
        )
    )
    assert first["validation"]["passed"] is True
    with pytest.raises(ToolError):
        _run(
            server.call_tool(
                "export_model",
                {
                    "model_name": "edge_cnn",
                    "output_path": path,
                    "opset": 17,
                    "input_shape": [1, 3, 32, 32],
                    "width": 8,
                    "depth": 1,
                    "num_classes": 10,
                },
            )
        )
    second = _payload(
        _run(
            server.call_tool(
                "export_model",
                {
                    "model_name": "edge_cnn",
                    "output_path": path,
                    "overwrite": True,
                    "opset": 17,
                    "input_shape": [1, 3, 32, 32],
                    "width": 8,
                    "depth": 1,
                    "num_classes": 10,
                },
            )
        )
    )
    assert second["validation"]["passed"] is True
    server = create_server(settings)
    with pytest.raises(ToolError):
        _run(
            server.call_tool(
                "export_model",
                {
                    "model_name": "edge_cnn",
                    "output_path": str(settings.workspace / "bad.txt"),
                    "opset": 17,
                    "input_shape": [1, 3, 32, 32],
                    "width": 8,
                    "depth": 1,
                    "num_classes": 10,
                },
            )
        )
    server = create_server(settings)
    tools = _run(server.list_tools())
    names = {tool.name for tool in tools}
    assert {
        "list_models",
        "list_hardware_profiles",
        "system_info",
        "inspect_model",
        "export_model",
        "quantize_model",
        "analyze_drift",
        "benchmark_model",
        "auto_tune",
        "list_reports",
        "generate_report",
        "read_report",
    } <= names
    resources = _run(server.list_resources())
    uris = {str(resource.uri) for resource in resources}
    assert "edge-forge://catalog/models" in uris
    assert "edge-forge://hardware/profiles" in uris
    templates = _run(server.list_resource_templates())
    assert any("reports/{run_id}" in str(template.uri_template) for template in templates)


def test_catalog_and_hardware_tools(settings) -> None:
    server = create_server(settings)
    models = _payload(_run(server.call_tool("list_models", {})))
    assert {entry["name"] for entry in models["models"]} >= {
        "edge_cnn",
        "mobile_edge_net",
        "tiny_vit",
    }
    profiles = _payload(_run(server.call_tool("list_hardware_profiles", {})))
    assert {entry["name"] for entry in profiles["profiles"]} >= {
        "raspberry-pi-5",
        "coral-edge-tpu",
        "generic-x86",
    }
    info = _payload(_run(server.call_tool("system_info", {})))
    assert info["version"]
    assert "CPUExecutionProvider" in info["providers"]


def test_resources_return_json(settings) -> None:
    server = create_server(settings)
    contents = list(_run(server.read_resource("edge-forge://catalog/models")))
    payload = json.loads(contents[0].content)
    assert payload["models"]
    hardware = list(_run(server.read_resource("edge-forge://hardware/profiles")))
    assert json.loads(hardware[0].content)["profiles"]


def test_end_to_end_pipeline_over_mcp(settings) -> None:
    server = create_server(settings)
    exported = _payload(
        _run(
            server.call_tool(
                "export_model",
                {
                    "model_name": "edge_cnn",
                    "opset": 17,
                    "input_shape": [1, 3, 32, 32],
                    "width": 8,
                    "depth": 1,
                    "num_classes": 10,
                    "simplify": True,
                },
            )
        )
    )
    model_path = exported["model_path"]
    assert exported["validation"]["passed"] is True

    inspected = _payload(_run(server.call_tool("inspect_model", {"model_path": model_path})))
    assert inspected["opset"] == 17

    quantized = _payload(
        _run(
            server.call_tool(
                "quantize_model",
                {
                    "model_path": model_path,
                    "precision": "int8_dynamic",
                    "calibration_samples": 4,
                },
            )
        )
    )
    assert quantized["precision"] == "int8_dynamic"

    drift = _payload(
        _run(
            server.call_tool(
                "analyze_drift",
                {
                    "candidate_path": quantized["output_path"],
                    "reference_model": "edge_cnn",
                    "input_shape": [1, 3, 32, 32],
                    "num_samples": 3,
                    "width": 8,
                    "depth": 1,
                },
            )
        )
    )
    assert drift["passed"] is True
    assert drift["metrics"]["cosine_similarity"] > 0.99

    benchmark = _payload(
        _run(
            server.call_tool(
                "benchmark_model",
                {
                    "model_path": quantized["output_path"],
                    "input_shape": [1, 3, 32, 32],
                    "iterations": 3,
                    "warmup_runs": 1,
                },
            )
        )
    )
    assert benchmark["latency"]["count"] == 3
    assert benchmark["latency"]["p99_ms"] > 0

    tuned = _payload(
        _run(
            server.call_tool(
                "auto_tune",
                {
                    "model_name": "edge_cnn",
                    "input_shape": [1, 3, 32, 32],
                    "max_size_mb": 10.0,
                    "allowed_precisions": ["fp32", "fp16"],
                    "opsets": [17],
                    "iterations": 2,
                    "warmup_runs": 1,
                    "width": 8,
                    "depth": 1,
                },
            )
        )
    )
    assert tuned["best_candidate_id"] is not None
    run_id = tuned["run_id"]
    assert tuned["feasible"] >= 1

    runs = _payload(_run(server.call_tool("list_reports", {})))
    assert any(entry["run_id"] == run_id for entry in runs["runs"])

    artifacts = _payload(
        _run(
            server.call_tool("generate_report", {"run_id": run_id, "formats": ["markdown", "html"]})
        )
    )
    assert set(artifacts["files"]) == {"markdown", "html"}

    markdown = _run(
        server.call_tool("read_report", {"run_id": run_id, "report_format": "markdown"})
    )
    assert markdown.is_error is False
    assert "Edge Vision Forge report" in markdown.content[0].text

    resource = list(_run(server.read_resource(f"edge-forge://reports/{run_id}")))
    assert "Edge Vision Forge report" in resource[0].content


def test_unknown_tool_returns_error(settings) -> None:
    server = create_server(settings)
    try:
        result = _run(server.call_tool("does_not_exist", {}))
    except Exception as error:
        assert "does_not_exist" in str(error) or "not found" in str(error).lower()
        return
    assert result.is_error is True


def test_path_confinement_rejects_artifacts_outside_roots(settings, fp32_onnx: Path) -> None:
    server = create_server(settings)
    outside = fp32_onnx.parent.parent.parent / "external" / "model.onnx"
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_bytes(fp32_onnx.read_bytes())

    with pytest.raises(ToolError) as blocked:
        _run(server.call_tool("inspect_model", {"model_path": str(outside)}))
    text = str(blocked.value)
    assert "PathAccessError" in text
    assert "outside the permitted roots" in text

    unconfined = settings.model_copy(update={"restrict_paths": False})
    allowed = _payload(
        _run(create_server(unconfined).call_tool("inspect_model", {"model_path": str(outside)}))
    )
    assert allowed["opset"] == 17


def test_invalid_arguments_return_structured_errors(settings) -> None:
    server = create_server(settings)
    with pytest.raises(ToolError) as failure:
        _run(
            server.call_tool(
                "export_model",
                {"model_name": "edge_cnn", "input_shape": [1, 3, 0, 0]},
            )
        )
    text = str(failure.value)
    assert "ValueError" in text
    assert "strictly positive" in text


def test_oversized_model_request_is_rejected(settings) -> None:
    server = create_server(settings)
    with pytest.raises(ToolError) as failure:
        _run(
            server.call_tool(
                "export_model",
                {"model_name": "edge_cnn", "width": 100000, "depth": 1},
            )
        )
    assert "width must be between" in str(failure.value)


def test_system_info_reports_path_policy(settings) -> None:
    server = create_server(settings)
    info = _payload(_run(server.call_tool("system_info", {})))
    assert info["restrict_paths"] is True
    assert str(settings.workspace.resolve()) in info["permitted_roots"]


def test_main_dispatches_transport(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    class StubServer:
        def run(self, **kwargs: Any) -> None:
            calls.append(kwargs)

    monkeypatch.setattr(
        "edge_forge.mcp_server.server.create_server", lambda _settings: StubServer()
    )
    main(
        [
            "--transport",
            "stdio",
            "--workspace",
            str(tmp_path / "artifacts"),
            "--reports-dir",
            str(tmp_path / "reports"),
            "--log-level",
            "WARNING",
        ]
    )
    assert calls == [{"transport": "stdio"}]

    with pytest.raises(ConfigurationError):
        main(["--transport", "streamable-http", "--host", "0.0.0.0", "--port", "9000"])

    main(
        [
            "--transport",
            "streamable-http",
            "--host",
            "127.0.0.1",
            "--port",
            "9000",
            "--allow-network",
        ]
    )
    assert calls[-1] == {"transport": "streamable-http", "host": "127.0.0.1", "port": 9000}
