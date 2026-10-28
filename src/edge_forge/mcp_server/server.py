"""Official MCP server exposing edge optimization tools and resources.

The module targets the modern ``mcp`` Python SDK (``MCPServer``, formerly
``FastMCP``) and keeps a compatibility shim for SDK 1.x deployments where the
class is still exported as ``FastMCP``. Every tool returns JSON encoded text so
that any MCP client can consume the result without custom decoders.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Literal, cast

from edge_forge import __version__
from edge_forge.config import ForgeSettings, get_settings
from edge_forge.core.auto_tuner import AutoTuner, summarize_tuning
from edge_forge.core.drift_analyzer import DriftAnalyzer
from edge_forge.core.exporter import OnnxExporter, inspect_onnx
from edge_forge.core.profiler import ModelProfiler, available_providers, runtime_version
from edge_forge.core.quantizer import QuantizationEngine
from edge_forge.errors import EdgeForgeError
from edge_forge.models.registry import (
    build_model,
    describe_models,
    inspect_torch_model,
    load_model,
)
from edge_forge.models.schemas import (
    CalibrationMethod,
    DriftThresholds,
    ExportOptions,
    Precision,
    QuantizationOptions,
    TuningConstraints,
    list_hardware_profiles,
)
from edge_forge.reporting.report_gen import (
    ReportGenerator,
    RunStore,
    build_forge_report,
    render_markdown,
)
from edge_forge.utils import iso_timestamp, short_id

_LOGGER = logging.getLogger(__name__)

try:
    from mcp.server.mcpserver import MCPServer as _ServerBackend

    _SDK_FLAVOR = "mcp.server.mcpserver.MCPServer"
except ImportError:
    from mcp.server.fastmcp import FastMCP as _ServerBackend  # type: ignore[attr-defined,no-redef]

    _SDK_FLAVOR = "mcp.server.fastmcp.FastMCP"

INSTRUCTIONS = (
    "Edge Vision Forge compresses and audits computer-vision models for edge deployment. "
    "Use list_models to discover reproducible reference architectures, export_model to produce "
    "ONNX, quantize_model for FP16 or INT8, analyze_drift to verify numerical fidelity, "
    "benchmark_model for latency and memory, and auto_tune to search the configuration space "
    "under deployment constraints. Tuning runs are persisted and can be rendered with "
    "generate_report."
)


def _json(payload: Any) -> str:
    return json.dumps(payload, indent=2, default=str, ensure_ascii=True)


def _dump(model: Any) -> Any:
    return model.model_dump(mode="json")


def _default_output(directory: Path, stem: str, suffix: str = "onnx") -> Path:
    return directory / f"{stem}.{suffix}"


def _precision(value: str | Precision | None, default: Precision) -> Precision:
    if value is None:
        return default
    if isinstance(value, Precision):
        return value
    return Precision(value)


def _parse_shape(value: list[int] | None, fallback: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return fallback
    if not value or any(int(dimension) <= 0 for dimension in value):
        raise ValueError("input_shape must contain strictly positive dimensions")
    return tuple(int(dimension) for dimension in value)


def create_server(settings: ForgeSettings | None = None) -> Any:
    """Build and configure an MCP server instance.

    Args:
        settings: Optional runtime settings; the process singleton is used when
            omitted.

    Returns:
        A configured MCP server compatible with the installed SDK generation.
    """
    active = settings or get_settings()
    active.ensure_directories()
    server = _ServerBackend(
        name="mcp-edge-vision-forge",
        version=__version__,
        instructions=INSTRUCTIONS,
    )
    exporter = OnnxExporter(active)
    quantizer = QuantizationEngine(active)
    analyzer = DriftAnalyzer(active)
    profiler = ModelProfiler(active)
    tuner = AutoTuner(
        active,
        exporter=exporter,
        quantizer=quantizer,
        analyzer=analyzer,
        profiler=profiler,
    )
    reports = ReportGenerator(active)
    store = RunStore(active)

    @server.tool(
        name="list_models",
        description="List the reproducible reference architectures available for benchmarking.",
    )
    def list_models_tool() -> str:
        return _json({"models": describe_models()})

    @server.tool(
        name="list_hardware_profiles",
        description="List built-in edge hardware profiles with latency and size budgets.",
    )
    def list_hardware_profiles_tool() -> str:
        return _json({"profiles": [_dump(profile) for profile in list_hardware_profiles()]})

    @server.tool(
        name="system_info",
        description="Report package, ONNX Runtime and execution provider information.",
    )
    def system_info_tool() -> str:
        return _json(
            {
                "version": __version__,
                "sdk": _SDK_FLAVOR,
                "onnxruntime": runtime_version(),
                "providers": available_providers(),
                "workspace": str(active.workspace),
                "reports_dir": str(active.reports_dir),
                "timestamp": iso_timestamp(),
            }
        )

    @server.tool(
        name="inspect_model",
        description="Inspect an ONNX or PyTorch artifact and return its tensor metadata.",
    )
    def inspect_model_tool(model_path: str) -> str:
        path = Path(model_path).expanduser().resolve()
        if path.suffix.lower() == ".onnx":
            return _json(_dump(inspect_onnx(path)))
        loaded = load_model(path)
        info = inspect_torch_model(
            loaded.model,
            loaded.input_shape,
            name=path.stem,
            architecture=loaded.architecture,
            seed=active.seed,
        )
        return _json(_dump(info))

    @server.tool(
        name="export_model",
        description=(
            "Export a registered PyTorch model or checkpoint to ONNX with dynamic axes, "
            "graph simplification and numerical validation."
        ),
    )
    def export_model_tool(
        model_name: str = "edge_cnn",
        output_path: str | None = None,
        opset: int = 17,
        input_shape: list[int] | None = None,
        dynamic_batch: bool = True,
        dynamic_spatial: bool = False,
        simplify: bool = True,
        width: int = 16,
        depth: int = 3,
        num_classes: int = 10,
    ) -> str:
        shape = _parse_shape(input_shape, (1, 3, 32, 32))
        model = build_model(
            model_name,
            width=width,
            depth=depth,
            num_classes=num_classes,
        )
        destination = (
            Path(output_path).expanduser().resolve()
            if output_path
            else _default_output(
                active.exports_dir,
                f"{model_name}-opset{opset}-{short_id()}",
            )
        )
        result = exporter.export(
            model,
            destination,
            input_shape=shape,
            options=ExportOptions(
                opset=opset,
                dynamic_batch=dynamic_batch,
                dynamic_spatial=dynamic_spatial,
                simplify=simplify,
            ),
            name=model_name,
        )
        return _json(_dump(result))

    @server.tool(
        name="quantize_model",
        description=(
            "Convert an ONNX model to FP16, dynamic INT8 or static INT8 with synthetic or "
            "real calibration data."
        ),
    )
    def quantize_model_tool(
        model_path: str,
        precision: str = "int8_dynamic",
        output_path: str | None = None,
        calibration_samples: int = 16,
        calibration_method: str = "gaussian",
        calibration_data_path: str | None = None,
        per_channel: bool = True,
        reduce_range: bool = False,
        quant_format: str = "qdq",
    ) -> str:
        selected = _precision(precision, Precision.INT8_DYNAMIC)
        destination = (
            Path(output_path).expanduser().resolve()
            if output_path
            else _default_output(
                active.quantized_dir,
                f"{Path(model_path).stem}.{selected.value}",
            )
        )
        result = quantizer.quantize(
            model_path,
            destination,
            options=QuantizationOptions(
                precision=selected,
                calibration_samples=calibration_samples,
                calibration_method=CalibrationMethod(calibration_method),
                calibration_data_path=calibration_data_path,
                per_channel=per_channel,
                reduce_range=reduce_range,
                quant_format=cast(Literal["qdq", "qoperator"], quant_format),
            ),
        )
        return _json(_dump(result))

    @server.tool(
        name="analyze_drift",
        description=(
            "Compare a PyTorch reference model or ONNX reference against a candidate and "
            "return cosine similarity, MSE, MAE, max error and KL divergence."
        ),
    )
    def analyze_drift_tool(
        candidate_path: str,
        reference_model: str = "edge_cnn",
        reference_onnx_path: str | None = None,
        input_shape: list[int] | None = None,
        num_samples: int = 8,
        min_cosine_similarity: float = 0.99,
        width: int = 16,
        depth: int = 3,
        num_classes: int = 10,
    ) -> str:
        shape = _parse_shape(input_shape, (1, 3, 32, 32))
        thresholds = DriftThresholds(min_cosine_similarity=min_cosine_similarity)
        if reference_onnx_path:
            report = analyzer.compare_onnx(
                reference_onnx_path,
                candidate_path,
                input_shape=shape,
                num_samples=num_samples,
                thresholds=thresholds,
            )
        else:
            model = build_model(
                reference_model,
                width=width,
                depth=depth,
                num_classes=num_classes,
            )
            report = analyzer.compare(
                model,
                candidate_path,
                input_shape=shape,
                num_samples=num_samples,
                thresholds=thresholds,
                reference_name=reference_model,
            )
        return _json(_dump(report))

    @server.tool(
        name="benchmark_model",
        description=(
            "Benchmark an ONNX model with warmup, p50/p90/p99 latency, FPS, RAM and CPU usage."
        ),
    )
    def benchmark_model_tool(
        model_path: str,
        input_shape: list[int] | None = None,
        warmup_runs: int = 5,
        iterations: int = 30,
        hardware_profile: str = "generic-x86",
        precision: str | None = None,
    ) -> str:
        shape = _parse_shape(input_shape, (1, 3, 32, 32))
        report = profiler.benchmark(
            model_path,
            input_shape=shape,
            warmup_runs=warmup_runs,
            iterations=iterations,
            hardware_profile=hardware_profile,
            precision=_precision(precision, Precision.FP32) if precision else None,
        )
        return _json(_dump(report))

    @server.tool(
        name="auto_tune",
        description=(
            "Search opset, simplification and precision combinations under latency, size and "
            "fidelity constraints; persist the run and return the ranked candidates."
        ),
    )
    def auto_tune_tool(
        model_name: str = "edge_cnn",
        input_shape: list[int] | None = None,
        max_latency_ms: float | None = None,
        min_cosine_similarity: float = 0.99,
        max_size_mb: float | None = None,
        allowed_precisions: list[str] | None = None,
        opsets: list[int] | None = None,
        simplify_options: list[bool] | None = None,
        hardware_profile: str = "generic-x86",
        warmup_runs: int = 5,
        iterations: int = 30,
        width: int = 16,
        depth: int = 3,
        num_classes: int = 10,
        persist: bool = True,
    ) -> str:
        shape = _parse_shape(input_shape, (1, 3, 32, 32))
        constraints = TuningConstraints(
            max_latency_ms=max_latency_ms,
            min_cosine_similarity=min_cosine_similarity,
            max_size_mb=max_size_mb,
            allowed_precisions=(
                [Precision(value) for value in allowed_precisions]
                if allowed_precisions
                else [
                    Precision.FP32,
                    Precision.FP16,
                    Precision.INT8_DYNAMIC,
                    Precision.INT8_STATIC,
                ]
            ),
            opset_candidates=opsets or [17],
            simplify_candidates=simplify_options or [True],
            hardware_profile=hardware_profile,
            warmup_runs=warmup_runs,
            iterations=iterations,
        )
        model = build_model(
            model_name,
            width=width,
            depth=depth,
            num_classes=num_classes,
        )
        result = tuner.tune(
            model,
            constraints=constraints,
            input_shape=shape,
            model_name=model_name,
        )
        if persist:
            store.save(result)
        payload = summarize_tuning(result)
        payload["constraints"] = _dump(result.constraints)
        payload["hardware_profile"] = _dump(result.hardware_profile)
        payload["candidates"] = [
            {
                "candidate_id": candidate.candidate_id,
                "precision": candidate.precision.value,
                "opset": candidate.opset,
                "simplified": candidate.simplified,
                "size_mb": candidate.size_mb,
                "feasible": candidate.feasible,
                "score": candidate.score,
                "p99_ms": (
                    candidate.benchmark.latency.p99_ms if candidate.benchmark is not None else None
                ),
                "cosine_similarity": (
                    candidate.drift.metrics.cosine_similarity
                    if candidate.drift is not None
                    else None
                ),
                "rejection_reasons": candidate.rejection_reasons,
            }
            for candidate in result.candidates
        ]
        return _json(payload)

    @server.tool(
        name="list_reports",
        description="List persisted tuning runs available for report generation.",
    )
    def list_reports_tool() -> str:
        return _json({"runs": store.list_runs()})

    @server.tool(
        name="generate_report",
        description="Render a persisted tuning run as JSON, Markdown and self-contained HTML.",
    )
    def generate_report_tool(
        run_id: str,
        formats: list[str] | None = None,
    ) -> str:
        artifacts = reports.generate_from_run(
            run_id, formats=formats or ["json", "markdown", "html"]
        )
        return _json(_dump(artifacts))

    @server.tool(
        name="read_report",
        description="Return the Markdown or JSON report for a persisted tuning run.",
    )
    def read_report_tool(run_id: str, report_format: str = "markdown") -> str:
        result = store.load(run_id)
        report = build_forge_report(result)
        if report_format == "json":
            return report.model_dump_json(indent=2)
        if report_format == "html":
            artifacts = reports.generate(result, formats=["html"])
            return Path(artifacts.files["html"]).read_text(encoding="utf-8")
        return render_markdown(report)

    @server.resource(
        "edge-forge://catalog/models",
        name="model-catalog",
        description="Catalog of reproducible reference architectures.",
        mime_type="application/json",
    )
    def model_catalog_resource() -> str:
        return _json({"models": describe_models()})

    @server.resource(
        "edge-forge://hardware/profiles",
        name="hardware-profiles",
        description="Built-in edge hardware profiles and deployment budgets.",
        mime_type="application/json",
    )
    def hardware_profiles_resource() -> str:
        return _json({"profiles": [_dump(profile) for profile in list_hardware_profiles()]})

    @server.resource(
        "edge-forge://system/info",
        name="system-info",
        description="Runtime and execution provider information.",
        mime_type="application/json",
    )
    def system_info_resource() -> str:
        return _json(
            {
                "version": __version__,
                "sdk": _SDK_FLAVOR,
                "onnxruntime": runtime_version(),
                "providers": available_providers(),
                "timestamp": iso_timestamp(),
            }
        )

    @server.resource(
        "edge-forge://reports",
        name="report-index",
        description="Index of persisted tuning runs.",
        mime_type="application/json",
    )
    def report_index_resource() -> str:
        return _json({"runs": store.list_runs()})

    @server.resource(
        "edge-forge://reports/{run_id}",
        name="report",
        description="Markdown report for a persisted tuning run.",
        mime_type="text/markdown",
    )
    def report_resource(run_id: str) -> str:
        return render_markdown(build_forge_report(store.load(run_id)))

    return server


def main(argv: list[str] | None = None) -> None:
    """Run the MCP server over stdio, SSE or streamable HTTP."""
    parser = argparse.ArgumentParser(
        prog="edge-forge-mcp",
        description="MCP server for edge model compression, drift auditing and benchmarking.",
    )
    parser.add_argument(
        "--transport",
        choices=("stdio", "sse", "streamable-http"),
        default=None,
        help="MCP transport to use; defaults to EDGE_FORGE_SERVER_TRANSPORT.",
    )
    parser.add_argument("--host", default=None, help="Bind host for HTTP transports.")
    parser.add_argument("--port", type=int, default=None, help="Bind port for HTTP transports.")
    parser.add_argument("--workspace", default=None, help="Workspace directory for artifacts.")
    parser.add_argument("--reports-dir", default=None, help="Directory for reports and run state.")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default=None,
    )
    arguments = parser.parse_args(argv)
    overrides: dict[str, Any] = {}
    if arguments.workspace:
        overrides["workspace"] = Path(arguments.workspace)
    if arguments.reports_dir:
        overrides["reports_dir"] = Path(arguments.reports_dir)
    if arguments.log_level:
        overrides["log_level"] = arguments.log_level
    settings = ForgeSettings(**overrides) if overrides else get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    transport = arguments.transport or settings.server_transport
    host = arguments.host or settings.server_host
    port = arguments.port or settings.server_port
    server = create_server(settings)
    _LOGGER.info("starting MCP server over %s", transport)
    if transport == "stdio":
        server.run(transport="stdio")
    else:
        server.run(transport=transport, host=host, port=port)


if __name__ == "__main__":
    try:
        main()
    except EdgeForgeError as error:
        print(f"edge-forge-mcp error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
