"""Typer and Rich command line interface for human operators."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from edge_forge import __version__
from edge_forge.config import ForgeSettings
from edge_forge.core.auto_tuner import AutoTuner, summarize_tuning
from edge_forge.core.drift_analyzer import DriftAnalyzer
from edge_forge.core.exporter import OnnxExporter, inspect_onnx
from edge_forge.core.profiler import ModelProfiler, available_providers, runtime_version
from edge_forge.core.quantizer import QuantizationEngine
from edge_forge.errors import EdgeForgeError
from edge_forge.models.registry import build_model, describe_models, inspect_torch_model, load_model
from edge_forge.models.schemas import (
    CalibrationMethod,
    DriftThresholds,
    ExportOptions,
    Precision,
    QuantizationOptions,
    TuningConstraints,
    list_hardware_profiles,
)
from edge_forge.reporting.report_gen import ReportGenerator, RunStore
from edge_forge.utils import format_mb, format_ms

app = typer.Typer(
    name="edge-forge",
    help=(
        "Compress, quantize, audit and benchmark PyTorch and ONNX computer-vision models "
        "for edge deployment."
    ),
    add_completion=False,
    no_args_is_help=True,
)
console = Console()
error_console = Console(stderr=True)


def _settings(ctx: typer.Context) -> ForgeSettings:
    if isinstance(ctx.obj, ForgeSettings):
        return ctx.obj
    settings = ForgeSettings().ensure_directories()
    ctx.obj = settings
    return settings


def _fail(error: Exception) -> None:
    error_console.print(f"[bold red]error:[/bold red] {error}")
    raise typer.Exit(code=1) from error


def _parse_shape(value: str) -> tuple[int, ...]:
    tokens = [token for token in value.replace("x", ",").split(",") if token.strip()]
    try:
        shape = tuple(int(token) for token in tokens)
    except ValueError as exc:
        raise ValueError(f"invalid input shape '{value}'") from exc
    if not shape or any(dimension <= 0 for dimension in shape):
        raise ValueError("input shape dimensions must be strictly positive")
    return shape


def _parse_precisions(value: str) -> list[Precision]:
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens:
        raise ValueError("at least one precision must be provided")
    return [Precision(token) for token in tokens]


def _parse_opsets(value: str) -> list[int]:
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens:
        raise ValueError("at least one opset must be provided")
    return [int(token) for token in tokens]


def _parse_formats(value: str) -> list[str]:
    tokens = [token.strip().lower() for token in value.split(",") if token.strip()]
    if not tokens:
        raise ValueError("at least one report format must be provided")
    return tokens


def _print_json(payload: Any) -> None:
    if hasattr(payload, "model_dump_json"):
        typer.echo(payload.model_dump_json())
        return
    typer.echo(json.dumps(payload, default=str))


def _latency_table(report: Any) -> Table:
    table = Table(title=f"Benchmark: {report.model_name}", show_lines=False)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Precision", report.precision.value)
    table.add_row("Provider", report.provider)
    table.add_row("Hardware profile", report.hardware_profile)
    table.add_row("Parameters", f"{report.parameters:,}")
    table.add_row("Size", format_mb(report.resources.model_size_mb))
    table.add_row("Mean latency", format_ms(report.latency.mean_ms))
    table.add_row("p50 latency", format_ms(report.latency.p50_ms))
    table.add_row("p90 latency", format_ms(report.latency.p90_ms))
    table.add_row("p99 latency", format_ms(report.latency.p99_ms))
    table.add_row("FPS", f"{report.latency.fps:.2f}")
    table.add_row("Peak RSS", format_mb(report.resources.peak_rss_mb))
    table.add_row("CPU mean", f"{report.resources.cpu_percent_mean:.2f}%")
    if report.meets_budget is not None:
        table.add_row("Meets budget", "yes" if report.meets_budget else "no")
    return table


def _candidate_table(candidates: list[Any]) -> Table:
    table = Table(title="Tuning candidates", show_lines=False)
    for column in (
        "ID",
        "Precision",
        "Opset",
        "Simplified",
        "Size",
        "p99",
        "FPS",
        "Cosine",
        "Feasible",
        "Score",
    ):
        table.add_column(
            column, justify="right" if column not in ("ID", "Precision", "Feasible") else "left"
        )
    for candidate in candidates:
        latency = candidate.benchmark.latency if candidate.benchmark is not None else None
        drift = candidate.drift.metrics if candidate.drift is not None else None
        table.add_row(
            candidate.candidate_id,
            candidate.precision.value,
            str(candidate.opset),
            "yes" if candidate.simplified else "no",
            format_mb(candidate.size_mb),
            format_ms(latency.p99_ms) if latency is not None else "n/a",
            f"{latency.fps:.1f}" if latency is not None else "n/a",
            f"{drift.cosine_similarity:.5f}" if drift is not None else "n/a",
            "yes" if candidate.feasible else "no",
            f"{candidate.score:.4f}",
        )
    return table


@app.callback()
def configure(
    ctx: typer.Context,
    workspace: Annotated[
        Path | None,
        typer.Option("--workspace", help="Directory for exported and quantized artifacts."),
    ] = None,
    reports_dir: Annotated[
        Path | None,
        typer.Option("--reports-dir", help="Directory for reports and persisted runs."),
    ] = None,
    log_level: Annotated[
        str | None,
        typer.Option("--log-level", help="Logging level: DEBUG, INFO, WARNING, ERROR."),
    ] = None,
) -> None:
    """Configure shared runtime settings before a command executes."""
    overrides: dict[str, Any] = {}
    if workspace is not None:
        overrides["workspace"] = workspace
    if reports_dir is not None:
        overrides["reports_dir"] = reports_dir
    if log_level is not None:
        overrides["log_level"] = log_level.upper()
    try:
        settings = ForgeSettings(**overrides).ensure_directories()
    except Exception as exc:
        _fail(exc)
    ctx.obj = settings
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        stream=sys.stderr,
        format="%(levelname)s %(name)s: %(message)s",
    )


@app.command()
def version(
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON.")] = False,
) -> None:
    """Print the package version."""
    if json_output:
        _print_json({"version": __version__})
        return
    console.print(f"mcp-edge-vision-forge {__version__}")


@app.command()
def system(
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON.")] = False,
) -> None:
    """Show ONNX Runtime and execution provider information."""
    payload = {
        "version": __version__,
        "onnxruntime": runtime_version(),
        "providers": available_providers(),
        "python": sys.version.split()[0],
    }
    if json_output:
        _print_json(payload)
        return
    table = Table(title="Runtime")
    table.add_column("Component")
    table.add_column("Value")
    for key, value in payload.items():
        table.add_row(key, ", ".join(value) if isinstance(value, list) else str(value))
    console.print(table)


@app.command()
def models(
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON.")] = False,
) -> None:
    """List the reproducible reference architectures."""
    catalog = describe_models()
    if json_output:
        _print_json({"models": catalog})
        return
    table = Table(title="Reference models")
    table.add_column("Name")
    table.add_column("Family")
    table.add_column("Parameters", justify="right")
    table.add_column("Size", justify="right")
    table.add_column("Description")
    for entry in catalog:
        table.add_row(
            str(entry["name"]),
            str(entry["family"]),
            f"{entry['parameters']:,}",
            format_mb(float(entry["size_mb"])),
            str(entry["description"]),
        )
    console.print(table)


@app.command()
def hardware(
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON.")] = False,
) -> None:
    """List built-in edge hardware profiles."""
    profiles = list_hardware_profiles()
    if json_output:
        _print_json({"profiles": [profile.model_dump(mode="json") for profile in profiles]})
        return
    table = Table(title="Hardware profiles")
    table.add_column("Name")
    table.add_column("Accelerator")
    table.add_column("Cores", justify="right")
    table.add_column("RAM", justify="right")
    table.add_column("Latency budget", justify="right")
    table.add_column("Max size", justify="right")
    for profile in profiles:
        table.add_row(
            profile.name,
            profile.accelerator,
            str(profile.cpu_cores),
            f"{profile.ram_mb} MiB",
            f"{profile.typical_latency_budget_ms:.1f} ms",
            f"{profile.max_model_size_mb:.1f} MiB",
        )
    console.print(table)


@app.command()
def inspect(
    model_path: Annotated[Path, typer.Argument(help="ONNX or PyTorch artifact.")],
    input_shape: Annotated[
        str, typer.Option("--input-shape", help="NCHW shape, e.g. 1,3,32,32.")
    ] = "1,3,32,32",
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON.")] = False,
) -> None:
    """Inspect an ONNX or PyTorch artifact."""
    try:
        path = model_path.expanduser().resolve()
        if path.suffix.lower() == ".onnx":
            info = inspect_onnx(path)
        else:
            loaded = load_model(path)
            info = inspect_torch_model(
                loaded.model,
                _parse_shape(input_shape),
                name=path.stem,
                architecture=loaded.architecture,
            )
    except EdgeForgeError as exc:
        _fail(exc)
        return
    if json_output:
        _print_json(info)
        return
    table = Table(title=f"Model: {info.name}")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Architecture", info.architecture)
    table.add_row("Parameters", f"{info.parameters:,}")
    table.add_row("Size", format_mb(info.size_mb))
    table.add_row("Precision", info.precision.value)
    table.add_row("Opset", str(info.opset or "n/a"))
    table.add_row(
        "Inputs", ", ".join(f"{spec.name}{list(spec.shape)}" for spec in info.input_specs)
    )
    table.add_row(
        "Outputs", ", ".join(f"{spec.name}{list(spec.shape)}" for spec in info.output_specs)
    )
    console.print(table)


@app.command()
def export(
    ctx: typer.Context,
    model_name: Annotated[str, typer.Argument(help="Registered model name.")] = "edge_cnn",
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Destination ONNX path.")
    ] = None,
    opset: Annotated[int, typer.Option("--opset", help="ONNX opset version.")] = 17,
    input_shape: Annotated[str, typer.Option("--input-shape", help="NCHW shape.")] = "1,3,32,32",
    dynamic_batch: Annotated[bool, typer.Option("--dynamic-batch/--static-batch")] = True,
    dynamic_spatial: Annotated[bool, typer.Option("--dynamic-spatial/--static-spatial")] = False,
    simplify: Annotated[bool, typer.Option("--simplify/--no-simplify")] = True,
    width: Annotated[int, typer.Option("--width", help="Channel width.")] = 16,
    depth: Annotated[int, typer.Option("--depth", help="Number of stages.")] = 3,
    num_classes: Annotated[int, typer.Option("--num-classes")] = 10,
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON.")] = False,
) -> None:
    """Export a registered PyTorch model to ONNX."""
    settings = _settings(ctx)
    try:
        shape = _parse_shape(input_shape)
        model = build_model(model_name, width=width, depth=depth, num_classes=num_classes)
        destination = (
            output.expanduser().resolve()
            if output is not None
            else settings.exports_dir / f"{model_name}-opset{opset}.onnx"
        )
        result = OnnxExporter(settings).export(
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
    except (EdgeForgeError, ValueError) as exc:
        _fail(exc)
        return
    if json_output:
        _print_json(result)
        return
    table = Table(title="Export result")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Path", result.model_path)
    table.add_row("Opset", str(result.opset))
    table.add_row("Size", format_mb(result.size_mb))
    table.add_row("Parameters", f"{result.parameters:,}")
    table.add_row("Simplified", "yes" if result.simplified else "no")
    if result.validation is not None:
        table.add_row("Validated", "yes" if result.validation.passed else "no")
        table.add_row("Max abs error", f"{result.validation.max_absolute_error:.3e}")
        table.add_row("Cosine similarity", f"{result.validation.cosine_similarity:.6f}")
    table.add_row("SHA-256", result.checksum)
    console.print(table)


@app.command()
def quantize(
    ctx: typer.Context,
    model_path: Annotated[Path, typer.Argument(help="Source ONNX model.")],
    precision: Annotated[
        str, typer.Option("--precision", help="fp16, int8_dynamic or int8_static.")
    ] = "int8_dynamic",
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    calibration_samples: Annotated[int, typer.Option("--calibration-samples")] = 16,
    calibration_method: Annotated[
        str, typer.Option("--calibration-method", help="gaussian, uniform, zeros or real.")
    ] = "gaussian",
    calibration_data_path: Annotated[Path | None, typer.Option("--calibration-data")] = None,
    per_channel: Annotated[bool, typer.Option("--per-channel/--per-tensor")] = True,
    reduce_range: Annotated[bool, typer.Option("--reduce-range")] = False,
    quant_format: Annotated[str, typer.Option("--quant-format", help="qdq or qoperator.")] = "qdq",
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON.")] = False,
) -> None:
    """Quantize an ONNX model to FP16 or INT8."""
    settings = _settings(ctx)
    try:
        selected = Precision(precision)
        destination = (
            output.expanduser().resolve()
            if output is not None
            else settings.quantized_dir / f"{model_path.stem}.{selected.value}.onnx"
        )
        result = QuantizationEngine(settings).quantize(
            model_path,
            destination,
            options=QuantizationOptions(
                precision=selected,
                calibration_samples=calibration_samples,
                calibration_method=CalibrationMethod(calibration_method),
                calibration_data_path=(
                    str(calibration_data_path) if calibration_data_path is not None else None
                ),
                per_channel=per_channel,
                reduce_range=reduce_range,
                quant_format=cast(Literal["qdq", "qoperator"], quant_format),
            ),
        )
    except (EdgeForgeError, ValueError) as exc:
        _fail(exc)
        return
    if json_output:
        _print_json(result)
        return
    table = Table(title="Quantization result")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Output", result.output_path)
    table.add_row("Precision", result.precision.value)
    table.add_row("Source size", format_mb(result.source_size_mb))
    table.add_row("Output size", format_mb(result.size_mb))
    table.add_row("Compression", f"{result.compression_ratio:.2f}x")
    table.add_row("Calibration samples", str(result.calibration_samples))
    table.add_row("SHA-256", result.checksum)
    console.print(table)


@app.command()
def drift(
    ctx: typer.Context,
    candidate_path: Annotated[Path, typer.Argument(help="Candidate ONNX model.")],
    reference_model: Annotated[
        str, typer.Option("--reference-model", help="Registered reference architecture.")
    ] = "edge_cnn",
    reference_onnx: Annotated[
        Path | None, typer.Option("--reference-onnx", help="Use an ONNX reference instead.")
    ] = None,
    input_shape: Annotated[str, typer.Option("--input-shape")] = "1,3,32,32",
    samples: Annotated[int, typer.Option("--samples")] = 8,
    min_cosine: Annotated[float, typer.Option("--min-cosine")] = 0.99,
    width: Annotated[int, typer.Option("--width")] = 16,
    depth: Annotated[int, typer.Option("--depth")] = 3,
    num_classes: Annotated[int, typer.Option("--num-classes")] = 10,
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON.")] = False,
) -> None:
    """Measure numerical drift between a reference and a candidate model."""
    settings = _settings(ctx)
    try:
        shape = _parse_shape(input_shape)
        thresholds = DriftThresholds(min_cosine_similarity=min_cosine)
        analyzer = DriftAnalyzer(settings)
        if reference_onnx is not None:
            report = analyzer.compare_onnx(
                reference_onnx,
                candidate_path,
                input_shape=shape,
                num_samples=samples,
                thresholds=thresholds,
            )
        else:
            model = build_model(reference_model, width=width, depth=depth, num_classes=num_classes)
            report = analyzer.compare(
                model,
                candidate_path,
                input_shape=shape,
                num_samples=samples,
                thresholds=thresholds,
                reference_name=reference_model,
            )
    except (EdgeForgeError, ValueError) as exc:
        _fail(exc)
        return
    if json_output:
        _print_json(report)
        return
    table = Table(title=f"Drift: {report.candidate_model}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Passed", "yes" if report.passed else "no")
    table.add_row("Cosine similarity", f"{report.metrics.cosine_similarity:.6f}")
    table.add_row("MSE", f"{report.metrics.mse:.6g}")
    table.add_row("MAE", f"{report.metrics.mae:.6g}")
    table.add_row("Max absolute error", f"{report.metrics.max_absolute_error:.6g}")
    table.add_row("KL divergence", f"{report.metrics.kl_divergence:.6g}")
    table.add_row("SNR", f"{report.metrics.snr_db:.2f} dB")
    table.add_row("Top-1 agreement", f"{report.metrics.top1_agreement:.4f}")
    console.print(table)
    if report.metrics.violations:
        console.print(Panel("\n".join(report.metrics.violations), title="Violations"))


@app.command()
def benchmark(
    ctx: typer.Context,
    model_path: Annotated[Path, typer.Argument(help="ONNX model to benchmark.")],
    input_shape: Annotated[str, typer.Option("--input-shape")] = "1,3,32,32",
    warmup: Annotated[int, typer.Option("--warmup")] = 5,
    iterations: Annotated[int, typer.Option("--iterations")] = 30,
    hardware_profile: Annotated[
        str, typer.Option("--hardware", help="Built-in hardware profile.")
    ] = "generic-x86",
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON.")] = False,
) -> None:
    """Benchmark latency and resource usage on the current host."""
    settings = _settings(ctx)
    try:
        report = ModelProfiler(settings).benchmark(
            model_path,
            input_shape=_parse_shape(input_shape),
            warmup_runs=warmup,
            iterations=iterations,
            hardware_profile=hardware_profile,
        )
    except (EdgeForgeError, ValueError) as exc:
        _fail(exc)
        return
    if json_output:
        _print_json(report)
        return
    console.print(_latency_table(report))


@app.command()
def tune(
    ctx: typer.Context,
    model_name: Annotated[str, typer.Argument(help="Registered model name.")] = "edge_cnn",
    input_shape: Annotated[str, typer.Option("--input-shape")] = "1,3,32,32",
    max_latency_ms: Annotated[float | None, typer.Option("--max-latency-ms")] = None,
    min_cosine: Annotated[float, typer.Option("--min-cosine")] = 0.99,
    max_size_mb: Annotated[float | None, typer.Option("--max-size-mb")] = None,
    precisions: Annotated[
        str, typer.Option("--precisions", help="Comma separated precision list.")
    ] = "fp32,fp16,int8_dynamic,int8_static",
    opsets: Annotated[str, typer.Option("--opsets", help="Comma separated opset list.")] = "17",
    simplify: Annotated[bool, typer.Option("--simplify/--no-simplify")] = True,
    hardware_profile: Annotated[str, typer.Option("--hardware")] = "generic-x86",
    warmup: Annotated[int, typer.Option("--warmup")] = 5,
    iterations: Annotated[int, typer.Option("--iterations")] = 30,
    width: Annotated[int, typer.Option("--width")] = 16,
    depth: Annotated[int, typer.Option("--depth")] = 3,
    num_classes: Annotated[int, typer.Option("--num-classes")] = 10,
    report: Annotated[
        bool, typer.Option("--report/--no-report", help="Render reports after tuning.")
    ] = True,
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON.")] = False,
) -> None:
    """Search the configuration space under deployment constraints."""
    settings = _settings(ctx)
    try:
        constraints = TuningConstraints(
            max_latency_ms=max_latency_ms,
            min_cosine_similarity=min_cosine,
            max_size_mb=max_size_mb,
            allowed_precisions=_parse_precisions(precisions),
            opset_candidates=_parse_opsets(opsets),
            simplify_candidates=[simplify],
            hardware_profile=hardware_profile,
            warmup_runs=warmup,
            iterations=iterations,
        )
        model = build_model(model_name, width=width, depth=depth, num_classes=num_classes)
        result = AutoTuner(settings).tune(
            model,
            constraints=constraints,
            input_shape=_parse_shape(input_shape),
            model_name=model_name,
        )
        store = RunStore(settings)
        store.save(result)
        artifacts = (
            ReportGenerator(settings).generate(result, formats=("json", "markdown", "html"))
            if report
            else None
        )
    except (EdgeForgeError, ValueError) as exc:
        _fail(exc)
        return
    if json_output:
        payload = summarize_tuning(result)
        payload["reports"] = artifacts.files if artifacts is not None else {}
        _print_json(payload)
        return
    console.print(_candidate_table(result.candidates))
    summary = summarize_tuning(result)
    console.print(
        Panel(
            f"Best candidate: {summary['best_candidate_id'] or 'none'}\n"
            f"Feasible candidates: {summary['feasible']} of {summary['candidates']}\n"
            f"Duration: {summary['duration_s']:.2f} s\n"
            f"Workdir: {summary['workdir']}",
            title=f"Tuning run {result.run_id}",
        )
    )
    if artifacts is not None:
        for report_format, path in artifacts.files.items():
            console.print(f"{report_format}: {path}")


@app.command()
def report(
    ctx: typer.Context,
    run_id: Annotated[str, typer.Argument(help="Persisted tuning run identifier.")],
    formats: Annotated[
        str, typer.Option("--formats", help="json,markdown,html")
    ] = "json,markdown,html",
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON.")] = False,
) -> None:
    """Render a persisted tuning run in the requested formats."""
    settings = _settings(ctx)
    try:
        artifacts = ReportGenerator(settings).generate_from_run(
            run_id, formats=_parse_formats(formats)
        )
    except (EdgeForgeError, ValueError) as exc:
        _fail(exc)
        return
    if json_output:
        _print_json(artifacts)
        return
    table = Table(title=f"Report {artifacts.report_id}")
    table.add_column("Format")
    table.add_column("Path")
    for report_format, path in artifacts.files.items():
        table.add_row(report_format, path)
    console.print(table)


@app.command()
def serve(
    ctx: typer.Context,
    transport: Annotated[
        str, typer.Option("--transport", help="stdio, sse or streamable-http.")
    ] = "stdio",
    host: Annotated[str | None, typer.Option("--host")] = None,
    port: Annotated[int | None, typer.Option("--port")] = None,
) -> None:
    """Run the MCP server."""
    from edge_forge.mcp_server.server import create_server

    settings = _settings(ctx)
    try:
        server = create_server(settings)
        if transport == "stdio":
            server.run(transport="stdio")
        else:
            server.run(
                transport=transport,
                host=host or settings.server_host,
                port=port or settings.server_port,
            )
    except EdgeForgeError as exc:
        _fail(exc)


if __name__ == "__main__":
    app()
