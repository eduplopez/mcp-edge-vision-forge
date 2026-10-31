"""Comparative report generation in JSON, Markdown and self-contained HTML."""

from __future__ import annotations

import html
import json
import platform
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from edge_forge.config import ForgeSettings, get_settings
from edge_forge.errors import ReportError, RunNotFoundError
from edge_forge.models.schemas import (
    ForgeReport,
    TuningCandidate,
    TuningResult,
)
from edge_forge.utils import format_mb, format_ms, iso_timestamp, utc_now

_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9._-]+$")
_SUPPORTED_FORMATS = ("json", "markdown", "html")


def validate_report_id(value: str) -> str:
    """Validate a report identifier before it is used as a file name.

    Raises:
        ReportError: If the identifier could escape the report directory.
    """
    if not _SAFE_RUN_ID.fullmatch(value) or ".." in value:
        raise ReportError(f"invalid report identifier: {value}")
    return value


class ReportArtifacts(BaseModel):
    """Paths produced by one report generation call."""

    report_id: str
    generated_at: str
    files: dict[str, str] = Field(default_factory=dict)


class RunStore:
    """Persist and retrieve tuning runs from the reports directory."""

    def __init__(self, settings: ForgeSettings | None = None) -> None:
        self._settings = settings or get_settings()
        self._settings.ensure_directories()

    @property
    def directory(self) -> Path:
        """Directory containing persisted tuning runs."""
        path = self._settings.reports_dir / "runs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _path(self, run_id: str) -> Path:
        if not _SAFE_RUN_ID.fullmatch(run_id) or ".." in run_id:
            raise RunNotFoundError(f"invalid run identifier: {run_id}")
        return self.directory / f"{run_id}.json"

    def save(self, result: TuningResult) -> Path:
        """Write a tuning result as indented JSON."""
        path = self._path(result.run_id)
        try:
            path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        except OSError as exc:
            raise ReportError(f"unable to persist tuning run {result.run_id}: {exc}") from exc
        return path

    def load(self, run_id: str) -> TuningResult:
        """Load a persisted tuning result."""
        path = self._path(run_id)
        if not path.exists():
            raise RunNotFoundError(f"tuning run '{run_id}' does not exist")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return TuningResult.model_validate(payload)
        except (OSError, ValueError) as exc:
            raise ReportError(f"unable to read tuning run {run_id}: {exc}") from exc

    def list_runs(self) -> list[dict[str, Any]]:
        """Return a lightweight index of every persisted run."""
        entries: list[dict[str, Any]] = []
        for path in sorted(self.directory.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            candidates = payload.get("candidates") or []
            entries.append(
                {
                    "run_id": payload.get("run_id", path.stem),
                    "source_model": (payload.get("source_model") or {}).get("name", "unknown"),
                    "created_at": payload.get("created_at"),
                    "candidates": len(candidates),
                    "feasible": sum(1 for item in candidates if item.get("feasible")),
                    "best_candidate_id": payload.get("best_candidate_id"),
                    "duration_s": payload.get("duration_s"),
                }
            )
        return entries

    def exists(self, run_id: str) -> bool:
        """Return whether a run identifier has a persisted result."""
        try:
            return self._path(run_id).exists()
        except RunNotFoundError:
            return False


def build_forge_report(result: TuningResult) -> ForgeReport:
    """Convert a tuning result into a presentation-ready report model."""
    environment = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "generated_at": iso_timestamp(),
    }
    try:
        import onnx

        environment["onnx"] = str(onnx.__version__)
    except ImportError:
        environment["onnx"] = "unavailable"
    try:
        import onnxruntime

        environment["onnxruntime"] = str(onnxruntime.__version__)
    except ImportError:
        environment["onnxruntime"] = "unavailable"
    try:
        import torch

        environment["torch"] = str(torch.__version__)
    except ImportError:
        environment["torch"] = "unavailable"
    return ForgeReport(
        report_id=result.run_id,
        title=f"Edge Vision Forge report: {result.source_model.name}",
        source_model=result.source_model,
        hardware_profile=result.hardware_profile,
        constraints=result.constraints,
        candidates=result.candidates,
        best_candidate_id=result.best_candidate_id,
        environment=environment,
        notes=result.notes,
    )


def _cell(value: Any) -> str:
    return html.escape(str(value)).replace("|", "\\|").replace("\n", " ")


def _candidate_row(candidate: TuningCandidate) -> str:
    latency = candidate.benchmark.latency if candidate.benchmark is not None else None
    drift = candidate.drift.metrics if candidate.drift is not None else None
    return (
        "| "
        + " | ".join(
            [
                _cell(candidate.candidate_id),
                _cell(candidate.precision.value),
                _cell(candidate.opset),
                "yes" if candidate.simplified else "no",
                f"{candidate.size_mb:.4f}",
                f"{latency.p50_ms:.4f}" if latency is not None else "n/a",
                f"{latency.p99_ms:.4f}" if latency is not None else "n/a",
                f"{latency.fps:.2f}" if latency is not None else "n/a",
                f"{drift.cosine_similarity:.6f}" if drift is not None else "n/a",
                f"{drift.mse:.6g}" if drift is not None else "n/a",
                "yes" if candidate.feasible else "no",
                f"{candidate.score:.4f}",
            ]
        )
        + " |"
    )


def render_markdown(report: ForgeReport) -> str:
    """Render a report as GitHub-flavored Markdown."""
    best = report.best
    lines: list[str] = [
        f"# {_cell(report.title)}",
        "",
        f"Generated at: {iso_timestamp(report.generated_at)}",
        "",
        "## Summary",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Report ID | {_cell(report.report_id)} |",
        f"| Source model | {_cell(report.source_model.name)} |",
        f"| Architecture | {_cell(report.source_model.architecture)} |",
        f"| Parameters | {report.source_model.parameters} |",
        f"| Target hardware | {_cell(report.hardware_profile.name)} |",
        f"| Accelerator | {_cell(report.hardware_profile.accelerator)} |",
        f"| Latency budget | {report.hardware_profile.typical_latency_budget_ms:.2f} ms |",
        f"| Candidates | {len(report.candidates)} |",
        f"| Best candidate | {_cell(report.best_candidate_id or 'none')} |",
    ]
    if best is not None:
        lines.extend(
            [
                f"| Best precision | {_cell(best.precision.value)} |",
                f"| Best size | {format_mb(best.size_mb)} |",
                (
                    f"| Best p99 latency | {format_ms(best.benchmark.latency.p99_ms)} |"
                    if best.benchmark is not None
                    else "| Best p99 latency | n/a |"
                ),
                (
                    f"| Best cosine similarity | {best.drift.metrics.cosine_similarity:.6f} |"
                    if best.drift is not None
                    else "| Best cosine similarity | n/a |"
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Constraints",
            "",
            "| Constraint | Value |",
            "| --- | --- |",
            f"| Max latency (p99) | {_optional(report.constraints.max_latency_ms, ' ms')} |",
            f"| Max size | {_optional(report.constraints.max_size_mb, ' MiB')} |",
            f"| Min cosine similarity | {report.constraints.min_cosine_similarity:.6f} |",
            f"| Max MSE | {_optional(report.constraints.max_mse, '')} |",
            f"| Max absolute error | {_optional(report.constraints.max_absolute_error, '')} |",
            "| Allowed precisions | "
            + ", ".join(item.value for item in report.constraints.allowed_precisions)
            + " |",
            "",
            "## Candidates",
            "",
            "| ID | Precision | Opset | Simplified | Size (MiB) | p50 (ms) | p99 (ms) | "
            "FPS | Cosine | MSE | Feasible | Score |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    lines.extend(_candidate_row(candidate) for candidate in report.candidates)
    lines.extend(["", "## Drift detail", ""])
    for candidate in report.candidates:
        if candidate.drift is None:
            lines.append(f"### {candidate.candidate_id}: no drift report")
            lines.append("")
            continue
        drift = candidate.drift
        lines.extend(
            [
                f"### {candidate.candidate_id} ({candidate.precision.value})",
                "",
                f"- Passed: {'yes' if drift.passed else 'no'}",
                f"- Cosine similarity: {drift.metrics.cosine_similarity:.6f}",
                f"- MSE: {drift.metrics.mse:.6g}",
                f"- MAE: {drift.metrics.mae:.6g}",
                f"- Max absolute error: {drift.metrics.max_absolute_error:.6g}",
                f"- KL divergence: {drift.metrics.kl_divergence:.6g}",
                f"- SNR: {drift.metrics.snr_db:.2f} dB",
                f"- Top-1 agreement: {drift.metrics.top1_agreement:.4f}",
                "",
            ]
        )
        if drift.metrics.violations:
            lines.append("Violations:")
            lines.extend(f"- {violation}" for violation in drift.metrics.violations)
            lines.append("")
    lines.extend(["## Environment", ""])
    lines.extend(f"- {_cell(key)}: {_cell(value)}" for key, value in report.environment.items())
    if report.notes:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {_cell(note)}" for note in report.notes)
    lines.append("")
    return "\n".join(lines)


def _optional(value: float | None, suffix: str) -> str:
    return "unbounded" if value is None else f"{value:.4f}{suffix}"


def _badge(candidate: TuningCandidate) -> str:
    if candidate.feasible:
        return '<span class="badge ok">feasible</span>'
    return '<span class="badge ko">rejected</span>'


def render_html(report: ForgeReport) -> str:
    """Render a self-contained HTML report with inline CSS only."""
    best = report.best
    rows: list[str] = []
    for candidate in report.candidates:
        latency = candidate.benchmark.latency if candidate.benchmark is not None else None
        drift_metrics = candidate.drift.metrics if candidate.drift is not None else None
        rows.append(
            "<tr>"
            f"<td>{html.escape(candidate.candidate_id)}</td>"
            f"<td>{html.escape(candidate.precision.value)}</td>"
            f"<td>{candidate.opset}</td>"
            f"<td>{'yes' if candidate.simplified else 'no'}</td>"
            f"<td>{candidate.size_mb:.4f}</td>"
            f"<td>{f'{latency.p50_ms:.4f}' if latency is not None else 'n/a'}</td>"
            f"<td>{f'{latency.p99_ms:.4f}' if latency is not None else 'n/a'}</td>"
            f"<td>{f'{latency.fps:.2f}' if latency is not None else 'n/a'}</td>"
            "<td>"
            f"{f'{drift_metrics.cosine_similarity:.6f}' if drift_metrics is not None else 'n/a'}"
            "</td>"
            f"<td>{f'{drift_metrics.mse:.6g}' if drift_metrics is not None else 'n/a'}</td>"
            f"<td>{_badge(candidate)}</td>"
            f"<td>{candidate.score:.4f}</td>"
            "</tr>"
        )
    drift_sections: list[str] = []
    for candidate in report.candidates:
        if candidate.drift is None:
            continue
        drift_report = candidate.drift
        violations = "".join(
            f"<li>{html.escape(item)}</li>" for item in drift_report.metrics.violations
        )
        drift_sections.append(
            f"<details><summary>{html.escape(candidate.candidate_id)} "
            f"({html.escape(candidate.precision.value)})</summary>"
            "<table><tbody>"
            f"<tr><th>Cosine similarity</th>"
            f"<td>{drift_report.metrics.cosine_similarity:.6f}</td></tr>"
            f"<tr><th>MSE</th><td>{drift_report.metrics.mse:.6g}</td></tr>"
            f"<tr><th>MAE</th><td>{drift_report.metrics.mae:.6g}</td></tr>"
            f"<tr><th>Max absolute error</th>"
            f"<td>{drift_report.metrics.max_absolute_error:.6g}</td></tr>"
            f"<tr><th>KL divergence</th>"
            f"<td>{drift_report.metrics.kl_divergence:.6g}</td></tr>"
            f"<tr><th>SNR</th><td>{drift_report.metrics.snr_db:.2f} dB</td></tr>"
            f"<tr><th>Top-1 agreement</th><td>{drift_report.metrics.top1_agreement:.4f}</td></tr>"
            "</tbody></table>" + (f"<ul>{violations}</ul>" if violations else "") + "</details>"
        )
    notes = "".join(f"<li>{html.escape(note)}</li>" for note in report.notes)
    environment = "".join(
        f"<li><strong>{html.escape(key)}</strong>: {html.escape(value)}</li>"
        for key, value in report.environment.items()
    )
    best_summary = (
        f"<p><strong>Best candidate:</strong> {html.escape(best.candidate_id)} "
        f"({html.escape(best.precision.value)}, {format_mb(best.size_mb)}"
        + (
            f", p99 {format_ms(best.benchmark.latency.p99_ms)}"
            if best.benchmark is not None
            else ""
        )
        + ")</p>"
        if best is not None
        else "<p><strong>No feasible candidate was found.</strong></p>"
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(report.title)}</title>
<style>
:root {{ color-scheme: light dark; --bg: #f6f7fb; --panel: #ffffff; --ink: #16181d;
  --muted: #5b6270; --line: #dfe3ea; --accent: #1f5eff; --ok: #0b7a4b; --ko: #b42318; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--bg); color: var(--ink);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
  Helvetica, Arial, sans-serif; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 32px 20px 64px; }}
h1 {{ font-size: 1.8rem; margin: 0 0 4px; }}
h2 {{ font-size: 1.2rem; margin: 32px 0 12px; }}
p, li {{ line-height: 1.55; }}
.meta {{ color: var(--muted); font-size: 0.9rem; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 12px; margin: 20px 0; }}
.card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
  padding: 14px 16px; }}
.card span {{ display: block; color: var(--muted); font-size: 0.78rem;
  text-transform: uppercase; letter-spacing: 0.04em; }}
.card strong {{ display: block; margin-top: 6px; font-size: 1.1rem; }}
.panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
  padding: 16px; overflow-x: auto; }}
table {{ width: 100%; border-collapse: collapse; font-size: 0.86rem; }}
th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line);
  white-space: nowrap; }}
th {{ color: var(--muted); font-weight: 600; }}
tbody tr:last-child td {{ border-bottom: none; }}
.badge {{ display: inline-block; border-radius: 999px; padding: 2px 9px;
  font-size: 0.75rem; font-weight: 600; }}
.badge.ok {{ color: var(--ok); background: rgba(11, 122, 75, 0.12); }}
.badge.ko {{ color: var(--ko); background: rgba(180, 35, 24, 0.12); }}
details {{ margin: 8px 0; border: 1px solid var(--line); border-radius: 8px;
  padding: 10px 12px; background: var(--panel); }}
summary {{ cursor: pointer; font-weight: 600; }}
ul {{ margin: 8px 0 0; padding-left: 20px; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg: #111318; --panel: #191c22; --ink: #e9ecf2; --muted: #9aa3b2;
    --line: #2b303a; --accent: #7aa2ff; }}
}}
</style>
</head>
<body>
<main>
<h1>{html.escape(report.title)}</h1>
<p class="meta">Generated at {html.escape(iso_timestamp(report.generated_at))} |
Report ID {html.escape(report.report_id)}</p>
<div class="cards">
<div class="card"><span>Source model</span>
<strong>{html.escape(report.source_model.name)}</strong></div>
<div class="card"><span>Parameters</span>
<strong>{report.source_model.parameters}</strong></div>
<div class="card"><span>Target hardware</span>
<strong>{html.escape(report.hardware_profile.name)}</strong></div>
<div class="card"><span>Latency budget</span>
<strong>{report.hardware_profile.typical_latency_budget_ms:.2f} ms</strong></div>
<div class="card"><span>Candidates</span><strong>{len(report.candidates)}</strong></div>
</div>
{best_summary}
<h2>Candidates</h2>
<div class="panel">
<table>
<thead><tr><th>ID</th><th>Precision</th><th>Opset</th><th>Simplified</th><th>Size (MiB)</th>
<th>p50 (ms)</th><th>p99 (ms)</th><th>FPS</th><th>Cosine</th><th>MSE</th>
<th>Status</th><th>Score</th></tr></thead>
<tbody>
{"".join(rows)}
</tbody>
</table>
</div>
<h2>Constraints</h2>
<div class="panel">
<table><tbody>
<tr><th>Max latency (p99)</th>
<td>{html.escape(_optional(report.constraints.max_latency_ms, " ms"))}</td></tr>
<tr><th>Max size</th>
<td>{html.escape(_optional(report.constraints.max_size_mb, " MiB"))}</td></tr>
<tr><th>Min cosine similarity</th><td>{report.constraints.min_cosine_similarity:.6f}</td></tr>
<tr><th>Allowed precisions</th>
<td>{html.escape(", ".join(item.value for item in report.constraints.allowed_precisions))}</td></tr>
</tbody></table>
</div>
<h2>Drift detail</h2>
{"".join(drift_sections)}
<h2>Environment</h2>
<div class="panel"><ul>{environment}</ul></div>
{f'<h2>Notes</h2><div class="panel"><ul>{notes}</ul></div>' if notes else ""}
</main>
</body>
</html>
"""


class ReportGenerator:
    """Render and persist comparative reports for a tuning run."""

    def __init__(self, settings: ForgeSettings | None = None) -> None:
        self._settings = settings or get_settings()
        self._settings.ensure_directories()
        self._store = RunStore(self._settings)
        self._output_dir = self._settings.reports_dir / "reports"
        self._output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def store(self) -> RunStore:
        """Run store used to persist and load tuning results."""
        return self._store

    @property
    def output_dir(self) -> Path:
        """Directory where rendered reports are written."""
        return self._output_dir

    def generate(
        self,
        result: TuningResult,
        *,
        formats: Iterable[str] = _SUPPORTED_FORMATS,
    ) -> ReportArtifacts:
        """Render a tuning result in every requested format."""
        selected = tuple(dict.fromkeys(item.lower() for item in formats))
        if not selected:
            raise ReportError("at least one report format must be requested")
        unknown = [item for item in selected if item not in _SUPPORTED_FORMATS]
        if unknown:
            raise ReportError(
                f"unsupported report formats: {', '.join(unknown)}; "
                f"supported: {', '.join(_SUPPORTED_FORMATS)}"
            )
        report = build_forge_report(result)
        report_id = validate_report_id(report.report_id)
        files: dict[str, str] = {}
        for report_format in selected:
            if report_format == "json":
                content = render_json(report)
                suffix = "json"
            elif report_format == "markdown":
                content = render_markdown(report)
                suffix = "md"
            else:
                content = render_html(report)
                suffix = "html"
            path = self._output_dir / f"{report_id}.{suffix}"
            try:
                path.write_text(content, encoding="utf-8")
            except OSError as exc:
                raise ReportError(f"unable to write report {path}: {exc}") from exc
            files[report_format] = str(path)
        return ReportArtifacts(
            report_id=report_id,
            generated_at=iso_timestamp(utc_now()),
            files=files,
        )

    def generate_from_run(
        self,
        run_id: str,
        *,
        formats: Iterable[str] = _SUPPORTED_FORMATS,
    ) -> ReportArtifacts:
        """Load a persisted run and render its report."""
        return self.generate(self._store.load(run_id), formats=formats)


def render_json(report: ForgeReport) -> str:
    """Render a report as indented JSON."""
    return report.model_dump_json(indent=2)
