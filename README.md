# mcp-edge-vision-forge

Production-grade Model Context Protocol server and MLOps CLI for compressing,
quantizing, auditing and benchmarking computer-vision models destined for edge
devices.

`mcp-edge-vision-forge` turns a PyTorch model into an optimized ONNX artifact,
measures the numerical drift introduced by quantization, profiles latency and
memory with ONNX Runtime, and searches export and precision configurations under
explicit deployment constraints. Everything is exposed both to humans through a
Typer CLI and to AI agents through the official MCP Python SDK.

## Contents

- [Highlights](#highlights)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [CLI reference](#cli-reference)
- [MCP server](#mcp-server)
- [Concepts](#concepts)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Development](#development)
- [Docker](#docker)
- [Security](#security)
- [Limitations](#limitations)
- [License](#license)

## Highlights

- PyTorch to ONNX export with dynamic batch and spatial axes, graph
  simplification and numerical validation against the reference model.
- FP16 conversion, dynamic INT8 and static INT8 quantization with synthetic or
  real calibration data.
- Drift analysis with cosine similarity, MSE, MAE, maximum absolute error, KL
  divergence, SNR and top-1 agreement.
- Benchmarking with warmup, p50/p90/p99 latency, FPS, peak RSS and CPU usage.
- Autonomous tuning over opsets, simplification and precision under latency,
  size and fidelity budgets.
- JSON, Markdown and self-contained HTML reports.
- Official MCP server exposing twelve tools and five resource endpoints.
- Typer CLI with Rich tables and machine-readable JSON output.
- Deterministic reference architectures for reproducible experiments.
- Strict typing with Pydantic v2 and mypy, tested on Python 3.11, 3.12 and 3.13.

## Requirements

- Python 3.11 or newer.
- PyTorch 2.2 or newer.
- ONNX 1.16 or newer.
- ONNX Runtime 1.18 or newer.
- Optional: `onnxsim` for the strongest graph simplification. The project falls
  back to a built-in deterministic simplifier when it is unavailable.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

On Windows, activate the environment with `.venv\Scripts\activate`.

For CPU-only environments, install PyTorch from the CPU index first to avoid
downloading CUDA libraries:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2"
pip install -e ".[dev]"
```

The default dependency set installs the CPU build of ONNX Runtime. For NVIDIA
GPUs replace it in the same environment, because both packages ship the same
`onnxruntime` module and cannot be installed side by side:

```bash
pip uninstall -y onnxruntime
pip install "onnxruntime-gpu>=1.18"
```

The CLI and the MCP server still request `CPUExecutionProvider` by default;
pass another provider through the Python API (`ModelProfiler(provider=...)`,
`OnnxExporter(provider=...)`) once the GPU package is active.

## Quick start

```bash
edge-forge models
edge-forge export edge_cnn --output artifacts/edge_cnn.onnx
edge-forge quantize artifacts/edge_cnn.onnx --precision int8_dynamic
edge-forge drift artifacts/edge_cnn.int8_dynamic.onnx --reference-model edge_cnn
edge-forge benchmark artifacts/edge_cnn.int8_dynamic.onnx --hardware raspberry-pi-5
edge-forge tune edge_cnn --max-latency-ms 25 --min-cosine 0.99 --max-size-mb 2
```

A tuning run prints a ranked candidate table and writes JSON, Markdown and HTML
reports. The persisted run identifier can be reused later:

```bash
edge-forge report run-1a2b3c4d5e --formats markdown,html
```

Every command accepts `--json` for scripting:

```bash
edge-forge benchmark artifacts/edge_cnn.onnx --json | jq '.latency.p99_ms'
```

## CLI reference

| Command | Purpose |
| --- | --- |
| `edge-forge version` | Print the package version. |
| `edge-forge system` | Show ONNX Runtime version and execution providers. |
| `edge-forge models` | List reproducible reference architectures. |
| `edge-forge hardware` | List built-in edge hardware profiles. |
| `edge-forge inspect PATH` | Inspect ONNX or PyTorch tensor metadata. |
| `edge-forge export MODEL` | Export a registered PyTorch model to ONNX. |
| `edge-forge quantize PATH` | Produce FP16, dynamic INT8 or static INT8. |
| `edge-forge drift PATH` | Compare a reference model with a candidate. |
| `edge-forge benchmark PATH` | Measure latency, FPS, RSS and CPU. |
| `edge-forge tune MODEL` | Search configurations under constraints. |
| `edge-forge report RUN_ID` | Render a persisted tuning run. |
| `edge-forge serve` | Run the MCP server. |

Useful options:

```text
--workspace PATH          Artifact directory (default: artifacts)
--reports-dir PATH        Report and run directory (default: reports)
--log-level LEVEL         DEBUG, INFO, WARNING, ERROR or CRITICAL
--input-shape 1,3,32,32   NCHW input shape
--opset 17                ONNX opset between 11 and 23
--dynamic-batch           Symbolic batch dimension (default)
--dynamic-spatial         Symbolic height and width dimensions
--simplify / --no-simplify
--precision fp16|int8_dynamic|int8_static
--calibration-samples 16
--calibration-method gaussian|uniform|zeros|real
--hardware raspberry-pi-5|jetson-orin-nano|coral-edge-tpu|smartphone-midrange|generic-x86
--max-latency-ms FLOAT
--max-size-mb FLOAT
--min-cosine FLOAT
```

## MCP server

Start the server over stdio:

```bash
edge-forge serve --transport stdio
```

Or over streamable HTTP:

```bash
edge-forge serve --transport streamable-http --host 0.0.0.0 --port 8000
```

The dedicated `edge-forge-mcp` entry point accepts the same transport options and
is convenient for desktop client configuration:

```json
{
  "mcpServers": {
    "edge-vision-forge": {
      "command": "edge-forge-mcp",
      "args": ["--transport", "stdio"],
      "env": {
        "EDGE_FORGE_WORKSPACE": "/data/edge-forge/artifacts",
        "EDGE_FORGE_REPORTS_DIR": "/data/edge-forge/reports"
      }
    }
  }
}
```

### Tools

| Tool | Key arguments | Purpose |
| --- | --- | --- |
| `list_models` | none | List reproducible reference architectures. |
| `list_hardware_profiles` | none | List hardware latency and size budgets. |
| `system_info` | none | Report versions and execution providers. |
| `inspect_model` | `model_path` | Inspect ONNX or PyTorch tensor metadata. |
| `export_model` | `model_name`, `opset`, `input_shape`, `dynamic_batch`, `simplify` | Export a model to validated ONNX. |
| `quantize_model` | `model_path`, `precision`, `calibration_samples`, `calibration_method`, `nodes_to_exclude` | Produce FP16 or INT8 artifacts. |
| `analyze_drift` | `candidate_path`, `reference_model` or `reference_onnx_path`, `num_samples` | Quantify fidelity loss. |
| `benchmark_model` | `model_path`, `input_shape`, `warmup_runs`, `iterations`, `hardware_profile` | Measure latency and resources. |
| `auto_tune` | `model_name`, `max_latency_ms`, `max_size_mb`, `min_cosine_similarity`, `allowed_precisions` | Search configurations and persist the run. |
| `list_reports` | none | List persisted tuning runs. |
| `generate_report` | `run_id`, `formats` | Render JSON, Markdown and HTML. |
| `read_report` | `run_id`, `report_format` | Return report content directly. |

Tool results are JSON encoded text, so any MCP client can consume them without a
custom decoder. Anticipated failures are returned as `ToolError` results whose
text is a JSON object with `error`, `message` and `retryable` fields.

Paths sent by the client are confined to the workspace, reports and cache
directories plus the process working directory. Add extra directories with
`EDGE_FORGE_EXTRA_ALLOWED_ROOTS`, or disable the guard for a trusted deployment
with `EDGE_FORGE_RESTRICT_PATHS=false` or `edge-forge-mcp --allow-unconfined-paths`.

### Resources

| URI | Description |
| --- | --- |
| `edge-forge://catalog/models` | Reference architecture catalog. |
| `edge-forge://hardware/profiles` | Built-in hardware profiles. |
| `edge-forge://system/info` | Runtime and provider information. |
| `edge-forge://reports` | Index of persisted tuning runs. |
| `edge-forge://reports/{run_id}` | Markdown report for a specific run. |

## Concepts

### Precision matrix

| Precision | Strategy | Typical use |
| --- | --- | --- |
| `fp32` | Unmodified ONNX graph. | Reference baseline. |
| `fp16` | Weight and activation conversion with float32 I/O. | GPU and NPU targets with FP16 support. |
| `int8_dynamic` | Dynamic quantization of weights, activations quantized at runtime. | CPU targets with minimal calibration effort. |
| `int8_static` | QDQ graph with MinMax calibration. | Edge TPU and latency-critical CPU deployments. |

Static quantization needs calibration data. The default is synthetic Gaussian
noise, which is adequate for smoke tests and architecture comparisons. For real
deployments, pass a `.npy` file with shape `(samples, channels, height, width)`,
an `.npz` archive keyed by input name, or a directory containing one `.npy` file
per input.

### Drift metrics

- Cosine similarity: directional agreement between flattened outputs.
- MSE and MAE: absolute error magnitude.
- Maximum absolute error: worst-case element-wise deviation.
- KL divergence: divergence between softmax distributions.
- SNR: signal-to-noise ratio in decibels.
- Top-1 agreement: fraction of samples with the same predicted class.

Default acceptance thresholds are cosine similarity `>= 0.99`, MSE `<= 0.01`,
MAE `<= 0.1`, maximum absolute error `<= 1.0`, KL divergence `<= 0.1` and SNR
`>= 20 dB`. A candidate is feasible only when every configured threshold passes.

### Tuning score

Candidates that satisfy every constraint are ranked with a composite score:

```text
score = 0.5 * (baseline_p50 / candidate_p50)
      + 0.3 * (baseline_size / candidate_size)
      + 0.2 * cosine_similarity
```

The FP32 export of the same configuration is the baseline. Higher is better.
Infeasible candidates are still reported with their rejection reasons for
auditing. Latency constraints are enforced on p99, not on the mean, so a
configuration is only selected when its tail latency fits the budget.

The three weights are configurable through `TuningConstraints`
(`score_weight_latency`, `score_weight_size`, `score_weight_drift`); they only
need to sum to a positive value. The defaults above reproduce the published
formula and keep the ranking stable across runs.

### Hardware profiles

| Profile | Accelerator | Cores | RAM | Latency budget | Max model size |
| --- | --- | --- | --- | --- | --- |
| `raspberry-pi-5` | CPU | 4 | 8192 MiB | 100 ms | 128 MiB |
| `jetson-orin-nano` | GPU | 6 | 8192 MiB | 30 ms | 512 MiB |
| `coral-edge-tpu` | NPU | 4 | 4096 MiB | 20 ms | 64 MiB |
| `smartphone-midrange` | CPU | 8 | 6144 MiB | 40 ms | 200 MiB |
| `generic-x86` | CPU | 8 | 16384 MiB | 50 ms | 1024 MiB |

Profiles also restrict the precisions that are considered feasible, which is why
a Coral target only accepts static INT8 candidates.

### Reproducibility

Reference architectures are initialized deterministically from a fixed seed, so
rebuilding `edge_cnn` reproduces the exact weights used during export. This is
what makes `analyze_drift` and `auto_tune` meaningful across CLI invocations.
Pass `seed=None` to the Python registry API when random initialization is
desired.

## Configuration

All settings use the `EDGE_FORGE_` prefix and can be provided through environment
variables or a `.env` file. See [.env.example](.env.example).

| Variable | Default | Description |
| --- | --- | --- |
| `EDGE_FORGE_WORKSPACE` | `artifacts` | Exported and quantized artifacts. |
| `EDGE_FORGE_CACHE_DIR` | `.edge_forge_cache` | Preprocessing scratch space. |
| `EDGE_FORGE_REPORTS_DIR` | `reports` | Reports and persisted runs. |
| `EDGE_FORGE_DEFAULT_OPSET` | `17` | Default ONNX opset. |
| `EDGE_FORGE_DEFAULT_INPUT_SHAPE` | `[1, 3, 224, 224]` | Default input shape. |
| `EDGE_FORGE_DEFAULT_PRECISION` | `fp32` | Default quantization precision. |
| `EDGE_FORGE_SEED` | `42` | Seed for calibration and dummy inputs. |
| `EDGE_FORGE_INTRA_OP_THREADS` | `0` | ONNX Runtime intra-op threads, zero means automatic. |
| `EDGE_FORGE_INTER_OP_THREADS` | `1` | ONNX Runtime inter-op threads. |
| `EDGE_FORGE_CALIBRATION_SAMPLES` | `16` | Default calibration batch count (max 4096). |
| `EDGE_FORGE_MAX_WORKERS` | `1` | Worker budget reserved for future parallel search. |
| `EDGE_FORGE_RESTRICT_PATHS` | `true` | Confine MCP tool paths to the permitted roots. |
| `EDGE_FORGE_EXTRA_ALLOWED_ROOTS` | *(empty)* | Extra directories for the MCP server, separated by `os.pathsep`. |
| `EDGE_FORGE_SERVER_TRANSPORT` | `stdio` | Default MCP transport. |
| `EDGE_FORGE_SERVER_HOST` | `127.0.0.1` | HTTP bind host. |
| `EDGE_FORGE_SERVER_PORT` | `8000` | HTTP bind port. |
| `EDGE_FORGE_LOG_LEVEL` | `INFO` | Logging level. |

## Architecture

```text
PyTorch model
    |
    v
OnnxExporter -----> GraphSimplifier -----> validation (PyTorch vs ONNX Runtime)
    |
    v
QuantizationEngine -----> FP16 / dynamic INT8 / static INT8
    |
    v
DriftAnalyzer -----> cosine, MSE, MAE, max error, KL, SNR, top-1
    |
    v
ModelProfiler -----> p50/p90/p99, FPS, peak RSS, CPU
    |
    v
AutoTuner -----> ranked candidates under constraints
    |
    v
ReportGenerator -----> JSON / Markdown / self-contained HTML
```

### Module map

| Path | Responsibility |
| --- | --- |
| `edge_forge/config.py` | Validated Pydantic settings and directory layout. |
| `edge_forge/models/schemas.py` | Reports, metrics, constraints and hardware profiles. |
| `edge_forge/models/registry.py` | Reference CNNs, depthwise network, tiny ViT and checkpoint loader. |
| `edge_forge/core/exporter.py` | ONNX export, opset validation, simplification and numerical checks. |
| `edge_forge/core/quantizer.py` | FP16, dynamic INT8, static INT8 and calibration readers. |
| `edge_forge/core/drift_analyzer.py` | PyTorch versus ONNX and ONNX versus ONNX fidelity analysis. |
| `edge_forge/core/profiler.py` | Warmup, latency percentiles, FPS and resource sampling. |
| `edge_forge/core/auto_tuner.py` | Constraint-aware configuration search and scoring. |
| `edge_forge/core/numerics.py` | Vectorized metric primitives. |
| `edge_forge/reporting/report_gen.py` | Run persistence and JSON, Markdown and HTML rendering. |
| `edge_forge/mcp_server/server.py` | MCP tools, resources and transport entry point. |
| `edge_forge/cli/main.py` | Typer and Rich command line interface. |

## Development

```bash
make install
make test
make lint
make typecheck
make check
```

The test-suite uses tiny deterministic models so the full export, quantization,
drift and tuning pipeline runs in seconds. It covers exporter validation, graph
simplification, all quantization modes, synthetic and real calibration, metric
primitives, profiling, tuning feasibility, report escaping, MCP tool and
resource calls, and the CLI.

Continuous integration runs lint, formatting, strict mypy and pytest on Python
3.11, 3.12 and 3.13, then builds the Docker image.

## Docker

```bash
docker build -t mcp-edge-vision-forge .
docker run --rm -it -v "$PWD/artifacts:/app/artifacts" mcp-edge-vision-forge
```

The default container command starts the MCP server over streamable HTTP on port
8000. Use `docker compose up --build` for the full setup with persistent artifact
and report volumes.

## Security

- Report identifiers are validated against path traversal before any file access.
- Full PyTorch module pickles and TorchScript archives are rejected by default
  because deserialization can execute arbitrary code. State dictionaries are
  loaded with the restricted unpickler; Python callers can opt into the unsafe
  formats with `load_model(..., allow_unsafe=True)` for trusted files only.
- HTML reports are fully escaped and contain no scripts, external stylesheets or
  remote assets.
- Every path supplied by an MCP client is resolved (following `..` and symbolic
  links) and confined to the workspace, reports and cache directories, the
  process working directory and any directory listed in
  `EDGE_FORGE_EXTRA_ALLOWED_ROOTS`. The CLI is not confined. Disable the guard
  only for a trusted single-user deployment with
  `EDGE_FORGE_RESTRICT_PATHS=false` or `edge-forge-mcp --allow-unconfined-paths`.
- Anticipated failures reach the client as structured `ToolError` payloads
  (`error`, `message`, `retryable`) instead of opaque protocol errors.
- Workloads are bounded: input tensors are capped at 25 million elements,
  benchmarks at 1000 iterations and warmups, calibration at 4096 samples, drift
  analysis at 512 samples, reference architectures at width 512, depth 32 and
  4096 classes, and a tuning search at 64 candidates. Every MCP tool runs its
  blocking work in a worker thread so the protocol event loop keeps answering.
- The MCP server can still read any file inside its permitted roots. Run it in a
  container or with restricted filesystem permissions when exposing it to
  untrusted agents.

## Limitations

- The exporter supports a single model input. Drift analysis also compares a
  single-input reference. Multi-input calibration is supported by the quantizer.
- Benchmarks measure the current host. A Raspberry Pi profile describes the
  deployment budget, not the machine executing the benchmark.
- Static INT8 quality depends on the calibration distribution. Use real data for
  production decisions.
- Dynamic INT8 can increase file size for very small models because quantized
  graphs add metadata and scale tensors. Compare against the FP32 baseline
  instead of assuming a fixed compression ratio.
- ONNX Runtime execution providers other than CPU can be injected through the
  Python API; the CLI and MCP defaults target CPU. Install `onnxruntime-gpu`
  instead of `onnxruntime` to make CUDA available.
- TorchScript loading is opt-in through `allow_unsafe=True`; the exporter and the
  MCP server only load registered state dictionaries by default.
- Graph simplification relies on `onnxsim` when installed. The fallback
  simplifier is deterministic and removes identity nodes, dead branches and
  unused initializers, but does not fuse operators.

## License

MIT. See [LICENSE](LICENSE).
