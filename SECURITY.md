# Security policy

## Reporting a vulnerability

Open a private security advisory on the GitHub repository or email the maintainer
listed in `pyproject.toml`. Please include the affected version, a minimal
reproduction and the expected impact. Do not open a public issue before a fix is
available.

## Threat model

`mcp-edge-vision-forge` runs untrusted models and receives tool arguments from an
AI agent. The default deployment is a local stdio server started by a single
user. The HTTP transports are disabled unless the operator opts in.

The project defends against:

- path traversal and symlink escapes from MCP tool arguments;
- report identifiers that could escape the reports directory;
- pickle deserialization through checkpoints, calibration arrays and ONNX
  external data;
- oversized inputs, calibration files, artifacts and search spaces that could
  exhaust the host;
- HTML and Markdown injection in generated reports;
- accidental network exposure of the tool surface.

It does not defend against a malicious model that exploits a memory-safety bug in
PyTorch, ONNX or ONNX Runtime, nor against an operator who deliberately disables
the guards.

## Defaults

- `EDGE_FORGE_RESTRICT_PATHS=true` confines every MCP path to the workspace,
  reports and cache directories, the process working directory and
  `EDGE_FORGE_EXTRA_ALLOWED_ROOTS`.
- `EDGE_FORGE_ALLOW_NETWORK=false` keeps the sse and streamable-http transports
  disabled.
- State dictionaries are loaded with `torch.load(weights_only=True)`. TorchScript
  and full module pickles require `allow_unsafe=True` from a trusted caller.
- Calibration arrays are loaded with `numpy.load(allow_pickle=False)`.
- ONNX external data must stay inside the model directory.
- `EDGE_FORGE_OPERATION_TIMEOUT_S=0` disables the per-tool deadline.

## Exposing the server over HTTP

Enable the transport explicitly and protect it:

```bash
EDGE_FORGE_ALLOW_NETWORK=true \
EDGE_FORGE_AUTH_TOKEN="$(openssl rand -hex 32)" \
edge-forge serve --transport streamable-http --host 127.0.0.1 --port 8000
```

The built-in guard is a static bearer token. TLS termination, rate limiting, IP
allow-lists and audit logging belong to the reverse proxy in front of the
process. Without `EDGE_FORGE_AUTH_TOKEN` the server logs a warning and relies on
the network boundary alone.

## Hard limits

| Resource | Limit |
| --- | --- |
| Input tensor elements | 25,000,000 |
| Benchmark iterations and warmups | 1,000 each |
| Calibration samples | 4,096 |
| Drift samples | 512 |
| Tuning candidates | 64 |
| Reference architecture width, depth, classes | 512, 32, 4,096 |
| Model file size | 2 GiB |
| Calibration file size | 512 MiB |
