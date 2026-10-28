# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS runtime

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    EDGE_FORGE_WORKSPACE=/app/artifacts \
    EDGE_FORGE_REPORTS_DIR=/app/reports \
    EDGE_FORGE_CACHE_DIR=/app/.cache \
    EDGE_FORGE_SERVER_TRANSPORT=streamable-http \
    EDGE_FORGE_SERVER_HOST=0.0.0.0 \
    EDGE_FORGE_SERVER_PORT=8000 \
    EDGE_FORGE_ALLOW_NETWORK=true

WORKDIR /app

RUN groupadd --system --gid 10001 edgeforge \
    && useradd --system --uid 10001 --gid edgeforge --home-dir /app --shell /usr/sbin/nologin edgeforge

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url "${TORCH_INDEX_URL}" "torch>=2.2" \
    && python -m pip install . \
    && mkdir -p /app/artifacts /app/reports /app/.cache \
    && chown -R edgeforge:edgeforge /app

USER edgeforge

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["edge-forge", "version"]

ENTRYPOINT ["edge-forge"]
CMD ["serve", "--transport", "streamable-http", "--host", "0.0.0.0", "--port", "8000"]
