"""Validated runtime configuration for mcp-edge-vision-forge.

The configuration is expressed with Pydantic v2 settings so that every value can
be supplied through constructor arguments, environment variables prefixed with
``EDGE_FORGE_`` or a local ``.env`` file.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from edge_forge.errors import ConfigurationError
from edge_forge.utils import ensure_directory

SUPPORTED_OPSETS: tuple[int, ...] = tuple(range(11, 24))
MIN_OPSET: int = SUPPORTED_OPSETS[0]
MAX_OPSET: int = SUPPORTED_OPSETS[-1]

TransportName = Literal["stdio", "sse", "streamable-http"]
LogLevelName = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class ForgeSettings(BaseSettings):
    """Runtime settings shared by the CLI, the MCP server and the core engine."""

    model_config = SettingsConfigDict(
        env_prefix="EDGE_FORGE_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        validate_assignment=True,
    )

    workspace: Path = Field(
        default=Path("artifacts"),
        description="Directory where exported and quantized artifacts are stored.",
    )
    cache_dir: Path = Field(
        default=Path(".edge_forge_cache"),
        description="Directory reserved for intermediate preprocessing files.",
    )
    reports_dir: Path = Field(
        default=Path("reports"),
        description="Directory where reports and persisted tuning runs are written.",
    )
    default_opset: int = Field(default=17, ge=MIN_OPSET, le=MAX_OPSET)
    default_input_shape: tuple[int, ...] = Field(
        default=(1, 3, 224, 224),
        description="Default NCHW input shape used when none is supplied.",
    )
    default_precision: Literal["fp32", "fp16", "int8_dynamic", "int8_static"] = "fp32"
    seed: int = Field(default=42, ge=0)
    intra_op_threads: int = Field(default=0, ge=0, le=256)
    inter_op_threads: int = Field(default=1, ge=1, le=64)
    calibration_samples: int = Field(default=16, ge=1, le=4096)
    max_workers: int = Field(default=1, ge=1, le=64)
    server_host: str = Field(default="127.0.0.1", min_length=1)
    server_port: int = Field(default=8000, ge=1, le=65535)
    server_transport: TransportName = "stdio"
    log_level: LogLevelName = "INFO"

    @field_validator("workspace", "cache_dir", "reports_dir")
    @classmethod
    def _expand_path(cls, value: Path) -> Path:
        expanded = value.expanduser()
        return expanded if expanded.is_absolute() else (Path.cwd() / expanded).resolve()

    @field_validator("default_input_shape")
    @classmethod
    def _validate_input_shape(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value:
            raise ValueError("default_input_shape must contain at least one dimension")
        if any(dimension <= 0 for dimension in value):
            raise ValueError("default_input_shape dimensions must be strictly positive")
        return value

    @model_validator(mode="after")
    def _validate_workspace_layout(self) -> ForgeSettings:
        if self.cache_dir == self.workspace:
            raise ValueError("cache_dir and workspace must be different directories")
        return self

    def ensure_directories(self) -> ForgeSettings:
        """Create workspace, cache and reports directories if they are missing."""
        try:
            ensure_directory(self.workspace)
            ensure_directory(self.cache_dir)
            ensure_directory(self.reports_dir)
        except OSError as exc:
            raise ConfigurationError(f"unable to create runtime directories: {exc}") from exc
        return self

    @property
    def runs_dir(self) -> Path:
        """Directory holding one sub-directory per autonomous tuning run."""
        return self.workspace / "runs"

    @property
    def exports_dir(self) -> Path:
        """Directory holding exported ONNX artifacts."""
        return self.workspace / "exports"

    @property
    def quantized_dir(self) -> Path:
        """Directory holding quantized ONNX artifacts."""
        return self.workspace / "quantized"


@lru_cache(maxsize=1)
def get_settings() -> ForgeSettings:
    """Return the process-wide settings singleton."""
    return ForgeSettings().ensure_directories()


def reset_settings_cache() -> None:
    """Clear the settings singleton, mainly useful in tests."""
    get_settings.cache_clear()
