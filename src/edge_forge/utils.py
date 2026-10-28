"""Shared, dependency-light helpers used across the package."""

from __future__ import annotations

import hashlib
import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

_CHUNK_SIZE: Final[int] = 1024 * 1024
_SLUG_PATTERN: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9]+")


def utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""
    return datetime.now(tz=UTC)


def iso_timestamp(moment: datetime | None = None) -> str:
    """Return an ISO-8601 representation with second precision."""
    value = moment or utc_now()
    return value.astimezone(UTC).replace(microsecond=0).isoformat()


def slugify(value: str, *, max_length: int = 48) -> str:
    """Convert arbitrary text into a filesystem-safe slug."""
    lowered = value.strip().lower()
    slug = _SLUG_PATTERN.sub("-", lowered).strip("-")
    if not slug:
        slug = "artifact"
    return slug[:max_length].strip("-") or "artifact"


def short_id(prefix: str = "") -> str:
    """Return a compact, collision-resistant identifier."""
    token = uuid.uuid4().hex[:10]
    return f"{prefix}{token}" if prefix else token


def sha256_file(path: Path) -> str:
    """Compute the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_to_mb(size_bytes: int) -> float:
    """Convert a byte count to mebibytes rounded to four decimals."""
    return round(size_bytes / (1024 * 1024), 4)


def file_size_mb(path: Path) -> float:
    """Return the size of a file in mebibytes."""
    return bytes_to_mb(path.stat().st_size)


def ensure_directory(path: Path) -> Path:
    """Create a directory (and parents) and return it resolved."""
    resolved = path.expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def resolve_path(value: str | os.PathLike[str], *, base: Path | None = None) -> Path:
    """Resolve a user supplied path relative to an optional base directory."""
    candidate = Path(value).expanduser()
    if not candidate.is_absolute() and base is not None:
        candidate = base / candidate
    return candidate.resolve()


def format_ms(value: float) -> str:
    """Format a latency value for human readable output."""
    if value < 1.0:
        return f"{value * 1000:.1f} us"
    return f"{value:.2f} ms"


def format_mb(value: float) -> str:
    """Format a size in MiB for human readable output."""
    if value < 1.0:
        return f"{value * 1024:.1f} KiB"
    return f"{value:.2f} MiB"
