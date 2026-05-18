"""On-disk cache for parsed + enriched lap telemetry.

The CSV → DataFrame parse runs at ~125 ms per lap (pandas C engine
ceiling) and the per-lap enrichment adds another ~20-150 ms depending
on stint length. Both are pure functions of the source CSV bytes, so
we memoize the result on disk keyed by ``(path, mtime, size)``.

A warm hit returns in roughly 10-30 ms (raw + enriched DataFrames
unpickled in one go), which is what makes "click capture → chart"
feel instantaneous on a second visit.

The cache lives under ``%LOCALAPPDATA%\\lfs-telemetry-viewer\\cache\\`` on
Windows (and the platform equivalent elsewhere). Each entry is a
single ``.pkl`` file whose name encodes the source key, so stale
entries can be evicted by simply deleting the directory.

We deliberately do *not* gzip-compress: benchmarks on a 14k-row /
131-column enriched DataFrame show plain pickle dumps at ~16 ms vs
gzip-1 at ~350 ms (writes happen on the hot cold-load path), and
pickle loads at ~10 ms vs gzip-1 decompress at ~50 ms. The ~2x disk
footprint is a fair trade for ~22x faster saves and ~5x faster loads.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import pickle
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Bumped when the on-disk format or any enrichment column set changes.
# Mismatch invalidates every existing entry transparently.
_CACHE_FORMAT_VERSION = 3

_DEFAULT_DIRNAME = "lfs-telemetry-viewer"
_CACHE_SUFFIX = ".pkl"

# Single shared lock so concurrent writers can't race on the same key.
# Reads are fine without locking (atomic file replace on write).
_write_lock = threading.Lock()


def cache_dir() -> Path:
    """Return the user-scoped cache directory, creating it if needed."""
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        root = Path(base) / _DEFAULT_DIRNAME / "cache"
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Caches" / _DEFAULT_DIRNAME
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        root = (Path(xdg) if xdg else Path.home() / ".cache") / _DEFAULT_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    return root


@dataclass(frozen=True)
class CacheKey:
    """Stable key derived from the source CSV's identity on disk."""

    path: Path
    mtime_ns: int
    size: int

    @classmethod
    def for_path(cls, path: str | Path) -> CacheKey | None:
        p = Path(path)
        try:
            st = p.stat()
        except OSError:
            return None
        return cls(path=p.resolve(), mtime_ns=st.st_mtime_ns, size=st.st_size)

    def filename(self) -> str:
        # Hash the absolute path so different captures with the same
        # basename do not collide; mtime+size make the key change
        # whenever the source bytes change.
        h = hashlib.sha1(
            f"{self.path}|{self.mtime_ns}|{self.size}|"
            f"v{_CACHE_FORMAT_VERSION}".encode()
        ).hexdigest()[:16]
        stem = self.path.stem[:32].replace(" ", "_")
        return f"{stem}__{h}{_CACHE_SUFFIX}"


def load(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Return ``(raw, enriched)`` from cache, or ``None`` on miss/error."""
    key = CacheKey.for_path(path)
    if key is None:
        return None
    target = cache_dir() / key.filename()
    if not target.is_file():
        return None
    try:
        with open(target, "rb") as fp:
            payload = pickle.load(fp)
    except (OSError, pickle.UnpicklingError, EOFError):
        # Corrupt cache entry — drop it so the next save replaces it.
        with contextlib.suppress(OSError):
            target.unlink()
        return None
    if (
        not isinstance(payload, dict)
        or payload.get("format") != _CACHE_FORMAT_VERSION
    ):
        return None
    raw = payload.get("raw")
    enriched = payload.get("enriched")
    if not isinstance(raw, pd.DataFrame) or not isinstance(enriched, pd.DataFrame):
        return None
    return raw, enriched


def save(
    path: str | Path, raw: pd.DataFrame, enriched: pd.DataFrame,
) -> None:
    """Persist ``(raw, enriched)`` for ``path``. Best-effort; errors swallowed."""
    key = CacheKey.for_path(path)
    if key is None:
        return
    target = cache_dir() / key.filename()
    tmp = target.with_suffix(target.suffix + ".tmp")
    payload = {
        "format": _CACHE_FORMAT_VERSION,
        "raw": raw,
        "enriched": enriched,
    }
    try:
        with _write_lock:
            with open(tmp, "wb") as fp:
                pickle.dump(payload, fp, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, target)
    except OSError:
        # Disk full / permission denied → silently skip caching.
        with contextlib.suppress(OSError):
            tmp.unlink()


def clear() -> int:
    """Delete every cached entry. Returns the count removed."""
    n = 0
    for child in cache_dir().glob(f"*{_CACHE_SUFFIX}"):
        try:
            child.unlink()
            n += 1
        except OSError:
            pass
    return n


__all__ = ["CacheKey", "cache_dir", "clear", "load", "save"]
