"""Atomic artifact IO and content hashing.

Every artifact this project publishes is written atomically (temp file in the same
directory, then ``os.replace``) so a crashed or killed run never leaves a half-written
metrics file that a later stage would silently read.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import yaml

__all__ = [
    "atomic_write_bytes",
    "atomic_write_text",
    "file_sha256",
    "hash_mapping",
    "read_json",
    "read_yaml",
    "write_csv",
    "write_json",
    "write_parquet",
    "write_yaml",
]

_HASH_CHUNK = 1 << 20


@contextmanager
def _atomic_path(path: Path) -> Iterator[Path]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        yield tmp
        os.replace(tmp, path)  # noqa: PTH105 - atomic rename primitive
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_write_bytes(path: Path | str, payload: bytes) -> Path:
    """Write ``payload`` to ``path`` atomically."""
    target = Path(path)
    with _atomic_path(target) as tmp:
        tmp.write_bytes(payload)
    return target


def atomic_write_text(path: Path | str, text: str, *, encoding: str = "utf-8") -> Path:
    """Write ``text`` to ``path`` atomically."""
    return atomic_write_bytes(path, text.encode(encoding))


def file_sha256(path: Path | str) -> str:
    """Stream a SHA-256 of a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


class _CanonicalEncoder(json.JSONEncoder):
    """JSON encoder that renders NumPy scalars/arrays and Paths deterministically."""

    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return o.as_posix()
        if isinstance(o, set | frozenset):
            return sorted(o)
        return super().default(o)


def hash_mapping(mapping: Mapping[str, Any], *, length: int = 16) -> str:
    """Hash a configuration mapping in a key-order-independent way.

    Used for the ``config_hash`` recorded on every prediction row, so that two runs whose
    resolved configuration differs anywhere produce different hashes.
    """
    canonical = json.dumps(mapping, sort_keys=True, separators=(",", ":"), cls=_CanonicalEncoder)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:length]


def write_json(path: Path | str, payload: Any, *, indent: int = 2) -> Path:
    """Serialise ``payload`` as sorted-key JSON, atomically."""
    text = json.dumps(payload, indent=indent, sort_keys=True, cls=_CanonicalEncoder) + "\n"
    return atomic_write_text(path, text)


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_yaml(path: Path | str, payload: Any) -> Path:
    """Serialise ``payload`` as YAML, atomically, preserving key order."""
    text = yaml.safe_dump(payload, sort_keys=False, default_flow_style=False, allow_unicode=True)
    return atomic_write_text(path, text)


def read_yaml(path: Path | str) -> Any:
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"configuration file not found: {resolved}")
    return yaml.safe_load(resolved.read_text(encoding="utf-8"))


def write_parquet(path: Path | str, frame: Any) -> Path:
    """Write a pandas DataFrame to Parquet atomically."""
    target = Path(path)
    with _atomic_path(target) as tmp:
        frame.to_parquet(tmp, index=False)
    return target


def write_csv(path: Path | str, frame: Any, **kwargs: Any) -> Path:
    """Write a pandas DataFrame to CSV atomically."""
    target = Path(path)
    kwargs.setdefault("index", False)
    with _atomic_path(target) as tmp:
        frame.to_csv(tmp, **kwargs)
    return target
