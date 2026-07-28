"""Seeding, determinism and environment capture.

Every production run records enough state here to be reconstructed later: the exact
seeds, the deterministic-algorithm settings, the interpreter and library versions, the
hardware, and the source-tree state (including a diff hash when the tree is dirty).
"""

from __future__ import annotations

import hashlib
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "SeedBundle",
    "SourceState",
    "collect_environment",
    "collect_source_state",
    "seed_everything",
    "worker_init_fn",
]


@dataclass(frozen=True)
class SeedBundle:
    """The complete set of seeds applied to a single run.

    A run is identified by ``base_seed``; the per-library seeds are derived from it so
    that quoting one integer is enough to reproduce the run.
    """

    base_seed: int
    python_seed: int
    numpy_seed: int
    torch_seed: int
    dataloader_seed: int
    augmentation_seed: int

    @classmethod
    def from_base(cls, base_seed: int) -> SeedBundle:
        """Derive all library seeds deterministically from a single integer."""
        if not 0 <= base_seed < 2**31:
            raise ValueError(f"base_seed must be in [0, 2**31), got {base_seed}")
        return cls(
            base_seed=base_seed,
            python_seed=base_seed,
            numpy_seed=(base_seed * 2_654_435_761 + 1) % (2**31),
            torch_seed=(base_seed * 40_503 + 7) % (2**31),
            dataloader_seed=(base_seed * 2_246_822_519 + 13) % (2**31),
            augmentation_seed=(base_seed * 3_266_489_917 + 17) % (2**31),
        )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def seed_everything(seed: int | SeedBundle, *, deterministic: bool = True) -> SeedBundle:
    """Seed Python, NumPy and PyTorch, optionally forcing deterministic kernels.

    Parameters
    ----------
    seed
        Either a base integer or an already-derived :class:`SeedBundle`.
    deterministic
        When true, request deterministic cuDNN/cuBLAS kernels. This is slower and a few
        operations have no deterministic implementation; those raise rather than
        silently falling back, which is the intended behaviour for a published run.

    Returns
    -------
    SeedBundle
        The bundle that was applied, for recording in the run manifest.
    """
    import torch

    bundle = seed if isinstance(seed, SeedBundle) else SeedBundle.from_base(seed)

    random.seed(bundle.python_seed)
    # The legacy global NumPy RNG is seeded deliberately: scikit-learn estimators and any
    # third-party code that calls np.random.* directly read from it. Package-internal
    # randomness uses np.random.Generator instances seeded from this bundle.
    np.random.seed(bundle.numpy_seed)  # noqa: NPY002
    torch.manual_seed(bundle.torch_seed)
    torch.cuda.manual_seed_all(bundle.torch_seed)

    if deterministic:
        # cuBLAS needs this before the first CUDA context to make GEMMs reproducible.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=False)
    return bundle


def worker_init_fn(worker_id: int) -> None:
    """Seed a DataLoader worker from the parent's torch seed.

    Without this each worker inherits the parent NumPy/random state, so augmentation and
    any NumPy-side randomness repeat across workers.
    """
    import torch

    base = torch.initial_seed() % (2**31)
    seed = (base + worker_id) % (2**31)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - see seed_everything


def _run_git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


@dataclass(frozen=True)
class SourceState:
    """Git provenance of the code that produced a run."""

    commit: str | None
    branch: str | None
    is_dirty: bool
    dirty_files: list[str] = field(default_factory=list)
    diff_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def collect_source_state(repo_root: Path | str | None = None) -> SourceState:
    """Capture the commit, branch and (if dirty) a hash of the uncommitted diff.

    A dirty tree is not an error -- it is recorded, so a reviewer can tell that a run did
    not come from a clean commit and can ask for the diff.
    """
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    commit = _run_git(["rev-parse", "HEAD"], root)
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], root)
    status = _run_git(["status", "--porcelain"], root)
    dirty_files = [line[3:] for line in status.splitlines()] if status else []
    diff_sha = None
    if dirty_files:
        diff = _run_git(["diff", "HEAD"], root) or ""
        diff_sha = hashlib.sha256(diff.encode("utf-8")).hexdigest()
    return SourceState(
        commit=commit,
        branch=branch,
        is_dirty=bool(dirty_files),
        dirty_files=dirty_files,
        diff_sha256=diff_sha,
    )


def collect_environment() -> dict[str, Any]:
    """Capture interpreter, library, OS and accelerator details for a run bundle."""
    import importlib.metadata as md

    packages: dict[str, str | None] = {}
    for name in (
        "numpy",
        "scipy",
        "pandas",
        "pyarrow",
        "scikit-learn",
        "PyWavelets",
        "torch",
        "matplotlib",
        "PyYAML",
        "opencv-python-headless",
    ):
        try:
            packages[name] = md.version(name)
        except md.PackageNotFoundError:
            packages[name] = None

    env: dict[str, Any] = {
        "python": sys.version,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "packages": packages,
        "env_vars": {
            key: os.environ.get(key)
            for key in ("CUBLAS_WORKSPACE_CONFIG", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }

    try:
        import torch

        env["torch"] = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": (
                torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
            ),
            "cuda_available": torch.cuda.is_available(),
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "default_dtype": str(torch.get_default_dtype()),
        }
    except ImportError:  # pragma: no cover - torch is a hard dependency
        env["torch"] = None
    return env
