#!/usr/bin/env python3
"""Thin entry point. All logic lives in `bruxism.cli.plot_preprocessing`.

Equivalent invocations:

    python scripts/data/plot_preprocessing.py --help
    python -m bruxism.cli.plot_preprocessing --help
    bruxism-... --help          # console script, after `pip install -e .`
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from a source checkout without installing the package.
_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from bruxism.cli.plot_preprocessing import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
