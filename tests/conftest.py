# Copyright 2026. Tesseract Hackathon submission.
"""Ensures the repo root is importable as `policy.*` / `baselines.*` /
`tesseracts.*` regardless of how pytest is invoked (bare `pytest` inserts
this file's own directory onto sys.path, not the repo root).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
