# Apache-2.0
"""The platform's modules live beside it, not on the install path."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
