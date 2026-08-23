# DISCERN FORK LOCAL ADDITION -- not part of upstream SAM3 (see Discern ADR 0002).
"""Test bootstrap for the fork's own tests.

``sam3/__init__.py`` imports the model, and therefore torch. The Zoom Anchor
resolver deliberately imports nothing heavy, and its tests must run with nothing
installed but pytest -- no GPU, no model, no network.

Registering ``sam3`` as a bare namespace package pointed at the source tree lets
those tests import ``sam3.zoom_anchor`` by its real name without executing that
``__init__``. If the resolver ever grows a heavy import, these tests stop
collecting, which is the point.
"""

import sys
import types
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

if "sam3" not in sys.modules:
    _package = types.ModuleType("sam3")
    _package.__path__ = [str(_ROOT / "sam3")]
    sys.modules["sam3"] = _package
