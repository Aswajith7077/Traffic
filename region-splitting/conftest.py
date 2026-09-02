"""Pytest configuration for the region-splitting module.

Ensures the module root is importable so tests can use the same relative
imports as ``main.py`` (``from services import ...``, ``from schema import ...``).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
