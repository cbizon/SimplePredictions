#!/usr/bin/env python3
"""Pytest configuration for local package imports."""

import sys
from pathlib import Path


# Ensure tests can import the repo's `src` package during collection.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
