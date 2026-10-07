#!/usr/bin/env python3
"""Compatibility launcher for the archived OpenMX symmetry helper."""
from pathlib import Path
import runpy

globals().update(runpy.run_path(str(Path(__file__).resolve().parent / 'legacy' / 'openmx' / 'openmx_symm_hs_python.py'), run_name=__name__))
