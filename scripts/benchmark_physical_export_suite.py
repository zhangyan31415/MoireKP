#!/usr/bin/env python3
"""Compatibility launcher for the physical-export benchmark suite."""
from pathlib import Path
import runpy

globals().update(runpy.run_path(str(Path(__file__).resolve().parent / 'benchmarks' / 'benchmark_physical_export_suite.py'), run_name=__name__))
