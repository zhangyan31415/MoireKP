"""Small formatting helper for TAPW command-line status output."""

from __future__ import annotations

import pprint
from typing import Iterable

import numpy as np


_SECTION_RULE = "=" * 72
_SUBSECTION_RULE = "-" * 72


class TapwReporter:
    """Emit TAPW status lines through a logger or stdout with one style."""

    def __init__(self, logger=None, *, verbose: bool = False):
        self.logger = logger
        self.verbose = bool(verbose)
        self._has_output = False

    def line(self, message: str = "") -> None:
        text = str(message)
        if self.logger is not None:
            self.logger.info(text)
        else:
            print(text)
        self._has_output = True

    def section(self, title: str) -> None:
        if self._has_output:
            self.line("")
        self.line(_SECTION_RULE)
        self.line(f"[TAPW] {title}")
        self.line(_SUBSECTION_RULE)

    def stage(self, title: str, description: str | None = None) -> None:
        self.section(title)
        if description:
            self.line(f"  {description}")

    def step(self, label: str, summary: str) -> None:
        self.line(f"  - {label}: {summary}")

    def kv(self, key: str, value) -> None:
        lines = self._format_value_lines(value)
        force_block = isinstance(value, np.ndarray) and value.ndim > 0
        if len(lines) == 1 and not force_block:
            self.line(f"  {key}: {lines[0]}")
            return
        self.line(f"  {key}:")
        for line in lines:
            self.line(f"    {line}")

    def array(self, label: str, value, *, precision: int = 6) -> None:
        self.line(f"  {label}:")
        arr = np.asarray(value)
        rendered = np.array2string(arr, precision=precision, suppress_small=False)
        for line in rendered.splitlines():
            self.line(f"    {line}")

    def check(self, label: str, *, passed: bool, detail: str = "") -> None:
        status = "PASS" if passed else "WARN"
        suffix = f" ({detail})" if detail else ""
        self.line(f"  {label}: {status}{suffix}")

    def detail(self, label: str, value=None, *, purpose: str | None = None) -> None:
        if not self.verbose:
            return
        self.line(f"  {label}:")
        if purpose:
            self.line(f"    Purpose: {purpose}")
        if value is None:
            return
        for line in self._format_value_lines(value):
            self.line(f"    {line}")

    def summary_block(self, title: str, lines: Iterable[str]) -> None:
        self.section(title)
        for line in lines:
            self.line(f"  {line}")

    def _format_value_lines(self, value) -> list[str]:
        if isinstance(value, float):
            return [f"{value:.6f}"]
        if isinstance(value, np.ndarray):
            return np.array2string(value, precision=6, suppress_small=False).splitlines()
        if isinstance(value, (dict, list, tuple)):
            return pprint.pformat(value, compact=True, sort_dicts=False).splitlines()
        text = str(value)
        return text.splitlines() or [""]


def ensure_reporter(reporter=None) -> TapwReporter:
    if isinstance(reporter, TapwReporter):
        return reporter
    return TapwReporter(reporter)
