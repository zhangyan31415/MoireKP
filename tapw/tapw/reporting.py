"""Small formatting helper for TAPW command-line status output."""

from __future__ import annotations

import os
import pprint
import sys
from pathlib import Path
from typing import Iterable, TextIO

import numpy as np


class TapwReporter:
    """Emit TAPW status lines through a logger or stdout with one style."""

    def __init__(
        self,
        logger=None,
        *,
        command: str = "tapw",
        stream: TextIO | None = None,
        color: bool | None = None,
        verbose: bool = False,
    ):
        self.logger = logger
        self.command = str(command).strip()
        self.stream = sys.stdout if stream is None else stream
        self.verbose = bool(verbose)
        tty = bool(getattr(self.stream, "isatty", lambda: False)())
        requested = tty if color is None else bool(color)
        self.use_color = self.logger is None and requested and os.environ.get("NO_COLOR") is None
        self._has_output = False
        self._last_was_blank = False

    def _style(self, text, code: str) -> str:
        rendered = str(text)
        if not self.use_color:
            return rendered
        return f"\033[{code}m{rendered}\033[0m"

    def _prefix(self) -> str:
        return self._style(f"[{self.command}]", "36")

    def line(self, message: str = "") -> None:
        text = str(message)
        if self.logger is not None:
            self.logger.info(text)
        else:
            print(text, file=self.stream, flush=True)
        self._has_output = True
        self._last_was_blank = text == ""

    def _gap(self) -> None:
        if self._has_output and not self._last_was_blank:
            self.line()

    def title(self, text: str) -> None:
        self.line(f"{self._prefix()} {self._style(text, '1')}")

    def section(self, title: str) -> None:
        self._gap()
        self.line(f"{self._prefix()} {self._style(title, '1')}")

    def stage(
        self,
        title: str,
        description: str | None = None,
        *,
        current: int | None = None,
        total: int | None = None,
    ) -> None:
        self._gap()
        if current is None and total is None:
            heading = self._style(title, "1")
        elif current is not None and total is not None:
            count = self._style(f"[{int(current)}/{int(total)}]", "1")
            heading = f"{count} {self._style(title, '1')}"
        else:
            raise ValueError("stage requires both current and total when numbering progress")
        self.line(f"{self._prefix()} {heading}")
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

    def fields(self, rows: Iterable[tuple[str, object]]) -> None:
        entries = [(str(label), value) for label, value in rows]
        if not entries:
            return
        width = max(len(label) for label, _ in entries)
        for label, value in entries:
            self._field(label, value, width=width)

    def _field(
        self,
        label: str,
        value,
        *,
        width: int | None = None,
        value_style: str | None = None,
    ) -> None:
        label_width = len(label) if width is None else int(width)
        lines = str(value).splitlines() or [""]
        rendered = self._style(lines[0], value_style) if value_style else lines[0]
        self.line(f"  {label:<{label_width}}  {rendered}")
        continuation = " " * (label_width + 4)
        for line in lines[1:]:
            rendered = self._style(line, value_style) if value_style else line
            self.line(f"{continuation}{rendered}")

    def path(self, label: str, value: str | Path) -> None:
        self._field(str(label), value, value_style="35")

    def array(self, label: str, value, *, precision: int = 6) -> None:
        self.line(f"  {label}:")
        arr = np.asarray(value)
        rendered = np.array2string(arr, precision=precision, suppress_small=False)
        for line in rendered.splitlines():
            self.line(f"    {line}")

    def check(self, label: str, *, passed: bool, detail: str = "") -> None:
        status = "OK" if passed else "WARN"
        code = "32" if passed else "33"
        suffix = f"  {detail}" if detail else ""
        self.line(f"  {label}  {self._style(status, code)}{suffix}")

    def warning(self, text: str) -> None:
        self.line(f"  {self._style('WARN', '33')}  {text}")

    def complete(self, *, elapsed: float | None = None) -> None:
        self._gap()
        timing = "" if elapsed is None else f" in {float(elapsed):.2f} s"
        self.line(
            f"{self._prefix()} {self._style('OK', '32')}  "
            f"{self._style(f'Completed{timing}', '1')}"
        )

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
