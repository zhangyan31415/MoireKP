"""Formatting helpers for release-facing KP command output."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence, TextIO


class KpReporter:
    """Render concise scientific CLI output without owning workflow state."""

    def __init__(
        self,
        command: str,
        *,
        stream: TextIO | None = None,
        color: bool | None = None,
    ) -> None:
        self.command = str(command).strip()
        self.stream = sys.stdout if stream is None else stream
        tty = bool(getattr(self.stream, "isatty", lambda: False)())
        requested = tty if color is None else bool(color)
        self.use_color = requested and os.environ.get("NO_COLOR") is None
        self._has_output = False
        self._last_was_blank = False

    def _style(self, text: object, code: str) -> str:
        rendered = str(text)
        if not self.use_color:
            return rendered
        return f"\033[{code}m{rendered}\033[0m"

    def _prefix(self) -> str:
        return self._style(f"[{self.command}]", "36")

    def line(self, message: object = "") -> None:
        text = str(message)
        print(text, file=self.stream, flush=True)
        self._has_output = True
        self._last_was_blank = text == ""

    def _gap(self) -> None:
        if self._has_output and not self._last_was_blank:
            self.line()

    def title(self, text: str) -> None:
        self.line(f"{self._prefix()} {self._style(text, '1')}")

    def section(self, text: str) -> None:
        self._gap()
        self.line(f"{self._prefix()} {self._style(text, '1')}")

    def status(self, text: str, status: str) -> None:
        """Render a section-like heading with a colored terminal status."""
        self._gap()
        normalized = str(status).strip().upper()
        code = "32" if normalized in {"OK", "PASS"} else "33"
        self.line(
            f"{self._prefix()} {self._style(text, '1')}  "
            f"{self._style(normalized, code)}"
        )

    def stage(self, current: int, total: int, text: str) -> None:
        self._gap()
        label = self._style(f"[{int(current)}/{int(total)}]", "1")
        self.line(f"{self._prefix()} {label} {self._style(text, '1')}")

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
        value: object,
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

    def path(self, label: str, value: str | Path, *, width: int | None = None) -> None:
        self._field(str(label), value, width=width, value_style="2")

    def field(self, label: str, value: object, *, width: int | None = None) -> None:
        self._field(str(label), value, width=width)

    def metric(self, label: str, value: object) -> None:
        self._field(str(label), value, value_style="33")

    def table(
        self,
        headers: Sequence[object],
        rows: Iterable[Sequence[object]],
        *,
        right_align: Iterable[int] = (),
        column_styles: Mapping[int, str] | None = None,
    ) -> None:
        """Render a compact table whose alignment is independent of ANSI codes."""
        header_cells = tuple(str(value) for value in headers)
        body = tuple(tuple(str(value) for value in row) for row in rows)
        if not header_cells:
            return
        if any(len(row) != len(header_cells) for row in body):
            raise ValueError("table rows must have the same column count as headers")
        widths = [len(cell) for cell in header_cells]
        for row in body:
            for index, cell in enumerate(row):
                widths[index] = max(widths[index], len(cell))
        right = {int(index) for index in right_align}
        styles = dict(column_styles or {})

        def render(row: Sequence[str], *, header: bool = False) -> str:
            cells: list[str] = []
            for index, value in enumerate(row):
                if not header and index in styles:
                    styled = self._style(value, styles[index])
                    if index in right:
                        padded = " " * (widths[index] - len(value)) + styled
                    elif index == len(row) - 1:
                        padded = styled
                    else:
                        padded = styled + " " * (widths[index] - len(value))
                elif index in right:
                    padded = value.rjust(widths[index])
                elif index == len(row) - 1:
                    padded = value
                else:
                    padded = value.ljust(widths[index])
                cells.append(padded)
            return "  " + "  ".join(cells)

        self.line(render(header_cells, header=True))
        for row in body:
            self.line(render(row))

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
