#!/usr/bin/env python3
"""Discover and convert ABACUS real-space H/S CSR matrix sets.

GPL-3.0-only. Modified for MoireKP on 2026-09-23: preserve independent sparse
translation supports and full floating-point output precision (see NOTICE.md).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import (
    BinaryIO,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    TextIO,
    Tuple,
    Union,
)


RY_TO_EV = 13.605693122994
DEFAULT_H_CUTOFF_EV = 1.0e-7
DEFAULT_S_CUTOFF = 1.0e-7


class CSRFormatError(ValueError):
    """Raised when an ABACUS CSR file is incomplete or structurally invalid."""


@dataclass(frozen=True)
class CSRBlock:
    """One sparse matrix block for a Bravais lattice vector."""

    r: Tuple[int, int, int]
    values: Tuple[complex, ...]
    column_indices: Tuple[int, ...]
    row_pointers: Tuple[int, ...]
    complex_values: bool


@dataclass(frozen=True)
class CSRSection:
    """Immutable descriptor for one ionic section in an ABACUS CSR file."""

    source: Path
    format: str
    step: int
    matrix: str
    dimension: int
    r_count: int
    nspin: Optional[int]
    spin_index: Optional[int]
    blocks_offset: int
    end_offset: int

    def iter_blocks(self) -> Iterator[CSRBlock]:
        """Stream validated blocks, retaining at most the current CSR block."""

        return _iter_section_blocks(self)


@dataclass(frozen=True)
class ConversionStats:
    """Summary of one completed H/S matrix-set conversion."""

    source_dimension: int
    output_dimension: int
    r_count: int
    h_written: int
    s_written: int
    layout: str
    cleanup_warnings: Tuple[str, ...] = ()


@dataclass(frozen=True)
class MatrixSelection:
    """One strictly matched set of standard ABACUS H/S outputs."""

    input_path: Path
    case_dir: Optional[Path]
    input_file: Optional[Path]
    out_dir: Path
    family: str
    nspin: int
    step: int
    h_sections: Tuple[CSRSection, ...]
    s_section: CSRSection
    source_files: Tuple[Path, ...]
    inference: str


@dataclass(frozen=True)
class _SectionMarker:
    format: str
    start_offset: int
    end_offset: int
    step_hint: int


_CURRENT_STEP_RE = re.compile(r"^\s*---\s*Ionic\s+Step\s+([+-]?\d+)\s*---\s*$")
_LEGACY_STEP_RE = re.compile(r"^\s*STEP\s*:\s*([+-]?\d+)\s*$", re.IGNORECASE)
_DIMENSION_RE = re.compile(
    r"^\s*Matrix\s+Dimension\s+of\s+([HS])\(R\)\s*:\s*([+-]?\d+)\s*$",
    re.IGNORECASE,
)
_R_COUNT_RE = re.compile(
    r"^\s*Matrix\s+number\s+of\s+([HS])\(R\)\s*:\s*([+-]?\d+)\s*$",
    re.IGNORECASE,
)
_COMPLEX_RE = re.compile(r"^\(([^,()]+),([^,()]+)\)$")
_CURRENT_MATRIX_NAME_RE = re.compile(
    r"^(hrs([12])|srs1)(g([0-9]+))?_nao[.]csr$"
)
_LEGACY_H1_NAME = "data-HR-sparse_SPIN0.csr"
_LEGACY_H2_NAME = "data-HR-sparse_SPIN1.csr"
_LEGACY_S_NAME = "data-SR-sparse_SPIN0.csr"


def _error(source: Union[str, Path], context: str, message: str) -> CSRFormatError:
    return CSRFormatError("{}: {}: {}".format(source, context, message))


def parse_scalar(
    token: str,
    *,
    source: Union[str, Path] = "<scalar>",
    context: str = "value",
) -> complex:
    """Parse one finite ABACUS real or ``(real,imag)`` scalar."""

    match = _COMPLEX_RE.fullmatch(token)
    try:
        if match is None:
            if any(character in token for character in "(),"):
                raise ValueError
            value = complex(float(token), 0.0)
        else:
            value = complex(float(match.group(1)), float(match.group(2)))
    except ValueError:
        raise _error(source, context, "malformed value token {!r}".format(token))

    if not (math.isfinite(value.real) and math.isfinite(value.imag)):
        raise _error(source, context, "nonfinite value token {!r}".format(token))
    return value


class _LineReader:
    _CHUNK_SIZE = 64 * 1024

    def __init__(
        self,
        handle: BinaryIO,
        end_offset: int,
        source: Path,
        context: str = "CSR file",
    ):
        self.handle = handle
        self.end_offset = end_offset
        self.source = source
        self.context = context
        self._buffer = bytearray()
        self._buffer_offset = handle.tell()

    def tell(self) -> int:
        return self._buffer_offset

    def _fill(self) -> bool:
        remaining = self.end_offset - self.handle.tell()
        if remaining <= 0:
            return False
        chunk = bytearray(min(self._CHUNK_SIZE, remaining))
        count = self.handle.readinto(chunk)
        if not count:
            return False
        self._buffer.extend(memoryview(chunk)[:count])
        return True

    def _decode(self, raw_line: bytes, offset: int) -> str:
        try:
            return raw_line.decode("utf-8")
        except UnicodeDecodeError as error:
            raise _error(
                self.source,
                "{} at byte offset {}".format(self.context, offset + error.start),
                "invalid UTF-8",
            )

    def readline(self) -> Optional[Tuple[int, str]]:
        while True:
            lf_index = self._buffer.find(b"\n")
            cr_index = self._buffer.find(b"\r")
            separator_indices = [
                index for index in (lf_index, cr_index) if index >= 0
            ]
            if separator_indices:
                separator_index = min(separator_indices)
                if (
                    self._buffer[separator_index] == 13
                    and separator_index + 1 == len(self._buffer)
                    and self._fill()
                ):
                    continue

                separator_size = 1
                if (
                    self._buffer[separator_index] == 13
                    and separator_index + 1 < len(self._buffer)
                    and self._buffer[separator_index + 1] == 10
                ):
                    separator_size = 2
                offset = self._buffer_offset
                raw_line = bytes(self._buffer[:separator_index])
                consumed = separator_index + separator_size
                del self._buffer[:consumed]
                self._buffer_offset += consumed
                return offset, self._decode(raw_line, offset)

            if self._fill():
                continue
            if not self._buffer:
                return None
            offset = self._buffer_offset
            raw_line = bytes(self._buffer)
            self._buffer_offset += len(self._buffer)
            self._buffer.clear()
            return offset, self._decode(raw_line, offset)


def _scan_section_markers(source: Path) -> Tuple[_SectionMarker, ...]:
    starts: List[Tuple[str, int, int]] = []
    pending_legacy_step = False
    end_offset = source.stat().st_size
    with source.open("rb") as handle:
        reader = _LineReader(handle, end_offset, source, "section scan")
        while True:
            item = reader.readline()
            if item is None:
                break
            offset, line = item

            current_match = _CURRENT_STEP_RE.fullmatch(line)
            if current_match is not None:
                starts.append(("current", offset, int(current_match.group(1))))
                pending_legacy_step = False
                continue

            legacy_match = _LEGACY_STEP_RE.fullmatch(line)
            if legacy_match is not None:
                starts.append(("legacy", offset, int(legacy_match.group(1))))
                pending_legacy_step = True
                continue

            dimension_match = _DIMENSION_RE.fullmatch(line)
            if dimension_match is not None:
                if pending_legacy_step:
                    pending_legacy_step = False
                else:
                    starts.append(("legacy", offset, 0))

    markers = []
    for index, (section_format, start_offset, step_hint) in enumerate(starts):
        next_offset = starts[index + 1][1] if index + 1 < len(starts) else end_offset
        markers.append(
            _SectionMarker(
                format=section_format,
                start_offset=start_offset,
                end_offset=next_offset,
                step_hint=step_hint,
            )
        )
    return tuple(markers)


def _parse_integer(token: str, source: Path, context: str) -> int:
    if re.fullmatch(r"[+-]?\d+", token) is None:
        raise _error(source, context, "expected an integer, got {!r}".format(token))
    return int(token)


def _next_data_line(reader: _LineReader) -> Optional[Tuple[int, str]]:
    while True:
        item = reader.readline()
        if item is None:
            return None
        offset, line = item
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return offset, stripped


def _parse_legacy_header(source: Path, marker: _SectionMarker) -> CSRSection:
    with source.open("rb") as handle:
        handle.seek(marker.start_offset)
        reader = _LineReader(
            handle, marker.end_offset, source, "legacy header"
        )
        first = _next_data_line(reader)
        if first is None:
            raise _error(source, "legacy section", "truncated header")

        first_line = first[1]
        step_match = _LEGACY_STEP_RE.fullmatch(first_line)
        if step_match is not None:
            step = int(step_match.group(1))
            dimension_item = _next_data_line(reader)
            if dimension_item is None:
                raise _error(source, "legacy step {}".format(step), "truncated header")
            dimension_line = dimension_item[1]
        else:
            step = 0
            dimension_line = first_line

        context = "legacy step {}".format(step)
        dimension_match = _DIMENSION_RE.fullmatch(dimension_line)
        if dimension_match is None:
            raise _error(source, context, "missing Matrix Dimension header")
        matrix = dimension_match.group(1).upper()
        dimension = int(dimension_match.group(2))

        count_item = _next_data_line(reader)
        if count_item is None:
            raise _error(source, context, "truncated Matrix number header")
        count_match = _R_COUNT_RE.fullmatch(count_item[1])
        if count_match is None or count_match.group(1).upper() != matrix:
            raise _error(source, context, "missing matching Matrix number header")
        r_count = int(count_match.group(2))

        if dimension <= 0:
            raise _error(source, context, "matrix dimension must be positive")
        if r_count < 0:
            raise _error(source, context, "R block count must be non-negative")

        return CSRSection(
            source=source,
            format="legacy",
            step=step,
            matrix=matrix,
            dimension=dimension,
            r_count=r_count,
            nspin=None,
            spin_index=None,
            blocks_offset=reader.tell(),
            end_offset=marker.end_offset,
        )


def _metadata_integer(line: str, label: str) -> Optional[int]:
    match = re.fullmatch(
        r"\s*([+-]?\d+)\s*#\s*{}\s*".format(re.escape(label)),
        line,
        re.IGNORECASE,
    )
    return None if match is None else int(match.group(1))


def _merge_metadata(
    source: Path,
    context: str,
    label: str,
    current: Optional[int],
    parsed: Optional[int],
) -> Optional[int]:
    if parsed is None:
        return current
    if current is not None and parsed != current:
        raise _error(
            source,
            context,
            "conflicting duplicate {}: {} then {}".format(label, current, parsed),
        )
    return parsed


def _parse_current_header(source: Path, marker: _SectionMarker) -> CSRSection:
    with source.open("rb") as handle:
        handle.seek(marker.start_offset)
        reader = _LineReader(
            handle,
            marker.end_offset,
            source,
            "ionic step {} header".format(marker.step_hint),
        )
        marker_item = reader.readline()
        if marker_item is None:
            raise _error(source, "ionic step {}".format(marker.step_hint), "truncated header")
        marker_match = _CURRENT_STEP_RE.fullmatch(marker_item[1])
        if marker_match is None:
            raise _error(source, "current section", "missing Ionic Step header")
        step = int(marker_match.group(1))
        context = "ionic step {}".format(step)

        matrix: Optional[str] = None
        nspin: Optional[int] = None
        spin_index: Optional[int] = None
        dimension: Optional[int] = None
        r_count: Optional[int] = None
        saw_csr_banner = False
        blocks_offset = marker.end_offset

        while True:
            item = reader.readline()
            if item is None:
                break
            offset, line = item
            stripped = line.strip()

            if saw_csr_banner and stripped and not stripped.startswith("#"):
                blocks_offset = offset
                break

            if stripped.startswith("#") and "CSR Format" in stripped:
                saw_csr_banner = True
                continue

            matrix_match = re.search(r"\b([HS])\(R\)", stripped, re.IGNORECASE)
            if stripped.startswith("#") and "matrix in real space" in stripped and matrix_match:
                matrix = matrix_match.group(1).upper()

            nspin = _merge_metadata(
                source,
                context,
                "number of spin directions",
                nspin,
                _metadata_integer(stripped, "number of spin directions"),
            )
            spin_index = _merge_metadata(
                source,
                context,
                "spin index",
                spin_index,
                _metadata_integer(stripped, "spin index"),
            )
            dimension = _merge_metadata(
                source,
                context,
                "number of localized basis",
                dimension,
                _metadata_integer(stripped, "number of localized basis"),
            )
            r_count = _merge_metadata(
                source,
                context,
                "number of Bravais lattice vector R",
                r_count,
                _metadata_integer(stripped, "number of Bravais lattice vector R"),
            )

        missing = [
            label
            for label, value in (
                ("matrix comment", matrix),
                ("number of spin directions", nspin),
                ("spin index", spin_index),
                ("number of localized basis", dimension),
                ("number of Bravais lattice vector R", r_count),
                ("CSR Format banner", saw_csr_banner),
            )
            if value is None or value is False
        ]
        if missing:
            raise _error(source, context, "truncated or missing {}".format(", ".join(missing)))
        assert matrix is not None
        assert nspin is not None
        assert spin_index is not None
        assert dimension is not None
        assert r_count is not None

        if dimension <= 0:
            raise _error(source, context, "matrix dimension must be positive")
        if r_count < 0:
            raise _error(source, context, "R block count must be non-negative")
        if nspin not in (1, 2, 4):
            raise _error(source, context, "number of spin directions must be 1, 2, or 4")
        if not 1 <= spin_index <= nspin:
            raise _error(
                source,
                context,
                "spin index must be between 1 and the number of spin directions ({})".format(
                    nspin
                ),
            )

        return CSRSection(
            source=source,
            format="current",
            step=step,
            matrix=matrix,
            dimension=dimension,
            r_count=r_count,
            nspin=nspin,
            spin_index=spin_index,
            blocks_offset=blocks_offset,
            end_offset=marker.end_offset,
        )


def _parse_section_header(source: Path, marker: _SectionMarker) -> CSRSection:
    if marker.format == "legacy":
        return _parse_legacy_header(source, marker)
    return _parse_current_header(source, marker)


def _section_context(section: CSRSection) -> str:
    if section.format == "current":
        return "ionic step {}".format(section.step)
    return "legacy step {}".format(section.step)


def _parse_r_header(section: CSRSection, line: str, block_number: int) -> Tuple[Tuple[int, int, int], int]:
    context = "{}, R block {}".format(_section_context(section), block_number)
    tokens = line.split()
    if len(tokens) != 4:
        raise _error(section.source, context, "expected Rx Ry Rz NNZ")
    integers = tuple(_parse_integer(token, section.source, context) for token in tokens)
    r = (integers[0], integers[1], integers[2])
    nnz = integers[3]
    if nnz < 0:
        raise _error(section.source, "{}, R={}".format(context, r), "negative NNZ")
    return r, nnz


def _legacy_array_line(
    reader: _LineReader,
    section: CSRSection,
    r: Tuple[int, int, int],
    label: str,
) -> Sequence[str]:
    item = reader.readline()
    if item is None:
        raise _error(
            section.source,
            "{}, R={}".format(_section_context(section), r),
            "truncated {} array".format(label),
        )
    return item[1].split()


def _seek_array_label(
    reader: _LineReader,
    section: CSRSection,
    r: Tuple[int, int, int],
    label: str,
) -> None:
    while True:
        item = reader.readline()
        if item is None:
            raise _error(
                section.source,
                "{}, R={}".format(_section_context(section), r),
                "truncated before {} array".format(label),
            )
        stripped = item[1].strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            if label.lower() in stripped.lower():
                return
            continue
        raise _error(
            section.source,
            "{}, R={}".format(_section_context(section), r),
            "expected {} banner".format(label),
        )


def _collect_until_label(
    reader: _LineReader,
    section: CSRSection,
    r: Tuple[int, int, int],
    label: str,
    next_label: str,
    expected: int,
) -> Tuple[str, ...]:
    tokens: List[str] = []
    while True:
        item = reader.readline()
        if item is None:
            raise _error(
                section.source,
                "{}, R={}".format(_section_context(section), r),
                "truncated {} array; {} length {}, expected {}".format(
                    label, label, len(tokens), expected
                ),
            )
        stripped = item[1].strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            if next_label.lower() in stripped.lower():
                if len(tokens) != expected:
                    raise _error(
                        section.source,
                        "{}, R={}".format(_section_context(section), r),
                        "{} length {}, expected {}".format(label, len(tokens), expected),
                    )
                return tuple(tokens)
            continue

        line_tokens = stripped.split()
        if len(tokens) + len(line_tokens) > expected:
            raise _error(
                section.source,
                "{}, R={}".format(_section_context(section), r),
                "extra {} tokens: expected {}".format(label, expected),
            )
        tokens.extend(line_tokens)


def _collect_exact_tokens(
    reader: _LineReader,
    section: CSRSection,
    r: Tuple[int, int, int],
    label: str,
    expected: int,
) -> Tuple[str, ...]:
    tokens: List[str] = []
    while len(tokens) < expected:
        item = reader.readline()
        if item is None:
            raise _error(
                section.source,
                "{}, R={}".format(_section_context(section), r),
                "truncated {} array; {} length {}, expected {}".format(
                    label, label, len(tokens), expected
                ),
            )
        stripped = item[1].strip()
        if not stripped or stripped.startswith("#"):
            continue
        line_tokens = stripped.split()
        if len(tokens) + len(line_tokens) > expected:
            raise _error(
                section.source,
                "{}, R={}".format(_section_context(section), r),
                "extra {} tokens: expected {}".format(label, expected),
            )
        tokens.extend(line_tokens)
    return tuple(tokens)


def _validate_block(
    section: CSRSection,
    r: Tuple[int, int, int],
    nnz: int,
    value_tokens: Sequence[str],
    column_tokens: Sequence[str],
    row_pointer_tokens: Sequence[str],
) -> CSRBlock:
    context = "{}, R={}".format(_section_context(section), r)
    if len(value_tokens) != nnz:
        raise _error(
            section.source, context, "values length {}, expected {}".format(len(value_tokens), nnz)
        )
    if len(column_tokens) != nnz:
        raise _error(
            section.source,
            context,
            "column length {}, expected {}".format(len(column_tokens), nnz),
        )
    if len(row_pointer_tokens) != section.dimension + 1:
        raise _error(
            section.source,
            context,
            "row pointer length {}, expected {}".format(
                len(row_pointer_tokens), section.dimension + 1
            ),
        )

    values = tuple(
        parse_scalar(token, source=section.source, context="{} value".format(context))
        for token in value_tokens
    )
    columns = tuple(
        _parse_integer(token, section.source, "{} column".format(context))
        for token in column_tokens
    )
    row_pointers = tuple(
        _parse_integer(token, section.source, "{} row pointer".format(context))
        for token in row_pointer_tokens
    )

    if any(column < 0 or column >= section.dimension for column in columns):
        raise _error(
            section.source,
            context,
            "column range violation; expected indices in [0, {})".format(section.dimension),
        )
    if row_pointers[0] != 0:
        raise _error(section.source, context, "row pointer start must be 0")
    if any(left > right for left, right in zip(row_pointers, row_pointers[1:])):
        raise _error(section.source, context, "row pointer monotonic violation")
    if row_pointers[-1] != nnz:
        raise _error(
            section.source,
            context,
            "row pointer end {}, expected NNZ {}".format(row_pointers[-1], nnz),
        )

    return CSRBlock(
        r=r,
        values=values,
        column_indices=columns,
        row_pointers=row_pointers,
        complex_values=any(_COMPLEX_RE.fullmatch(token) for token in value_tokens),
    )


def _validate_trailing(reader: _LineReader, section: CSRSection) -> None:
    while True:
        item = reader.readline()
        if item is None:
            return
        stripped = item[1].strip()
        if stripped and not stripped.startswith("#"):
            raise _error(
                section.source,
                _section_context(section),
                "unexpected trailing content {!r}".format(stripped),
            )


def _iter_legacy_blocks(reader: _LineReader, section: CSRSection) -> Iterator[CSRBlock]:
    for block_number in range(1, section.r_count + 1):
        header_item = _next_data_line(reader)
        if header_item is None:
            raise _error(
                section.source,
                _section_context(section),
                "R block count truncated at {}; expected {}".format(
                    block_number - 1, section.r_count
                ),
            )
        r, nnz = _parse_r_header(section, header_item[1], block_number)
        if nnz == 0:
            yield CSRBlock(
                r=r,
                values=(),
                column_indices=(),
                row_pointers=(0,) * (section.dimension + 1),
                complex_values=False,
            )
            continue
        value_tokens = _legacy_array_line(reader, section, r, "values")
        column_tokens = _legacy_array_line(reader, section, r, "column")
        row_pointer_tokens = _legacy_array_line(reader, section, r, "row pointer")
        yield _validate_block(
            section, r, nnz, value_tokens, column_tokens, row_pointer_tokens
        )
    _validate_trailing(reader, section)


def _iter_current_blocks(reader: _LineReader, section: CSRSection) -> Iterator[CSRBlock]:
    for block_number in range(1, section.r_count + 1):
        header_item = _next_data_line(reader)
        if header_item is None:
            raise _error(
                section.source,
                _section_context(section),
                "R block count truncated at {}; expected {}".format(
                    block_number - 1, section.r_count
                ),
            )
        r, nnz = _parse_r_header(section, header_item[1], block_number)
        _seek_array_label(reader, section, r, "CSR values")
        value_tokens = _collect_until_label(
            reader, section, r, "values", "CSR column indices", nnz
        )
        column_tokens = _collect_until_label(
            reader, section, r, "column", "CSR row pointers", nnz
        )
        row_pointer_tokens = _collect_exact_tokens(
            reader, section, r, "row pointer", section.dimension + 1
        )
        yield _validate_block(
            section, r, nnz, value_tokens, column_tokens, row_pointer_tokens
        )
    _validate_trailing(reader, section)


def _iter_section_blocks(section: CSRSection) -> Iterator[CSRBlock]:
    with section.source.open("rb") as handle:
        handle.seek(section.blocks_offset)
        reader = _LineReader(
            handle, section.end_offset, section.source, _section_context(section)
        )
        if section.format == "legacy":
            yield from _iter_legacy_blocks(reader, section)
        else:
            yield from _iter_current_blocks(reader, section)


def _validate_complete(section: CSRSection) -> None:
    for _block in section.iter_blocks():
        pass


def select_csr_section(
    source: Union[str, Path], step: Union[str, int] = "latest"
) -> CSRSection:
    """Select a complete CSR section by ionic step without whole-file reads.

    ``step="latest"`` selects the newest complete appended section. An explicit
    integer selects that ionic step and reports a contextual error if it is
    absent or incomplete.
    """

    source_path = Path(source).expanduser().resolve()
    if not source_path.is_file():
        raise CSRFormatError("{}: CSR source is not a regular file".format(source_path))
    if step != "latest" and not isinstance(step, int):
        raise ValueError("step must be 'latest' or an integer")

    markers = _scan_section_markers(source_path)
    if not markers:
        raise CSRFormatError(
            "{}: no legacy Matrix Dimension or current Ionic Step section found".format(
                source_path
            )
        )

    if step == "latest":
        newest_error: Optional[CSRFormatError] = None
        for marker in reversed(markers):
            try:
                section = _parse_section_header(source_path, marker)
                _validate_complete(section)
                return section
            except CSRFormatError as error:
                if newest_error is None:
                    newest_error = error
        assert newest_error is not None
        raise CSRFormatError(
            "{}: no complete CSR section; latest candidate failed: {}".format(
                source_path, newest_error
            )
        )

    candidates = [marker for marker in markers if marker.step_hint == step]
    if not candidates:
        available = sorted({marker.step_hint for marker in markers})
        raise CSRFormatError(
            "{}: step {} not found; available steps: {}".format(
                source_path, step, ", ".join(str(value) for value in available)
            )
        )

    marker = candidates[-1]
    section = _parse_section_header(source_path, marker)
    _validate_complete(section)
    return section


def _canonical_input_parameter(
    key: str, value: str, path: Path, line_number: int
) -> object:
    if key == "suffix":
        return value
    if key == "nspin":
        try:
            parsed = int(value)
        except ValueError:
            raise ValueError(
                "{}:{}: nspin must be 1, 2, or 4".format(path, line_number)
            )
        if parsed not in (1, 2, 4):
            raise ValueError(
                "{}:{}: nspin must be 1, 2, or 4".format(path, line_number)
            )
        return parsed

    normalized = value.strip().lower()
    if normalized in ("1", "true", ".true.", "yes", "on"):
        return True
    if normalized in ("0", "false", ".false.", "no", "off"):
        return False
    raise ValueError(
        "{}:{}: {} boolean value must be one of 1/0, true/false, "
        ".true./.false., yes/no, or on/off; got {!r}".format(
            path, line_number, key, value
        )
    )


def _read_input_parameters(path: Path) -> Dict[str, str]:
    parameters: Dict[str, str] = {}
    canonical_parameters: Dict[str, object] = {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                comment_offsets = [
                    offset for marker in ("#", "!") if (offset := line.find(marker)) >= 0
                ]
                if comment_offsets:
                    line = line[: min(comment_offsets)]
                tokens = line.split()
                if not tokens or tokens[0].lower() == "input_parameters":
                    continue
                key = tokens[0].lower()
                if key not in ("suffix", "nspin", "noncolin", "lspinorb"):
                    continue
                if len(tokens) < 2:
                    raise ValueError(
                        "{}:{}: {} requires a value".format(path, line_number, key)
                    )
                value = tokens[1]
                canonical = _canonical_input_parameter(
                    key, value, path, line_number
                )
                if (
                    key in canonical_parameters
                    and canonical_parameters[key] != canonical
                ):
                    raise ValueError(
                        "{}:{}: {} has conflicting duplicate values {!r} and {!r}".format(
                            path, line_number, key, parameters[key], value
                        )
                    )
                parameters[key] = value
                canonical_parameters[key] = canonical
    except UnicodeDecodeError as error:
        raise ValueError(
            "{}: invalid UTF-8 near byte {}".format(path, error.start)
        )
    return parameters


def _parse_input_bool(value: str, path: Path, key: str) -> bool:
    normalized = value.strip().lower()
    if normalized in ("1", "true", ".true.", "yes", "on"):
        return True
    if normalized in ("0", "false", ".false.", "no", "off"):
        return False
    raise ValueError(
        "{}: {} boolean value must be one of 1/0, true/false, "
        ".true./.false., yes/no, or on/off; got {!r}".format(path, key, value)
    )


def _input_nspin(parameters: Dict[str, str], path: Path) -> int:
    raw_nspin = parameters.get("nspin", "1")
    try:
        nspin = int(raw_nspin)
    except ValueError:
        raise ValueError("{}: nspin must be 1, 2, or 4".format(path))
    if nspin not in (1, 2, 4):
        raise ValueError("{}: nspin must be 1, 2, or 4".format(path))
    if _parse_input_bool(parameters.get("noncolin", "false"), path, "noncolin"):
        return 4
    if _parse_input_bool(parameters.get("lspinorb", "false"), path, "lspinorb"):
        return 4
    return nspin


def _complete_sections_by_step(path: Path) -> Dict[int, CSRSection]:
    complete: Dict[int, CSRSection] = {}
    for marker in _scan_section_markers(path):
        try:
            section = _parse_section_header(path, marker)
            _validate_complete(section)
        except CSRFormatError:
            continue
        complete[section.step] = section
    return complete


def _matching_sections(
    paths: Sequence[Path], step: Union[str, int]
) -> Optional[Tuple[CSRSection, ...]]:
    if step != "latest":
        assert isinstance(step, int)
        selected = []
        for path in paths:
            try:
                selected.append(select_csr_section(path, step=step))
            except CSRFormatError:
                markers = _scan_section_markers(path)
                if not markers or any(marker.step_hint == step for marker in markers):
                    raise
                return None
        sections = tuple(selected)
        if any(section.step != step for section in sections):
            return None
        return sections

    per_file_list = []
    for path in paths:
        complete = _complete_sections_by_step(path)
        if not complete:
            select_csr_section(path, step="latest")
            raise AssertionError("select_csr_section returned without a complete section")
        per_file_list.append(complete)
    per_file = tuple(per_file_list)
    common_steps = set(per_file[0])
    for sections in per_file[1:]:
        common_steps.intersection_update(sections)
    if not common_steps:
        return None
    selected_step = max(common_steps)
    return tuple(sections[selected_step] for sections in per_file)


def _contains_complex_tokens(sections: Sequence[CSRSection]) -> bool:
    for section in sections:
        for block in section.iter_blocks():
            if block.complex_values:
                return True
    return False


def _no_matrix_error(out_dir: Path) -> ValueError:
    expected_names = (
        "hrs1_nao.csr",
        "hrs2_nao.csr",
        "srs1_nao.csr",
        _LEGACY_H1_NAME,
        _LEGACY_H2_NAME,
        _LEGACY_S_NAME,
    )
    return ValueError(
        "{}: no standard ABACUS H/S matrix set found. "
        "Scanned expected paths: {}. "
        "Required ABACUS settings: basis_type lcao; gamma_only false; "
        "out_mat_hs2 1 12".format(
            out_dir,
            ", ".join(str((out_dir / name).resolve()) for name in expected_names),
        )
    )


def _has_standard_matrix_name(directory: Path) -> bool:
    for path in directory.iterdir():
        if not path.is_file():
            continue
        if _CURRENT_MATRIX_NAME_RE.fullmatch(path.name) is not None:
            return True
        if path.name in (_LEGACY_H1_NAME, _LEGACY_H2_NAME, _LEGACY_S_NAME):
            return True
    return False


def _resolve_input_location(
    input_path: Union[str, Path],
) -> Tuple[Path, Optional[Path], Optional[Path], Path, Dict[str, str]]:
    resolved = Path(input_path).expanduser().resolve()
    if resolved.is_file() and resolved.name == "INPUT":
        case_dir = resolved.parent
        input_file = resolved
    elif resolved.is_dir() and _has_standard_matrix_name(resolved):
        return resolved, None, None, resolved, {}
    elif resolved.is_dir() and (resolved / "INPUT").is_file():
        case_dir = resolved
        input_file = resolved / "INPUT"
    elif resolved.is_dir():
        return resolved, None, None, resolved, {}
    else:
        raise ValueError(
            "{}: expected an ABACUS case directory, INPUT file, or OUT directory".format(
                resolved
            )
        )

    parameters = _read_input_parameters(input_file)
    suffix = parameters.get("suffix", "ABACUS")
    if not suffix or "/" in suffix or suffix in (".", ".."):
        raise ValueError("{}: invalid suffix {!r}".format(input_file, suffix))
    out_dir = (case_dir / "OUT.{}".format(suffix)).resolve()
    if not out_dir.is_dir():
        raise ValueError(
            "{}: standard output directory does not exist (from suffix={!r})".format(
                out_dir, suffix
            )
        )
    return resolved, case_dir, input_file, out_dir, parameters


def _discover_current_groups(
    out_dir: Path,
) -> Dict[str, Dict[str, Path]]:
    groups: Dict[str, Dict[str, Path]] = {}
    for path in out_dir.iterdir():
        if not path.is_file():
            continue
        match = _CURRENT_MATRIX_NAME_RE.fullmatch(path.name)
        if match is None:
            continue
        stem = match.group(1)
        group = match.group(3) or ""
        role = {"hrs1": "h1", "hrs2": "h2", "srs1": "s"}[stem]
        groups.setdefault(group, {})[role] = path.resolve()
    return groups


def _group_sort_key(group: str) -> int:
    return -1 if not group else int(group[1:])


def _select_current_set(
    groups: Dict[str, Dict[str, Path]],
    nspin: int,
    step: Union[str, int],
    out_dir: Path,
) -> Tuple[Tuple[CSRSection, ...], CSRSection, Tuple[Path, ...]]:
    need_roles = ("h1", "h2", "s") if nspin == 2 else ("h1", "s")
    candidates = []
    incomplete = []
    parse_errors = []
    for group, files in groups.items():
        missing = tuple(role for role in need_roles if role not in files)
        if missing:
            incomplete.append((group, missing))
            continue
        if nspin != 2 and "h2" in files:
            raise ValueError(
                "{}: second H channel is present but effective nspin={}".format(
                    out_dir, nspin
                )
            )
        paths = tuple(files[role] for role in need_roles)
        try:
            sections = _matching_sections(paths, step)
        except CSRFormatError as error:
            parse_errors.append(error)
            continue
        if sections is None:
            continue
        candidates.append((sections[0].step, _group_sort_key(group), paths, sections))

    if not candidates:
        if parse_errors:
            raise parse_errors[-1]
        if step != "latest":
            raise ValueError(
                "{}: step {} has no complete same-group H/S matrix set".format(
                    out_dir, step
                )
            )
        details = []
        for group, missing in incomplete:
            suffix = group or "<base>"
            expected_names = {
                "h1": "hrs1{}_nao.csr".format(group),
                "h2": "hrs2{}_nao.csr".format(group),
                "s": "srs1{}_nao.csr".format(group),
            }
            details.append(
                "{} missing {}".format(
                    suffix, ", ".join(expected_names[role] for role in missing)
                )
            )
        raise ValueError(
            "{}: incomplete current ABACUS matrix set{}".format(
                out_dir, ": " + "; ".join(details) if details else ""
            )
        )

    _selected_step, _group_rank, paths, sections = max(
        candidates, key=lambda item: (item[0], item[1])
    )
    h_count = 2 if nspin == 2 else 1
    return tuple(sections[:h_count]), sections[h_count], paths


def _select_legacy_set(
    files: Dict[str, Path],
    nspin: int,
    step: Union[str, int],
    out_dir: Path,
) -> Tuple[Tuple[CSRSection, ...], CSRSection, Tuple[Path, ...]]:
    roles = ("h1", "h2", "s") if nspin == 2 else ("h1", "s")
    if nspin != 2 and "h2" in files:
        raise ValueError(
            "{}: second H channel is present but effective nspin={}".format(
                out_dir, nspin
            )
        )
    missing = tuple(role for role in roles if role not in files)
    if missing:
        names = {
            "h1": _LEGACY_H1_NAME,
            "h2": _LEGACY_H2_NAME,
            "s": _LEGACY_S_NAME,
        }
        raise ValueError(
            "{}: incomplete legacy ABACUS matrix set; missing {}".format(
                out_dir, ", ".join(names[role] for role in missing)
            )
        )
    paths = tuple(files[role] for role in roles)
    sections = _matching_sections(paths, step)
    if sections is None:
        if step == "latest":
            raise ValueError(
                "{}: legacy H/S files have no complete matching ionic step".format(
                    out_dir
                )
            )
        raise ValueError(
            "{}: step {} has no complete legacy H/S matrix set".format(out_dir, step)
        )
    h_count = 2 if nspin == 2 else 1
    return tuple(sections[:h_count]), sections[h_count], paths


def discover_matrix_set(
    input_path: Union[str, Path],
    *,
    step: Union[str, int] = "latest",
    nspin: Optional[int] = None,
) -> MatrixSelection:
    """Discover one complete standard ABACUS H/S matrix set."""

    if step != "latest" and (not isinstance(step, int) or isinstance(step, bool)):
        raise ValueError("step must be 'latest' or an integer")
    if nspin is not None and nspin not in (1, 2, 4):
        raise ValueError("nspin must be 1, 2, or 4")

    resolved, case_dir, input_file, out_dir, parameters = _resolve_input_location(
        input_path
    )
    current_groups = _discover_current_groups(out_dir)
    legacy_names = {
        _LEGACY_H1_NAME: "h1",
        _LEGACY_H2_NAME: "h2",
        _LEGACY_S_NAME: "s",
    }
    legacy_files = {
        role: (out_dir / name).resolve()
        for name, role in legacy_names.items()
        if (out_dir / name).is_file()
    }

    if current_groups and legacy_files:
        raise ValueError(
            "{}: mixed current and legacy standard matrix families".format(out_dir)
        )
    if not current_groups and not legacy_files:
        raise _no_matrix_error(out_dir)

    second_channel = (
        any("h2" in files for files in current_groups.values())
        if current_groups
        else "h2" in legacy_files
    )
    if input_file is not None:
        effective_nspin = _input_nspin(parameters, input_file)
        if nspin is not None and nspin != effective_nspin:
            raise ValueError(
                "{}: --nspin={} conflicts with INPUT effective nspin={}".format(
                    input_file, nspin, effective_nspin
                )
            )
        selected_nspin = effective_nspin
        inference = "input-metadata"
    elif nspin is not None:
        selected_nspin = nspin
        inference = "cli-override"
    elif second_channel:
        selected_nspin = 2
        inference = "second-H-channel"
    else:
        selected_nspin = 1
        inference = "pending-value-inspection"

    if current_groups:
        h_sections, s_section, source_files = _select_current_set(
            current_groups, selected_nspin, step, out_dir
        )
        family = "current"
    else:
        h_sections, s_section, source_files = _select_legacy_set(
            legacy_files, selected_nspin, step, out_dir
        )
        family = "legacy"

    if inference == "pending-value-inspection":
        if _contains_complex_tokens(h_sections + (s_section,)):
            selected_nspin = 4
            inference = "complex-value-token"
        else:
            inference = "real-values-default"

    _validate_matrix_set(h_sections, s_section, selected_nspin)

    return MatrixSelection(
        input_path=resolved,
        case_dir=case_dir,
        input_file=input_file,
        out_dir=out_dir,
        family=family,
        nspin=selected_nspin,
        step=h_sections[0].step,
        h_sections=h_sections,
        s_section=s_section,
        source_files=source_files,
        inference=inference,
    )


def _validate_cutoff(value: float, label: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        raise ValueError("{} must be a finite non-negative number".format(label))
    if not math.isfinite(converted) or converted < 0.0:
        raise ValueError("{} must be a finite non-negative number".format(label))
    return converted


def _write_sparse_record(
    handle: TextIO,
    r: Tuple[int, int, int],
    row: int,
    column: int,
    value: complex,
) -> None:
    handle.write(
        "{:5d}  {:5d}  {:5d}  {:5d}  {:5d}  {:.17g}  {:.17g}\n".format(
            r[0], r[1], r[2], row + 1, column + 1, value.real, value.imag
        )
    )


def _grouped_spinor_index(index: int, dimension: int) -> int:
    orbital, spin = divmod(index, 2)
    return orbital + spin * (dimension // 2)


def _write_block(
    handle: TextIO,
    block: CSRBlock,
    cutoff: float,
    scale: float,
    row_offset: int = 0,
    column_offset: int = 0,
    index_map: Optional[Callable[[int], int]] = None,
) -> int:
    written = 0
    for row in range(len(block.row_pointers) - 1):
        for index in range(block.row_pointers[row], block.row_pointers[row + 1]):
            value = block.values[index] * scale
            if not (math.isfinite(value.real) and math.isfinite(value.imag)):
                raise ValueError(
                    "nonfinite converted matrix value at R={}, row={}, column={}".format(
                        block.r, row, block.column_indices[index]
                    )
                )
            if abs(value) > cutoff:
                output_row = index_map(row) if index_map is not None else row
                source_column = block.column_indices[index]
                output_column = (
                    index_map(source_column)
                    if index_map is not None
                    else source_column
                )
                _write_sparse_record(
                    handle,
                    block.r,
                    output_row + row_offset,
                    output_column + column_offset,
                    value,
                )
                written += 1
    return written


def _write_final_sparse_file(
    path: Path,
    body_path: Path,
    matrix: str,
    count: int,
    dimension: int,
    r_count: int,
    noncollinear: bool,
) -> Path:
    temporary = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=".{}.tmp-".format(path.name),
        dir=str(path.parent),
        delete=False,
    )
    temporary_path = Path(temporary.name)
    try:
        suffix = (
            " with non-collinear calculation"
            if noncollinear
            else ""
        )
        temporary.write(" ! Sparse format of {}{} \n".format(matrix, suffix))
        temporary.write(
            " {} ! Number of non-zeros lines of {}mnR\n".format(
                count, matrix
            )
        )
        temporary.write("  {} ! Number of orbitals \n".format(dimension))
        temporary.write("  {} ! Number of R points \n".format(r_count))
        with body_path.open("r", encoding="utf-8", newline="") as body:
            shutil.copyfileobj(body, temporary)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary.close()
        return temporary_path
    except BaseException:
        temporary.close()
        temporary_path.unlink(missing_ok=True)
        raise


def _backup_target(target: Path) -> Path:
    temporary = tempfile.NamedTemporaryFile(
        mode="w+b",
        prefix=".{}.backup-".format(target.name),
        dir=str(target.parent),
        delete=False,
    )
    backup_path = Path(temporary.name)
    try:
        with target.open("rb") as source:
            shutil.copyfileobj(source, temporary)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary.close()
        return backup_path
    except BaseException:
        temporary.close()
        backup_path.unlink(missing_ok=True)
        raise


def _annotate_matrix_rollback_error(
    original_error: BaseException, diagnostics: Sequence[str]
) -> None:
    diagnostic_tuple = tuple(diagnostics)
    diagnostic_message = "matrix-set rollback/cleanup errors: {}".format(
        "; ".join(diagnostic_tuple)
    )
    try:
        original_error.matrix_rollback_errors = diagnostic_tuple
    except BaseException:
        pass
    if isinstance(original_error, OSError) and original_error.strerror is not None:
        original_error.strerror = "{}; {}".format(
            original_error.strerror, diagnostic_message
        )
    else:
        original_error.args = tuple(original_error.args) + (
            diagnostic_message,
        )
    if hasattr(original_error, "add_note"):
        original_error.add_note(diagnostic_message)


def _commit_matrix_files(
    staged_targets: Sequence[Tuple[Path, Path]],
) -> Tuple[str, ...]:
    """Publish all staged files, restoring the original matrix set on failure."""

    backups = {}
    installed_targets: List[Path] = []
    try:
        for _staged, target in staged_targets:
            if target.exists():
                backups[target] = _backup_target(target)

        for staged, target in staged_targets:
            os.replace(staged, target)
            installed_targets.append(target)
    except BaseException as original_error:
        rollback_errors = []
        retained_backups = set()
        for target in reversed(installed_targets):
            backup = backups.get(target)
            try:
                if backup is None:
                    target.unlink(missing_ok=True)
                else:
                    os.replace(backup, target)
                    backups.pop(target)
            except BaseException as rollback_error:
                if backup is None:
                    rollback_errors.append(
                        "{}: rollback removal failed: {}".format(
                            target, rollback_error
                        )
                    )
                else:
                    retained_backups.add(backup)
                    rollback_errors.append(
                        "{}: rollback restore failed: {}; "
                        "backup retained at {}".format(
                            target, rollback_error, backup
                        )
                    )

        for backup in backups.values():
            if backup in retained_backups:
                continue
            try:
                backup.unlink(missing_ok=True)
            except BaseException as cleanup_error:
                retained_backups.add(backup)
                rollback_errors.append(
                    "backup cleanup failed: {}; backup retained at {}".format(
                        cleanup_error, backup
                    )
                )
        if rollback_errors:
            _annotate_matrix_rollback_error(original_error, rollback_errors)
        raise
    else:
        cleanup_diagnostics = []
        for target, backup in backups.items():
            try:
                backup.unlink(missing_ok=True)
            except OSError as cleanup_error:
                cleanup_diagnostics.append(
                    "{}: backup cleanup failed: {}; backup retained at {}".format(
                        target, cleanup_error, backup
                    )
                )
        return tuple(cleanup_diagnostics)


def _validate_matrix_metadata(
    h_sections: Sequence[CSRSection],
    s_section: CSRSection,
    nspin: int,
) -> CSRSection:
    if nspin not in (1, 2, 4):
        raise ValueError("nspin must be 1, 2, or 4")
    expected_h_sections = 2 if nspin == 2 else 1
    if len(h_sections) != expected_h_sections:
        raise ValueError(
            "nspin={} requires exactly {} H section{}".format(
                nspin, expected_h_sections, "" if expected_h_sections == 1 else "s"
            )
        )

    h_section = h_sections[0]
    if any(section.matrix != "H" for section in h_sections) or s_section.matrix != "S":
        raise ValueError("matrix set must contain H section(s) and one S section")
    for section in h_sections:
        if section.dimension != s_section.dimension:
            raise ValueError("H/S matrix dimension mismatch")
        if section.step != s_section.step:
            raise ValueError("H/S ionic step mismatch")
    if nspin == 4 and h_section.dimension % 2:
        raise ValueError("nspin=4 source matrix dimension must be even")
    if nspin == 1:
        for section in tuple(h_sections) + (s_section,):
            if (
                section.format == "current"
                and (section.nspin != 1 or section.spin_index != 1)
            ):
                raise ValueError(
                    "nspin=1 requires current matrix headers with nspin=1, spin index 1"
                )
    if nspin == 2:
        for expected_spin_index, section in enumerate(h_sections, 1):
            if (
                section.format == "current"
                and (section.nspin != 2 or section.spin_index != expected_spin_index)
            ):
                raise ValueError(
                    "nspin=2 H channel {} must have current spin index {}".format(
                        expected_spin_index, expected_spin_index
                    )
                )
        if (
            s_section.format == "current"
            and (s_section.nspin != 1 or s_section.spin_index != 1)
        ):
            raise ValueError(
                "nspin=2 S section must have current nspin=1, spin index 1"
            )
    if nspin == 4:
        for matrix, section in (("H", h_section), ("S", s_section)):
            if (
                section.format == "current"
                and (section.nspin != 1 or section.spin_index != 1)
            ):
                raise ValueError(
                    "nspin=4 {} section must have current nspin=1, spin index 1".format(
                        matrix
                    )
                )
    return h_section


def _iter_validated_matrix_blocks(
    h_sections: Sequence[CSRSection], s_section: CSRSection
) -> Iterator[Tuple[CSRBlock, ...]]:
    """Stream independent sparse supports as additive channel contributions.

An absent R block is zero, not a malformed matrix. Contributions for the same
R from different spin channels have disjoint matrix indices. Only duplicate
R blocks within one source are invalid. Memory stays bounded by one CSR block.
"""
    sections = tuple(h_sections) + (s_section,)
    zero_pointers = (0,) * (s_section.dimension + 1)
    for index, section in enumerate(sections):
        seen_r = set()
        for block in section.iter_blocks():
            if block.r in seen_r:
                raise ValueError("duplicate R block {}".format(block.r))
            seen_r.add(block.r)
            empty = CSRBlock(block.r, (), (), zero_pointers, False)
            contributions = [empty] * len(sections)
            contributions[index] = block
            yield tuple(contributions)


def _validate_matrix_set(
    h_sections: Sequence[CSRSection], s_section: CSRSection, nspin: int
) -> None:
    _validate_matrix_metadata(h_sections, s_section, nspin)
    for _blocks in _iter_validated_matrix_blocks(h_sections, s_section):
        pass


def convert_matrix_set(
    h_sections: Sequence[CSRSection],
    s_section: CSRSection,
    nspin: int,
    output_dir: Union[str, Path],
    h_cutoff_ev: float = DEFAULT_H_CUTOFF_EV,
    s_cutoff: float = DEFAULT_S_CUTOFF,
) -> ConversionStats:
    """Convert one matched ABACUS H/S section set to ``H.dat`` and ``S.dat``."""

    h_cutoff_ev = _validate_cutoff(h_cutoff_ev, "H cutoff")
    s_cutoff = _validate_cutoff(s_cutoff, "S cutoff")
    h_section = _validate_matrix_metadata(h_sections, s_section, nspin)

    output_directory = Path(output_dir).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    temporary_paths: List[Path] = []
    h_count = 0
    s_count = 0
    union_r = set()
    try:
        h_body_handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=".H.body-",
            dir=str(output_directory),
            delete=False,
        )
        h_body = Path(h_body_handle.name)
        temporary_paths.append(h_body)
        s_body_handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=".S.body-",
            dir=str(output_directory),
            delete=False,
        )
        s_body = Path(s_body_handle.name)
        temporary_paths.append(s_body)
        try:
            for complete_blocks in _iter_validated_matrix_blocks(
                h_sections, s_section
            ):
                union_r.add(complete_blocks[0].r)
                if nspin in (1, 4):
                    index_map = None
                    if nspin == 4:
                        def index_map(index: int) -> int:
                            return _grouped_spinor_index(
                                index, h_section.dimension
                            )

                    h_count += _write_block(
                        h_body_handle,
                        complete_blocks[0],
                        h_cutoff_ev,
                        RY_TO_EV,
                        index_map=index_map,
                    )
                    s_count += _write_block(
                        s_body_handle,
                        complete_blocks[1],
                        s_cutoff,
                        1.0,
                        index_map=index_map,
                    )
                else:
                    dimension = h_section.dimension
                    h_count += _write_block(
                        h_body_handle,
                        complete_blocks[0],
                        h_cutoff_ev,
                        RY_TO_EV,
                    )
                    h_count += _write_block(
                        h_body_handle,
                        complete_blocks[1],
                        h_cutoff_ev,
                        RY_TO_EV,
                        dimension,
                        dimension,
                    )
                    s_count += _write_block(
                        s_body_handle, complete_blocks[2], s_cutoff, 1.0
                    )
                    s_count += _write_block(
                        s_body_handle,
                        complete_blocks[2],
                        s_cutoff,
                        1.0,
                        dimension,
                        dimension,
                    )
        finally:
            h_body_handle.close()
            s_body_handle.close()

        output_dimension = h_section.dimension * (2 if nspin == 2 else 1)
        noncollinear = nspin in (2, 4)
        h_final = _write_final_sparse_file(
            output_directory / "H.dat",
            h_body,
            "H",
            h_count,
            output_dimension,
            len(union_r),
            noncollinear,
        )
        temporary_paths.append(h_final)
        s_final = _write_final_sparse_file(
            output_directory / "S.dat",
            s_body,
            "S",
            s_count,
            output_dimension,
            len(union_r),
            noncollinear,
        )
        temporary_paths.append(s_final)
        cleanup_warnings = _commit_matrix_files(
            (
                (h_final, output_directory / "H.dat"),
                (s_final, output_directory / "S.dat"),
            )
        )
        return ConversionStats(
            source_dimension=h_section.dimension,
            output_dimension=output_dimension,
            r_count=len(union_r),
            h_written=h_count,
            s_written=s_count,
            layout=(
                "spatial"
                if nspin == 1
                else "block"
                if nspin == 2
                else "grouped-spinor"
            ),
            cleanup_warnings=cleanup_warnings,
        )
    finally:
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)


def _step_argument(value: str) -> Union[str, int]:
    if value.lower() == "latest":
        return "latest"
    if re.fullmatch(r"[+-]?[0-9]+", value) is None:
        raise argparse.ArgumentTypeError("step must be 'latest' or an integer")
    return int(value)


def _cutoff_argument(value: str) -> float:
    try:
        cutoff = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("cutoff must be a finite non-negative number")
    if not math.isfinite(cutoff) or cutoff < 0.0:
        raise argparse.ArgumentTypeError("cutoff must be a finite non-negative number")
    return cutoff


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert standard ABACUS H(R)/S(R) outputs to H.dat and S.dat."
    )
    parser.add_argument("INPUT_PATH", help="ABACUS case, INPUT, or OUT directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="output directory (default: current working directory)",
    )
    parser.add_argument(
        "--step",
        type=_step_argument,
        default="latest",
        metavar="latest|N",
        help="ionic step (default: latest complete matching set)",
    )
    parser.add_argument(
        "--nspin",
        type=int,
        choices=(1, 2, 4),
        help=(
            "override direct OUT inference only; a second H channel implies 2, "
            "parenthesized complex tokens imply 4, and real values default to 1; "
            "case INPUT conflicts are errors"
        ),
    )
    parser.add_argument(
        "--h-cutoff-ev",
        type=_cutoff_argument,
        default=DEFAULT_H_CUTOFF_EV,
        help="strict H magnitude cutoff in eV (default: 1e-7)",
    )
    parser.add_argument(
        "--s-cutoff",
        type=_cutoff_argument,
        default=DEFAULT_S_CUTOFF,
        help="strict dimensionless S magnitude cutoff (default: 1e-7)",
    )
    return parser


def _metadata(
    selection: MatrixSelection,
    stats: ConversionStats,
    h_cutoff_ev: float,
    s_cutoff: float,
) -> Dict[str, object]:
    spatial_orbitals = (
        stats.source_dimension // 2
        if selection.nspin == 4
        else stats.source_dimension
    )
    return {
        "H_unit": "eV",
        "S_unit": "dimensionless",
        "case": (
            str(selection.case_dir)
            if selection.case_dir is not None
            else None
        ),
        "counts": {"H": stats.h_written, "S": stats.s_written},
        "family": selection.family,
        "input": str(selection.input_path),
        "naming_family": selection.family,
        "out": str(selection.out_dir),
        "output_orbitals": stats.output_dimension,
        "r_count": stats.r_count,
        "source_files": [str(path) for path in selection.source_files],
        "source_nspin": selection.nspin,
        "spatial_orbitals": spatial_orbitals,
        "spin_inference": selection.inference,
        "spin_layout": stats.layout,
        "step": selection.step,
        "thresholds": {"H_eV": h_cutoff_ev, "S": s_cutoff},
    }


def _write_staged_metadata(path: Path, metadata: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the command-line converter, returning a process status."""

    parser = _argument_parser()
    try:
        arguments = parser.parse_args(argv)
    except SystemExit as error:
        return int(error.code)

    try:
        selection = discover_matrix_set(
            arguments.INPUT_PATH,
            step=arguments.step,
            nspin=arguments.nspin,
        )
        output_dir = arguments.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".analysis_abacus.tmp-", dir=str(output_dir)
        ) as temporary_directory:
            staging_dir = Path(temporary_directory)
            stats = convert_matrix_set(
                selection.h_sections,
                selection.s_section,
                selection.nspin,
                staging_dir,
                h_cutoff_ev=arguments.h_cutoff_ev,
                s_cutoff=arguments.s_cutoff,
            )
            staged_metadata = staging_dir / "matrix_meta.json"
            _write_staged_metadata(
                staged_metadata,
                _metadata(
                    selection,
                    stats,
                    arguments.h_cutoff_ev,
                    arguments.s_cutoff,
                ),
            )
            targets = tuple(
                (staging_dir / name, output_dir / name)
                for name in ("H.dat", "S.dat", "matrix_meta.json")
            )
            cleanup_warnings = (
                stats.cleanup_warnings + _commit_matrix_files(targets)
            )
    except (OSError, ValueError) as error:
        print("error: {}".format(error), file=sys.stderr)
        return 2

    print(
        "family={} nspin={} step={}".format(
            selection.family, selection.nspin, selection.step
        )
    )
    for name in ("H.dat", "S.dat", "matrix_meta.json"):
        print(str((output_dir / name).resolve()))
    for diagnostic in cleanup_warnings:
        print("warning: {}".format(diagnostic), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
