from __future__ import annotations

import re
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from typing import Any


def _expand_orbital_pattern(segment: str) -> list[str]:
    """Expand one species segment such as ``Bi-s3p2d2`` into AO labels."""
    segment = str(segment).strip()
    if not segment:
        return []
    if "-" not in segment:
        return [segment]
    species, patterns = segment.split("-", 1)
    s_list = ["s"]
    p_list = ["px", "py", "pz"]
    d_list = ["dz2", "dx2-y2", "dxy", "dxz", "dyz"]
    f_list = ["f5z2", "f5xz2", "f5yz2", "fzx2", "fxyz", "fx3", "f3yx2"]
    labels: list[str] = []
    for match in re.finditer(r"([spdf])(\d+)", patterns):
        orb = match.group(1)
        rep = int(match.group(2))
        base = {"s": s_list, "p": p_list, "d": d_list, "f": f_list}[orb]
        for repeat in range(1, rep + 1):
            for item in base:
                labels.append(f"{species}_{item}_r{repeat}")
    return labels


def expand_orbital_order_pattern(pattern: str) -> list[str]:
    """Expand a comma-separated orbital-order pattern into flat AO labels."""
    labels: list[str] = []
    for segment in str(pattern).split(","):
        if segment.strip():
            labels.extend(_expand_orbital_pattern(segment))
    return labels


def _normalize_labels(labels: Sequence[str], expected_count: int | None) -> list[str]:
    out = [str(label) for label in labels]
    if expected_count is None:
        return out
    if len(out) < int(expected_count):
        out.extend(f"orb_{idx}" for idx in range(len(out), int(expected_count)))
    return out[: int(expected_count)]


def _sector_aliases(sector: str, index: int) -> list[str]:
    raw = str(sector)
    aliases = [raw, raw.lower(), str(index), str(index + 1)]
    match = re.fullmatch(r"[Ll](\d+)", raw)
    if match is not None:
        number = int(match.group(1))
        aliases.extend([f"L{number}", f"l{number}", f"layer{number}", f"layer_{number}", str(number)])
    aliases.extend([f"L{index + 1}", f"l{index + 1}", f"layer{index + 1}", f"layer_{index + 1}"])
    seen: set[str] = set()
    out: list[str] = []
    for alias in aliases:
        if alias not in seen:
            seen.add(alias)
            out.append(alias)
    return out


def expand_orbital_order_by_sector(
    raw: Any,
    sector_names: Sequence[str],
    *,
    expected_count: int | None = None,
) -> dict[str, list[str]] | None:
    """Expand legacy or per-sector ``orbital_order`` into labels per sector/layer.

    Accepted forms:
    - ``"Bi-s3p2d2,Te-s3p2d2"``: same order for every sector.
    - ``["Bi-...", "I-..."]``: one pattern per sector in ``sector_names`` order.
    - ``{"L1": "Bi-...", "L2": "I-..."}``: explicit sector/layer mapping.
    """
    if raw is None:
        return None
    names = [str(name) for name in sector_names]
    if not names:
        return None
    if isinstance(raw, str):
        labels = _normalize_labels(expand_orbital_order_pattern(raw), expected_count)
        return {name: list(labels) for name in names}
    if isinstance(raw, Mapping):
        out: dict[str, list[str]] = {}
        by_key = {str(key): value for key, value in raw.items()}
        for idx, name in enumerate(names):
            value = None
            for alias in _sector_aliases(name, idx):
                if alias in by_key:
                    value = by_key[alias]
                    break
            if value is None:
                continue
            if isinstance(value, str):
                labels = expand_orbital_order_pattern(value)
            elif isinstance(value, Sequence):
                labels = [str(item) for item in value]
            else:
                raise ValueError(f"orbital_order entry for {name!r} must be a string or list")
            out[name] = _normalize_labels(labels, expected_count)
        return out or None
    if isinstance(raw, Sequence):
        items = list(raw)
        if len(items) == 1:
            labels = (
                expand_orbital_order_pattern(items[0])
                if isinstance(items[0], str)
                else [str(item) for item in items[0]]
            )
            labels = _normalize_labels(labels, expected_count)
            return {name: list(labels) for name in names}
        if len(items) != len(names):
            raise ValueError(
                f"orbital_order list must have length 1 or match sectors/layers ({len(names)}); got {len(items)}"
            )
        out = {}
        for name, item in zip(names, items):
            labels = expand_orbital_order_pattern(item) if isinstance(item, str) else [str(value) for value in item]
            out[name] = _normalize_labels(labels, expected_count)
        return out
    raise ValueError(f"Unsupported orbital_order type: {type(raw).__name__}")


def orbital_slot_map_from_labels(
    source_labels: Sequence[str],
    target_labels: Sequence[str],
    *,
    one_based: bool = False,
) -> dict[int, int]:
    """Return source-slot to target-slot map using label identity.

    Duplicate labels are matched in first-unmatched order.
    """
    target_by_label: dict[str, deque[int]] = defaultdict(deque)
    offset = 1 if one_based else 0
    for idx, label in enumerate(target_labels):
        target_by_label[str(label)].append(idx + offset)
    out: dict[int, int] = {}
    for idx, label in enumerate(source_labels):
        choices = target_by_label.get(str(label))
        if not choices:
            continue
        out[idx + offset] = int(choices.popleft())
    return out
