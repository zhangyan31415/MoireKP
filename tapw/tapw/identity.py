from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


IDENTITY_SCHEMA = "moirekp.artifact-identity.v1"
_FILE_SCHEMA = b"moirekp:file:v1\0"
_ARRAY_SCHEMA = b"moirekp:array:v1\0"
_MAPPING_SCHEMA = b"moirekp:mapping:v1\0"


def hash_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    digest.update(_FILE_SCHEMA)
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def hash_array(array: Any) -> str:
    value = np.asarray(array)
    if value.dtype.hasobject:
        raise TypeError("object-dtype arrays cannot be used for artifact identity")
    contiguous = np.ascontiguousarray(value)
    descriptor = json.dumps(
        {"dtype": contiguous.dtype.str, "shape": list(contiguous.shape)},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(_ARRAY_SCHEMA)
    digest.update(descriptor)
    digest.update(b"\0")
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def hash_mapping(mapping: Mapping[str, Any]) -> str:
    if not isinstance(mapping, Mapping):
        raise TypeError("hash_mapping expects a mapping")
    encoded = json.dumps(
        _canonical_value(mapping),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(_MAPPING_SCHEMA)
    digest.update(encoded)
    return digest.hexdigest()


def require_identity_fields(
    metadata: Mapping[str, Any],
    fields: Sequence[str],
    context: str,
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    missing: list[str] = []
    for field in fields:
        if field not in metadata:
            missing.append(str(field))
            continue
        value = _metadata_scalar(metadata[field], field=str(field), context=context)
        if value is None or value == "":
            missing.append(str(field))
            continue
        values[str(field)] = value
    if missing:
        raise KeyError(f"{context} is missing required identity fields: {', '.join(missing)}")
    return values


def require_matching_identity(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    fields: Sequence[str],
    context: str,
) -> None:
    expected = require_identity_fields(left, fields, context)
    actual = require_identity_fields(right, fields, context)
    mismatched = [field for field in fields if expected[str(field)] != actual[str(field)]]
    if mismatched:
        details = ", ".join(
            f"{field}: {expected[str(field)]!r} != {actual[str(field)]!r}"
            for field in mismatched
        )
        raise ValueError(f"{context} identity mismatch ({details})")


def _metadata_scalar(value: Any, *, field: str, context: str) -> Any:
    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise ValueError(f"{context} identity field {field} must be scalar, got shape {value.shape}")
        value = value.item()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("identity mapping keys must be strings")
            output[key] = _canonical_value(item)
        return output
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("object-dtype arrays cannot be used for artifact identity")
        return _canonical_value(value.tolist())
    if isinstance(value, np.generic):
        return _canonical_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {"__bytes__": value.hex()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonical_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported identity value type: {type(value).__name__}")
