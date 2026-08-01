"""Persistent, fail-closed cache for target-independent response bases.

This module deliberately does not choose a cache directory or integrate with the
model pipeline.  Callers must provide both the directory and the complete
target-independent compiler input used to construct the cache key.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import numpy as np

from .response_basis import (
    COMPLETE_LINEAR_V2,
    CompiledResponseBasis,
    _canonical_json,
    _validate_persistent_cache_input_record,
)


CACHE_FORMAT_VERSION = "compiled_response_basis_cache_v1"

_TARGET_DEPENDENT_FIELDS = frozenset(
    {
        "heff",
        "heff_file",
        "target",
        "target_hamiltonians",
        "fit_indices",
        "response_fit_indices",
        "response_fit_objective",
        "fit_objective",
        "spectral_weighting",
        "spectral_weight_hash",
        "regularization",
        "band_window",
        "bandwindow",
        "fit_pivots",
        "fit_hash",
        "fitted_coefficients",
        "coefficients_fitted",
        "nonzero_channel_ids",
    }
)
_CACHE_KEY_PATTERN = re.compile(r"[0-9a-f]{64}")


class TargetDependentCacheKeyError(ValueError):
    """Raised when a cache-key payload contains fit/target state."""


class ResponseBasisCacheCorruptionError(ValueError):
    """Raised when an existing persistent cache cannot be certified."""


def _find_target_dependent_field(value: Any, *, path: tuple[str, ...] = ()) -> tuple[str, ...] | None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            child_path = (*path, key)
            if key in _TARGET_DEPENDENT_FIELDS:
                return child_path
            found = _find_target_dependent_field(child, path=child_path)
            if found is not None:
                return found
        return None
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found = _find_target_dependent_field(child, path=(*path, str(index)))
            if found is not None:
                return found
    return None


def target_independent_basis_key(compiler_input: Mapping[str, Any]) -> str:
    """Return a canonical key after rejecting known target-dependent fields.

    The payload should be the compiler input record: actual layout, Q vectors,
    exactified actions, coordinate convention, support policy, candidate seeds,
    normalization versions and reducer mode.  It must not contain Heff or fit
    state.  Canonical serialization is shared with ``CompiledResponseBasis`` so
    array byte order does not alter the key.
    """

    if not isinstance(compiler_input, Mapping):
        raise TypeError("response-basis compiler input must be a mapping")
    forbidden_path = _find_target_dependent_field(compiler_input)
    if forbidden_path is not None:
        dotted = ".".join(forbidden_path)
        raise TargetDependentCacheKeyError(
            f"target-dependent field {dotted!r} is forbidden in a response-basis cache key"
        )
    _validate_persistent_cache_input_record(compiler_input)
    import hashlib

    return hashlib.sha256(_canonical_json(compiler_input).encode("utf-8")).hexdigest()


class PersistentResponseBasisCache:
    """Disk cache for frozen ``CompiledResponseBasis`` objects.

    Statistics are process-local and per cache instance.  A successful disk hit
    is warm; invoking the supplied compiler after a missing file is cold.  An
    existing invalid file is corruption and never falls back to compilation.
    """

    def __init__(self, cache_directory: str | os.PathLike[str]) -> None:
        self.cache_directory = Path(cache_directory)
        self._cold_compile_count_by_key: dict[str, int] = {}
        self._warm_load_count_by_key: dict[str, int] = {}
        self._corrupt_load_count_by_key: dict[str, int] = {}

    def _path_for_key(self, key: str) -> Path:
        if _CACHE_KEY_PATTERN.fullmatch(key) is None:
            raise ValueError(f"invalid response-basis cache key: {key!r}")
        return self.cache_directory / f"{key}.npz"

    def path_for(self, compiler_input: Mapping[str, Any]) -> Path:
        return self._path_for_key(target_independent_basis_key(compiler_input))

    @contextmanager
    def _key_lock(self, key: str):
        self.cache_directory.mkdir(parents=True, exist_ok=True)
        lock_path = self.cache_directory / f".{key}.lock"
        with lock_path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _validate_basis(basis: CompiledResponseBasis) -> None:
        if not isinstance(basis, CompiledResponseBasis):
            raise TypeError("persistent response-basis cache accepts only CompiledResponseBasis")
        if basis.response_semantics != COMPLETE_LINEAR_V2:
            raise ValueError(
                "persistent response-basis cache accepts only complete_linear_v2; "
                f"got {basis.response_semantics!r}"
            )

    def store(
        self,
        compiler_input: Mapping[str, Any],
        basis: CompiledResponseBasis,
    ) -> Path:
        key = target_independent_basis_key(compiler_input)
        self._validate_basis(basis)
        with self._key_lock(key):
            destination = self._path_for_key(key)
            if destination.is_file():
                existing = self.load(compiler_input)
                if existing.basis_hash != basis.basis_hash:
                    raise ResponseBasisCacheCorruptionError(
                        "response-basis cache key already contains a different basis hash"
                    )
            return self._store_unlocked(key, basis)

    def _store_unlocked(self, key: str, basis: CompiledResponseBasis) -> Path:
        self._validate_basis(basis)
        frozen = basis.freeze()
        metadata = {
            "cache_format_version": CACHE_FORMAT_VERSION,
            "compiler_input_key": key,
            "response_semantics": basis.response_semantics,
            "basis_hash": basis.basis_hash,
        }
        arrays = {
            "cache_metadata_json": np.asarray(_canonical_json(metadata)),
            **{f"basis_{name}": np.asarray(value) for name, value in frozen.items()},
        }

        destination = self._path_for_key(key)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{key}.",
            suffix=".tmp",
            dir=self.cache_directory,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w+b") as handle:
                np.savez_compressed(handle, **arrays)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            self._fsync_directory()
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    def _fsync_directory(self) -> None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        try:
            descriptor = os.open(self.cache_directory, flags)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def load(self, compiler_input: Mapping[str, Any]) -> CompiledResponseBasis:
        key = target_independent_basis_key(compiler_input)
        path = self._path_for_key(key)
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            with np.load(path, allow_pickle=False) as archive:
                arrays = {name: np.asarray(archive[name]) for name in archive.files}
            basis = self._certify_arrays(key, arrays)
        except ResponseBasisCacheCorruptionError:
            self._increment(self._corrupt_load_count_by_key, key)
            raise
        except Exception as exc:
            self._increment(self._corrupt_load_count_by_key, key)
            raise ResponseBasisCacheCorruptionError(
                f"response-basis cache {path.name!r} could not be read and will not be recompiled"
            ) from exc
        self._increment(self._warm_load_count_by_key, key)
        return basis

    def _certify_arrays(
        self,
        expected_key: str,
        arrays: Mapping[str, np.ndarray],
    ) -> CompiledResponseBasis:
        if "cache_metadata_json" not in arrays:
            raise ResponseBasisCacheCorruptionError("cache metadata is missing")
        try:
            metadata = json.loads(str(np.asarray(arrays["cache_metadata_json"]).item()))
        except Exception as exc:
            raise ResponseBasisCacheCorruptionError("cache metadata is invalid") from exc
        if metadata.get("cache_format_version") != CACHE_FORMAT_VERSION:
            raise ResponseBasisCacheCorruptionError("cache format version mismatch")
        if metadata.get("compiler_input_key") != expected_key:
            raise ResponseBasisCacheCorruptionError("cache compiler input key mismatch")
        if metadata.get("response_semantics") != COMPLETE_LINEAR_V2:
            raise ResponseBasisCacheCorruptionError("cache response semantics mismatch")
        basis_arrays = {
            name[len("basis_") :]: value
            for name, value in arrays.items()
            if name.startswith("basis_")
        }
        try:
            basis = CompiledResponseBasis.from_frozen(basis_arrays)
        except Exception as exc:
            raise ResponseBasisCacheCorruptionError(
                "cached frozen basis failed internal hash certification"
            ) from exc
        if basis.response_semantics != COMPLETE_LINEAR_V2:
            raise ResponseBasisCacheCorruptionError("frozen basis response semantics mismatch")
        if metadata.get("basis_hash") != basis.basis_hash:
            raise ResponseBasisCacheCorruptionError("cache basis hash mismatch")
        return basis

    def get_or_compile(
        self,
        compiler_input: Mapping[str, Any],
        compiler: Callable[[], CompiledResponseBasis],
    ) -> CompiledResponseBasis:
        key = target_independent_basis_key(compiler_input)
        try:
            return self.load(compiler_input)
        except FileNotFoundError:
            pass
        with self._key_lock(key):
            try:
                return self.load(compiler_input)
            except FileNotFoundError:
                self._increment(self._cold_compile_count_by_key, key)
                basis = compiler()
                self._validate_basis(basis)
                destination = self._path_for_key(key)
                if destination.is_file():
                    existing = self.load(compiler_input)
                    if existing.basis_hash != basis.basis_hash:
                        raise ResponseBasisCacheCorruptionError(
                            "response-basis cache key already contains a different basis hash"
                        )
                    return existing
                self._store_unlocked(key, basis)
                return basis

    @staticmethod
    def _increment(counts: dict[str, int], key: str) -> None:
        counts[key] = counts.get(key, 0) + 1

    def stats(self) -> dict[str, Any]:
        keys = sorted(
            set(self._cold_compile_count_by_key)
            | set(self._warm_load_count_by_key)
            | set(self._corrupt_load_count_by_key)
        )
        cold = {key: int(self._cold_compile_count_by_key.get(key, 0)) for key in keys}
        warm = {key: int(self._warm_load_count_by_key.get(key, 0)) for key in keys}
        corrupt = {key: int(self._corrupt_load_count_by_key.get(key, 0)) for key in keys}
        return {
            "keys": keys,
            "cold_compile_count": int(sum(cold.values())),
            "warm_load_count": int(sum(warm.values())),
            "corrupt_load_count": int(sum(corrupt.values())),
            "cold_compile_count_by_key": cold,
            "warm_load_count_by_key": warm,
            "corrupt_load_count_by_key": corrupt,
        }


__all__ = [
    "CACHE_FORMAT_VERSION",
    "PersistentResponseBasisCache",
    "ResponseBasisCacheCorruptionError",
    "TargetDependentCacheKeyError",
    "target_independent_basis_key",
]
