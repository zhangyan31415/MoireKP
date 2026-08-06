#!/usr/bin/env python3
"""Compare production local-fixed and forced-Reynolds response subspaces."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
import socket
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy.linalg
from scipy import sparse

import kp.model.response_basis as response_basis
import kp.model.response_basis_graded as response_basis_graded
from kp.model.core import build_model
from kp.model.pipeline import build_moire_config_from_file
from kp.model.response_basis import (
    _channel_sparse_vector,
    clear_response_basis_cache,
    compile_model_response_basis,
)


def _load_config(path: Path, *, output_dir: Path) -> Any:
    with contextlib.redirect_stdout(io.StringIO()):
        value = build_moire_config_from_file(path)
    config = value[0] if isinstance(value, tuple) else value
    config.output_dir = output_dir
    return config


def _compile(
    path: Path,
    *,
    output_dir: Path,
    force_reynolds: bool,
    reuse_existing: bool,
) -> tuple[Any, float]:
    if not reuse_existing:
        shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _load_config(path, output_dir=output_dir)
    with contextlib.redirect_stdout(io.StringIO()):
        model = build_model(config)
    clear_response_basis_cache()

    original = response_basis_graded.compile_graded_candidate_group
    if force_reynolds:

        def forced(*args: Any, **kwargs: Any) -> Any:
            kwargs["force_reynolds"] = True
            return original(*args, **kwargs)

        response_basis_graded.compile_graded_candidate_group = forced
    started = time.perf_counter()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            compiled = compile_model_response_basis(model, config, reduce=True)
    finally:
        response_basis_graded.compile_graded_candidate_group = original
    return compiled, float(time.perf_counter() - started)


def _coefficient_matrix(compiled: Any) -> sparse.csc_matrix:
    columns = [
        _channel_sparse_vector(
            channel,
            compiled.coordinate,
            compiled.dim,
        )
        for channel in compiled.channels
    ]
    if not columns:
        return sparse.csc_matrix((0, 0), dtype=np.float64)
    return sparse.hstack(columns, format="csc")


def _normalize_columns(matrix: sparse.csc_matrix) -> tuple[sparse.csc_matrix, np.ndarray]:
    norms = np.sqrt(np.asarray(matrix.power(2).sum(axis=0)).ravel())
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise RuntimeError("compiled response basis contains a zero or non-finite column")
    normalized = matrix @ sparse.diags(1.0 / norms, format="csc")
    return sparse.csc_matrix(normalized), norms


def _inverse_square_root_spd(matrix: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = scipy.linalg.eigh(symmetric, check_finite=False)
    if np.any(eigenvalues <= 0.0):
        raise RuntimeError("whitened response metric is not positive definite")
    return (
        eigenvectors
        * (1.0 / np.sqrt(eigenvalues))[None, :]
    ) @ eigenvectors.T


def _orthonormal_overlap(
    fast_gram: np.ndarray,
    cross: np.ndarray,
    oracle_gram: np.ndarray,
) -> tuple[np.ndarray, int, int, float, float]:
    fast_symmetric = 0.5 * (fast_gram + fast_gram.T)
    oracle_symmetric = 0.5 * (oracle_gram + oracle_gram.T)
    try:
        fast_lower = scipy.linalg.cholesky(
            fast_symmetric, lower=True, check_finite=False
        )
        oracle_lower = scipy.linalg.cholesky(
            oracle_symmetric, lower=True, check_finite=False
        )
    except scipy.linalg.LinAlgError:
        fast_values, fast_vectors = scipy.linalg.eigh(
            fast_symmetric, check_finite=False
        )
        oracle_values, oracle_vectors = scipy.linalg.eigh(
            oracle_symmetric, check_finite=False
        )
        fast_threshold = float(
            max(
                128.0 * np.finfo(np.float64).eps * max(1, fast_gram.shape[0]),
                1.0e-14,
            )
        )
        oracle_threshold = float(
            max(
                128.0 * np.finfo(np.float64).eps * max(1, oracle_gram.shape[0]),
                1.0e-14,
            )
        )
        fast_active = fast_values > fast_threshold
        oracle_active = oracle_values > oracle_threshold
        fast_transform = (
            fast_vectors[:, fast_active] / np.sqrt(fast_values[fast_active])[None, :]
        )
        oracle_transform = (
            oracle_vectors[:, oracle_active]
            / np.sqrt(oracle_values[oracle_active])[None, :]
        )
        overlap = fast_transform.T @ cross @ oracle_transform
        fast_self = fast_transform.T @ fast_symmetric @ fast_transform
        oracle_self = oracle_transform.T @ oracle_symmetric @ oracle_transform
        overlap = (
            _inverse_square_root_spd(fast_self)
            @ overlap
            @ _inverse_square_root_spd(oracle_self)
        )
        return (
            overlap,
            int(np.count_nonzero(fast_active)),
            int(np.count_nonzero(oracle_active)),
            fast_threshold,
            oracle_threshold,
        )
    left_whitened = scipy.linalg.solve_triangular(
        fast_lower,
        cross,
        lower=True,
        check_finite=False,
    )
    overlap = scipy.linalg.solve_triangular(
        oracle_lower,
        left_whitened.T,
        lower=True,
        check_finite=False,
    ).T
    fast_self_left = scipy.linalg.solve_triangular(
        fast_lower,
        fast_symmetric,
        lower=True,
        check_finite=False,
    )
    fast_self = scipy.linalg.solve_triangular(
        fast_lower,
        fast_self_left.T,
        lower=True,
        check_finite=False,
    ).T
    oracle_self_left = scipy.linalg.solve_triangular(
        oracle_lower,
        oracle_symmetric,
        lower=True,
        check_finite=False,
    )
    oracle_self = scipy.linalg.solve_triangular(
        oracle_lower,
        oracle_self_left.T,
        lower=True,
        check_finite=False,
    ).T
    overlap = (
        _inverse_square_root_spd(fast_self)
        @ overlap
        @ _inverse_square_root_spd(oracle_self)
    )
    threshold = float(
        128.0 * np.finfo(np.float64).eps * max(1, fast_gram.shape[0])
    )
    return overlap, fast_gram.shape[0], oracle_gram.shape[0], threshold, threshold


def _certified_minimum_singular_value(gram: np.ndarray) -> tuple[float, float]:
    symmetric = 0.5 * (gram + gram.T)
    minimum = float(
        scipy.linalg.eigvalsh(
            symmetric,
            subset_by_index=(0, 0),
            check_finite=False,
            driver="evr",
        )[0]
    )
    scale = max(1.0, float(np.linalg.norm(symmetric, ord=2)))
    epsilon = np.finfo(np.float64).eps
    dimension = max(1, gram.shape[0])
    gamma = float((dimension * epsilon) / (1.0 - dimension * epsilon))
    backward_error = float(64.0 * gamma * scale)
    lower_eigenvalue = minimum - backward_error
    if lower_eigenvalue <= 0.0:
        raise RuntimeError("normalized response Gram has no certified positive lower bound")
    return float(np.sqrt(lower_eigenvalue)), backward_error


def _one_sided_projector_bound(
    source: sparse.csc_matrix,
    target: sparse.csc_matrix,
    *,
    source_channel_ids: tuple[str, ...],
    target_channel_ids: tuple[str, ...],
    source_gram: np.ndarray,
    target_gram: np.ndarray,
) -> tuple[float, float, int, int]:
    target_position = {
        channel_id: index for index, channel_id in enumerate(target_channel_ids)
    }
    target_factor = scipy.linalg.cho_factor(
        0.5 * (target_gram + target_gram.T),
        lower=True,
        check_finite=False,
    )
    residual_squared_sum = 0.0
    matched_count = 0
    projected_count = 0
    for source_index, channel_id in enumerate(source_channel_ids):
        source_column = source.getcol(source_index)
        matching_index = target_position.get(channel_id)
        if matching_index is not None:
            target_column = target.getcol(matching_index)
            overlap = float((target_column.T @ source_column).toarray()[0, 0])
            sign = 1.0 if overlap >= 0.0 else -1.0
            residual = source_column - sign * target_column
            residual_squared_sum += float(residual.power(2).sum())
            matched_count += 1
            continue
        cross = np.asarray(target.T @ source_column).reshape(-1)
        coefficients = scipy.linalg.cho_solve(
            target_factor,
            cross,
            check_finite=False,
        )
        projected = np.asarray(target @ coefficients).reshape(-1)
        residual = source_column.toarray().reshape(-1) - projected
        residual_squared_sum += float(residual @ residual)
        projected_count += 1
    minimum_singular_value, backward_error = _certified_minimum_singular_value(
        source_gram
    )
    bound = float(np.sqrt(residual_squared_sum) / minimum_singular_value)
    return bound, backward_error, matched_count, projected_count


def _compare(fast: Any, oracle: Any) -> dict[str, Any]:
    fast_matrix, fast_norms = _normalize_columns(_coefficient_matrix(fast))
    oracle_matrix, oracle_norms = _normalize_columns(_coefficient_matrix(oracle))
    if fast_matrix.shape[0] != oracle_matrix.shape[0]:
        raise RuntimeError("fast and Reynolds coefficient ambients differ")
    fast_gram = np.asarray((fast_matrix.T @ fast_matrix).toarray(), dtype=np.float64)
    oracle_gram = np.asarray((oracle_matrix.T @ oracle_matrix).toarray(), dtype=np.float64)
    cross = np.asarray((fast_matrix.T @ oracle_matrix).toarray(), dtype=np.float64)
    (
        overlap,
        fast_rank,
        oracle_rank,
        fast_threshold,
        oracle_threshold,
    ) = _orthonormal_overlap(fast_gram, cross, oracle_gram)
    if fast_rank and oracle_rank:
        singular_values = scipy.linalg.svdvals(overlap, check_finite=False)
        singular_values = np.clip(singular_values, 0.0, 1.0)
        minimum_overlap = float(singular_values[-1])
        projector_error = float(np.sqrt(max(0.0, 1.0 - minimum_overlap**2)))
    else:
        singular_values = np.empty(0, dtype=np.float64)
        minimum_overlap = 1.0 if fast_rank == oracle_rank else 0.0
        projector_error = 0.0 if fast_rank == oracle_rank else 1.0
    certification_bound = float(
        max(
            [0.0]
            + [
                float(channel.propagated_error_bound) / float(norm)
                for channel, norm in zip(fast.channels, fast_norms)
            ]
            + [
                float(channel.propagated_error_bound) / float(norm)
                for channel, norm in zip(oracle.channels, oracle_norms)
            ]
        )
    )
    gate = float(max(1.0e-10, 10.0 * certification_bound))
    fast_ids = tuple(str(channel.channel_id) for channel in fast.channels)
    oracle_ids = tuple(str(channel.channel_id) for channel in oracle.channels)
    (
        fast_to_oracle_bound,
        fast_gram_backward_error,
        common_fast_count,
        projected_fast_count,
    ) = _one_sided_projector_bound(
        fast_matrix,
        oracle_matrix,
        source_channel_ids=fast_ids,
        target_channel_ids=oracle_ids,
        source_gram=fast_gram,
        target_gram=oracle_gram,
    )
    (
        oracle_to_fast_bound,
        oracle_gram_backward_error,
        common_oracle_count,
        projected_oracle_count,
    ) = _one_sided_projector_bound(
        oracle_matrix,
        fast_matrix,
        source_channel_ids=oracle_ids,
        target_channel_ids=fast_ids,
        source_gram=oracle_gram,
        target_gram=fast_gram,
    )
    projector_upper_bound = max(fast_to_oracle_bound, oracle_to_fast_bound)
    return {
        "ambient_real_dimension": int(fast_matrix.shape[0]),
        "fast_channel_count": len(fast.channels),
        "oracle_channel_count": len(oracle.channels),
        "fast_rank": fast_rank,
        "oracle_rank": oracle_rank,
        "fast_rank_threshold": fast_threshold,
        "oracle_rank_threshold": oracle_threshold,
        "minimum_principal_overlap": minimum_overlap,
        "maximum_principal_angle_sine": projector_error,
        "projector_error_spectral_estimate": projector_error,
        "projector_error_spectral_upper_bound": projector_upper_bound,
        "fast_to_oracle_projector_upper_bound": fast_to_oracle_bound,
        "oracle_to_fast_projector_upper_bound": oracle_to_fast_bound,
        "fast_gram_eigenvalue_backward_error": fast_gram_backward_error,
        "oracle_gram_eigenvalue_backward_error": oracle_gram_backward_error,
        "common_channel_count": min(common_fast_count, common_oracle_count),
        "fast_columns_explicitly_projected": projected_fast_count,
        "oracle_columns_explicitly_projected": projected_oracle_count,
        "certification_bound": certification_bound,
        "projector_gate": gate,
        "projector_gate_passed": bool(
            fast_rank == oracle_rank and projector_upper_bound <= gate
        ),
        "overlap_singular_value_min": (
            float(singular_values[-1]) if singular_values.size else None
        ),
        "overlap_singular_value_max": (
            float(singular_values[0]) if singular_values.size else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    fast, fast_seconds = _compile(
        args.config,
        output_dir=args.output_root / "fast",
        force_reynolds=False,
        reuse_existing=bool(args.reuse_existing),
    )
    oracle, oracle_seconds = _compile(
        args.config,
        output_dir=args.output_root / "reynolds",
        force_reynolds=True,
        reuse_existing=bool(args.reuse_existing),
    )
    result = {
        "schema_version": "response-compiler-projector-comparison-v1",
        "hostname": socket.gethostname(),
        "config": str(args.config),
        "compiler_version": response_basis.COMPILER_VERSION,
        "fast_compile_wall_seconds": fast_seconds,
        "reynolds_compile_wall_seconds": oracle_seconds,
        "fast_basis_hash": str(fast.basis_hash),
        "reynolds_basis_hash": str(oracle.basis_hash),
        **_compare(fast, oracle),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
