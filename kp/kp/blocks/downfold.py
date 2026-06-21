from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import scipy.linalg


MEV_PER_EV = 1000.0
PROJECTOR_BLAS_THREADS = 8


class NearPoleError(RuntimeError):
    """Raised when a downfolding denominator is too close to a discarded pole."""


@dataclass(frozen=True)
class DownfoldingOptions:
    method: str = "fixed_schur"
    e_ref: float | None = None
    pole_warning_mev: float = 10.0
    pole_danger_mev: float = 1.0
    fail_on_near_pole: bool = False
    compute_pole_diagnostics: bool = False
    compute_condition_number: bool = False


@dataclass
class DownfoldingResult:
    heff: np.ndarray
    method: str
    e_ref: float | None = None
    pole_distance_min_mev: float | None = None
    pole_condition_number: float | None = None
    near_pole: bool = False
    danger_pole: bool = False
    hermiticity_residual: float = 0.0
    b_min_eig: float | None = None
    b_condition_number: float | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _ProjectorGroup:
    rows: np.ndarray
    cols: np.ndarray
    local: np.ndarray
    local_h: np.ndarray


@dataclass(frozen=True)
class _ProjectorGroups:
    n_columns: int
    groups: tuple[_ProjectorGroup, ...]


# Reporting-only helpers kept here for reference, but intentionally not part of
# the active downfolding core.
#
# import csv
# import json
# from pathlib import Path
# from typing import Iterable, Mapping, Sequence
#
# @dataclass(frozen=True)
# class TopNMetric:
#     n: int
#     rms_mev: float
#     max_abs_mev: float
#     mean_mev: float
#
#
# @dataclass(frozen=True)
# class SweepRow:
#     e_ref: float
#     global_pole_distance_min_mev: float | None
#     metrics: Mapping[int, TopNMetric]


def _hermitize(matrix: np.ndarray) -> np.ndarray:
    mat = np.asarray(matrix, dtype=np.complex128)
    return 0.5 * (mat + mat.conj().T)


def _hermiticity_residual(matrix: np.ndarray) -> float:
    mat = np.asarray(matrix, dtype=np.complex128)
    denom = np.linalg.norm(mat)
    if denom == 0.0:
        return 0.0
    return float(np.linalg.norm(mat - mat.conj().T) / denom)


def _bounded_blas_threads() -> Any:
    try:
        from threadpoolctl import threadpool_limits
    except Exception:  # pragma: no cover - optional runtime dependency
        return nullcontext()
    return threadpool_limits(limits=PROJECTOR_BLAS_THREADS, user_api="blas")


def set_projector_blas_threads(threads: int) -> None:
    global PROJECTOR_BLAS_THREADS
    PROJECTOR_BLAS_THREADS = max(1, int(threads))


def _shifted_hamiltonian_matrix(h_hh: np.ndarray, e_ref: float) -> np.ndarray:
    a = -np.array(h_hh, dtype=np.complex128, copy=True)
    a.ravel()[:: a.shape[0] + 1] += float(e_ref)
    return a


def _solve_hermitian_shifted(h_hh: np.ndarray, rhs: np.ndarray, e_ref: float) -> np.ndarray:
    a = _shifted_hamiltonian_matrix(h_hh, e_ref)
    with _bounded_blas_threads():
        return np.linalg.solve(a, rhs)


def _pole_diagnostics(
    e_ref: float,
    h_hh: np.ndarray,
    *,
    pole_warning_mev: float,
    pole_danger_mev: float,
    fail_on_near_pole: bool,
    compute_pole_diagnostics: bool,
    compute_condition_number: bool,
) -> tuple[float | None, float | None, bool, bool, list[str]]:
    min_distance_mev: float | None = None
    near = False
    danger = False
    warnings: list[str] = []
    if compute_pole_diagnostics or fail_on_near_pole:
        h_hh = _hermitize(h_hh)
        with _bounded_blas_threads():
            high_eigs = np.linalg.eigvalsh(h_hh)
        min_distance_mev = float(np.min(np.abs(e_ref - high_eigs)) * MEV_PER_EV)
        near = min_distance_mev < pole_warning_mev
        danger = min_distance_mev < pole_danger_mev
        if danger:
            warnings.append(
                f"DANGER: min |E_ref - E_high| = {min_distance_mev:.3f} meV "
                f"is below {pole_danger_mev:.3f} meV"
            )
        elif near:
            warnings.append(
                f"WARNING: min |E_ref - E_high| = {min_distance_mev:.3f} meV "
                f"is below {pole_warning_mev:.3f} meV"
            )
        if fail_on_near_pole and near:
            raise NearPoleError(warnings[-1])

    cond = None
    if compute_condition_number:
        a = _shifted_hamiltonian_matrix(h_hh, e_ref)
        with _bounded_blas_threads():
            cond = float(np.linalg.cond(a))
    return min_distance_mev, cond, near, danger, warnings


def _projected_hamiltonian(
    ham: np.ndarray,
    u_left: np.ndarray,
    u_right: np.ndarray | None = None,
    *,
    chunk_cols: int = 512,
    chunk_rows: int = 512,
) -> np.ndarray:
    """Compute U_left^dagger H U_right with bounded GEMM workspace."""
    if u_right is None:
        u_right = u_left
    if chunk_cols <= 0:
        raise ValueError(f"chunk_cols must be positive, got {chunk_cols}")
    if chunk_rows <= 0:
        raise ValueError(f"chunk_rows must be positive, got {chunk_rows}")

    ham = np.asarray(ham, dtype=np.complex128)
    u_left = np.asarray(u_left, dtype=np.complex128)
    u_right = np.asarray(u_right, dtype=np.complex128)
    hu = _left_multiply_hamiltonian(
        ham,
        u_right,
        chunk_cols=chunk_cols,
        chunk_rows=chunk_rows,
    )
    with _bounded_blas_threads():
        return u_left.conj().T @ hu


def _left_multiply_hamiltonian(
    ham: np.ndarray,
    basis: np.ndarray,
    *,
    chunk_cols: int = 512,
    chunk_rows: int = 512,
) -> np.ndarray:
    ham = np.asarray(ham, dtype=np.complex128)
    basis = np.asarray(basis, dtype=np.complex128)
    if ham.ndim != 2 or ham.shape[0] != ham.shape[1]:
        raise ValueError(f"ham must be a square matrix, got shape={ham.shape}")
    if basis.ndim != 2 or basis.shape[0] != ham.shape[1]:
        raise ValueError(f"basis shape {basis.shape} is incompatible with ham shape {ham.shape}")
    out = np.empty((ham.shape[0], basis.shape[1]), dtype=np.complex128)
    with _bounded_blas_threads():
        for col_start in range(0, basis.shape[1], chunk_cols):
            col_stop = min(col_start + chunk_cols, basis.shape[1])
            basis_block = basis[:, col_start:col_stop]
            for row_start in range(0, ham.shape[0], chunk_rows):
                row_stop = min(row_start + chunk_rows, ham.shape[0])
                out[row_start:row_stop, col_start:col_stop] = (
                    ham[row_start:row_stop, :] @ basis_block
                )
    return out


def _build_projector_groups(basis: np.ndarray) -> _ProjectorGroups:
    """Group projector columns that have identical nonzero row support."""
    basis = np.asarray(basis, dtype=np.complex128)
    if basis.ndim != 2:
        raise ValueError(f"basis must be a matrix, got shape={basis.shape}")

    by_rows: dict[tuple[int, ...], list[int]] = {}
    nz_cols, nz_rows = np.nonzero(basis.T)
    if nz_cols.size:
        starts = np.r_[0, np.flatnonzero(np.diff(nz_cols)) + 1]
        stops = np.r_[starts[1:], nz_cols.size]
        for start, stop in zip(starts, stops):
            col = int(nz_cols[start])
            rows = tuple(int(row) for row in nz_rows[start:stop])
            by_rows.setdefault(rows, []).append(col)

    groups: list[_ProjectorGroup] = []
    for row_tuple, col_list in by_rows.items():
        if not row_tuple:
            continue
        rows = np.array(row_tuple, dtype=int)
        cols = np.array(col_list, dtype=int)
        local = np.ascontiguousarray(basis[np.ix_(rows, cols)], dtype=np.complex128)
        groups.append(_ProjectorGroup(rows=rows, cols=cols, local=local, local_h=local.conj().T))
    return _ProjectorGroups(n_columns=int(basis.shape[1]), groups=tuple(groups))


def _projector_column_groups(basis: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    return [(group.rows, group.cols, group.local) for group in _build_projector_groups(basis).groups]


def projector_groups_from_block_columns(
    block_indices: Sequence[np.ndarray],
    local_columns: Sequence[np.ndarray],
    column_indices: Sequence[np.ndarray],
    *,
    n_columns: int,
) -> _ProjectorGroups:
    groups: list[_ProjectorGroup] = []
    for rows_raw, local_raw, cols_raw in zip(block_indices, local_columns, column_indices):
        rows = np.asarray(rows_raw, dtype=int)
        cols = np.asarray(cols_raw, dtype=int)
        local = np.ascontiguousarray(local_raw, dtype=np.complex128)
        if cols.size == 0:
            continue
        if rows.ndim != 1 or cols.ndim != 1:
            raise ValueError("projector group rows and columns must be one-dimensional")
        if local.shape != (rows.size, cols.size):
            raise ValueError(
                f"projector local block shape {local.shape} does not match "
                f"rows={rows.size}, cols={cols.size}"
            )
        groups.append(_ProjectorGroup(rows=rows, cols=cols, local=local, local_h=local.conj().T))
    return _ProjectorGroups(n_columns=int(n_columns), groups=tuple(groups))


def _grouped_projected_hamiltonian_from_groups(
    ham: np.ndarray,
    left: _ProjectorGroups,
    right: _ProjectorGroups | None = None,
    *,
    assume_hermitian: bool = False,
) -> np.ndarray:
    if right is None:
        right = left
    out = np.zeros((left.n_columns, right.n_columns), dtype=np.complex128)
    with _bounded_blas_threads():
        if assume_hermitian and right is left:
            for left_pos, left_group in enumerate(left.groups):
                for right_pos in range(left_pos, len(left.groups)):
                    right_group = left.groups[right_pos]
                    block = (
                        left_group.local_h
                        @ ham[np.ix_(left_group.rows, right_group.rows)]
                        @ right_group.local
                    )
                    if right_pos == left_pos:
                        block = _hermitize(block)
                    out[np.ix_(left_group.cols, right_group.cols)] = block
                    if right_pos != left_pos:
                        out[np.ix_(right_group.cols, left_group.cols)] = block.conj().T
            return out

        for left_group in left.groups:
            for right_group in right.groups:
                out[np.ix_(left_group.cols, right_group.cols)] = (
                    left_group.local_h
                    @ ham[np.ix_(left_group.rows, right_group.rows)]
                    @ right_group.local
                )
    return out


def _grouped_projected_hamiltonian(
    ham: np.ndarray,
    u_left: np.ndarray,
    u_right: np.ndarray | None = None,
) -> np.ndarray:
    """Compute U_left^dagger H U_right using block-sparse projector support."""
    if u_right is None:
        ham = np.asarray(ham, dtype=np.complex128)
        u_left = np.asarray(u_left, dtype=np.complex128)
        groups = _build_projector_groups(u_left)
        return _grouped_projected_hamiltonian_from_groups(ham, groups)

    ham = np.asarray(ham, dtype=np.complex128)
    u_left = np.asarray(u_left, dtype=np.complex128)
    u_right = np.asarray(u_right, dtype=np.complex128)
    left_groups = _build_projector_groups(u_left)
    right_groups = left_groups if u_right is u_left else _build_projector_groups(u_right)
    return _grouped_projected_hamiltonian_from_groups(
        ham,
        left_groups,
        right_groups,
    )


def _matching_projector_groups(left: _ProjectorGroups, right: _ProjectorGroups) -> bool:
    if len(left.groups) != len(right.groups):
        return False
    return all(
        np.array_equal(left_group.rows, right_group.rows)
        for left_group, right_group in zip(left.groups, right.groups)
    )


def _downfold_blocks_from_matching_projector_groups(
    ham: np.ndarray,
    low: _ProjectorGroups,
    high: _ProjectorGroups,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h_pp = np.zeros((low.n_columns, low.n_columns), dtype=np.complex128)
    h_ph = np.zeros((low.n_columns, high.n_columns), dtype=np.complex128)
    h_hh = np.zeros((high.n_columns, high.n_columns), dtype=np.complex128)

    with _bounded_blas_threads():
        for left_pos, (low_left, high_left) in enumerate(zip(low.groups, high.groups)):
            left_h = np.ascontiguousarray(
                np.vstack((low_left.local_h, high_left.local_h)),
                dtype=np.complex128,
            )
            low_left_n = low_left.cols.size
            for right_pos in range(left_pos, len(low.groups)):
                low_right = low.groups[right_pos]
                high_right = high.groups[right_pos]
                right_basis = np.ascontiguousarray(
                    np.hstack((low_right.local, high_right.local)),
                    dtype=np.complex128,
                )
                low_right_n = low_right.cols.size
                projected = (
                    left_h
                    @ ham[np.ix_(low_left.rows, low_right.rows)]
                    @ right_basis
                )

                ll = projected[:low_left_n, :low_right_n]
                lh = projected[:low_left_n, low_right_n:]
                hl = projected[low_left_n:, :low_right_n]
                hh = projected[low_left_n:, low_right_n:]

                if right_pos == left_pos:
                    ll = _hermitize(ll)
                    hh = _hermitize(hh)
                h_pp[np.ix_(low_left.cols, low_right.cols)] = ll
                h_ph[np.ix_(low_left.cols, high_right.cols)] = lh
                h_hh[np.ix_(high_left.cols, high_right.cols)] = hh
                if right_pos != left_pos:
                    h_pp[np.ix_(low_right.cols, low_left.cols)] = ll.conj().T
                    h_ph[np.ix_(low_right.cols, high_left.cols)] = hl.conj().T
                    h_hh[np.ix_(high_right.cols, high_left.cols)] = hh.conj().T

    return h_pp, h_ph, h_hh


def downfold_blocks(
    h_pp: np.ndarray,
    h_ph: np.ndarray | None,
    h_hh: np.ndarray | None,
    options: DownfoldingOptions,
    *,
    assume_hermitian_blocks: bool = False,
) -> DownfoldingResult:
    """Build a static effective Hamiltonian from P/H block matrices.

    Supported methods:
    - first_order: H_eff = H_PP.
    - fixed_schur: H_eff(E_ref) = H_PP + H_PH (E_ref I - H_HH)^(-1) H_HP.
    - linearized_lowdin: linearize Sigma(E) around E_ref and solve A psi = E B psi.

    Energies are in eV. Diagnostics report pole distances in meV.
    """
    method = options.method.lower()
    h_pp = np.asarray(h_pp, dtype=np.complex128) if assume_hermitian_blocks else _hermitize(h_pp)

    if method == "first_order":
        heff = _hermitize(h_pp)
        return DownfoldingResult(
            heff=heff,
            method=method,
            e_ref=options.e_ref,
            hermiticity_residual=_hermiticity_residual(heff),
        )

    if h_ph is None or h_hh is None:
        raise ValueError(f"method={method!r} requires h_ph and h_hh")
    if options.e_ref is None:
        raise ValueError(f"method={method!r} requires configurable e_ref")

    h_ph = np.asarray(h_ph, dtype=np.complex128)
    h_hh = np.asarray(h_hh, dtype=np.complex128) if assume_hermitian_blocks else _hermitize(h_hh)
    h_hp = h_ph.conj().T
    e_ref = float(options.e_ref)
    pole_min, pole_cond, near, danger, warnings = _pole_diagnostics(
        e_ref,
        h_hh,
        pole_warning_mev=float(options.pole_warning_mev),
        pole_danger_mev=float(options.pole_danger_mev),
        fail_on_near_pole=bool(options.fail_on_near_pole),
        compute_pole_diagnostics=bool(options.compute_pole_diagnostics),
        compute_condition_number=bool(options.compute_condition_number),
    )
    if method == "fixed_schur":
        y = _solve_hermitian_shifted(h_hh, h_hp, e_ref)
        with _bounded_blas_threads():
            heff = _hermitize(h_pp + h_ph @ y)
        return DownfoldingResult(
            heff=heff,
            method=method,
            e_ref=e_ref,
            pole_distance_min_mev=pole_min,
            pole_condition_number=pole_cond,
            near_pole=near,
            danger_pole=danger,
            hermiticity_residual=_hermiticity_residual(heff),
            warnings=warnings,
        )

    if method == "linearized_lowdin":
        a = _shifted_hamiltonian_matrix(h_hh, e_ref)
        with _bounded_blas_threads():
            lu, piv = scipy.linalg.lu_factor(a, check_finite=False, overwrite_a=True)
            y = scipy.linalg.lu_solve((lu, piv), h_hp, check_finite=False)
            sigma0 = h_ph @ y
        # d/dE (E I - H_HH)^(-1) = - (E I - H_HH)^(-2).
        # Therefore Sigma1 = dSigma/dE = -H_PH A^{-2} H_HP.
        with _bounded_blas_threads():
            y2 = scipy.linalg.lu_solve((lu, piv), y, check_finite=False)
            sigma1 = -(h_ph @ y2)
        big_a = _hermitize(h_pp + sigma0 - e_ref * sigma1)
        big_b = _hermitize(np.eye(h_pp.shape[0], dtype=np.complex128) - sigma1)
        with _bounded_blas_threads():
            b_eigs, b_vecs = np.linalg.eigh(big_b)
        b_min = float(np.min(b_eigs))
        if b_min <= 0.0:
            msg = f"linearized_lowdin B is not positive definite; min eig={b_min:.6e}"
            if options.fail_on_near_pole:
                raise ValueError(msg)
            warnings.append(msg)
        b_cond = float(np.max(np.abs(b_eigs)) / max(np.min(np.abs(b_eigs)), np.finfo(float).eps))
        if b_min <= 0.0:
            # Fall back to generalized eigensolver for diagnostics, but the returned
            # static matrix is intentionally not produced for non-positive B.
            raise ValueError(warnings[-1])
        with _bounded_blas_threads():
            inv_sqrt = (b_vecs * (1.0 / np.sqrt(b_eigs))) @ b_vecs.conj().T
            heff = _hermitize(inv_sqrt @ big_a @ inv_sqrt)
        return DownfoldingResult(
            heff=heff,
            method=method,
            e_ref=e_ref,
            pole_distance_min_mev=pole_min,
            pole_condition_number=pole_cond,
            near_pole=near,
            danger_pole=danger,
            hermiticity_residual=_hermiticity_residual(heff),
            b_min_eig=b_min,
            b_condition_number=b_cond,
            warnings=warnings,
        )

    if method == "qdpt2":
        raise ValueError("downfolding method 'qdpt2' is not available in this release; use fixed_schur or linearized_lowdin")

    raise ValueError(f"Unknown downfolding method: {options.method!r}")


def downfold_from_projectors(
    hamk_full: np.ndarray,
    u_low: np.ndarray,
    u_high: np.ndarray | None,
    options: DownfoldingOptions,
) -> DownfoldingResult:
    ham = np.asarray(hamk_full, dtype=np.complex128)
    u_low = np.asarray(u_low, dtype=np.complex128)
    low_groups = _build_projector_groups(u_low)
    if options.method.lower() == "first_order":
        h_pp = _grouped_projected_hamiltonian_from_groups(ham, low_groups, assume_hermitian=True)
        return downfold_blocks(h_pp, None, None, options, assume_hermitian_blocks=True)
    if u_high is None:
        raise ValueError(f"method={options.method!r} requires high-space projectors")
    u_high = np.asarray(u_high, dtype=np.complex128)
    high_groups = _build_projector_groups(u_high)
    h_pp = _grouped_projected_hamiltonian_from_groups(ham, low_groups, assume_hermitian=True)
    h_ph = _grouped_projected_hamiltonian_from_groups(ham, low_groups, high_groups)
    h_hh = _grouped_projected_hamiltonian_from_groups(ham, high_groups, assume_hermitian=True)
    return downfold_blocks(h_pp, h_ph, h_hh, options, assume_hermitian_blocks=True)


def downfold_from_projector_groups(
    hamk_full: np.ndarray,
    low_groups: _ProjectorGroups,
    high_groups: _ProjectorGroups | None,
    options: DownfoldingOptions,
) -> DownfoldingResult:
    ham = np.asarray(hamk_full, dtype=np.complex128)
    if options.method.lower() == "first_order":
        h_pp = _grouped_projected_hamiltonian_from_groups(ham, low_groups, assume_hermitian=True)
        return downfold_blocks(h_pp, None, None, options, assume_hermitian_blocks=True)
    if high_groups is None:
        raise ValueError(f"method={options.method!r} requires high-space projectors")
    if _matching_projector_groups(low_groups, high_groups):
        h_pp, h_ph, h_hh = _downfold_blocks_from_matching_projector_groups(
            ham,
            low_groups,
            high_groups,
        )
    else:
        h_pp = _grouped_projected_hamiltonian_from_groups(ham, low_groups, assume_hermitian=True)
        h_ph = _grouped_projected_hamiltonian_from_groups(ham, low_groups, high_groups)
        h_hh = _grouped_projected_hamiltonian_from_groups(ham, high_groups, assume_hermitian=True)
    return downfold_blocks(h_pp, h_ph, h_hh, options, assume_hermitian_blocks=True)


# def parse_int_list(value: str | Sequence[int] | None) -> list[int]:
#     if value is None:
#         return []
#     if isinstance(value, str):
#         return [int(part.strip()) for part in value.split(",") if part.strip()]
#     return [int(v) for v in value]
#
#
# def parse_float_list(value: str | Sequence[float] | None) -> list[float]:
#     if value is None:
#         return []
#     if isinstance(value, str):
#         return [float(part.strip()) for part in value.split(",") if part.strip()]
#     return [float(v) for v in value]
#
#
# def compute_top_n_metrics(
#     reference_eigs: Sequence[np.ndarray],
#     effective_eigs: Sequence[np.ndarray],
#     top_n_list: Iterable[int],
# ) -> dict[int, TopNMetric]:
#     if len(reference_eigs) != len(effective_eigs):
#         raise ValueError(
#             f"reference rows ({len(reference_eigs)}) and effective rows ({len(effective_eigs)}) differ"
#         )
#     metrics: dict[int, TopNMetric] = {}
#     for n in sorted({int(x) for x in top_n_list}):
#         if n <= 0:
#             raise ValueError(f"top-N values must be positive, got {n}")
#         errors: list[np.ndarray] = []
#         for ref, eff in zip(reference_eigs, effective_eigs):
#             ref_sorted = np.sort(np.asarray(ref, dtype=float).ravel())
#             eff_sorted = np.sort(np.asarray(eff, dtype=float).ravel())
#             if ref_sorted.size < n or eff_sorted.size < n:
#                 raise ValueError(
#                     f"Cannot compare top {n}: reference has {ref_sorted.size}, effective has {eff_sorted.size}"
#                 )
#             errors.append(eff_sorted[-n:] - ref_sorted[-n:])
#         err = np.concatenate(errors) * MEV_PER_EV
#         metrics[n] = TopNMetric(
#             n=n,
#             rms_mev=float(np.sqrt(np.mean(err * err))),
#             max_abs_mev=float(np.max(np.abs(err))),
#             mean_mev=float(np.mean(err)),
#         )
#     return metrics
#
#
# def print_top_n_metrics(metrics: Mapping[int, TopNMetric], *, prefix: str = "[kp]") -> None:
#     if not metrics:
#         return
#     print(f"{prefix} Top-band accuracy (effective - reference, meV):")
#     for n in sorted(metrics):
#         m = metrics[n]
#         marker = " main" if n in {2, 4, 6} else ""
#         print(
#             f"{prefix}   top {n:>2}{marker}: "
#             f"RMS={m.rms_mev:.3f}, max={m.max_abs_mev:.3f}, mean={m.mean_mev:.3f}"
#         )
#
#
# def write_sweep_csv(rows: Sequence[SweepRow], path: str | Path) -> None:
#     path = Path(path)
#     path.parent.mkdir(parents=True, exist_ok=True)
#     all_top_n = sorted({n for row in rows for n in row.metrics})
#     fieldnames = ["e_ref", "global_pole_distance_min_mev"]
#     for n in all_top_n:
#         fieldnames.extend([f"top{n}_rms_mev", f"top{n}_max_abs_mev", f"top{n}_mean_mev"])
#     with path.open("w", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writeheader()
#         for row in rows:
#             item: dict[str, float | str] = {
#                 "e_ref": row.e_ref,
#                 "global_pole_distance_min_mev": (
#                     "" if row.global_pole_distance_min_mev is None else row.global_pole_distance_min_mev
#                 ),
#             }
#             for n in all_top_n:
#                 metric = row.metrics.get(n)
#                 if metric is None:
#                     item[f"top{n}_rms_mev"] = ""
#                     item[f"top{n}_max_abs_mev"] = ""
#                     item[f"top{n}_mean_mev"] = ""
#                 else:
#                     item[f"top{n}_rms_mev"] = metric.rms_mev
#                     item[f"top{n}_max_abs_mev"] = metric.max_abs_mev
#                     item[f"top{n}_mean_mev"] = metric.mean_mev
#             writer.writerow(item)
#
#
# def write_sweep_json(rows: Sequence[SweepRow], path: str | Path) -> None:
#     path = Path(path)
#     path.parent.mkdir(parents=True, exist_ok=True)
#     payload = [
#         {
#             "e_ref": row.e_ref,
#             "global_pole_distance_min_mev": row.global_pole_distance_min_mev,
#             "metrics": {
#                 str(n): {
#                     "rms_mev": m.rms_mev,
#                     "max_abs_mev": m.max_abs_mev,
#                     "mean_mev": m.mean_mev,
#                 }
#                 for n, m in sorted(row.metrics.items())
#             },
#         }
#         for row in rows
#     ]
#     path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
