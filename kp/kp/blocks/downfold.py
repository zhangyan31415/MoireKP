from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


MEV_PER_EV = 1000.0


class NearPoleError(RuntimeError):
    """Raised when a downfolding denominator is too close to a discarded pole."""


@dataclass(frozen=True)
class DownfoldingOptions:
    method: str = "fixed_schur"
    e_ref: float | None = None
    pole_warning_mev: float = 10.0
    pole_danger_mev: float = 1.0
    fail_on_near_pole: bool = False


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


def _pole_diagnostics(
    e_ref: float,
    h_hh: np.ndarray,
    *,
    pole_warning_mev: float,
    pole_danger_mev: float,
    fail_on_near_pole: bool,
) -> tuple[float, float, bool, bool, list[str]]:
    h_hh = _hermitize(h_hh)
    high_eigs = np.linalg.eigvalsh(h_hh)
    min_distance_mev = float(np.min(np.abs(e_ref - high_eigs)) * MEV_PER_EV)
    a = e_ref * np.eye(h_hh.shape[0], dtype=np.complex128) - h_hh
    cond = float(np.linalg.cond(a))
    near = min_distance_mev < pole_warning_mev
    danger = min_distance_mev < pole_danger_mev
    warnings: list[str] = []
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
    return min_distance_mev, cond, near, danger, warnings


def downfold_blocks(
    h_pp: np.ndarray,
    h_ph: np.ndarray | None,
    h_hh: np.ndarray | None,
    options: DownfoldingOptions,
) -> DownfoldingResult:
    """Build a static effective Hamiltonian from P/H block matrices.

    Supported methods:
    - first_order: H_eff = H_PP.
    - fixed_schur: H_eff(E_ref) = H_PP + H_PH (E_ref I - H_HH)^(-1) H_HP.
    - linearized_lowdin: linearize Sigma(E) around E_ref and solve A psi = E B psi.

    Energies are in eV. Diagnostics report pole distances in meV.
    """
    method = options.method.lower()
    h_pp = _hermitize(h_pp)

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
    h_hh = _hermitize(h_hh)
    h_hp = h_ph.conj().T
    e_ref = float(options.e_ref)
    pole_min, pole_cond, near, danger, warnings = _pole_diagnostics(
        e_ref,
        h_hh,
        pole_warning_mev=float(options.pole_warning_mev),
        pole_danger_mev=float(options.pole_danger_mev),
        fail_on_near_pole=bool(options.fail_on_near_pole),
    )
    a = e_ref * np.eye(h_hh.shape[0], dtype=np.complex128) - h_hh

    if method == "fixed_schur":
        y = np.linalg.solve(a, h_hp)
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
        y = np.linalg.solve(a, h_hp)
        sigma0 = h_ph @ y
        # d/dE (E I - H_HH)^(-1) = - (E I - H_HH)^(-2).
        # Therefore Sigma1 = dSigma/dE = -H_PH A^{-2} H_HP.
        y2 = np.linalg.solve(a, y)
        sigma1 = -(h_ph @ y2)
        big_a = _hermitize(h_pp + sigma0 - e_ref * sigma1)
        big_b = _hermitize(np.eye(h_pp.shape[0], dtype=np.complex128) - sigma1)
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
    h_pp = u_low.conj().T @ ham @ u_low
    if options.method.lower() == "first_order":
        return downfold_blocks(h_pp, None, None, options)
    if u_high is None:
        raise ValueError(f"method={options.method!r} requires high-space projectors")
    u_high = np.asarray(u_high, dtype=np.complex128)
    x = ham @ u_high
    h_hh = u_high.conj().T @ x
    h_ph = u_low.conj().T @ x
    return downfold_blocks(h_pp, h_ph, h_hh, options)


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
