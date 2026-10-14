from __future__ import annotations
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import scipy.linalg
import scipy.optimize
import scipy.sparse
from .band_degeneracy import (
    BandDegeneracyGuard,
    DEFAULT_DEGENERACY_TOL_EV,
    retain_coefficients_on_degeneracy,
)
from .core import (
    ContinuumModelBuilder,
    MoireConfig,
    _prepare_band_state,
    _reduce_and_hermitize_hamiltonian,
)
from . import configuration


def _refinement_tag_alias(tag: str) -> str:
    aliases = {
        "kinetic": "Kinect",
        "onsite": "Onsite",
        "intralayer": "intra",
        "interlayer": "inter",
        "tunneling": "inter",
        "moire": "intra",
    }
    return aliases.get(str(tag), str(tag))



def _refinement_band_slice(config: Mapping[str, Any], model_config: configuration.ConfiguredModel, dim: int) -> tuple[int, int]:
    if config.get("band_slice") is not None:
        raw = list(config["band_slice"])
        if len(raw) != 2:
            raise ValueError(f"fit.refine_bands.band_slice must be [start, stop], got {raw!r}")
        start, stop = int(raw[0]), int(raw[1])
    elif config.get("top_bands") is not None:
        count = int(config["top_bands"])
        start, stop = max(0, int(dim) - count), int(dim)
    elif model_config.band_plot_config.get("top_bands") is not None:
        count = int(model_config.band_plot_config["top_bands"])
        start, stop = max(0, int(dim) - count), int(dim)
    elif model_config.band_slice is not None:
        start, stop = int(model_config.band_slice[0]), int(model_config.band_slice[1])
    else:
        start, stop = 0, int(dim)
    if start < 0 or stop > dim or start >= stop:
        raise ValueError(f"Invalid band refinement slice [{start}, {stop}] for dimension {dim}")
    return start, stop



def _model_hamiltonians_for_kpoints(moire_config: MoireConfig, model: Any, kpoints: np.ndarray) -> np.ndarray:
    basis = getattr(model, "_compiled_response_basis", None)
    fitted = getattr(model, "_fitted_response_model", None)
    if basis is None or fitted is None:
        raise ValueError("Hamiltonian evaluation requires a fitted complete response basis")
    from .response_basis import CompiledResponseRuntime

    runtime = CompiledResponseRuntime(basis=basis, fitted=fitted)
    dim_full = int(basis.dim)
    keep = (
        np.arange(dim_full, dtype=int)
        if moire_config.keep_indices is None
        else np.asarray(moire_config.keep_indices, dtype=int)
    )
    remove = (
        np.asarray(moire_config.remove_indices, dtype=int)
        if moire_config.remove_indices is not None
        else np.array([], dtype=int)
    )
    reduction_state = type("_ResponseReductionState", (), {"keep": keep, "remove": remove})()
    return np.asarray(
        [
            _reduce_and_hermitize_hamiltonian(hamiltonian, reduction_state)
            for hamiltonian in runtime.hamiltonians(np.asarray(kpoints, dtype=float))
        ],
        dtype=np.complex128,
    )



def _term_component_hamiltonians_for_kpoints(
    moire_config: MoireConfig,
    state: Any,
    term: Any,
    component: str,
    kpoints: np.ndarray,
) -> np.ndarray:
    """Return the linear Hamiltonian response for one coefficient component."""
    remove = np.asarray(getattr(state, "remove", np.array([], dtype=int)), dtype=int)
    if remove.size:
        raise NotImplementedError("direct term response is only linear when no Schur remove subspace is active")
    keep = np.asarray(getattr(state, "keep", np.array([], dtype=int)), dtype=int)
    dim_full = int(getattr(state, "dim_full"))
    real = 1.0 if component == "real" else 0.0
    imag = 1.0 if component == "imag" else 0.0
    if component not in {"real", "imag"}:
        raise ValueError(f"unknown coefficient component {component!r}")
    hamiltonians = []
    for kval in np.asarray(kpoints, dtype=float):
        h_full = np.zeros((dim_full, dim_full), dtype=np.complex128)
        ContinuumModelBuilder.add_symmetrized_term_to_matrix_static(
            h_full,
            getattr(term, "Y_basis"),
            kval,
            getattr(term, "symmetry_ops", []),
            real,
            imag,
            symmetry_gen=getattr(state, "symmetry_gen", None),
            term=term,
        )
        h = h_full[np.ix_(keep, keep)] if keep.size != dim_full else h_full
        hamiltonians.append(0.5 * (h + h.conj().T))
    return np.asarray(hamiltonians)



def _term_response_pair_hamiltonians_for_kpoints(
    state: Any,
    term: Any,
    kpoints: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return real/imag coefficient responses while symmetrizing each term once."""
    remove = np.asarray(getattr(state, "remove", np.array([], dtype=int)), dtype=int)
    if remove.size:
        raise NotImplementedError("direct term response is only linear when no Schur remove subspace is active")
    keep = np.asarray(getattr(state, "keep", np.array([], dtype=int)), dtype=int)
    dim_full = int(getattr(state, "dim_full"))
    real_list: list[np.ndarray] = []
    imag_list: list[np.ndarray] = []
    y_basis = getattr(term, "Y_basis")
    sym_ops = getattr(term, "symmetry_ops", [])
    symmetry_gen = getattr(state, "symmetry_gen", None)
    sparse_available = (
        ContinuumModelBuilder._SYMM_USE_SPARSE_BASIS
        and hasattr(y_basis, "eval_sparse")
        and symmetry_gen is not None
    )
    k_array = np.asarray(kpoints, dtype=float)
    if sparse_available and k_array.size and (
        not hasattr(term, "_moire_needs_hermitize_real")
        or not hasattr(term, "_moire_needs_hermitize_imag")
    ):
        ContinuumModelBuilder.symmetrize_Y_and_iY_basis_static(
            y_basis,
            k_array[0],
            sym_ops,
            symmetry_gen=symmetry_gen,
            term=term,
            use_cache=False,
        )
    needs_herm_real = bool(getattr(term, "_moire_needs_hermitize_real", False))
    needs_herm_imag = bool(getattr(term, "_moire_needs_hermitize_imag", False))
    has_herm_flags = hasattr(term, "_moire_needs_hermitize_real") and hasattr(term, "_moire_needs_hermitize_imag")
    inconsistent = bool(getattr(term, "_moire_hermitize_flags_inconsistent", False))
    use_sparse_direct = (
        sparse_available
        and has_herm_flags
        and not inconsistent
    )
    for kval in k_array:
        if use_sparse_direct:
            h_real = np.zeros((dim_full, dim_full), dtype=np.complex128)
            h_imag = np.zeros((dim_full, dim_full), dtype=np.complex128)
            orbit_actions = (
                ContinuumModelBuilder._get_symmetry_orbit_actions_cached(kval, sym_ops, symmetry_gen)
                if sym_ops
                else [(kval, None, tuple(), False)]
            )
            for kk, action, op_seq_applied, is_anti in orbit_actions:
                if op_seq_applied != tuple():
                    use_sparse_direct = False
                    break
                use_add_at = not getattr(y_basis, "_moire_sparse_unique", True)
                rows, cols, vals0 = y_basis.eval_sparse(kk)
                if action is not None:
                    _perm, inv_perm, vals, inv_vals, is_anti_total = action
                    rr = inv_perm[rows]
                    cc = inv_perm[cols]
                    vv = np.conjugate(vals0) if is_anti_total else vals0
                    vv = vv * vals[rr] * inv_vals[cc]
                else:
                    rr, cc, vv = rows, cols, vals0
                    is_anti_total = bool(is_anti)
                support = ContinuumModelBuilder.term_output_support_mask(term, rr, cc, dim_full)
                if not np.all(support):
                    rr = rr[support]
                    cc = cc[support]
                    vv = vv[support]
                sign = -1.0 if is_anti_total else 1.0
                if use_add_at:
                    np.add.at(h_real, (rr, cc), vv)
                    np.add.at(h_imag, (rr, cc), 1j * sign * vv)
                else:
                    h_real[rr, cc] += vv
                    h_imag[rr, cc] += 1j * sign * vv
                if needs_herm_real:
                    vv_h = np.conjugate(vv)
                    if use_add_at:
                        np.add.at(h_real, (cc, rr), vv_h)
                    else:
                        h_real[cc, rr] += vv_h
                if needs_herm_imag:
                    vv_h = np.conjugate(vv)
                    if use_add_at:
                        np.add.at(h_imag, (cc, rr), -1j * sign * vv_h)
                    else:
                        h_imag[cc, rr] += -1j * sign * vv_h
            else:
                if keep.size != dim_full:
                    h_real = h_real[np.ix_(keep, keep)]
                    h_imag = h_imag[np.ix_(keep, keep)]
                real_list.append(0.5 * (h_real + h_real.conj().T))
                imag_list.append(0.5 * (h_imag + h_imag.conj().T))
                continue
        h_real, h_imag = ContinuumModelBuilder.symmetrize_Y_and_iY_basis_static(
            y_basis,
            kval,
            sym_ops,
            symmetry_gen=symmetry_gen,
            term=term,
            use_cache=False,
        )
        if keep.size != dim_full:
            h_real = h_real[np.ix_(keep, keep)]
            h_imag = h_imag[np.ix_(keep, keep)]
        real_list.append(0.5 * (h_real + h_real.conj().T))
        imag_list.append(0.5 * (h_imag + h_imag.conj().T))
    return np.asarray(real_list), np.asarray(imag_list)



def _select_band_refinement_variables(model: Any, config: Mapping[str, Any]) -> list[tuple[Any, str]]:
    raw_tags = config.get("variable_tags", config.get("tags", ["Kinect", "Onsite", "inter"]))
    if isinstance(raw_tags, str):
        raw_tags = [raw_tags]
    tags = {_refinement_tag_alias(str(tag)) for tag in raw_tags}
    raw_components = config.get("components", ["real"])
    if isinstance(raw_components, str):
        raw_components = [raw_components]
    components = {str(item).strip().lower() for item in raw_components}
    if not components <= {"real", "imag"}:
        raise ValueError(f"fit.refine_bands.components must contain only 'real'/'imag', got {sorted(components)!r}")
    variables: list[tuple[Any, str]] = []
    for term in getattr(model, "terms", {}).values():
        if not getattr(term, "active", True):
            continue
        if str(getattr(term, "tag", "")) not in tags:
            continue
        if "real" in components:
            variables.append((term, "real"))
        if "imag" in components:
            variables.append((term, "imag"))
    return variables



def _term_component_value(term: Any, component: str) -> float:
    if component == "real":
        return float(getattr(term, "r_value_real", 0.0))
    if component == "imag":
        return float(getattr(term, "r_value_imag", 0.0))
    raise ValueError(f"unknown coefficient component {component!r}")



def _set_term_component_value(term: Any, component: str, value: float) -> None:
    if component == "real":
        term.r_value_real = float(value)
        return
    if component == "imag":
        term.r_value_imag = float(value)
        return
    raise ValueError(f"unknown coefficient component {component!r}")



def _band_refinement_metrics(
    hamiltonians: np.ndarray,
    target_eig: np.ndarray,
    target_h: np.ndarray,
    *,
    band_slice: tuple[int, int],
    align: str,
    eigenvalues: np.ndarray | None = None,
) -> dict[str, float]:
    eig = (
        np.linalg.eigvalsh(hamiltonians)
        if eigenvalues is None
        else np.asarray(eigenvalues, dtype=float)
    )
    if eig.shape != target_eig.shape:
        raise ValueError(
            f"model eigenvalues have shape {eig.shape}, expected {target_eig.shape}"
        )
    start, stop = band_slice
    model_sel = eig[:, start:stop]
    target_sel = target_eig[:, start:stop]
    shift = 0.0
    if align == "top":
        shift = float(target_sel[0, -1] - model_sel[0, -1])
        model_sel = model_sel + shift
    elif align == "bottom":
        shift = float(target_sel[0, 0] - model_sel[0, 0])
        model_sel = model_sel + shift
    elif align not in {"none", ""}:
        raise ValueError(f"fit.refine_bands.align currently supports 'top', 'bottom', or 'none', got {align!r}")
    diff = model_sel - target_sel
    all_diff = eig - target_eig
    matrix_diff = hamiltonians - target_h
    return {
        "top_band_rms_mev": float(np.sqrt(np.mean(diff**2)) * 1000.0),
        "top_band_max_mev": float(np.max(np.abs(diff)) * 1000.0),
        "all_band_rms_mev": float(np.sqrt(np.mean(all_diff**2)) * 1000.0),
        "all_band_max_mev": float(np.max(np.abs(all_diff)) * 1000.0),
        "matrix_element_rms_mev": float(np.sqrt(np.mean(np.abs(matrix_diff) ** 2)) * 1000.0),
        "matrix_element_max_mev": float(np.max(np.abs(matrix_diff)) * 1000.0),
        "alignment_shift_mev": float(shift * 1000.0),
    }



def _band_refinement_band_residual(
    model_selected: np.ndarray,
    target_selected: np.ndarray,
    *,
    band_sigma: float,
    normalize: bool,
) -> np.ndarray:
    residual = (np.asarray(model_selected, dtype=float) - np.asarray(target_selected, dtype=float)) / float(band_sigma)
    if normalize and residual.size:
        residual = residual / np.sqrt(float(residual.size))
    return residual



def _band_refinement_eigenvalue_jacobian(
    eigvecs: np.ndarray,
    basis: np.ndarray,
    *,
    band_slice: tuple[int, int],
    align: str,
    band_sigma: float,
    normalize: bool,
    edge_band_weights: np.ndarray | None = None,
    band_weights: np.ndarray | None = None,
    point_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Analytic Jacobian of the selected eigenvalue residual.

    For a real coefficient y_j and Hermitian response B_j(k),
    d lambda_n(k) / d y_j = <u_n(k)|B_j(k)|u_n(k)>.
    """
    vec = np.asarray(eigvecs, dtype=np.complex128)
    responses = np.asarray(basis, dtype=np.complex128)
    if vec.ndim != 3:
        raise ValueError(f"eigvecs must have shape (n_k, dim, dim), got {vec.shape}")
    if responses.ndim != 4:
        raise ValueError(f"basis must have shape (n_variables, n_k, dim, dim), got {responses.shape}")
    if responses.shape[1:] != (vec.shape[0], vec.shape[1], vec.shape[1]):
        raise ValueError(
            "basis/eigvec shape mismatch: "
            f"basis={responses.shape}, eigvecs={vec.shape}"
        )
    start, stop = band_slice
    selected_vec = vec[:, :, start:stop]
    selected = np.einsum("kdn,jkde,ken->knj", selected_vec.conj(), responses, selected_vec, optimize=True).real
    if align == "top":
        selected = selected - selected[0:1, -1:, :]
    elif align == "bottom":
        selected = selected - selected[0:1, 0:1, :]
    elif align not in {"none", ""}:
        raise ValueError(f"fit.refine_bands.align currently supports 'top', 'bottom', or 'none', got {align!r}")
    if point_mask is not None:
        selected = selected[np.asarray(point_mask, dtype=bool)]
    selected = selected / float(band_sigma)
    if normalize and selected.size:
        selected = selected / np.sqrt(float(selected.shape[0] * selected.shape[1]))
    if edge_band_weights is not None:
        weights = np.asarray(edge_band_weights, dtype=float)
        if weights.shape != (selected.shape[1],):
            raise ValueError(f"edge_band_weights must have shape ({selected.shape[1]},), got {weights.shape}")
        selected = selected * np.sqrt(weights.reshape(1, -1, 1))
    if band_weights is not None:
        weights = np.asarray(band_weights, dtype=float)
        if weights.shape != selected.shape[:2]:
            raise ValueError(f"band_weights must have shape {selected.shape[:2]}, got {weights.shape}")
        selected = selected * np.sqrt(weights[:, :, None])
    return selected.reshape(selected.shape[0] * selected.shape[1], selected.shape[2])



@dataclass
class _BandRefinementOptimizeResult:
    x: np.ndarray
    cost: float
    nfev: int
    njev: int
    status: int
    message: str



def _band_refinement_gauss_newton_available(
    *,
    analytic_jacobian_enabled: bool,
    matrix_jacobian: Any | None,
) -> bool:
    """Return whether the dense/reduced analytic Jacobian can use GN.

    A reduced matrix-loss block is optional: a band-only analytic Jacobian is
    already a valid dense Gauss--Newton system.  Only the explicit sparse
    matrix-loss Jacobian stays on SciPy's sparse trust-region path.
    """

    return bool(analytic_jacobian_enabled and matrix_jacobian is None)



def _solve_band_refinement_gauss_newton(
    residual_fn,
    jacobian_fn,
    y0: np.ndarray,
    *,
    max_nfev: int,
    xtol: float,
    ftol: float,
    gtol: float,
    initial_damping: float = 1.0e-6,
    max_line_search_steps: int = 8,
) -> _BandRefinementOptimizeResult:
    """Small dense damped Gauss-Newton solver for refinement coefficients."""
    y = np.asarray(y0, dtype=float).copy()
    max_eval = max(1, int(max_nfev))
    damping = max(float(initial_damping), 0.0)
    line_steps = max(1, int(max_line_search_steps))
    residual = np.asarray(residual_fn(y), dtype=float)
    cost = 0.5 * float(residual @ residual)
    nfev = 1
    njev = 0
    message = "max_nfev reached"
    status = 0
    for _iteration in range(max_eval):
        jac = np.asarray(jacobian_fn(y), dtype=float)
        njev += 1
        gradient = jac.T @ residual
        grad_norm = float(np.linalg.norm(gradient, ord=np.inf)) if gradient.size else 0.0
        if grad_norm <= float(gtol):
            status = 1
            message = "gtol satisfied"
            break
        normal = jac.T @ jac
        diag = np.maximum(np.diag(normal), 1.0)
        local_damping = damping
        accepted = False
        step = np.zeros_like(y)
        trial_cost = cost
        trial_residual = residual
        for _damping_round in range(8):
            lhs = normal + local_damping * np.diag(diag)
            try:
                step = scipy.linalg.solve(lhs, -gradient, assume_a="pos", check_finite=False)
            except (scipy.linalg.LinAlgError, ValueError):
                step = scipy.linalg.lstsq(lhs, -gradient, check_finite=False)[0]
            if not np.all(np.isfinite(step)):
                local_damping = max(local_damping * 10.0, 1.0e-12)
                continue
            for line_index in range(line_steps):
                alpha = 0.5**line_index
                trial_y = y + alpha * step
                trial_residual = np.asarray(residual_fn(trial_y), dtype=float)
                nfev += 1
                trial_cost = 0.5 * float(trial_residual @ trial_residual)
                if trial_cost < cost:
                    y = trial_y
                    residual = trial_residual
                    accepted = True
                    break
                if nfev >= max_eval:
                    break
            if accepted or nfev >= max_eval:
                break
            local_damping = max(local_damping * 10.0, 1.0e-12)
        improvement = cost - trial_cost
        if accepted:
            step_norm = float(np.linalg.norm(y - (y - alpha * step)))
            scale_norm = float(np.linalg.norm(y) + float(xtol))
            prev_cost = cost
            cost = trial_cost
            damping = max(local_damping * 0.3, 1.0e-14)
            if improvement <= float(ftol) * max(prev_cost, 1.0):
                status = 2
                message = "ftol satisfied"
                break
            if step_norm <= float(xtol) * max(scale_norm, 1.0):
                status = 3
                message = "xtol satisfied"
                break
        else:
            damping = max(local_damping * 10.0, 1.0e-12)
            if nfev >= max_eval:
                break
            status = 4
            message = "line search failed to reduce cost"
            break
        if nfev >= max_eval:
            break
    return _BandRefinementOptimizeResult(
        x=y,
        cost=cost,
        nfev=int(nfev),
        njev=int(njev),
        status=int(status),
        message=message,
    )



def _band_refinement_global_matrix_jacobian(
    basis: np.ndarray,
    *,
    matrix_weight: float,
    matrix_sigma: float,
) -> np.ndarray:
    responses = np.asarray(basis, dtype=np.complex128)
    if responses.ndim != 4:
        raise ValueError(f"basis must have shape (n_variables, n_k, dim, dim), got {responses.shape}")
    scale = np.sqrt(float(matrix_weight)) / float(matrix_sigma)
    flat = responses.reshape(responses.shape[0], -1).T
    return scale * np.vstack([flat.real, flat.imag])



def _band_refinement_global_matrix_jacobian_sparse(
    basis: np.ndarray,
    *,
    matrix_weight: float,
    matrix_sigma: float,
) -> scipy.sparse.csr_matrix:
    responses = np.asarray(basis, dtype=np.complex128)
    if responses.ndim != 4:
        raise ValueError(f"basis must have shape (n_variables, n_k, dim, dim), got {responses.shape}")
    scale = np.sqrt(float(matrix_weight)) / float(matrix_sigma)
    flat = responses.reshape(responses.shape[0], -1).T
    return scale * scipy.sparse.vstack(
        [
            scipy.sparse.csr_matrix(flat.real),
            scipy.sparse.csr_matrix(flat.imag),
        ],
        format="csr",
    )



def _band_refinement_reduced_global_matrix_loss(
    basis: np.ndarray,
    target_delta: np.ndarray,
    *,
    matrix_weight: float,
    matrix_sigma: float,
    rcond: float = 1.0e-12,
) -> dict[str, Any]:
    """Compress the linear full-matrix loss to coefficient space.

    The matrix residual is A d + c, where d = y - y0.  Since A is
    linear, ||A d + c||^2 differs from ||L (d - d*)||^2 only by a constant
    when L.T @ L = A.T @ A and d* is a least-squares minimizer.  The optimizer
    sees the same gradient/curvature but with at most n_variables rows.
    """
    responses = np.asarray(basis, dtype=np.complex128)
    delta = np.asarray(target_delta, dtype=np.complex128)
    if responses.ndim != 4:
        raise ValueError(f"basis must have shape (n_variables, n_k, dim, dim), got {responses.shape}")
    if delta.shape != responses.shape[1:]:
        raise ValueError(f"target_delta must have shape {responses.shape[1:]}, got {delta.shape}")
    if matrix_weight <= 0.0:
        raise ValueError(f"matrix_weight must be positive, got {matrix_weight}")
    if matrix_sigma <= 0.0:
        raise ValueError(f"matrix_sigma must be positive, got {matrix_sigma}")
    if rcond < 0.0:
        raise ValueError(f"rcond must be non-negative, got {rcond}")

    flat = responses.reshape(responses.shape[0], -1)
    target = delta.reshape(-1)
    scale_sq = float(matrix_weight) / (float(matrix_sigma) ** 2)
    gram = scale_sq * np.real(flat.conj() @ flat.T)
    gram = 0.5 * (gram + gram.T)
    linear = scale_sq * np.real(flat.conj() @ target)
    evals, evecs = scipy.linalg.eigh(gram)
    if evals.size == 0:
        return {
            "jacobian": np.zeros((0, responses.shape[0]), dtype=float),
            "center_delta": np.zeros(responses.shape[0], dtype=float),
            "rank": 0,
            "condition_number": None,
            "source_rows": int(2 * target.size),
            "compressed_rows": 0,
            "rcond": float(rcond),
        }
    max_eval = float(np.max(evals))
    cutoff = max(float(rcond) * max(max_eval, 1.0), np.finfo(float).eps * max(responses.shape[0], 1) * max_eval)
    keep = evals > cutoff
    if not np.any(keep):
        return {
            "jacobian": np.zeros((0, responses.shape[0]), dtype=float),
            "center_delta": np.zeros(responses.shape[0], dtype=float),
            "rank": 0,
            "condition_number": None,
            "source_rows": int(2 * target.size),
            "compressed_rows": 0,
            "rcond": float(rcond),
        }
    kept_evals = np.asarray(evals[keep], dtype=float)
    kept_vecs = np.asarray(evecs[:, keep], dtype=float)
    jac = np.sqrt(kept_evals)[:, None] * kept_vecs.T
    center_delta = -kept_vecs @ ((kept_vecs.T @ linear) / kept_evals)
    return {
        "jacobian": jac,
        "center_delta": center_delta,
        "rank": int(kept_evals.size),
        "condition_number": float(np.sqrt(float(kept_evals[-1] / kept_evals[0]))) if kept_evals[0] > 0.0 else None,
        "source_rows": int(2 * target.size),
        "compressed_rows": int(kept_evals.size),
        "rcond": float(rcond),
    }



def _edge_weighted_band_weights(
    n_bands: int,
    *,
    primary_bands: int,
    target_bands: str,
    decay: float = 0.65,
    floor: float = 0.15,
    normalize_mean: bool = True,
) -> np.ndarray:
    """Return deterministic edge-decaying weights for a top/bottom low-energy band window."""
    n = int(n_bands)
    primary = int(primary_bands)
    if n <= 0:
        raise ValueError(f"n_bands must be positive, got {n_bands}")
    if primary <= 0:
        raise ValueError(f"primary_bands must be positive, got {primary_bands}")
    if primary > n:
        raise ValueError(f"primary_bands={primary} exceeds n_bands={n}")
    decay_value = float(decay)
    floor_value = float(floor)
    if not (0.0 < decay_value <= 1.0):
        raise ValueError(f"decay must be in (0, 1], got {decay}")
    if not (0.0 <= floor_value <= 1.0):
        raise ValueError(f"floor must be in [0, 1], got {floor}")
    target = str(target_bands or "top").strip().lower()
    if target not in {"top", "bottom"}:
        raise ValueError("model.target_bands must be 'top' or 'bottom'")
    indices = np.arange(n, dtype=int)
    if target == "top":
        distance = np.maximum(0, (n - 1 - indices) - (primary - 1))
    else:
        distance = np.maximum(0, indices - (primary - 1))
    weights = floor_value + (1.0 - floor_value) * np.power(decay_value, distance.astype(float))
    if normalize_mean:
        mean = float(np.mean(weights))
        if mean <= 0.0 or not np.isfinite(mean):
            raise ValueError("weighted band loss produced invalid zero/NaN mean weight")
        weights = weights / mean
    return weights.astype(float)



def _boundary_gap_mev_by_k(eigvals: np.ndarray, band_slice: Sequence[int]) -> np.ndarray:
    eig = np.asarray(eigvals, dtype=float)
    if eig.ndim != 2:
        raise ValueError(f"boundary gap expects eigvals shape (Nk,dim), got {eig.shape}")
    if len(band_slice) != 2:
        raise ValueError(f"band_slice must be [start, stop], got {band_slice!r}")
    start, stop = int(band_slice[0]), int(band_slice[1])
    dim = eig.shape[1]
    if start < 0 or stop > dim or start >= stop:
        raise ValueError(f"Invalid band_slice {band_slice!r} for dim={dim}")
    gaps: list[np.ndarray] = []
    if start > 0:
        gaps.append(np.abs(eig[:, start] - eig[:, start - 1]))
    if stop < dim:
        gaps.append(np.abs(eig[:, stop] - eig[:, stop - 1]))
    if not gaps:
        return np.full(eig.shape[0], np.inf, dtype=float)
    return np.min(np.stack(gaps, axis=0), axis=0) * 1000.0



def _gap_weights_from_eigvals(
    eigvals: np.ndarray,
    band_slice: Sequence[int],
    *,
    gap_tolerance_mev: float,
) -> np.ndarray:
    gap_mev = _boundary_gap_mev_by_k(eigvals, band_slice)
    if gap_tolerance_mev <= 0.0:
        return np.ones_like(gap_mev, dtype=float)
    weights = np.clip(gap_mev / float(gap_tolerance_mev), 0.0, 1.0)
    weights[~np.isfinite(weights)] = 1.0
    return weights



def _subspace_overlap_metrics(
    model_basis: np.ndarray,
    target_basis: np.ndarray,
    *,
    gap_weights: np.ndarray | None = None,
) -> dict[str, Any]:
    model_u = np.asarray(model_basis, dtype=np.complex128)
    target_u = np.asarray(target_basis, dtype=np.complex128)
    if model_u.shape != target_u.shape or model_u.ndim != 3:
        raise ValueError(
            "subspace overlap expects model/target basis arrays with matching "
            f"shape (Nk,dim,N), got {model_u.shape} and {target_u.shape}"
        )
    n_k, _dim, n_state = model_u.shape
    if n_state <= 0:
        raise ValueError("subspace overlap requires at least one state")
    overlaps: list[float] = []
    leakages: list[float] = []
    min_sv: list[float] = []
    for idx in range(n_k):
        singular_values = np.linalg.svd(target_u[idx].conj().T @ model_u[idx], compute_uv=False)
        clipped = np.clip(singular_values, 0.0, 1.0)
        overlap = float(np.sum(clipped**2) / n_state)
        leakage = float(max(0.0, 1.0 - overlap))
        overlaps.append(overlap)
        leakages.append(leakage)
        min_sv.append(float(np.min(singular_values)))
    overlap_arr = np.asarray(overlaps, dtype=float)
    leakage_arr = np.asarray(leakages, dtype=float)
    min_sv_arr = np.asarray(min_sv, dtype=float)
    report = {
        "n_k": int(n_k),
        "n_state": int(n_state),
        "mean_overlap": float(np.mean(overlap_arr)),
        "min_overlap": float(np.min(overlap_arr)),
        "p05_overlap": float(np.percentile(overlap_arr, 5)),
        "mean_leakage": float(np.mean(leakage_arr)),
        "max_leakage": float(np.max(leakage_arr)),
        "p95_leakage": float(np.percentile(leakage_arr, 95)),
        "mean_min_singular_value": float(np.mean(min_sv_arr)),
        "min_singular_value": float(np.min(min_sv_arr)),
    }
    if gap_weights is not None:
        weights = np.asarray(gap_weights, dtype=float)
        if weights.shape != (n_k,):
            raise ValueError(f"gap_weights must have shape ({n_k},), got {weights.shape}")
        report["gap_weight_min"] = float(np.min(weights))
        report["gap_weight_mean"] = float(np.mean(weights))
    return report



def _principal_angle_subspace_residual(
    model_basis: np.ndarray,
    target_basis: np.ndarray,
    *,
    weight: float,
    normalize: bool,
    gap_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if weight < 0.0:
        raise ValueError("fit.refine_bands.subspace_loss.weight must be non-negative")
    model_u = np.asarray(model_basis, dtype=np.complex128)
    target_u = np.asarray(target_basis, dtype=np.complex128)
    if model_u.shape != target_u.shape or model_u.ndim != 3:
        raise ValueError(
            "subspace loss expects model/target basis arrays with matching "
            f"shape (Nk,dim,N), got {model_u.shape} and {target_u.shape}"
        )
    n_k = int(model_u.shape[0])
    if gap_weights is None:
        weights = np.ones(n_k, dtype=float)
    else:
        weights = np.asarray(gap_weights, dtype=float)
        if weights.shape != (n_k,):
            raise ValueError(f"gap_weights must have shape ({n_k},), got {weights.shape}")
    parts: list[np.ndarray] = []
    for idx in range(n_k):
        singular_values = np.linalg.svd(target_u[idx].conj().T @ model_u[idx], compute_uv=False)
        principal = np.sqrt(np.maximum(0.0, 1.0 - np.clip(singular_values, 0.0, 1.0) ** 2))
        parts.append(np.sqrt(weight * max(float(weights[idx]), 0.0)) * principal)
    residual = np.concatenate(parts) if parts else np.zeros(0, dtype=float)
    if normalize and residual.size:
        residual = residual / np.sqrt(float(residual.size))
    report = _subspace_overlap_metrics(model_u, target_u, gap_weights=weights)
    report.update(
        {
            "enabled": True,
            "mode": "principal_angles",
            "weight": float(weight),
            "normalize": bool(normalize),
            "loss_norm_sq": float(np.sum(residual**2)),
        }
    )
    return residual.astype(float), report



def _low_subspace_matrix_residual(
    matrix_delta: np.ndarray,
    target_basis: np.ndarray,
    *,
    weight: float,
    sigma_mev: float,
    normalize: bool,
    gap_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    delta = np.asarray(matrix_delta, dtype=np.complex128)
    basis = np.asarray(target_basis, dtype=np.complex128)
    if delta.ndim != 3 or delta.shape[1] != delta.shape[2]:
        raise ValueError(f"low-subspace matrix loss expects delta shape (Nk,dim,dim), got {delta.shape}")
    if basis.ndim != 3 or basis.shape[0] != delta.shape[0] or basis.shape[1] != delta.shape[1]:
        raise ValueError(
            "low-subspace matrix loss expects target basis shape (Nk,dim,N), "
            f"got delta={delta.shape}, basis={basis.shape}"
        )
    if weight < 0.0:
        raise ValueError("fit.refine_bands.low_subspace_matrix_loss.weight must be non-negative")
    if sigma_mev <= 0.0:
        raise ValueError("fit.refine_bands.low_subspace_matrix_loss.sigma_mev must be positive")
    n_k = int(delta.shape[0])
    if gap_weights is None:
        weights = np.ones(n_k, dtype=float)
    else:
        weights = np.asarray(gap_weights, dtype=float)
        if weights.shape != (n_k,):
            raise ValueError(f"gap_weights must have shape ({n_k},), got {weights.shape}")
    sigma = float(sigma_mev) * 1.0e-3
    values: list[np.ndarray] = []
    scaled: list[np.ndarray] = []
    for idx in range(n_k):
        projected = basis[idx].conj().T @ delta[idx] @ basis[idx]
        values.append(projected.reshape(-1))
        scaled.append(np.sqrt(weight * max(float(weights[idx]), 0.0)) * projected.reshape(-1) / sigma)
    flat_values = np.concatenate(values) if values else np.zeros(0, dtype=np.complex128)
    flat_scaled = np.concatenate(scaled) if scaled else np.zeros(0, dtype=np.complex128)
    residual = np.concatenate([flat_scaled.real, flat_scaled.imag])
    if normalize and residual.size:
        residual = residual / np.sqrt(float(flat_values.size))
    abs_values = np.abs(flat_values)
    report = {
        "enabled": True,
        "mode": "target_subspace_projected",
        "weight": float(weight),
        "sigma_mev": float(sigma_mev),
        "normalize": bool(normalize),
        "n_k": int(n_k),
        "n_state": int(basis.shape[2]),
        "rms_mev": float(np.sqrt(np.mean(abs_values**2)) * 1000.0) if abs_values.size else None,
        "max_mev": float(np.max(abs_values) * 1000.0) if abs_values.size else None,
        "loss_norm_sq": float(np.sum(residual**2)),
    }
    return residual.astype(float), report



def _matrix_loss_block_name(name: str) -> str:
    from . import fit_sampling

    key = str(name).strip().lower()
    if key not in fit_sampling._MATRIX_LOSS_BLOCK_ALIASES:
        raise ValueError(
            "fit.refine_bands.matrix_loss.blocks contains unknown block "
            f"{name!r}; expected one of {sorted(fit_sampling._MATRIX_LOSS_BLOCK_ALIASES)}"
        )
    return fit_sampling._MATRIX_LOSS_BLOCK_ALIASES[key]



def _matrix_loss_block_masks(moire_config: MoireConfig, dim: int) -> dict[str, np.ndarray]:
    q1_count = int(np.asarray(moire_config.Q_set1).shape[0])
    q2_count = int(np.asarray(moire_config.Q_set2).shape[0])
    n_orb1 = int(moire_config.n_orb1)
    n_orb2 = int(moire_config.n_orb2)
    expected = q1_count * n_orb1 + q2_count * n_orb2
    if expected != int(dim):
        raise ValueError(
            "fit.refine_bands.matrix_loss cannot build qset_orbital blocks: "
            f"model dimension is {dim}, but qset/orbital layout implies {expected}"
        )

    sectors: list[int] = []
    q_indices: list[int] = []
    orbital_indices: list[int] = []
    for sector, q_count, n_orb in ((1, q1_count, n_orb1), (2, q2_count, n_orb2)):
        for orbital_index in range(n_orb):
            for q_index in range(q_count):
                sectors.append(sector)
                q_indices.append(q_index)
                orbital_indices.append(orbital_index)

    sector_arr = np.asarray(sectors, dtype=int)
    q_arr = np.asarray(q_indices, dtype=int)
    orbital_arr = np.asarray(orbital_indices, dtype=int)
    same_sector = sector_arr[:, None] == sector_arr[None, :]
    same_q = q_arr[:, None] == q_arr[None, :]
    same_orbital = orbital_arr[:, None] == orbital_arr[None, :]
    kinetic_diagonal = same_sector & same_q & same_orbital
    intralayer = same_sector & ~kinetic_diagonal
    interlayer = ~same_sector
    return {
        "all": np.ones((dim, dim), dtype=bool),
        "kinetic_diagonal": kinetic_diagonal,
        "intralayer": intralayer,
        "interlayer": interlayer,
    }



def _matrix_loss_config(raw_cfg: Mapping[str, Any], *, matrix_weight: float, matrix_sigma_mev: float) -> dict[str, Any] | None:
    raw = raw_cfg.get("matrix_loss")
    if raw in (None, False):
        return None
    if raw is True:
        cfg: dict[str, Any] = {"enabled": True}
    elif isinstance(raw, Mapping):
        cfg = dict(raw)
    else:
        raise ValueError("fit.refine_bands.matrix_loss must be a mapping or boolean when provided")
    if not bool(cfg.get("enabled", True)):
        return None
    mode = str(cfg.get("mode", "block_normalized")).strip().lower()
    if mode in {"block", "blocks"}:
        mode = "block_normalized"
    if mode not in {"global", "dense", "block_normalized"}:
        raise ValueError("fit.refine_bands.matrix_loss.mode must be 'global' or 'block_normalized'")
    weight = float(cfg.get("weight", matrix_weight if matrix_weight > 0.0 else 1.0))
    sigma_mev = float(cfg.get("sigma_mev", matrix_sigma_mev))
    if weight < 0.0:
        raise ValueError("fit.refine_bands.matrix_loss.weight must be non-negative")
    if sigma_mev <= 0.0:
        raise ValueError("fit.refine_bands.matrix_loss.sigma_mev must be positive")
    blocks_raw = cfg.get("blocks")
    if blocks_raw is None:
        blocks = {"kinetic_diagonal": 1.0, "intralayer": 1.0, "interlayer": 1.0}
    elif not isinstance(blocks_raw, Mapping):
        raise ValueError("fit.refine_bands.matrix_loss.blocks must be a mapping")
    else:
        blocks = {}
        for block_name, block_weight in blocks_raw.items():
            canonical = _matrix_loss_block_name(str(block_name))
            value = float(block_weight)
            if value < 0.0:
                raise ValueError("fit.refine_bands.matrix_loss block weights must be non-negative")
            blocks[canonical] = value
    return {
        "enabled": True,
        "mode": "global" if mode == "dense" else mode,
        "weight": weight,
        "sigma_mev": sigma_mev,
        "normalize_band_loss": bool(cfg.get("normalize_band_loss", True)),
        "blocks": blocks,
    }



def _matrix_loss_residual(
    matrix_delta: np.ndarray,
    masks: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
    *,
    report: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    delta = np.asarray(matrix_delta, dtype=np.complex128)
    if delta.ndim != 3 or delta.shape[1] != delta.shape[2]:
        raise ValueError(f"matrix loss expects delta shape (Nk,dim,dim), got {delta.shape}")
    mode = str(config.get("mode", "block_normalized")).strip().lower()
    weight = float(config.get("weight", 1.0))
    sigma_mev = float(config.get("sigma_mev", 10.0))
    if weight < 0.0:
        raise ValueError("fit.refine_bands.matrix_loss.weight must be non-negative")
    if sigma_mev <= 0.0:
        raise ValueError("fit.refine_bands.matrix_loss.sigma_mev must be positive")
    sigma = sigma_mev * 1.0e-3
    if mode == "global":
        scaled = np.sqrt(weight) * delta.reshape(-1) / sigma
        vector = np.concatenate([scaled.real, scaled.imag])
        if not report:
            return vector, {"enabled": True, "mode": "global"}
        abs_delta = np.abs(delta)
        report = {
            "enabled": True,
            "mode": "global",
            "weight": weight,
            "sigma_mev": sigma_mev,
            "rms_mev": float(np.sqrt(np.mean(abs_delta**2)) * 1000.0),
            "max_mev": float(np.max(abs_delta) * 1000.0),
            "loss_norm_sq": float(np.sum(vector**2)),
        }
        return vector, report
    if mode != "block_normalized":
        raise ValueError("fit.refine_bands.matrix_loss.mode must be 'global' or 'block_normalized'")

    blocks_raw = config.get("blocks", {"kinetic_diagonal": 1.0, "intralayer": 1.0, "interlayer": 1.0})
    if not isinstance(blocks_raw, Mapping):
        raise ValueError("fit.refine_bands.matrix_loss.blocks must be a mapping")
    parts: list[np.ndarray] = []
    block_reports: dict[str, Any] = {}
    for raw_name, raw_weight in blocks_raw.items():
        name = _matrix_loss_block_name(str(raw_name))
        block_weight = float(raw_weight)
        if block_weight < 0.0:
            raise ValueError("fit.refine_bands.matrix_loss block weights must be non-negative")
        mask = np.asarray(masks.get(name), dtype=bool)
        if mask.shape != delta.shape[1:]:
            raise ValueError(f"fit.refine_bands.matrix_loss block {name!r} has shape {mask.shape}, expected {delta.shape[1:]}")
        count = int(np.count_nonzero(mask))
        if count == 0 or block_weight == 0.0 or weight == 0.0:
            block_reports[name] = {
                "weight": block_weight,
                "count": count,
                "rms_mev": None,
                "max_mev": None,
                "loss_norm_sq": 0.0,
            }
            continue
        values = delta[:, mask].reshape(-1)
        normalizer = float(np.sqrt(values.size))
        scaled = np.sqrt(weight * block_weight) * values / (sigma * normalizer)
        vector = np.concatenate([scaled.real, scaled.imag])
        parts.append(vector)
        if not report:
            continue
        abs_values = np.abs(values)
        block_reports[name] = {
            "weight": block_weight,
            "count": count,
            "rms_mev": float(np.sqrt(np.mean(abs_values**2)) * 1000.0),
            "max_mev": float(np.max(abs_values) * 1000.0),
            "loss_norm_sq": float(np.sum(vector**2)),
        }
    residual = np.concatenate(parts) if parts else np.zeros(0, dtype=float)
    if not report:
        return residual, {"enabled": True, "mode": "block_normalized"}
    return residual, {
        "enabled": True,
        "mode": "block_normalized",
        "weight": weight,
        "sigma_mev": sigma_mev,
        "normalization": "per_block_rms",
        "blocks": block_reports,
    }



def _refinement_loss_band_slice(
    raw: Mapping[str, Any],
    *,
    default_slice: tuple[int, int],
    dim: int,
    target_bands: str,
) -> tuple[int, int]:
    from . import fit_sampling

    if raw.get("band_slice") is not None:
        values = configuration._as_int_list(raw.get("band_slice"), name="fit.refine_bands loss band_slice")
        if len(values) != 2:
            raise ValueError(f"fit.refine_bands loss band_slice must be [start, stop], got {values!r}")
        start, stop = int(values[0]), int(values[1])
    elif raw.get("top_bands") is not None:
        start, stop = fit_sampling._auto_low_energy_band_slice(dim, int(raw["top_bands"]), "top")
    elif raw.get("bottom_bands") is not None:
        start, stop = fit_sampling._auto_low_energy_band_slice(dim, int(raw["bottom_bands"]), "bottom")
    elif raw.get("n_bands") is not None:
        start, stop = fit_sampling._auto_low_energy_band_slice(dim, int(raw["n_bands"]), target_bands)
    else:
        start, stop = int(default_slice[0]), int(default_slice[1])
    if start < 0 or stop > dim or start >= stop:
        raise ValueError(f"Invalid refinement loss slice [{start}, {stop}] for dimension {dim}")
    return int(start), int(stop)



def _subspace_loss_config(
    raw_cfg: Mapping[str, Any],
    *,
    default_slice: tuple[int, int],
    dim: int,
    target_bands: str,
) -> dict[str, Any] | None:
    raw = raw_cfg.get("subspace_loss")
    if raw in (None, False):
        return None
    if raw is True:
        cfg: dict[str, Any] = {"enabled": True}
    elif isinstance(raw, Mapping):
        cfg = dict(raw)
    else:
        raise ValueError("fit.refine_bands.subspace_loss must be a mapping or boolean when provided")
    if not bool(cfg.get("enabled", True)):
        return None
    mode = str(cfg.get("mode", "principal_angles")).strip().lower()
    if mode not in {"principal_angles", "projector"}:
        raise ValueError("fit.refine_bands.subspace_loss.mode must be 'principal_angles' or 'projector'")
    weight = float(cfg.get("weight", 1.0))
    if weight < 0.0:
        raise ValueError("fit.refine_bands.subspace_loss.weight must be non-negative")
    band_slice = _refinement_loss_band_slice(cfg, default_slice=default_slice, dim=dim, target_bands=target_bands)
    return {
        "enabled": True,
        "mode": mode,
        "weight": weight,
        "normalize": bool(cfg.get("normalize", True)),
        "gap_tolerance_mev": float(cfg.get("gap_tolerance_mev", raw_cfg.get("gap_tolerance_mev", 0.1))),
        "band_slice": [int(band_slice[0]), int(band_slice[1])],
    }



def _low_subspace_matrix_loss_config(
    raw_cfg: Mapping[str, Any],
    *,
    default_slice: tuple[int, int],
    dim: int,
    target_bands: str,
) -> dict[str, Any] | None:
    raw = raw_cfg.get("low_subspace_matrix_loss")
    if raw in (None, False):
        return None
    if raw is True:
        cfg: dict[str, Any] = {"enabled": True}
    elif isinstance(raw, Mapping):
        cfg = dict(raw)
    else:
        raise ValueError("fit.refine_bands.low_subspace_matrix_loss must be a mapping or boolean when provided")
    if not bool(cfg.get("enabled", True)):
        return None
    weight = float(cfg.get("weight", 1.0))
    sigma_mev = float(cfg.get("sigma_mev", cfg.get("matrix_sigma_mev", 10.0)))
    if weight < 0.0:
        raise ValueError("fit.refine_bands.low_subspace_matrix_loss.weight must be non-negative")
    if sigma_mev <= 0.0:
        raise ValueError("fit.refine_bands.low_subspace_matrix_loss.sigma_mev must be positive")
    band_slice = _refinement_loss_band_slice(cfg, default_slice=default_slice, dim=dim, target_bands=target_bands)
    return {
        "enabled": True,
        "weight": weight,
        "sigma_mev": sigma_mev,
        "normalize": bool(cfg.get("normalize", True)),
        "gap_tolerance_mev": float(cfg.get("gap_tolerance_mev", raw_cfg.get("gap_tolerance_mev", 0.1))),
        "band_slice": [int(band_slice[0]), int(band_slice[1])],
    }



def _shell_projected_matrix_loss_config(raw_cfg: Mapping[str, Any]) -> dict[str, Any] | None:
    raw = raw_cfg.get("shell_projected_matrix_loss")
    source_key = "shell_projected_matrix_loss"
    if "shell_subspace_loss" in raw_cfg:
        raise ValueError("unsupported fit.refine_bands.shell_subspace_loss; use shell_projected_matrix_loss")
    if raw in (None, False):
        return None
    if raw is True:
        cfg: dict[str, Any] = {"enabled": True}
    elif isinstance(raw, Mapping):
        cfg = dict(raw)
    else:
        raise ValueError(f"fit.refine_bands.{source_key} must be a mapping or boolean when provided")
    if not bool(cfg.get("enabled", True)):
        return None
    weight = float(cfg.get("weight", 1.0))
    if weight < 0.0:
        raise ValueError(f"fit.refine_bands.{source_key}.weight must be non-negative")
    max_shells_raw = cfg.get("max_shells", 3)
    max_shells = int(max_shells_raw) if max_shells_raw is not None else None
    if max_shells is not None and max_shells <= 0:
        raise ValueError(f"fit.refine_bands.{source_key}.max_shells must be positive")
    raw_window = cfg.get("window")
    if raw_window is None:
        window = {
            "mode": "gap_aware",
            "center_fraction": float(cfg.get("subspace_fraction", 0.5)),
            "search_fraction": [0.35, 0.65],
            "gap_tolerance_mev": float(cfg.get("gap_tolerance_mev", raw_cfg.get("gap_tolerance_mev", 0.1))),
        }
    elif isinstance(raw_window, Mapping):
        window = dict(raw_window)
        if "center_fraction" not in window and "subspace_fraction" in cfg:
            window["center_fraction"] = float(cfg["subspace_fraction"])
        window.setdefault("center_fraction", 0.5)
        window.setdefault("mode", "gap_aware")
        if str(window.get("mode", "")).strip().lower() == "gap_aware":
            window.setdefault("search_fraction", [0.35, 0.65])
            window.setdefault("gap_tolerance_mev", float(cfg.get("gap_tolerance_mev", raw_cfg.get("gap_tolerance_mev", 0.1))))
    else:
        raise ValueError(f"fit.refine_bands.{source_key}.window must be a mapping when provided")
    center_fraction = float(window.get("center_fraction", window.get("subspace_fraction", 0.5)))
    if center_fraction <= 0.0 or center_fraction > 1.0:
        raise ValueError(f"fit.refine_bands.{source_key}.window.center_fraction must be in (0, 1]")
    window["center_fraction"] = center_fraction
    shell_decay = float(cfg.get("shell_decay", 1.0))
    if shell_decay < 0.0:
        raise ValueError(f"fit.refine_bands.{source_key}.shell_decay must be non-negative")
    return {
        "enabled": True,
        "weight": weight,
        "max_shells": max_shells,
        "subspace_fraction": center_fraction,
        "window": window,
        "shell_decay": shell_decay,
        "tol": float(cfg.get("tol", 1.0e-6)),
        "source_key": source_key,
    }



def _refinement_acceptance_guard_config(
    raw_cfg: Mapping[str, Any],
    model_config: configuration.ConfiguredModel,
    *,
    dim: int,
) -> dict[str, Any] | None:
    from . import fit_sampling
    from . import validation

    raw = raw_cfg.get("acceptance_guard")
    if raw is False:
        return None
    if raw is None:
        enabled = bool(raw_cfg.get("enabled", False))
        cfg: dict[str, Any] = {}
    elif raw is True:
        enabled = True
        cfg = {}
    elif isinstance(raw, Mapping):
        cfg = dict(raw)
        enabled = bool(cfg.get("enabled", True))
    else:
        raise ValueError("fit.refine_bands.acceptance_guard must be a mapping or boolean when provided")
    if not enabled:
        return None

    plot_config = dict(model_config.band_plot_config or {})
    if "band_slice" in cfg:
        plot_config["band_slice"] = cfg["band_slice"]
    if "top_bands" in cfg:
        plot_config["top_bands"] = cfg["top_bands"]
        plot_config.pop("bottom_bands", None)
    if "bottom_bands" in cfg:
        plot_config["bottom_bands"] = cfg["bottom_bands"]
        plot_config.pop("top_bands", None)
    if "align" in cfg:
        plot_config["align"] = cfg["align"]
    start, stop = validation._resolve_plot_band_slice(
        nbands=int(dim),
        metric_band_slice=model_config.band_slice,
        plot_config=plot_config,
    )
    align = str(plot_config.get("align", raw_cfg.get("align", "none"))).strip().lower()
    max_rms_increase = float(cfg.get("max_rms_increase_mev", 0.05))
    max_max_increase = float(cfg.get("max_max_increase_mev", 0.25))
    target_bands = str(raw_cfg.get("target_bands", model_config.raw.get("model", {}).get("target_bands", "top"))).strip().lower()
    windows: list[dict[str, Any]] = [
        {
            "name": "validation_window",
            "band_slice": [int(start), int(stop)],
            "align": align,
            "max_rms_increase_mev": max_rms_increase,
            "max_max_increase_mev": max_max_increase,
        }
    ]
    if bool(cfg.get("guard_primary_window", True)):
        primary_count = max(1, min(int(sum(model_config.n_orb)), int(dim)))
        primary_slice = fit_sampling._auto_low_energy_band_slice(int(dim), primary_count, target_bands)
        windows.append(
            {
                "name": "primary_window",
                "band_slice": primary_slice,
                "align": str(cfg.get("primary_align", align)),
                "max_rms_increase_mev": float(cfg.get("max_primary_rms_increase_mev", max_rms_increase)),
                "max_max_increase_mev": float(cfg.get("max_primary_max_increase_mev", max_max_increase)),
            }
        )
    if bool(cfg.get("guard_all_bands", True)):
        windows.append(
            {
                "name": "all_bands",
                "band_slice": [0, int(dim)],
                "align": str(cfg.get("all_band_align", align)),
                "max_rms_increase_mev": float(cfg.get("max_all_band_rms_increase_mev", max_rms_increase)),
                "max_max_increase_mev": float(cfg.get("max_all_band_max_increase_mev", max_max_increase)),
            }
        )

    raw_alphas = cfg.get("line_search_alphas", [1.0, 0.75, 0.5, 0.25, 0.1, 0.05])
    if not isinstance(raw_alphas, Sequence) or isinstance(raw_alphas, (str, bytes)):
        raise ValueError("fit.refine_bands.acceptance_guard.line_search_alphas must be a list")
    alphas = sorted({float(value) for value in raw_alphas if float(value) > 0.0}, reverse=True)
    if not alphas:
        alphas = [1.0]
    selection = str(cfg.get("selection", "first_accepted")).strip().lower()
    if selection not in {"first_accepted", "best_validation_window"}:
        raise ValueError(
            "fit.refine_bands.acceptance_guard.selection must be "
            "'first_accepted' or 'best_validation_window'"
        )

    return {
        "enabled": True,
        "band_slice": [int(start), int(stop)],
        "align": align,
        "max_rms_increase_mev": max_rms_increase,
        "max_max_increase_mev": max_max_increase,
        "windows": windows,
        "line_search_alphas": alphas,
        "selection": selection,
    }



def _guard_window_acceptance(
    *,
    base_h: np.ndarray,
    candidate_h: np.ndarray,
    heff_eig: np.ndarray,
    heff_all: np.ndarray,
    window: Mapping[str, Any],
) -> dict[str, Any]:
    band_slice = tuple(int(value) for value in window["band_slice"])
    align = str(window.get("align", "none"))
    initial = _band_refinement_metrics(base_h, heff_eig, heff_all, band_slice=band_slice, align=align)
    candidate = _band_refinement_metrics(candidate_h, heff_eig, heff_all, band_slice=band_slice, align=align)
    rms_limit = float(initial["top_band_rms_mev"]) + float(window["max_rms_increase_mev"])
    max_limit = float(initial["top_band_max_mev"]) + float(window["max_max_increase_mev"])
    accepted = (
        float(candidate["top_band_rms_mev"]) <= rms_limit
        and float(candidate["top_band_max_mev"]) <= max_limit
    )
    return {
        **dict(window),
        "initial": initial,
        "candidate": candidate,
        "rms_limit_mev": float(rms_limit),
        "max_limit_mev": float(max_limit),
        "accepted": bool(accepted),
    }



def _apply_refinement_acceptance_guard(
    *,
    raw_cfg: Mapping[str, Any],
    model_config: configuration.ConfiguredModel,
    base_h: np.ndarray,
    heff_eig: np.ndarray,
    heff_all: np.ndarray,
    y0: np.ndarray,
    candidate_y: np.ndarray,
    h_from_y: Any,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    guard_cfg = _refinement_acceptance_guard_config(raw_cfg, model_config, dim=base_h.shape[-1])
    if guard_cfg is None:
        return np.asarray(candidate_y, dtype=float), None

    y0_arr = np.asarray(y0, dtype=float)
    candidate_arr = np.asarray(candidate_y, dtype=float)
    trials: list[dict[str, Any]] = []
    accepted_trials: list[tuple[np.ndarray, dict[str, Any]]] = []
    for alpha in guard_cfg["line_search_alphas"]:
        trial_y = y0_arr + float(alpha) * (candidate_arr - y0_arr)
        trial_h = h_from_y(trial_y)
        windows = [
            _guard_window_acceptance(
                base_h=base_h,
                candidate_h=trial_h,
                heff_eig=heff_eig,
                heff_all=heff_all,
                window=window,
            )
            for window in guard_cfg["windows"]
        ]
        failed = [str(window["name"]) for window in windows if not bool(window["accepted"])]
        trial_report = {
            "alpha": float(alpha),
            "accepted": not failed,
            "failed_windows": failed,
            "windows": windows,
        }
        trials.append(trial_report)
        if not failed:
            accepted_trials.append((trial_y, trial_report))
        if not failed and str(guard_cfg.get("selection", "first_accepted")) == "first_accepted":
            return trial_y, {
                **guard_cfg,
                "selected_alpha": float(alpha),
                "accepted": True,
                "reverted": False,
                "scaled": float(alpha) < 1.0,
                "failed_windows": [],
                "trials": trials,
                "windows": windows,
                "reason": None,
            }

    if accepted_trials:
        def trial_key(item: tuple[np.ndarray, dict[str, Any]]) -> tuple[float, float, float]:
            _trial_y, trial = item
            windows = list(trial.get("windows", []))
            validation = next((window for window in windows if window.get("name") == "validation_window"), windows[0])
            candidate = validation["candidate"]
            return (
                float(candidate["top_band_rms_mev"]),
                float(candidate["top_band_max_mev"]),
                -float(trial.get("alpha", 0.0)),
            )

        selected_y, selected_trial = min(accepted_trials, key=trial_key)
        return selected_y, {
            **guard_cfg,
            "selected_alpha": float(selected_trial["alpha"]),
            "accepted": True,
            "reverted": False,
            "scaled": float(selected_trial["alpha"]) < 1.0,
            "failed_windows": [],
            "trials": trials,
            "windows": list(selected_trial.get("windows", [])),
            "reason": None,
        }

    last = trials[-1] if trials else {"failed_windows": [], "windows": []}
    return y0_arr.copy(), {
        **guard_cfg,
        "selected_alpha": 0.0,
        "accepted": False,
        "reverted": True,
        "scaled": False,
        "failed_windows": list(last.get("failed_windows", [])),
        "trials": trials,
        "windows": list(last.get("windows", [])),
        "reason": "validation_window_worsened",
    }



def _sync_complete_response_coefficients_to_terms(model: Any, coefficients: np.ndarray) -> None:
    basis = getattr(model, "_compiled_response_basis")
    # ``term_index`` is authored against the complete candidate registry in
    # ``compile_model_response_basis``.  The semantic ``model.terms`` dictionary
    # may have already overwritten duplicate physical keys, so indexing that
    # dictionary can assign a fitted coefficient to the wrong authored seed.
    terms = list(
        getattr(model, "candidate_terms", ())
        or getattr(model, "terms", {}).values()
    )
    for term in terms:
        term.active = False
        term.r_value_real = 0.0
        term.r_value_imag = 0.0
    for channel, value in zip(basis.channels, np.asarray(coefficients, dtype=float)):
        term_index = int(channel.metadata.get("term_index", -1))
        if term_index < 0 or term_index >= len(terms):
            raise ValueError(
                f"compiled response channel {channel.channel_id!r} has invalid term_index={term_index}"
            )
        term = terms[term_index]
        if channel.component == "real":
            term.r_value_real = float(value)
        elif channel.component == "imag":
            term.r_value_imag = float(value)
        else:
            raise ValueError(f"compiled response channel has unknown component {channel.component!r}")
        if float(value) != 0.0:
            term.active = True



def _public_hamiltonian_quadratic_residual(
    basis: Any,
    channel_indices: np.ndarray,
    kpoints: np.ndarray,
    target: np.ndarray,
    *,
    target_bands: str,
    bands: int,
    one_sided_weight: float,
    two_sided_weight: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Compress the public normalized Hamiltonian loss to coefficient space."""

    from .response_basis import (
        NormalizedLowEnergyWeightingSpec,
        build_normalized_low_energy_weighting,
    )

    points = np.asarray(kpoints, dtype=float)
    target_array = np.asarray(target, dtype=np.complex128)
    selected = np.asarray(channel_indices, dtype=np.int64)
    complex_design, support = basis._response_design_on_union_support(points, selected)
    weighting = build_normalized_low_energy_weighting(
        target_array,
        NormalizedLowEnergyWeightingSpec(
            band_edge=str(target_bands),
            n_bands=int(bands),
            one_sided_weight=float(one_sided_weight),
            two_sided_weight=float(two_sided_weight),
        ),
    )
    complex_design, complex_target = basis._normalized_low_energy_complex_rows(
        complex_design,
        support,
        target_array,
        weighting,
    )
    n_kpoints = max(1, int(points.shape[0]))
    design = np.vstack((complex_design.real, complex_design.imag)) / np.sqrt(n_kpoints)
    target_vector = np.concatenate((complex_target.real, complex_target.imag)) / np.sqrt(
        n_kpoints
    )
    gram = design.T @ design
    linear = design.T @ target_vector
    gram = 0.5 * (gram + gram.T)
    values, vectors = np.linalg.eigh(gram)
    tolerance = np.finfo(float).eps * max(gram.shape) * max(
        float(np.max(np.abs(values))) if values.size else 0.0,
        1.0,
    )
    retained = values > tolerance
    if not np.any(retained):
        raise ValueError("public nonlinear Hamiltonian loss has zero coefficient rank")
    root = np.sqrt(values[retained])[:, None] * vectors[:, retained].T
    center = (vectors[:, retained].T @ linear) / np.sqrt(values[retained])
    constant = max(0.0, float(target_vector @ target_vector - center @ center))
    if constant > tolerance:
        root = np.vstack((root, np.zeros((1, root.shape[1]), dtype=float)))
        center = np.concatenate((center, np.asarray([np.sqrt(constant)], dtype=float)))
    return root, center, {
        **weighting.artifact(),
        "hamiltonian_kpoints": int(points.shape[0]),
        "coefficient_rank": int(np.count_nonzero(retained)),
        "coefficient_count": int(selected.size),
    }



def _solve_target_subspace_action_linear(
    response_actions: np.ndarray,
    base_hamiltonians: np.ndarray,
    target_hamiltonians: np.ndarray,
    target_frames: np.ndarray,
    *,
    regularization: float,
    response_scales: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit ``(H_model-H_target)V_target`` as one real linear system."""

    actions = np.asarray(response_actions, dtype=np.complex128)
    base = np.asarray(base_hamiltonians, dtype=np.complex128)
    target = np.asarray(target_hamiltonians, dtype=np.complex128)
    frames = np.asarray(target_frames, dtype=np.complex128)
    if actions.ndim != 4:
        raise ValueError(
            "response_actions must have shape (Nvar,Nk,dim,Nstate), got "
            f"{actions.shape}"
        )
    n_variables, n_kpoints, dim, n_states = actions.shape
    if base.shape != (n_kpoints, dim, dim) or target.shape != base.shape:
        raise ValueError(
            "base/target Hamiltonians must match response actions, got "
            f"actions={actions.shape}, base={base.shape}, target={target.shape}"
        )
    if frames.shape != (n_kpoints, dim, n_states):
        raise ValueError(
            "target_frames must have shape (Nk,dim,Nstate), got "
            f"{frames.shape}"
        )
    ridge = float(regularization)
    if ridge < 0.0:
        raise ValueError("target-subspace action regularization must be non-negative")
    scales = np.asarray(response_scales, dtype=float)
    if scales.shape != (n_variables,) or np.any(scales <= 0.0):
        raise ValueError("response_scales must be positive with shape (Nvar,)")

    target_action = np.einsum(
        "kmn,kns->kms",
        target - base,
        frames,
        optimize=True,
    ).reshape(-1)
    design_complex = np.moveaxis(actions, 0, -1).reshape(-1, n_variables)
    design = np.vstack((design_complex.real, design_complex.imag))
    right = np.concatenate((target_action.real, target_action.imag))
    physical_rows = int(design.shape[0])
    if ridge > 0.0:
        design = np.vstack((design, np.sqrt(ridge) * np.diag(scales)))
        right = np.concatenate((right, np.zeros(n_variables, dtype=float)))
    delta, residuals, rank, singular_values = np.linalg.lstsq(
        design,
        right,
        rcond=1.0e-10,
    )
    action_residual = design_complex @ delta - target_action
    return np.asarray(delta, dtype=float), {
        "physical_rows": physical_rows,
        "rows": int(design.shape[0]),
        "cols": int(design.shape[1]),
        "rank": int(rank),
        "regularization": ridge,
        "action_residual_norm": float(np.linalg.norm(action_residual)),
        "least_squares_residual_norm": (
            float(np.sqrt(float(residuals[0]))) if np.size(residuals) else None
        ),
        "condition_number": (
            float(singular_values[0] / singular_values[-1])
            if np.size(singular_values) and float(singular_values[-1]) > 0.0
            else None
        ),
    }



def _compiled_response_band_jacobian(
    basis: Any,
    channel_indices: np.ndarray,
    kpoints: np.ndarray,
    eigenvectors: np.ndarray,
    selected_masks: np.ndarray,
) -> np.ndarray:
    """Hellmann--Feynman derivatives without a dense response tensor."""

    points = np.asarray(kpoints, dtype=float)
    vectors = np.asarray(eigenvectors, dtype=np.complex128)
    masks = np.asarray(selected_masks, dtype=bool)
    selected = np.asarray(channel_indices, dtype=np.int64)
    dimensionless = basis.coordinate.to_dimensionless(points)
    w = dimensionless[:, 0] + 1j * dimensionless[:, 1]
    monomial_values = {
        monomial: (w ** monomial[0]) * (np.conjugate(w) ** monomial[1])
        for monomial in basis.coordinate.monomials
    }
    offsets = np.cumsum([0] + [int(np.count_nonzero(mask)) for mask in masks])
    jacobian = np.zeros((int(offsets[-1]), int(selected.size)), dtype=float)
    for output_index, channel_index in enumerate(selected):
        channel = basis.channels[int(channel_index)]
        for k_index in range(points.shape[0]):
            band_indices = np.flatnonzero(masks[k_index])
            selected_vectors = vectors[k_index][:, band_indices]
            derivatives = np.zeros(int(band_indices.size), dtype=np.complex128)
            for monomial, matrix in channel.coefficients.items():
                coo = matrix.tocoo()
                if coo.nnz == 0:
                    continue
                left = np.conjugate(selected_vectors[np.asarray(coo.row, dtype=int), :])
                right = selected_vectors[np.asarray(coo.col, dtype=int), :]
                derivatives += monomial_values[monomial][k_index] * np.sum(
                    left * np.asarray(coo.data, dtype=np.complex128)[:, None] * right,
                    axis=0,
                )
            jacobian[offsets[k_index] : offsets[k_index + 1], output_index] = (
                derivatives.real
            )
    return jacobian



@dataclass(frozen=True)
class _PublicBandResponseEvaluator:
    """Cached selected-channel responses for public nonlinear fitting."""

    basis: Any
    coefficients0: np.ndarray
    channel_indices: np.ndarray
    y0: np.ndarray
    kpoints: np.ndarray
    base_hamiltonians: np.ndarray
    support: np.ndarray
    design_by_k: np.ndarray | None
    backend: str
    cache_bytes: int

    def hamiltonians(self, y: np.ndarray) -> np.ndarray:
        values = np.asarray(y, dtype=float)
        if values.shape != self.y0.shape:
            raise ValueError(
                f"public nonlinear coefficient vector has shape {values.shape}, "
                f"expected {self.y0.shape}"
            )
        if self.design_by_k is None:
            coefficients = np.asarray(self.coefficients0, dtype=float).copy()
            coefficients[self.channel_indices] = values
            return self.basis.hamiltonians(self.kpoints, coefficients)
        delta_support = np.einsum(
            "ksj,j->ks",
            self.design_by_k,
            values - self.y0,
            optimize=True,
        )
        hamiltonians = np.asarray(self.base_hamiltonians, dtype=np.complex128).copy()
        hamiltonians.reshape(len(self.kpoints), -1)[:, self.support] += delta_support
        return hamiltonians

    def band_jacobian(
        self,
        eigenvectors: np.ndarray,
        selected_masks: np.ndarray,
    ) -> np.ndarray:
        vectors = np.asarray(eigenvectors, dtype=np.complex128)
        masks = np.asarray(selected_masks, dtype=bool)
        if self.design_by_k is None:
            return _compiled_response_band_jacobian(
                self.basis,
                self.channel_indices,
                self.kpoints,
                vectors,
                masks,
            )
        if vectors.shape != (
            len(self.kpoints),
            int(self.basis.dim),
            int(self.basis.dim),
        ):
            raise ValueError(
                "public nonlinear eigenvectors must have shape "
                f"(Nk,dim,dim), got {vectors.shape}"
            )
        if masks.shape != (len(self.kpoints), int(self.basis.dim)):
            raise ValueError(
                "public nonlinear selected masks must have shape "
                f"(Nk,dim), got {masks.shape}"
            )
        offsets = np.cumsum([0] + [int(np.count_nonzero(mask)) for mask in masks])
        jacobian = np.zeros(
            (int(offsets[-1]), int(self.channel_indices.size)),
            dtype=float,
        )
        support_rows = self.support // int(self.basis.dim)
        support_cols = self.support % int(self.basis.dim)
        for k_index, mask in enumerate(masks):
            band_indices = np.flatnonzero(mask)
            selected_vectors = vectors[k_index][:, band_indices]
            projected_support = (
                np.conjugate(selected_vectors[support_rows, :])
                * selected_vectors[support_cols, :]
            )
            jacobian[offsets[k_index] : offsets[k_index + 1]] = (
                projected_support.T @ self.design_by_k[k_index]
            ).real
        return jacobian



_PUBLIC_NONLINEAR_CACHE_LIMIT_BYTES = 768 * 1024**2



def _public_band_response_union_support(
    basis: Any,
    channel_indices: np.ndarray,
) -> np.ndarray:
    support_parts: list[np.ndarray] = []
    for channel_index in np.asarray(channel_indices, dtype=np.int64):
        channel = basis.channels[int(channel_index)]
        for matrix in channel.coefficients.values():
            coo = matrix.tocoo()
            if coo.nnz:
                support_parts.append(
                    np.asarray(coo.row, dtype=np.int64) * int(basis.dim)
                    + np.asarray(coo.col, dtype=np.int64)
                )
    if not support_parts:
        return np.zeros(0, dtype=np.int64)
    return np.unique(np.concatenate(support_parts))



def _prepare_public_band_response_evaluator(
    basis: Any,
    coefficients0: np.ndarray,
    channel_indices: np.ndarray,
    kpoints: np.ndarray,
    *,
    cache_limit_bytes: int = _PUBLIC_NONLINEAR_CACHE_LIMIT_BYTES,
) -> _PublicBandResponseEvaluator:
    """Compile selected complete-response channels once at band-loss k-points."""

    points = np.asarray(kpoints, dtype=float)
    indices = np.asarray(channel_indices, dtype=np.int64)
    coefficients = np.asarray(coefficients0, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
        raise ValueError(f"public nonlinear kpoints must have shape (Nk,2), got {points.shape}")
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("public nonlinear channel_indices must be non-empty and one-dimensional")
    if coefficients.shape != (len(basis.channels),):
        raise ValueError(
            f"public nonlinear coefficient vector has shape {coefficients.shape}, "
            f"expected {(len(basis.channels),)}"
        )
    limit = int(cache_limit_bytes)
    if limit < 0:
        raise ValueError("public nonlinear cache_limit_bytes must be non-negative")
    support_array = _public_band_response_union_support(basis, indices)
    estimated_bytes = int(
        points.shape[0]
        * support_array.size
        * indices.size
        * np.dtype(np.complex128).itemsize
    )
    if estimated_bytes <= limit:
        complex_design, compiled_support = basis._response_design_on_union_support(
            points,
            indices,
        )
        if not np.array_equal(support_array, np.asarray(compiled_support, dtype=np.int64)):
            raise ValueError("public nonlinear union-support discovery disagrees with compiler")
        design_by_k: np.ndarray | None = np.asarray(
            complex_design,
            dtype=np.complex128,
        ).reshape(
            points.shape[0],
            support_array.size,
            indices.size,
        )
        backend = "union_support_vectorized"
        cache_bytes = int(design_by_k.nbytes)
    else:
        design_by_k = None
        backend = "sparse_low_memory"
        cache_bytes = 0
    return _PublicBandResponseEvaluator(
        basis=basis,
        coefficients0=coefficients.copy(),
        channel_indices=indices.copy(),
        y0=coefficients[indices].copy(),
        kpoints=points.copy(),
        base_hamiltonians=np.asarray(
            basis.hamiltonians(points, coefficients),
            dtype=np.complex128,
        ),
        support=support_array,
        design_by_k=design_by_k,
        backend=backend,
        cache_bytes=cache_bytes,
    )



def _public_band_residual_and_jacobian(
    basis: Any,
    full_coefficients: np.ndarray,
    channel_indices: np.ndarray,
    y: np.ndarray,
    kpoints: np.ndarray,
    target: np.ndarray,
    *,
    target_bands: str,
    bands: int,
    band_loss_weight: float,
    evaluator: _PublicBandResponseEvaluator | None = None,
    weighting: Any | None = None,
    target_eigenvalues: np.ndarray | None = None,
    degeneracy_guard: BandDegeneracyGuard | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Evaluate the public eigenvalue residual and analytic Jacobian."""

    from .response_basis import (
        NormalizedLowEnergyWeightingSpec,
        build_normalized_low_energy_weighting,
    )

    target_array = np.asarray(target, dtype=np.complex128)
    points = np.asarray(kpoints, dtype=float)
    indices = np.asarray(channel_indices, dtype=np.int64)
    if evaluator is None:
        coefficients = np.asarray(full_coefficients, dtype=float).copy()
        coefficients[indices] = np.asarray(y, dtype=float)
        model_h = basis.hamiltonians(points, coefficients)
    else:
        model_h = evaluator.hamiltonians(np.asarray(y, dtype=float))
    model_eigenvalues, model_eigenvectors = np.linalg.eigh(model_h)
    if target_eigenvalues is None:
        target_eigenvalue_array = np.linalg.eigvalsh(target_array)
    else:
        target_eigenvalue_array = np.asarray(target_eigenvalues, dtype=float)
    resolved_weighting = weighting
    if resolved_weighting is None:
        resolved_weighting = build_normalized_low_energy_weighting(
            target_array,
            NormalizedLowEnergyWeightingSpec(
                band_edge=str(target_bands),
                n_bands=int(bands),
                one_sided_weight=0.0,
                two_sided_weight=0.0,
            ),
        )
    selected_mask = resolved_weighting.selected_mask
    active_points = np.ones(len(points), dtype=bool)
    if degeneracy_guard is not None:
        degeneracy_guard.check(model_eigenvalues)
        active_points = degeneracy_guard.active_points
        selected_mask = selected_mask & active_points[:, None]
    if evaluator is None:
        raw_jacobian = _compiled_response_band_jacobian(
            basis,
            indices,
            points,
            model_eigenvectors,
            selected_mask,
        )
    else:
        raw_jacobian = evaluator.band_jacobian(
            model_eigenvectors,
            selected_mask,
        )
    residual_parts: list[np.ndarray] = []
    jacobian_parts: list[np.ndarray] = []
    offset = 0
    n_kpoints = max(1, int(np.count_nonzero(active_points)))
    for k_index, count in enumerate(resolved_weighting.selected_counts):
        if not active_points[k_index]:
            continue
        mask = resolved_weighting.selected_mask[k_index]
        scale = np.sqrt(float(band_loss_weight) / (n_kpoints * int(count)))
        residual_parts.append(
            scale
            * (
                model_eigenvalues[k_index, mask]
                - target_eigenvalue_array[k_index, mask]
            )
        )
        jacobian_parts.append(scale * raw_jacobian[offset : offset + int(count)])
        offset += int(count)
    return (
        np.concatenate(residual_parts),
        np.vstack(jacobian_parts),
        tuple(int(value) if active_points[i] else 0
              for i, value in enumerate(resolved_weighting.selected_counts)),
    )



@retain_coefficients_on_degeneracy
def _refine_public_nonlinear_complete_response(
    moire_config: MoireConfig,
    model_config: configuration.ConfiguredModel,
    model: Any,
    raw_cfg: Mapping[str, Any],
    progress_callback: Callable[..., None] | None = None,
) -> dict[str, Any]:
    from . import validation

    """Add only the public band loss to the exact public linear objective."""

    basis = getattr(model, "_compiled_response_basis", None)
    fitted = getattr(model, "_fitted_response_model", None)
    if basis is None or fitted is None:
        raise ValueError("public nonlinear fitting requires the completed linear response fit")
    indices = np.asarray(fitted.fit_solver_channel_indices, dtype=np.int64)
    if indices.size == 0:
        raise ValueError("public nonlinear fitting has no retained linear solver channels")
    coefficients0 = np.asarray(fitted.coefficients, dtype=float)
    y0 = coefficients0[indices].copy()

    all_kpoints = configuration._load_kpoints(model_config)
    original_heff = np.asarray(
        configuration._load_heff_in_model_basis(model_config),
        dtype=np.complex128,
    )
    target_all = validation._current_heff_support_hamiltonians(
        original_heff,
        qset1=np.asarray(moire_config.Q_set1, dtype=float),
        qset2=np.asarray(moire_config.Q_set2, dtype=float),
        n_orb=tuple(int(value) for value in model_config.n_orb),
        harmonics_config=model_config.harmonics_config,
    )
    h_indices = np.asarray(raw_cfg["hamiltonian_kpoints"], dtype=int)
    band_indices = np.asarray(raw_cfg["band_kpoints"], dtype=int)
    target_bands = str(raw_cfg["target_bands"])
    requested_bands = int(raw_cfg["bands"])
    one_sided_weight = float(raw_cfg["one_sided_weight"])
    two_sided_weight = float(raw_cfg["two_sided_weight"])
    band_loss_weight = float(raw_cfg["band_loss_weight"])
    h_root, h_center, h_report = _public_hamiltonian_quadratic_residual(
        basis,
        indices,
        all_kpoints[h_indices],
        target_all[h_indices],
        target_bands=target_bands,
        bands=requested_bands,
        one_sided_weight=one_sided_weight,
        two_sided_weight=two_sided_weight,
    )
    band_points = all_kpoints[band_indices]
    band_target = target_all[band_indices]
    from .response_basis import (
        NormalizedLowEnergyWeightingSpec,
        build_normalized_low_energy_weighting,
    )

    band_weighting = build_normalized_low_energy_weighting(
        band_target,
        NormalizedLowEnergyWeightingSpec(
            band_edge=target_bands,
            n_bands=requested_bands,
            one_sided_weight=0.0,
            two_sided_weight=0.0,
        ),
    )
    band_target_eigenvalues = np.linalg.eigvalsh(band_target)
    band_response_start = time.perf_counter()
    band_response_evaluator = _prepare_public_band_response_evaluator(
        basis,
        coefficients0,
        indices,
        band_points,
    )
    band_guard = BandDegeneracyGuard(
        band_target_eigenvalues,
        np.linalg.eigvalsh(band_response_evaluator.base_hamiltonians),
        band_weighting.selected_mask,
        tolerance_ev=raw_cfg.get("degeneracy_tol_ev", DEFAULT_DEGENERACY_TOL_EV),
    )
    band_guard.require_usable()
    band_response_seconds = float(time.perf_counter() - band_response_start)
    band_response_message = (
        "nonlinear band response backend: "
        f"{band_response_evaluator.backend}, "
        f"support={band_response_evaluator.support.size}, "
        f"cache={band_response_evaluator.cache_bytes / (1024**2):.1f} MiB, "
        f"{band_response_seconds:.2f} s"
    )
    if progress_callback is not None:
        progress_callback(band_response_message, state="done")
    else:
        print(f"[kp model] {band_response_message}", flush=True)

    initial_band_residual, _initial_band_jacobian, resolved_counts = (
        _public_band_residual_and_jacobian(
            basis,
            coefficients0,
            indices,
            y0,
            band_points,
            band_target,
            target_bands=target_bands,
            bands=requested_bands,
            band_loss_weight=band_loss_weight,
            evaluator=band_response_evaluator,
            weighting=band_weighting,
            target_eigenvalues=band_target_eigenvalues,
            degeneracy_guard=band_guard,
        )
    )

    max_steps = int(raw_cfg["max_steps"])
    evaluation_cache: dict[str, Any] = {}
    loss_evaluations = 0

    def evaluate_band(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(y, dtype=float)
        cached_y = evaluation_cache.get("y")
        if cached_y is None or not np.array_equal(cached_y, values):
            band_residual, band_jacobian, _counts = _public_band_residual_and_jacobian(
                basis,
                coefficients0,
                indices,
                values,
                band_points,
                band_target,
                target_bands=target_bands,
                bands=requested_bands,
                band_loss_weight=band_loss_weight,
                evaluator=band_response_evaluator,
                weighting=band_weighting,
                target_eigenvalues=band_target_eigenvalues,
                degeneracy_guard=band_guard,
            )
            evaluation_cache.update(
                {
                    "y": values.copy(),
                    "residual": band_residual,
                    "jacobian": band_jacobian,
                }
            )
        return evaluation_cache["residual"], evaluation_cache["jacobian"]

    def objective(y: np.ndarray) -> np.ndarray:
        nonlocal loss_evaluations
        loss_evaluations += 1
        if progress_callback is not None:
            progress_callback(
                f"nonlinear loss evaluation {loss_evaluations}/{max_steps}",
                state="start",
            )
        band_residual, _band_jacobian = evaluate_band(y)
        return np.concatenate((h_root @ np.asarray(y, dtype=float) - h_center, band_residual))

    def jacobian(y: np.ndarray) -> np.ndarray:
        _band_residual, band_jacobian = evaluate_band(y)
        return np.vstack((h_root, band_jacobian))

    result = scipy.optimize.least_squares(
        objective,
        y0,
        jac=jacobian,
        max_nfev=max_steps,
    )
    final_y = np.asarray(result.x, dtype=float)
    final_band_residual, _final_jacobian, final_counts = _public_band_residual_and_jacobian(
        basis,
        coefficients0,
        indices,
        final_y,
        band_points,
        band_target,
        target_bands=target_bands,
        bands=requested_bands,
        band_loss_weight=band_loss_weight,
        evaluator=band_response_evaluator,
        weighting=band_weighting,
        target_eigenvalues=band_target_eigenvalues,
        degeneracy_guard=band_guard,
    )
    coefficients = coefficients0.copy()
    coefficients[indices] = final_y
    refined_fit = fitted.with_coefficients(
        coefficients,
        channel_ids=basis.channel_ids,
        response_scales=basis.response_scales,
        provenance={
            "operation": "public_nonlinear",
            "hamiltonian_kpoints": h_indices.tolist(),
            "band_kpoints": band_indices.tolist(),
            "requested_bands": requested_bands,
            "resolved_band_counts": list(final_counts),
            "band_loss_weight": band_loss_weight,
            "max_steps": max_steps,
        },
    )
    setattr(model, "_fitted_response_model", refined_fit)
    _sync_complete_response_coefficients_to_terms(model, coefficients)
    return {
        "enabled": True,
        "mode": "public_nonlinear",
        "solver": "scipy_least_squares_analytic_hf",
        "reference": "current_heff_support_mask",
        "hamiltonian_kpoints": h_indices.tolist(),
        "band_kpoints": band_indices.tolist(),
        "requested_bands": requested_bands,
        "resolved_band_counts": list(final_counts),
        "one_sided_weight": one_sided_weight,
        "two_sided_weight": two_sided_weight,
        "band_loss_weight": band_loss_weight,
        "degeneracy_guard": band_guard.report(),
        "band_response_backend": band_response_evaluator.backend,
        "band_response_support_entries": int(band_response_evaluator.support.size),
        "band_response_cache_bytes": int(band_response_evaluator.cache_bytes),
        "band_response_cache_seconds": band_response_seconds,
        "hamiltonian_loss": h_report,
        "linear_initial_band_loss": float(np.sum(initial_band_residual**2)),
        "nonlinear_final_band_loss": float(np.sum(final_band_residual**2)),
        "initial_coefficients": y0.tolist(),
        "final_coefficients": final_y.tolist(),
        "max_steps": max_steps,
        "nfev": int(result.nfev),
        "njev": None if result.njev is None else int(result.njev),
        "cost": float(result.cost),
        "status": int(result.status),
        "message": str(result.message),
        "accepted": True,
        "acceptance_guard": {"enabled": False, "reason": "public nonlinear result is not silently rolled back"},
    }



def _refinement_initial_fitted_model(
    raw_cfg: Mapping[str, Any],
    basis: Any,
    fitted: Any,
) -> tuple[Any, dict[str, Any] | None]:
    """Optionally initialize refinement from a certified fit on the same basis."""
    raw_path = raw_cfg.get("initial_model_data")
    if raw_path in {None, ""}:
        return fitted, None
    path = Path(str(raw_path)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"fit.refine_bands.initial_model_data not found: {path}")
    with np.load(path, allow_pickle=False) as arrays:
        if "fitted_coefficients" not in arrays or "basis_hash" not in arrays:
            raise ValueError(
                "fit.refine_bands.initial_model_data must contain fitted_coefficients and basis_hash"
            )
        coefficients = np.asarray(arrays["fitted_coefficients"], dtype=float)
        source_basis_hash = str(np.asarray(arrays["basis_hash"]).item())
    if source_basis_hash != str(basis.basis_hash):
        raise ValueError(
            "fit.refine_bands.initial_model_data belongs to a different response basis: "
            f"source={source_basis_hash}, current={basis.basis_hash}"
        )
    expected_shape = np.asarray(fitted.coefficients, dtype=float).shape
    if coefficients.shape != expected_shape:
        raise ValueError(
            "fit.refine_bands.initial_model_data coefficient shape mismatch: "
            f"source={coefficients.shape}, current={expected_shape}"
        )
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("fit.refine_bands.initial_model_data contains non-finite coefficients")
    initialized = fitted.with_coefficients(
        coefficients,
        channel_ids=basis.channel_ids,
        response_scales=basis.response_scales,
        provenance={
            "operation": "refinement_initialization",
            "source_model_data": str(path),
        },
    )
    return initialized, {
        "source": str(path),
        "basis_hash": source_basis_hash,
        "coefficient_count": int(coefficients.size),
    }



@retain_coefficients_on_degeneracy
def _refine_complete_response_basis(
    moire_config: MoireConfig,
    model_config: configuration.ConfiguredModel,
    model: Any,
    raw_cfg: Mapping[str, Any],
    progress_callback: Callable[..., None] | None = None,
) -> dict[str, Any]:
    from . import validation

    """Refine coefficients in the fitted complete response basis."""
    if str(raw_cfg.get("mode", "")).strip().lower() == "public_nonlinear":
        return _refine_public_nonlinear_complete_response(
            moire_config,
            model_config,
            model,
            raw_cfg,
            progress_callback=progress_callback,
        )
    basis = getattr(model, "_compiled_response_basis", None)
    fitted = getattr(model, "_fitted_response_model", None)
    if basis is None or fitted is None:
        raise ValueError("complete_linear refinement requires a fitted CompiledResponseBasis")
    if fitted.basis_hash != basis.basis_hash:
        raise ValueError("complete_linear refinement received a fit from a different basis")
    fitted, initialization_report = _refinement_initial_fitted_model(raw_cfg, basis, fitted)
    if initialization_report is not None:
        setattr(model, "_fitted_response_model", fitted)
        _sync_complete_response_coefficients_to_terms(model, fitted.coefficients)

    refine_row_selector: Sequence[int] | None = getattr(model_config, "band_indices", None)
    kpoints = np.asarray(moire_config.kpoints, dtype=float)
    fit_kpoints_report = {
        "source": "band_indices",
        "count": int(len(kpoints)),
        "selected_indices": [int(value) for value in (refine_row_selector or [])],
    }
    if raw_cfg.get("indices") is not None:
        explicit_indices = configuration._as_int_list(raw_cfg.get("indices"), name="fit.refine_bands.indices")
        for index in explicit_indices:
            if index < 0 or index >= len(kpoints):
                raise IndexError(
                    f"fit.refine_bands.indices contains {index}, but available k indices are 0..{len(kpoints) - 1}"
                )
        kpoints = configuration._select_rows(kpoints, explicit_indices)
        refine_row_selector = [int(value) for value in explicit_indices]
        fit_kpoints_report = {
            "source": "fit.refine_bands.indices",
            "count": int(len(kpoints)),
            "selected_indices": list(refine_row_selector),
        }
    else:
        use_fit_kpoints = bool(
            raw_cfg.get(
                "use_fit_kpoints",
                str(raw_cfg.get("mode", "")).strip().lower() == "auto_low_energy",
            )
        )
        fit_indices = list(getattr(model_config, "fit_indices", []) or [])
        if use_fit_kpoints and moire_config.kpoints_fit is not None and fit_indices:
            kpoints = np.asarray(moire_config.kpoints_fit, dtype=float)
            refine_row_selector = [int(value) for value in fit_indices]
            fit_kpoints_report = {
                "source": "fit.indices",
                "count": int(len(kpoints)),
                "selected_indices": list(refine_row_selector),
            }

    if getattr(model_config, "heff_file", None) is None:
        raise ValueError("fit.refine_bands requires the projected Heff file")
    heff_all, heff_eig, target_reference = validation._refinement_target_hamiltonians(
        moire_config,
        model_config,
        raw_cfg,
        refine_row_selector,
    )
    if int(heff_all.shape[0]) != int(len(kpoints)):
        raise ValueError(
            "fit.refine_bands k-point/Heff row mismatch: "
            f"kpoints={len(kpoints)}, heff_rows={heff_all.shape[0]}, source={fit_kpoints_report['source']}"
        )

    raw_tags = raw_cfg.get("variable_tags", ["Kinect", "Onsite", "inter"])
    if isinstance(raw_tags, str):
        raw_tags = [raw_tags]
    selected_tags = {_refinement_tag_alias(str(tag)) for tag in raw_tags}
    raw_components = raw_cfg.get("components", ["real"])
    if isinstance(raw_components, str):
        raw_components = [raw_components]
    selected_components = {str(component).strip().lower() for component in raw_components}
    if not selected_components <= {"real", "imag"}:
        raise ValueError("fit.refine_bands.components must contain only 'real' and/or 'imag'")

    solver = str(raw_cfg.get("solver", "")).strip().lower()
    if solver in {
        "linear_target_subspace_action",
        "target_subspace_action_linear",
        "linear_wavefunction_action",
    }:
        candidate_indices: list[int] = []
        excluded_unconfirmed_channels: list[str] = []
        for index, channel in enumerate(basis.channels):
            tag = _refinement_tag_alias(str(channel.metadata.get("tag", "")))
            if tag not in selected_tags or channel.component not in selected_components:
                continue
            if channel.classification != "confirmed_nonzero":
                excluded_unconfirmed_channels.append(channel.channel_id)
                continue
            candidate_indices.append(index)
        if not candidate_indices:
            raise ValueError("fit.refine_bands selected no retained complete-response channels")

        base_h = _model_hamiltonians_for_kpoints(moire_config, model, kpoints)
        band_slice = _refinement_band_slice(raw_cfg, model_config, base_h.shape[-1])
        align = str(
            raw_cfg.get(
                "align",
                getattr(model_config, "band_plot_config", {}).get("align", "top"),
            )
        ).strip().lower()
        initial_metrics = _band_refinement_metrics(
            base_h,
            heff_eig,
            heff_all,
            band_slice=band_slice,
            align=align,
        )
        _target_eigvals, target_eigenvectors = np.linalg.eigh(heff_all)
        slice_start, slice_stop = (int(value) for value in band_slice)
        target_frames = target_eigenvectors[:, :, slice_start:slice_stop]
        action_start = time.perf_counter()
        action_by_k = basis.response_action(
            kpoints,
            target_frames,
            channel_indices=np.asarray(candidate_indices, dtype=np.int64),
        )
        action_norms = np.sqrt(
            np.sum(np.abs(action_by_k) ** 2, axis=(0, 2, 3))
        )
        norm_tol = float(raw_cfg.get("variable_norm_tol", 1.0e-12))
        retained_mask = action_norms > norm_tol
        zero_grid_channels = [
            basis.channels[index].channel_id
            for index, retained in zip(candidate_indices, retained_mask)
            if not bool(retained)
        ]
        selected_indices = [
            index
            for index, retained in zip(candidate_indices, retained_mask)
            if bool(retained)
        ]
        if not selected_indices:
            raise ValueError("target-subspace action refinement has zero response rank")
        response_actions = np.moveaxis(action_by_k[:, retained_mask], 1, 0)
        action_seconds = float(time.perf_counter() - action_start)

        max_variables_raw = raw_cfg.get("max_variables")
        max_variables = int(max_variables_raw) if max_variables_raw is not None else None
        if max_variables is not None and max_variables > 0 and len(selected_indices) > max_variables:
            raise ValueError(
                "target-subspace action refinement selected "
                f"{len(selected_indices)} variables, exceeding max_variables={max_variables}"
            )
        selected = np.asarray(selected_indices, dtype=np.int64)
        coefficients0 = np.asarray(fitted.coefficients, dtype=float)
        y0 = coefficients0[selected].copy()
        response_scales = np.asarray(basis.response_scales[selected], dtype=float)
        delta, linear_report = _solve_target_subspace_action_linear(
            response_actions,
            base_h,
            heff_all,
            target_frames,
            regularization=float(
                raw_cfg.get("regularization", raw_cfg.get("linear_regularization", 0.0))
            ),
            response_scales=response_scales,
        )
        candidate_y = y0 + delta

        def h_from_y(y: np.ndarray) -> np.ndarray:
            coefficients = coefficients0.copy()
            coefficients[selected] = np.asarray(y, dtype=float)
            return basis.hamiltonians(kpoints, coefficients)

        accepted_y, acceptance_guard = _apply_refinement_acceptance_guard(
            raw_cfg=raw_cfg,
            model_config=model_config,
            base_h=base_h,
            heff_eig=heff_eig,
            heff_all=heff_all,
            y0=y0,
            candidate_y=candidate_y,
            h_from_y=h_from_y,
        )
        coefficients = coefficients0.copy()
        coefficients[selected] = accepted_y
        response_report = {
            "mode": "compiled_sparse_target_subspace_action",
            "basis_hash": basis.basis_hash,
            "candidate_channels": int(basis.candidate_artifact.get("candidate_channel_count", 0)),
            "retained_basis_channels": int(len(basis.channels)),
            "requested_refinement_channels": int(
                len(selected_indices) + len(zero_grid_channels) + len(excluded_unconfirmed_channels)
            ),
            "refinement_channels": int(len(selected_indices)),
            "zero_on_target_action": zero_grid_channels,
            "excluded_unconfirmed_channels": excluded_unconfirmed_channels,
            "selected_channel_ids": [basis.channels[index].channel_id for index in selected_indices],
            "wall_seconds": action_seconds,
            "dense_response_tensor_materialized": False,
        }
        refined_fit = fitted.with_coefficients(
            coefficients,
            channel_ids=basis.channel_ids,
            response_scales=basis.response_scales,
            provenance={
                "operation": "target_subspace_action_refinement",
                "reference": target_reference,
                "selected_channel_ids": response_report["selected_channel_ids"],
                "fit_kpoints": fit_kpoints_report,
                "band_slice": list(band_slice),
            },
        )
        setattr(model, "_fitted_response_model", refined_fit)
        _sync_complete_response_coefficients_to_terms(model, coefficients)
        refined_h = h_from_y(accepted_y)
        refined_metrics = _band_refinement_metrics(
            refined_h,
            heff_eig,
            heff_all,
            band_slice=band_slice,
            align=align,
        )
        normalized_drift = np.abs((accepted_y - y0) * response_scales)
        return {
            "enabled": True,
            "solver": "linear_target_subspace_action",
            "initialization": initialization_report,
            "reference": target_reference,
            "fit_kpoints": fit_kpoints_report,
            "band_slice": [slice_start, slice_stop],
            "align": align,
            "variable_tags": sorted(selected_tags),
            "components": sorted(selected_components),
            "n_variables": int(len(selected_indices)),
            "max_variables": max_variables,
            "response_basis": response_report,
            "linear_system": linear_report,
            "acceptance_guard": acceptance_guard if acceptance_guard is not None else {"enabled": False},
            "initial": initial_metrics,
            "refined": refined_metrics,
            "max_scaled_coefficient_drift": float(np.max(normalized_drift)) if normalized_drift.size else 0.0,
            "p95_scaled_coefficient_drift": (
                float(np.quantile(normalized_drift, 0.95)) if normalized_drift.size else 0.0
            ),
            "nfev": 0,
        }

    all_responses = basis.response_tensor(kpoints)
    norm_tol = float(raw_cfg.get("variable_norm_tol", 1.0e-12))
    selected_indices = []
    zero_grid_channels = []
    excluded_unconfirmed_channels = []
    for index, channel in enumerate(basis.channels):
        tag = _refinement_tag_alias(str(channel.metadata.get("tag", "")))
        if tag not in selected_tags or channel.component not in selected_components:
            continue
        if channel.classification != "confirmed_nonzero":
            excluded_unconfirmed_channels.append(channel.channel_id)
            continue
        if float(np.linalg.norm(all_responses[:, index])) <= norm_tol:
            zero_grid_channels.append(channel.channel_id)
            continue
        selected_indices.append(index)
    if not selected_indices:
        raise ValueError("fit.refine_bands selected no retained complete-response channels")

    max_variables_raw = raw_cfg.get("max_variables")
    max_variables = int(max_variables_raw) if max_variables_raw is not None else None
    base_h = _model_hamiltonians_for_kpoints(moire_config, model, kpoints)
    band_slice = _refinement_band_slice(raw_cfg, model_config, base_h.shape[-1])
    align = str(raw_cfg.get("align", getattr(model_config, "band_plot_config", {}).get("align", "top"))).strip().lower()
    initial_metrics = _band_refinement_metrics(
        base_h,
        heff_eig,
        heff_all,
        band_slice=band_slice,
        align=align,
    )
    response_report = {
        "mode": "compiled_response_basis",
        "basis_hash": basis.basis_hash,
        "candidate_channels": int(basis.candidate_artifact.get("candidate_channel_count", 0)),
        "retained_basis_channels": int(len(basis.channels)),
        "fit_solver_channels": int(len(fitted.fit_solver_channel_ids)),
        "fit_selected_channels": int(len(fitted.fit_solver_channel_ids)),
        "fit_rrqr_pivot_channels": int(len(fitted.fit_selected_channel_ids)),
        "requested_refinement_channels": int(
            len(selected_indices) + len(zero_grid_channels) + len(excluded_unconfirmed_channels)
        ),
        "confirmed_refinement_channels": int(len(selected_indices) + len(zero_grid_channels)),
        "refinement_channels": int(len(selected_indices)),
        "zero_on_refinement_grid": list(zero_grid_channels),
        "excluded_unconfirmed_channels": list(excluded_unconfirmed_channels),
        "selected_channel_ids": [basis.channels[index].channel_id for index in selected_indices],
    }
    if max_variables is not None and max_variables > 0 and len(selected_indices) > max_variables:
        return {
            "enabled": True,
            "skipped": True,
            "reason": "variable_count_exceeds_max_variables",
            "fit_kpoints": fit_kpoints_report,
            "band_slice": [int(band_slice[0]), int(band_slice[1])],
            "align": align,
            "n_variables": int(len(selected_indices)),
            "max_variables": int(max_variables),
            "response_basis": response_report,
            "initial": initial_metrics,
            "refined": initial_metrics,
            "max_nfev": int(raw_cfg.get("max_nfev", 250)),
            "nfev": 0,
        }

    indices = np.asarray(selected_indices, dtype=int)
    responses = np.moveaxis(all_responses[:, indices], 1, 0)
    coefficients0 = np.asarray(fitted.coefficients, dtype=float)
    y0 = coefficients0[indices].copy()
    target_slice = heff_eig[:, band_slice[0] : band_slice[1]]
    band_sigma = float(raw_cfg.get("band_sigma_mev", 1.0)) * 1.0e-3
    matrix_sigma = float(raw_cfg.get("matrix_sigma_mev", 10.0)) * 1.0e-3
    matrix_weight = float(raw_cfg.get("matrix_weight", 0.0))
    coefficient_weight = float(raw_cfg.get("coefficient_weight", 0.0))
    if band_sigma <= 0.0 or matrix_sigma <= 0.0:
        raise ValueError("fit.refine_bands band_sigma_mev and matrix_sigma_mev must be positive")
    if matrix_weight < 0.0 or coefficient_weight < 0.0:
        raise ValueError("fit.refine_bands matrix_weight and coefficient_weight must be non-negative")

    def h_from_y(y: np.ndarray) -> np.ndarray:
        return base_h + np.tensordot(np.asarray(y, dtype=float) - y0, responses, axes=(0, 0))

    def aligned_selected_eigs(hamiltonians: np.ndarray) -> np.ndarray:
        eigenvalues = np.linalg.eigvalsh(hamiltonians)
        band_guard.check(eigenvalues)
        selected = eigenvalues[:, band_slice[0] : band_slice[1]]
        if align == "top":
            selected = selected + (target_slice[0, -1] - selected[0, -1])
        elif align == "bottom":
            selected = selected + (target_slice[0, 0] - selected[0, 0])
        elif align not in {"none", ""}:
            raise ValueError(
                "fit.refine_bands.align currently supports 'top', 'bottom', or 'none', "
                f"got {align!r}"
            )
        return selected

    selected_mask = np.zeros_like(heff_eig, dtype=bool)
    selected_mask[:, band_slice[0]:band_slice[1]] = True
    band_guard = BandDegeneracyGuard(
        heff_eig, np.linalg.eigvalsh(base_h), selected_mask,
        tolerance_ev=raw_cfg.get("degeneracy_tol_ev", DEFAULT_DEGENERACY_TOL_EV),
        alignment=align,
    )
    band_guard.require_usable()
    active_band_points = band_guard.active_points
    response_scales = basis.response_scales[indices]

    def objective(y: np.ndarray) -> np.ndarray:
        hamiltonians = h_from_y(y)
        band_delta = (aligned_selected_eigs(hamiltonians)[active_band_points]
                      - target_slice[active_band_points])
        chunks = [band_delta.reshape(-1) / band_sigma]
        if matrix_weight > 0.0:
            matrix_delta = hamiltonians - heff_all
            matrix_scale = np.sqrt(matrix_weight) / (matrix_sigma * np.sqrt(max(1, matrix_delta.size)))
            chunks.extend(
                [
                    matrix_scale * matrix_delta.real.reshape(-1),
                    matrix_scale * matrix_delta.imag.reshape(-1),
                ]
            )
        if coefficient_weight > 0.0:
            chunks.append(np.sqrt(coefficient_weight) * (np.asarray(y) - y0) * response_scales)
        return np.concatenate(chunks)

    max_nfev = int(raw_cfg.get("max_nfev", 250))
    result = scipy.optimize.least_squares(objective, y0, max_nfev=max_nfev)
    candidate_y = np.asarray(result.x, dtype=float)
    accepted_y, acceptance_guard = _apply_refinement_acceptance_guard(
        raw_cfg=raw_cfg,
        model_config=model_config,
        base_h=base_h,
        heff_eig=heff_eig,
        heff_all=heff_all,
        y0=y0,
        candidate_y=candidate_y,
        h_from_y=h_from_y,
    )
    band_guard.check(np.linalg.eigvalsh(h_from_y(accepted_y)))
    coefficients = coefficients0.copy()
    coefficients[indices] = accepted_y
    refined_fit = fitted.with_coefficients(
        coefficients,
        channel_ids=basis.channel_ids,
        response_scales=basis.response_scales,
        provenance={
            "operation": "band_refinement",
            "selected_channel_ids": response_report["selected_channel_ids"],
            "fit_kpoints": fit_kpoints_report,
            "band_slice": list(band_slice),
            "align": align,
        },
    )
    setattr(model, "_fitted_response_model", refined_fit)
    _sync_complete_response_coefficients_to_terms(model, coefficients)
    refined_h = h_from_y(accepted_y)
    refined_metrics = _band_refinement_metrics(
        refined_h,
        heff_eig,
        heff_all,
        band_slice=band_slice,
        align=align,
    )
    normalized_drift = np.abs((accepted_y - y0) * response_scales)
    return {
        "enabled": True,
        "solver": "compiled_response_least_squares",
        "initialization": initialization_report,
        "reference": target_reference,
        "degeneracy_guard": band_guard.report(),
        "fit_kpoints": fit_kpoints_report,
        "band_slice": [int(band_slice[0]), int(band_slice[1])],
        "align": align,
        "variable_tags": sorted(selected_tags),
        "components": sorted(selected_components),
        "n_variables": int(len(selected_indices)),
        "max_variables": int(max_variables) if max_variables is not None else None,
        "response_basis": response_report,
        "band_sigma_mev": float(band_sigma * 1000.0),
        "matrix_weight": matrix_weight,
        "matrix_sigma_mev": float(matrix_sigma * 1000.0),
        "coefficient_weight": coefficient_weight,
        "acceptance_guard": acceptance_guard if acceptance_guard is not None else {"enabled": False},
        "max_nfev": max_nfev,
        "nfev": int(result.nfev),
        "cost": float(result.cost),
        "optimizer": {
            "mode": "scipy_least_squares",
            "status": int(result.status),
            "message": str(result.message),
        },
        "initial": initial_metrics,
        "refined": refined_metrics,
        "max_scaled_coefficient_drift": float(np.max(normalized_drift)) if normalized_drift.size else 0.0,
        "p95_scaled_coefficient_drift": (
            float(np.quantile(normalized_drift, 0.95)) if normalized_drift.size else 0.0
        ),
        "variables": [
            {
                "channel_id": basis.channels[index].channel_id,
                "tag": str(basis.channels[index].metadata.get("tag", "")),
                "component": basis.channels[index].component,
                "initial": float(initial),
                "refined": float(refined),
                "response_scale": float(basis.response_scales[index]),
            }
            for index, initial, refined in zip(selected_indices, y0, accepted_y)
        ],
    }



@retain_coefficients_on_degeneracy
def refine_band_coefficients(
    moire_config: MoireConfig,
    model_config: configuration.ConfiguredModel,
    model: Any,
    *,
    progress_callback: Callable[..., None] | None = None,
) -> dict[str, Any]:
    raw_cfg = dict(model_config.band_refinement_config or {})
    if not bool(raw_cfg.get("enabled", False)):
        return {"enabled": False}
    if moire_config.kpoints is None:
        raise ValueError("fit.refine_bands requires model kpoints")
    if getattr(model, "_compiled_response_basis", None) is None:
        raise ValueError("band refinement requires a fitted complete response basis")
    return _refine_complete_response_basis(
        moire_config, model_config, model, raw_cfg,
        progress_callback=progress_callback,
    )
