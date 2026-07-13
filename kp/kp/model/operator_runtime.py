from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import sparse

from .core import ContinuumModelBuilder, ContinuumTerm


_DENSE_ACTION_ENTRY_TOL = 1.0e-13
_ORBIT_RECORD_CACHE: dict[
    tuple[int, tuple[Any, ...]],
    list[tuple[np.ndarray, Mapping[str, Any] | None]],
] = {}
_RECIPE_VECTOR_KEYS = (
    "term_index",
    "row",
    "col",
    "mz",
    "mz_star",
    "prefactor_real",
    "prefactor_imag",
)


class CompiledOperatorUnsupported(NotImplementedError):
    """The model cannot be represented by the compiled operator runtime."""


def clear_operator_recipe_cache() -> None:
    _ORBIT_RECORD_CACHE.clear()


def compile_operator_recipe(
    runtime_terms: Sequence[ContinuumTerm],
    moire_config: Any,
    dim: int,
) -> dict[str, np.ndarray]:
    term_index: list[int] = []
    row: list[int] = []
    col: list[int] = []
    mz_values: list[int] = []
    mz_star_values: list[int] = []
    q_center: list[tuple[float, float]] = []
    prefactor_real: list[complex] = []
    prefactor_imag: list[complex] = []
    dynamic_hermitization_terms: list[int] = []
    monomial_transform_cache: dict[
        tuple[tuple[float, ...], tuple[float, float], int, int],
        tuple[int, int, tuple[float, float], complex],
    ] = {}

    def cached_transform_monomial(
        transform: np.ndarray,
        q_base: np.ndarray,
        mz: int,
        mz_star: int,
    ) -> tuple[int, int, tuple[float, float], complex]:
        transform_key = tuple(float(x) for x in np.asarray(transform, dtype=float).ravel())
        q_key = (float(q_base[0]), float(q_base[1]))
        cache_key = (transform_key, q_key, int(mz), int(mz_star))
        cached = monomial_transform_cache.get(cache_key)
        if cached is not None:
            return cached
        mz_new, mz_star_new, q_new, transform_prefactor = transform_monomial(
            transform,
            q_base,
            mz,
            mz_star,
        )
        cached = (
            int(mz_new),
            int(mz_star_new),
            (float(q_new[0]), float(q_new[1])),
            complex(transform_prefactor),
        )
        monomial_transform_cache[cache_key] = cached
        return cached

    for idx, term in enumerate(runtime_terms):
        if not hasattr(term.Y_basis, "eval_sparse"):
            raise CompiledOperatorUnsupported(
                "compiled operator runtime requires sparse term basis metadata"
            )

        rows = np.asarray(getattr(term.Y_basis, "_moire_sparse_rows"), dtype=int)
        cols = np.asarray(getattr(term.Y_basis, "_moire_sparse_cols"), dtype=int)
        row_q_idx = np.asarray(getattr(term.Y_basis, "_moire_sparse_row_q_idx"), dtype=int)
        q_rows = np.asarray(getattr(term.Y_basis, "_moire_sparse_Q_rows"), dtype=float)
        hermitize_in_basis = bool(
            getattr(term.Y_basis, "_moire_sparse_hermitize_in_basis", False)
        )
        if rows.size == 0:
            continue

        orbit_records = _orbit_records(term.symmetry_ops, moire_config.symmetry_gen)
        base_components = [(int(term.key.Mz), int(term.key.Mz_star), 1.0 + 0.0j)]
        if hermitize_in_basis:
            base_components.append(
                (int(term.key.Mz_star), int(term.key.Mz), 1.0 + 0.0j)
            )

        term_contributions: list[
            tuple[int, int, int, int, np.ndarray, complex, complex]
        ] = []
        for transform, action in orbit_records:
            for sparse_pos, r_out, c_out, matrix_factor, is_anti_total in iter_transformed_sparse_entries(
                action,
                rows,
                cols,
            ):
                q_base = q_rows[row_q_idx[sparse_pos]]
                for mz_base, mz_star_base, component_prefactor in base_components:
                    mz_new, mz_star_new, q_new, transform_prefactor = cached_transform_monomial(
                        transform,
                        q_base,
                        mz_base,
                        mz_star_base,
                    )
                    prefactor = component_prefactor * transform_prefactor
                    if is_anti_total:
                        prefactor = np.conjugate(prefactor)
                        mz_new, mz_star_new = mz_star_new, mz_new
                    prefactor *= matrix_factor
                    real_pref = prefactor
                    imag_pref = (1j * (-1.0 if is_anti_total else 1.0)) * prefactor
                    term_contributions.append(
                        (
                            int(r_out),
                            int(c_out),
                            int(mz_new),
                            int(mz_star_new),
                            np.asarray(q_new, dtype=float),
                            complex(real_pref),
                            complex(imag_pref),
                        )
                    )
        term_contributions = ContinuumModelBuilder.filter_operator_contributions_for_term_support(
            term,
            term_contributions,
            dim,
        )
        if not term_contributions:
            continue
        has_recorded_flags = hasattr(term, "_moire_needs_hermitize_real") and hasattr(
            term,
            "_moire_needs_hermitize_imag",
        )
        dynamic_hermitization = has_recorded_flags and bool(
            getattr(term, "_moire_hermitize_flags_inconsistent", False)
        )
        if dynamic_hermitization:
            dynamic_hermitization_terms.append(idx)
            needs_herm_real = False
            needs_herm_imag = False
        elif has_recorded_flags:
            needs_herm_real = bool(getattr(term, "_moire_needs_hermitize_real"))
            needs_herm_imag = bool(getattr(term, "_moire_needs_hermitize_imag"))
        else:
            needs_herm_real, needs_herm_imag = _hermitize_flags_from_operator_contributions(
                term_contributions,
                dim,
                probe_k=_production_hermitization_probe(moire_config),
            )
        for r_out, c_out, mz_new, mz_star_new, q_new, real_pref, imag_pref in term_contributions:
            _append_operator_contribution(
                idx,
                r_out,
                c_out,
                mz_new,
                mz_star_new,
                q_new,
                real_pref,
                imag_pref,
                term_index,
                row,
                col,
                mz_values,
                mz_star_values,
                q_center,
                prefactor_real,
                prefactor_imag,
            )
            herm_real_pref = np.conjugate(real_pref) if needs_herm_real else 0.0j
            herm_imag_pref = np.conjugate(imag_pref) if needs_herm_imag else 0.0j
            if herm_real_pref != 0.0j or herm_imag_pref != 0.0j:
                _append_operator_contribution(
                    idx,
                    c_out,
                    r_out,
                    mz_star_new,
                    mz_new,
                    q_new,
                    herm_real_pref,
                    herm_imag_pref,
                    term_index,
                    row,
                    col,
                    mz_values,
                    mz_star_values,
                    q_center,
                    prefactor_real,
                    prefactor_imag,
                )

    if not term_index:
        raise CompiledOperatorUnsupported(
            "compiled operator recipe produced no operator contributions"
        )
    recipe = {
        "term_index": np.asarray(term_index, dtype=np.int64),
        "row": np.asarray(row, dtype=np.int64),
        "col": np.asarray(col, dtype=np.int64),
        "mz": np.asarray(mz_values, dtype=np.int64),
        "mz_star": np.asarray(mz_star_values, dtype=np.int64),
        "q_center": np.asarray(q_center, dtype=float),
        "prefactor_real": np.asarray(prefactor_real, dtype=np.complex128),
        "prefactor_imag": np.asarray(prefactor_imag, dtype=np.complex128),
        "dynamic_hermitization_term_index": np.asarray(
            dynamic_hermitization_terms,
            dtype=np.int64,
        ),
    }
    return validate_operator_recipe(
        recipe,
        dim=int(dim),
        term_count=len(runtime_terms),
    )


def validate_operator_recipe(
    recipe: Mapping[str, Any],
    *,
    dim: int,
    term_count: int,
) -> dict[str, np.ndarray]:
    missing = [
        key
        for key in (*_RECIPE_VECTOR_KEYS, "q_center", "dynamic_hermitization_term_index")
        if key not in recipe
    ]
    if missing:
        raise ValueError(f"compiled operator recipe is missing arrays: {missing}")
    arrays = {
        "term_index": np.asarray(recipe["term_index"], dtype=np.int64),
        "row": np.asarray(recipe["row"], dtype=np.int64),
        "col": np.asarray(recipe["col"], dtype=np.int64),
        "mz": np.asarray(recipe["mz"], dtype=np.int64),
        "mz_star": np.asarray(recipe["mz_star"], dtype=np.int64),
        "q_center": np.asarray(recipe["q_center"], dtype=float),
        "prefactor_real": np.asarray(recipe["prefactor_real"], dtype=np.complex128),
        "prefactor_imag": np.asarray(recipe["prefactor_imag"], dtype=np.complex128),
        "dynamic_hermitization_term_index": np.asarray(
            recipe["dynamic_hermitization_term_index"],
            dtype=np.int64,
        ),
    }
    lengths = {key: int(arrays[key].size) for key in _RECIPE_VECTOR_KEYS}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"compiled operator recipe vector lengths differ: {lengths}")
    count = lengths["term_index"]
    if arrays["q_center"].shape != (count, 2):
        raise ValueError(
            "compiled operator recipe q_center must have shape "
            f"({count}, 2), got {arrays['q_center'].shape}"
        )
    if count == 0:
        raise ValueError("compiled operator recipe contains no contributions")
    if np.any(arrays["term_index"] < 0) or np.any(arrays["term_index"] >= int(term_count)):
        raise ValueError("compiled operator recipe references an out-of-range term index")
    if np.any(arrays["row"] < 0) or np.any(arrays["row"] >= int(dim)):
        raise ValueError("compiled operator recipe references an out-of-range row")
    if np.any(arrays["col"] < 0) or np.any(arrays["col"] >= int(dim)):
        raise ValueError("compiled operator recipe references an out-of-range column")
    dynamic = arrays["dynamic_hermitization_term_index"]
    if dynamic.ndim != 1:
        raise ValueError("dynamic_hermitization_term_index must be one-dimensional")
    if np.any(dynamic < 0) or np.any(dynamic >= int(term_count)):
        raise ValueError("compiled operator recipe references an out-of-range dynamic term")
    if not np.all(np.isfinite(arrays["q_center"])):
        raise ValueError("compiled operator recipe contains non-finite q centers")
    if not np.all(np.isfinite(arrays["prefactor_real"])) or not np.all(
        np.isfinite(arrays["prefactor_imag"])
    ):
        raise ValueError("compiled operator recipe contains non-finite prefactors")
    return arrays


@dataclass
class CompiledOperatorRuntime:
    recipe: dict[str, np.ndarray]
    r_real: np.ndarray
    r_imag: np.ndarray
    dim: int
    _dynamic_masks: dict[int, np.ndarray] = field(init=False, repr=False)
    _static_mask: np.ndarray = field(init=False, repr=False)
    _static_terms: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.r_real = np.asarray(self.r_real, dtype=float)
        self.r_imag = np.asarray(self.r_imag, dtype=float)
        if self.r_real.ndim != 1 or self.r_real.shape != self.r_imag.shape:
            raise ValueError("compiled operator coefficients must be equal-length vectors")
        self.recipe = validate_operator_recipe(
            self.recipe,
            dim=int(self.dim),
            term_count=int(self.r_real.size),
        )
        dynamic = self.recipe["dynamic_hermitization_term_index"]
        term_index = self.recipe["term_index"]
        self._dynamic_masks = {
            int(term): term_index == int(term)
            for term in dynamic
        }
        if self._dynamic_masks:
            dynamic_all = np.zeros(term_index.shape, dtype=bool)
            for mask in self._dynamic_masks.values():
                dynamic_all |= mask
            self._static_mask = ~dynamic_all
        else:
            self._static_mask = np.ones(term_index.shape, dtype=bool)
        self._static_terms = term_index[self._static_mask]

    @classmethod
    def from_terms(
        cls,
        runtime_terms: Sequence[ContinuumTerm],
        moire_config: Any,
        dim: int,
    ) -> "CompiledOperatorRuntime":
        r_real: list[float] = []
        r_imag: list[float] = []
        for term in runtime_terms:
            if term.r_value_real is None or term.r_value_imag is None:
                raise ValueError(f"Term {term.key} coefficients not assigned!")
            r_real.append(float(term.r_value_real))
            r_imag.append(float(term.r_value_imag))
        recipe = compile_operator_recipe(runtime_terms, moire_config, dim)
        return cls(
            recipe=recipe,
            r_real=np.asarray(r_real, dtype=float),
            r_imag=np.asarray(r_imag, dtype=float),
            dim=int(dim),
        )

    def hamiltonian(self, k: np.ndarray) -> np.ndarray:
        point = np.asarray(k, dtype=float)
        if point.shape != (2,):
            raise ValueError(f"k must have shape (2,), got {point.shape}")
        term_index = self.recipe["term_index"]
        rows = self.recipe["row"]
        cols = self.recipe["col"]
        mz = self.recipe["mz"]
        mz_star = self.recipe["mz_star"]
        q_center = self.recipe["q_center"]
        prefactor_real = self.recipe["prefactor_real"]
        prefactor_imag = self.recipe["prefactor_imag"]
        z = (point[0] - q_center[:, 0]) + 1j * (point[1] - q_center[:, 1])
        monomial = (z**mz) * (np.conjugate(z) ** mz_star)
        hamiltonian = np.zeros((int(self.dim), int(self.dim)), dtype=np.complex128)
        if np.any(self._static_mask):
            coefficient = (
                self.r_real[self._static_terms] * prefactor_real[self._static_mask]
                + self.r_imag[self._static_terms] * prefactor_imag[self._static_mask]
            )
            np.add.at(
                hamiltonian,
                (rows[self._static_mask], cols[self._static_mask]),
                coefficient * monomial[self._static_mask],
            )
        for term, mask in self._dynamic_masks.items():
            h_real = np.zeros_like(hamiltonian)
            h_imag = np.zeros_like(hamiltonian)
            np.add.at(
                h_real,
                (rows[mask], cols[mask]),
                prefactor_real[mask] * monomial[mask],
            )
            np.add.at(
                h_imag,
                (rows[mask], cols[mask]),
                prefactor_imag[mask] * monomial[mask],
            )
            if not np.allclose(h_real, h_real.conj().T):
                h_real += h_real.conj().T
            if not np.allclose(h_imag, h_imag.conj().T):
                h_imag += h_imag.conj().T
            hamiltonian += self.r_real[term] * h_real + self.r_imag[term] * h_imag
        return hamiltonian

    def export_arrays(self) -> dict[str, np.ndarray]:
        return {key: np.asarray(value) for key, value in self.recipe.items()}


def compiled_operator_response_matrix(
    recipe: Mapping[str, Any],
    kpoints: np.ndarray,
    *,
    dim: int,
    term_count: int,
) -> sparse.csc_matrix:
    """Return ``B_j(k) = dH(k)/dc_j`` for every production coefficient.

    Columns are interleaved real/imaginary coefficient channels for each seed
    term.  Rows are entries of the multi-k block-diagonal Hamiltonian.  The
    recipe, symmetry orbit, and conditional Hermitian completion are therefore
    exactly the same ones used by :class:`CompiledOperatorRuntime`.
    """

    arrays = validate_operator_recipe(
        recipe,
        dim=int(dim),
        term_count=int(term_count),
    )
    points = np.asarray(kpoints, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"kpoints must have shape (Nk, 2), got {points.shape}")

    nk = int(points.shape[0])
    block_dim = nk * int(dim)
    response_shape = (block_dim * block_dim, 2 * int(term_count))
    if nk == 0:
        return sparse.csc_matrix(response_shape, dtype=np.complex128)

    term_index = arrays["term_index"]
    rows = arrays["row"]
    cols = arrays["col"]
    mz = arrays["mz"]
    mz_star = arrays["mz_star"]
    q_center = arrays["q_center"]
    prefactor_real = arrays["prefactor_real"]
    prefactor_imag = arrays["prefactor_imag"]
    dynamic_terms = arrays["dynamic_hermitization_term_index"]
    dynamic_mask = (
        np.isin(term_index, dynamic_terms)
        if dynamic_terms.size
        else np.zeros(term_index.shape, dtype=bool)
    )
    static_mask = ~dynamic_mask

    response_rows: list[np.ndarray] = []
    response_cols: list[np.ndarray] = []
    response_values: list[np.ndarray] = []

    for k_index, point in enumerate(points):
        z = (point[0] - q_center[:, 0]) + 1j * (point[1] - q_center[:, 1])
        monomial = (z**mz) * (np.conjugate(z) ** mz_star)
        if np.any(static_mask):
            global_rows = k_index * int(dim) + rows[static_mask]
            global_cols = k_index * int(dim) + cols[static_mask]
            flat_rows = global_rows * block_dim + global_cols
            static_terms = term_index[static_mask]
            response_rows.extend((flat_rows, flat_rows))
            response_cols.extend((2 * static_terms, 2 * static_terms + 1))
            response_values.extend(
                (
                    prefactor_real[static_mask] * monomial[static_mask],
                    prefactor_imag[static_mask] * monomial[static_mask],
                )
            )

        for term in dynamic_terms:
            mask = term_index == int(term)
            h_real = np.zeros((int(dim), int(dim)), dtype=np.complex128)
            h_imag = np.zeros_like(h_real)
            np.add.at(
                h_real,
                (rows[mask], cols[mask]),
                prefactor_real[mask] * monomial[mask],
            )
            np.add.at(
                h_imag,
                (rows[mask], cols[mask]),
                prefactor_imag[mask] * monomial[mask],
            )
            if not np.allclose(h_real, h_real.conj().T):
                h_real += h_real.conj().T
            if not np.allclose(h_imag, h_imag.conj().T):
                h_imag += h_imag.conj().T
            for channel, matrix in enumerate((h_real, h_imag)):
                local_rows, local_cols = np.nonzero(matrix)
                if local_rows.size == 0:
                    continue
                global_rows = k_index * int(dim) + local_rows
                global_cols = k_index * int(dim) + local_cols
                response_rows.append(global_rows * block_dim + global_cols)
                response_cols.append(
                    np.full(local_rows.shape, 2 * int(term) + channel, dtype=np.int64)
                )
                response_values.append(matrix[local_rows, local_cols])

    if not response_values:
        return sparse.csc_matrix(response_shape, dtype=np.complex128)
    matrix = sparse.coo_matrix(
        (
            np.concatenate(response_values),
            (np.concatenate(response_rows), np.concatenate(response_cols)),
        ),
        shape=response_shape,
        dtype=np.complex128,
    ).tocsc()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return matrix


def _hermitize_flags_from_operator_contributions(
    contributions: Sequence[tuple[int, int, int, int, np.ndarray, complex, complex]],
    dim: int,
    *,
    probe_k: np.ndarray | None = None,
) -> tuple[bool, bool]:
    if not contributions:
        return False, False
    point = (
        np.array([0.137, -0.219], dtype=float)
        if probe_k is None
        else np.asarray(probe_k, dtype=float)
    )
    if point.shape != (2,):
        raise ValueError(f"hermitianization probe must have shape (2,), got {point.shape}")
    real_matrix = np.zeros((int(dim), int(dim)), dtype=np.complex128)
    imag_matrix = np.zeros_like(real_matrix)
    for row_idx, col_idx, mz, mz_star, q, real_pref, imag_pref in contributions:
        z = (point[0] - float(q[0])) + 1j * (point[1] - float(q[1]))
        monomial = (z ** int(mz)) * (np.conjugate(z) ** int(mz_star))
        real_matrix[int(row_idx), int(col_idx)] += complex(real_pref) * monomial
        imag_matrix[int(row_idx), int(col_idx)] += complex(imag_pref) * monomial
    return (
        not np.allclose(real_matrix, real_matrix.conj().T),
        not np.allclose(imag_matrix, imag_matrix.conj().T),
    )


def _production_hermitization_probe(moire_config: Any) -> np.ndarray:
    for attribute in ("kpoints_fit", "kpoints"):
        values = getattr(moire_config, attribute, None)
        if values is None:
            continue
        points = np.asarray(values, dtype=float)
        if points.ndim == 2 and points.shape[0] > 0 and points.shape[1] == 2:
            return points[0]
    return np.array([0.137, -0.219], dtype=float)


def _append_operator_contribution(
    term_idx: int,
    row_idx: int,
    col_idx: int,
    mz: int,
    mz_star: int,
    q: np.ndarray,
    real_prefactor: complex,
    imag_prefactor: complex,
    term_index: list[int],
    rows: list[int],
    cols: list[int],
    mz_values: list[int],
    mz_star_values: list[int],
    q_center: list[tuple[float, float]],
    prefactor_real: list[complex],
    prefactor_imag: list[complex],
) -> None:
    if abs(real_prefactor) <= _DENSE_ACTION_ENTRY_TOL and abs(
        imag_prefactor
    ) <= _DENSE_ACTION_ENTRY_TOL:
        return
    term_index.append(int(term_idx))
    rows.append(int(row_idx))
    cols.append(int(col_idx))
    mz_values.append(int(mz))
    mz_star_values.append(int(mz_star))
    q_center.append((float(q[0]), float(q[1])))
    prefactor_real.append(complex(real_prefactor))
    prefactor_imag.append(complex(imag_prefactor))


def iter_transformed_sparse_entries(
    action: Mapping[str, Any] | None,
    rows: np.ndarray,
    cols: np.ndarray,
):
    if action is None:
        for sparse_pos, (r_out, c_out) in enumerate(zip(rows, cols)):
            yield sparse_pos, int(r_out), int(c_out), 1.0 + 0.0j, False
        return

    kind = str(action.get("kind", ""))
    if kind == "monomial":
        perm = np.asarray(action["perm"], dtype=int)
        vals = np.asarray(action["vals"], dtype=np.complex128)
        inv_vals = np.asarray(action["inv_vals"], dtype=np.complex128)
        inv_perm = np.empty_like(perm)
        inv_perm[perm] = np.arange(perm.shape[0], dtype=int)
        rr = inv_perm[rows]
        cc = inv_perm[cols]
        matrix_factor = vals[rr] * inv_vals[cc]
        is_anti_total = bool(action.get("is_anti_total", False))
        for sparse_pos, (r_out, c_out, factor) in enumerate(zip(rr, cc, matrix_factor)):
            yield sparse_pos, int(r_out), int(c_out), complex(factor), is_anti_total
        return

    if kind == "dense":
        left = np.asarray(action["left"], dtype=np.complex128)
        right = np.asarray(action["right"], dtype=np.complex128)
        is_anti_total = bool(action.get("is_anti_total", False))
        for sparse_pos, (row_in, col_in) in enumerate(zip(rows, cols)):
            left_col = left[:, int(row_in)]
            right_row = right[int(col_in), :]
            out_rows = np.flatnonzero(np.abs(left_col) > _DENSE_ACTION_ENTRY_TOL)
            out_cols = np.flatnonzero(np.abs(right_row) > _DENSE_ACTION_ENTRY_TOL)
            for r_out in out_rows:
                left_value = left_col[int(r_out)]
                for c_out in out_cols:
                    factor = left_value * right_row[int(c_out)]
                    if abs(factor) <= _DENSE_ACTION_ENTRY_TOL:
                        continue
                    yield sparse_pos, int(r_out), int(c_out), complex(factor), is_anti_total
        return

    raise ValueError(f"unknown compiled symmetry action kind {kind!r}")


def _orbit_records(
    sym_ops: Sequence[Mapping[str, Any]],
    symmetry_gen: Any,
) -> list[tuple[np.ndarray, Mapping[str, Any] | None]]:
    if not sym_ops:
        return [(np.eye(2, dtype=float), None)]
    sym_ops_list = [dict(op) for op in sym_ops if isinstance(op, Mapping)]
    cache_key = (id(symmetry_gen), ContinuumModelBuilder._sym_ops_cache_key(sym_ops_list))
    cached = _ORBIT_RECORD_CACHE.get(cache_key)
    if cached is not None:
        return cached
    _points, op_seqs = ContinuumModelBuilder._generate_symmetry_orbit(
        np.zeros(2, dtype=float),
        sym_ops_list,
    )
    records: list[tuple[np.ndarray, Mapping[str, Any] | None]] = []
    for op_seq in op_seqs:
        transform = _linear_transform_for_op_seq(op_seq, sym_ops_list)
        if not op_seq:
            records.append((transform, None))
            continue
        op_seq_applied = tuple(reversed(op_seq))
        action = ContinuumModelBuilder._get_composed_symmetry_action(
            symmetry_gen,
            op_seq_applied,
        )
        if action is not None and ContinuumModelBuilder._validate_composed_symmetry_action_once(
            symmetry_gen,
            op_seq_applied,
            action,
        ):
            perm, vals, inv_vals, is_anti_total = action
            records.append(
                (
                    transform,
                    {
                        "kind": "monomial",
                        "perm": perm,
                        "vals": vals,
                        "inv_vals": inv_vals,
                        "is_anti_total": bool(is_anti_total),
                    },
                )
            )
            continue
        records.append(
            (transform, dense_composed_symmetry_action(symmetry_gen, op_seq_applied))
        )
    _ORBIT_RECORD_CACHE[cache_key] = records
    return records


def dense_composed_symmetry_action(
    symmetry_gen: Any,
    op_seq_applied: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    if symmetry_gen is None:
        raise CompiledOperatorUnsupported(
            "compiled operator runtime requires symmetry_gen when sym_ops is non-empty"
        )
    left_total: np.ndarray | None = None
    right_total: np.ndarray | None = None
    is_anti_total = False
    dim: int | None = None
    for op_name, param in op_seq_applied:
        left_op, right_op, is_anti_op = _dense_single_symmetry_action(
            symmetry_gen,
            str(op_name),
            param,
        )
        left_op = np.asarray(left_op, dtype=np.complex128)
        right_op = np.asarray(right_op, dtype=np.complex128)
        if left_op.ndim != 2 or left_op.shape[0] != left_op.shape[1]:
            raise ValueError(f"symmetry operator {op_name!r} must be square, got {left_op.shape}")
        if right_op.shape != left_op.shape:
            raise ValueError(
                f"right action for symmetry operator {op_name!r} has shape {right_op.shape}, "
                f"expected {left_op.shape}"
            )
        if dim is None:
            dim = int(left_op.shape[0])
            left_total = np.eye(dim, dtype=np.complex128)
            right_total = np.eye(dim, dtype=np.complex128)
        elif left_op.shape != (dim, dim):
            raise ValueError(
                f"symmetry operator {op_name!r} shape {left_op.shape} does not match {(dim, dim)}"
            )
        assert left_total is not None and right_total is not None
        if is_anti_op:
            left_total = left_op @ left_total.conj()
            right_total = right_total.conj() @ right_op
            is_anti_total = not is_anti_total
        else:
            left_total = left_op @ left_total
            right_total = right_total @ right_op
    if left_total is None or right_total is None:
        raise ValueError("dense composed symmetry action requires at least one operation")
    return {
        "kind": "dense",
        "left": left_total,
        "right": right_total,
        "is_anti_total": bool(is_anti_total),
    }


def _dense_single_symmetry_action(
    symmetry_gen: Any,
    op_name: str,
    param: Any,
) -> tuple[np.ndarray, np.ndarray, bool]:
    if op_name == "C3z":
        left = symmetry_gen.get_operator(op_name, param)
        right = symmetry_gen.get_operator(op_name, -int(param))
        return left, right, False
    if op_name in ContinuumModelBuilder._SYMM_ANTIUNITARY_OPS:
        left = symmetry_gen.get_operator(op_name, param)
        return left, np.asarray(left).conj().T, True
    if op_name in ContinuumModelBuilder._SYMM_UNITARY_OPS:
        left = symmetry_gen.get_operator(op_name, param)
        return left, np.asarray(left).conj().T, False
    raise ValueError(f"Unknown symmetry operation: {op_name}")


def _linear_transform_for_op_seq(
    op_seq: Sequence[tuple[str, Any]],
    sym_ops: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    op_by_name = {
        str(op["name"]): op
        for op in sym_ops
        if isinstance(op, Mapping) and "name" in op
    }
    columns = []
    for vector in np.eye(2, dtype=float):
        out = np.asarray(vector, dtype=float)
        for op_name, param in op_seq:
            operation = op_by_name[str(op_name)]
            out = ContinuumModelBuilder._apply_k_map_to_vector(
                out,
                operation,
                power=None if param is None else int(param),
            )
        columns.append(out)
    return np.column_stack(columns)


def transform_monomial(
    transform: np.ndarray,
    q_base: np.ndarray,
    mz: int,
    mz_star: int,
) -> tuple[int, int, np.ndarray, complex]:
    q_center = np.linalg.solve(transform, np.asarray(q_base, dtype=float))
    z_e1 = complex(transform[0, 0], transform[1, 0])
    z_e2 = complex(transform[0, 1], transform[1, 1])
    det = float(np.linalg.det(transform))
    if det > 0:
        alpha = z_e1
        if not np.allclose(z_e2, 1j * alpha, atol=1.0e-8, rtol=0.0):
            raise CompiledOperatorUnsupported(
                "compiled operator runtime supports only conformal rotation k maps"
            )
        prefactor = (alpha**mz) * (np.conjugate(alpha) ** mz_star)
        return int(mz), int(mz_star), q_center, complex(prefactor)
    alpha = z_e1
    if not np.allclose(z_e2, -1j * alpha, atol=1.0e-8, rtol=0.0):
        raise CompiledOperatorUnsupported(
            "compiled operator runtime supports only reflection k maps"
        )
    prefactor = (alpha**mz) * (np.conjugate(alpha) ** mz_star)
    return int(mz_star), int(mz), q_center, complex(prefactor)
