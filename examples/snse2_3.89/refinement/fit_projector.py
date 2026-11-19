"""Refine the Gamma model from the bundled projected Hamiltonian and checkpoint."""
from argparse import ArgumentParser
from dataclasses import replace
from pathlib import Path
import importlib.util
import json
import shutil

import numpy as np
import scipy.linalg
import scipy.optimize
import scipy.special
from scipy.linalg.blas import zherk
import yaml

from kp.model.response_basis import (
    CompiledResponseBasis,
    NormalizedLowEnergyWeightingSpec,
    build_normalized_low_energy_weighting,
)
from kp.model.refinement import _prepare_public_band_response_evaluator

parser = ArgumentParser(description=__doc__)
parser.add_argument("configuration", type=Path)
parser.add_argument("--output", type=Path, help="New directory for the refined model and checkpoints")
parser.add_argument("--check-inputs", action="store_true", help="Check the frozen linear spectrum without optimizing")
args = parser.parse_args()
if args.output is None and not args.check_inputs:
    parser.error("--output is required for refinement")
if args.output is not None and args.output.exists():
    parser.error("--output must be a new directory")

cfg = args.configuration.resolve()
settings = yaml.safe_load(cfg.read_text())
fit = settings["fit"]
indices = np.asarray(fit["kpoints"], dtype=int)
with np.load(cfg.parent / settings["input"], allow_pickle=False) as data:
    names = ("metadata_json", "entry_channel", "entry_monomial", "entry_row", "entry_col", "entry_value")
    frozen = {name: data[f"basis_{name}"] for name in names}
    meta = json.loads(frozen["metadata_json"].item())
    old_hash = meta["basis_hash"]
    meta["basis_hash"] = ""
    frozen["metadata_json"] = np.array(json.dumps(meta))
    basis = CompiledResponseBasis.from_frozen(frozen)
    basis = replace(basis, basis_hash=basis._compute_hash())
    linear_coefficients = data["fitted_coefficients"]
    y0 = data["starting_coefficients"]
    k = data["reference_kpoints"]
    basis_q = data["qset1"]
    h = data["h_eff"]
    ops = {op: data[f"exactified_{op}"] for op in ("C3z", "C2", "TR")}
    fit_metadata = json.loads(data["fitted_metadata_json"].item())
    cols = np.asarray(fit_metadata["fit_solver_channel_indices"], dtype=int)
    expected_energies = data["reference_eigvals"]

regen = np.linalg.eigvalsh(basis.hamiltonians(k, linear_coefficients))
verify = float(np.max(np.abs(regen - expected_energies)))
assert verify < 1e-8
assert y0.shape == linear_coefficients.shape
assert h.shape == (len(k), basis.dim, basis.dim)
assert np.allclose(h, h.swapaxes(-1, -2).conj(), atol=1e-10, rtol=0)
print("Frozen linear spectrum error (eV):", verify, "dimension:", basis.dim, "coefficients:", len(y0), flush=True)
if args.check_inputs:
    raise SystemExit(0)

out = args.output
out.mkdir(parents=True)
n_state = fit.get("fit_states", 6)
n_band = fit.get("energy_bands", 6)
gram = np.zeros((len(cols), len(cols)))
rhs = np.zeros(len(cols))
weighting = NormalizedLowEnergyWeightingSpec(
    band_edge="top",
    n_bands=n_state,
    one_sided_weight=fit["one_sided_weight"],
    two_sided_weight=fit["two_sided_weight"],
)
for j in indices:
    design, support = basis._response_design_on_union_support(k[j:j + 1], cols)
    weight = build_normalized_low_energy_weighting(h[j:j + 1], weighting)
    matrix, target = basis._normalized_low_energy_complex_rows(design, support, h[j:j + 1], weight)
    lower_gram = zherk(1.0, np.asfortranarray(matrix), trans=2, lower=1).real
    gram += lower_gram + np.tril(lower_gram, -1).T
    rhs += (matrix.conj().T @ target).real
gram /= len(indices)
rhs /= len(indices)
eigenvalues, eigenvectors = scipy.linalg.eigh(gram)
retained = eigenvalues > eigenvalues[-1] * 1e-11
rank = int(retained.sum())
whiten = eigenvectors[:, retained] / np.sqrt(eigenvalues[retained])
base = whiten.T @ (gram @ y0[cols] - rhs)
matrix_jacobian = np.eye(rank)
orthogonality_error = np.max(np.abs(whiten.T @ gram @ whiten - matrix_jacobian))
print("Whitened Gram rank:", rank, "maximum orthogonality residual:", orthogonality_error, flush=True)
assert orthogonality_error < 0.0001

response = _prepare_public_band_response_evaluator(
    basis, y0, cols, k[indices], cache_limit_bytes=6 * 1024**3,
)
print("Response cache:", response.backend, response.cache_bytes, flush=True)
full_energies = np.linalg.eigvalsh(h[indices])
wanted = full_energies[:, -n_band:]
tau = fit.get("smoothing_eV", 0.0005)
grid = np.stack([np.linspace(energies[-n_band] - 0.0015, energies[-1] + 0.003, 201)
                 for energies in full_energies])
step = grid[:, 1] - grid[:, 0]
count_ref = scipy.special.expit((full_energies[:, :, None] - grid[:, None, :]) / tau).sum(axis=1)
scales = np.sqrt(fit["band_loss_weight"] * tau * step / len(indices))

# A smooth rank-six projector penalizes both reference weight lost from
# the target subspace and weight entering it from the neighboring bands.
ref_energy, ref_vectors = np.linalg.eigh(h[indices])
top = 6
frames = ref_vectors[:, :, -top:]
mu = (ref_energy[:, -top] + ref_energy[:, -top - 1]) / 2
projector_tau = np.clip((ref_energy[:, -top] - ref_energy[:, -top - 1]) / 4, 0.00010, 0.00060)
identity = np.eye(basis.dim)
projector_scale = fit["projector_weight_eV2"] / (len(indices) * top)


def projector_rows(eigenvalues, eigenvectors):
    residuals = []
    jacobians = []
    for i, (energies, vectors) in enumerate(zip(eigenvalues, eigenvectors)):
        occupancy = scipy.special.expit((energies - mu[i]) / projector_tau[i])
        derivative = occupancy * (1 - occupancy) / projector_tau[i]
        rotated = vectors.conj().T @ frames[i]
        metric = identity - 2 * (rotated @ rotated.conj().T)
        mismatch = top + float(occupancy @ metric.diagonal().real)
        if mismatch < 0:
            raise ValueError(f"Negative soft projector mismatch {mismatch}")
        residual = np.sqrt(projector_scale * mismatch)
        energy_delta = energies[:, None] - energies
        occupancy_delta = occupancy[:, None] - occupancy
        # Use the diagonal limit for coinciding eigenvalues.
        divided_difference = (derivative[:, None] + derivative) * 0.5
        np.divide(occupancy_delta, energy_delta, out=divided_difference,
                  where=np.abs(energy_delta) > 1e-9)
        gradient = vectors @ (metric * divided_difference) @ vectors.conj().T
        partial = (gradient.ravel()[response.support].conj() @ response.design_by_k[i]).real
        residuals.append(residual)
        jacobians.append((projector_scale / (2 * residual)) * (partial @ whiten))
    return np.asarray(residuals), np.asarray(jacobians)


memo = {}
calls = 0


def evaluate(z):
    if "z" in memo and np.array_equal(z, memo["z"]):
        return memo["residual"], memo["jacobian"]
    coefficients = y0[cols] + whiten @ z
    energies, vectors = np.linalg.eigh(response.hamiltonians(coefficients))
    occupancy = scipy.special.expit((energies[:, :, None] - grid[:, None, :]) / tau)
    derivative = occupancy * (1 - occupancy) / tau
    selected = derivative.max(axis=-1) > 1e-9
    band_jacobian = response.band_jacobian(vectors, selected) @ whiten
    count_residuals = []
    count_jacobians = []
    offset = 0
    for i, mask in enumerate(selected):
        size = int(mask.sum())
        count_residuals.append(scales[i] * (occupancy[i].sum(axis=0) - count_ref[i]))
        count_jacobians.append(scales[i] * derivative[i, mask].T @ band_jacobian[offset:offset + size])
        offset += size
    projector_residual, projector_jacobian = projector_rows(energies, vectors)
    residual = np.r_[base + matrix_jacobian @ z, np.concatenate(count_residuals), projector_residual]
    jacobian = np.vstack([matrix_jacobian, *count_jacobians, projector_jacobian])
    memo.update(z=z.copy(), residual=residual, jacobian=jacobian, bands=energies[:, -n_band:])
    return residual, jacobian


z0 = min((np.zeros(rank), -base), key=lambda z: np.linalg.norm(evaluate(z)[0]))
_, jacobian = evaluate(z0)
rng = np.random.default_rng(419)
direction = rng.normal(size=len(z0))
direction /= np.linalg.norm(direction)
epsilon = 1e-7
forward, _ = evaluate(z0 + epsilon * direction)
backward, _ = evaluate(z0 - epsilon * direction)
finite_difference = (forward - backward) / (2 * epsilon)
jacobian_error = np.linalg.norm(jacobian @ direction - finite_difference) / np.linalg.norm(finite_difference)
assert jacobian_error < 0.002, jacobian_error


def fun(z):
    global calls
    residual, _ = evaluate(z)
    calls += 1
    if calls % 20 == 0:
        rms = 1000 * np.sqrt(np.mean((memo["bands"] - wanted) ** 2))
        print("Evaluation:", calls, "target RMS (meV):", rms, "loss:", residual @ residual, flush=True)
        current = y0.copy()
        current[cols] += whiten @ z
        np.save(out / "checkpoint_coefficients.npy", current)
    return residual


solution = scipy.optimize.least_squares(
    fun, z0, jac=lambda z: evaluate(z)[1], tr_solver="lsmr",
    tr_options={"maxiter": 20, "atol": 1e-5, "btol": 1e-5},
    max_nfev=fit["max_steps"], ftol=1e-9, xtol=1e-9, gtol=1e-9,
)
coefficients = y0.copy()
coefficients[cols] += whiten @ solution.x
hamiltonians = basis.hamiltonians(k, coefficients)
monomials = np.asarray(meta["coordinate"]["monomials"], dtype=int)
matrices = np.zeros((len(monomials), basis.dim, basis.dim), dtype=complex)
np.add.at(
    matrices, (frozen["entry_monomial"], frozen["entry_row"], frozen["entry_col"]),
    coefficients[frozen["entry_channel"]] * frozen["entry_value"],
)
np.savez_compressed(
    out / "polynomial.npz", monomials=monomials, matrices=matrices,
    origin=meta["coordinate"]["origin"], scale=meta["coordinate"]["scale"],
)
np.save(out / "coefficients.npy", coefficients)
shutil.copy2(Path(__file__).resolve().parents[1] / "model/Gamma/evaluate.py", out / "evaluate.py")
spec = importlib.util.spec_from_file_location("standalone", out / "evaluate.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
runtime = module.load_model(out)
fresh = np.stack([runtime.hamiltonian(momentum) for momentum in k])
reconstruction_error = float(np.abs(fresh - hamiltonians).max())
assert reconstruction_error < 1e-8

covariance = {}
q_indices = np.arange(4)[None, :] * len(basis_q) + np.arange(len(basis_q))[:, None]
for name, operator in ops.items():
    permutation = [
        int(np.argmax([np.linalg.norm(operator[np.ix_(rows, columns)]) for rows in q_indices]))
        for columns in q_indices
    ]
    rotation = np.linalg.lstsq(basis_q, basis_q[permutation], rcond=None)[0].T
    residuals = []
    for momentum in ([0.01, -0.012], [-0.013, 0.008]):
        matrix = runtime.hamiltonian(momentum)
        if name == "TR":
            matrix = matrix.conj()
        transformed = operator @ matrix @ operator.conj().T
        residuals.append(np.abs(transformed - runtime.hamiltonian(rotation @ momentum)).max())
    covariance[name] = float(max(residuals))
assert max(covariance.values()) < 1e-8
np.savez_compressed(out / "validation_matrices.npz", k=k, hamiltonians=fresh)
report = dict(
    frozen_basis_hash=old_hash, rank=rank, parameters=len(coefficients),
    reconstruction_max_eV=reconstruction_error, covariance_max_eV=covariance,
    nfev=solution.nfev, scipy_status=int(solution.status), scipy_message=str(solution.message),
    loss=float(solution.fun @ solution.fun), jacobian_relative_error=float(jacobian_error), fit=fit,
)
(out / "VERIFICATION.json").write_text(json.dumps(report, indent=2) + "\n")
print(report, flush=True)
