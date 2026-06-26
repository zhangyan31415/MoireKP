#!/usr/bin/env python
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor
import json
import math
import shutil
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.sparse
from scipy.spatial import cKDTree

from tapw.symmetry.representations import direct_sum, generate_direct_sum_params, spin_reps
from tapw.geometry.rotations import get_any_rot_orb_twostep


BOHR_TO_ANG = 0.529177210903

AtomBlockKey = tuple[tuple[int, int, int], int, int]
AtomBlockItem = tuple[AtomBlockKey, np.ndarray]
RawSymmetryAction = tuple[tuple[int, int, int], int, int, np.ndarray, np.ndarray, bool]
CanonicalSymmetryAction = tuple[AtomBlockKey, bool, np.ndarray, np.ndarray, bool]
SparseTriplets = dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray, np.ndarray]]


class AtomRecord:
    def __init__(
        self,
        index: int,
        species: str,
        frac: np.ndarray,
        cart: np.ndarray,
        orbital_spec: str,
        magnetic_moment: np.ndarray | None = None,
    ):
        self.index = int(index)
        self.species = str(species)
        self.frac = np.asarray(frac, dtype=float)
        self.cart = np.asarray(cart, dtype=float)
        self.orbital_spec = str(orbital_spec)
        self.magnetic_moment = None if magnetic_moment is None else np.asarray(magnetic_moment, dtype=float)
        self.orbital_count = orbital_count(self.orbital_spec)
        self.offset = 0


class OpenMXStructure:
    def __init__(
        self,
        path: Path,
        lattice: np.ndarray,
        atoms: list[AtomRecord],
        species_basis: dict[str, str],
        *,
        spinful: bool = False,
        spin_mode: str = "none",
    ):
        self.path = Path(path)
        self.lattice = np.asarray(lattice, dtype=float)
        self.atoms = list(atoms)
        self.species_basis = dict(species_basis)
        self.spinful = bool(spinful)
        self.spin_mode = str(spin_mode)
        offset = 0
        for atom in self.atoms:
            atom.offset = offset
            offset += atom.orbital_count
        self.nwann_spinless = int(offset)
        self.nwann = int(2 * offset if self.spinful else offset)


class SymmetryOperation:
    def __init__(
        self,
        index: int,
        rotation_frac: np.ndarray,
        translation_frac: np.ndarray,
        rotation_cart: np.ndarray,
        translation_cart: np.ndarray,
    ):
        self.index = int(index)
        self.rotation_frac = np.asarray(rotation_frac, dtype=float)
        self.translation_frac = np.asarray(translation_frac, dtype=float)
        self.rotation_cart = np.asarray(rotation_cart, dtype=float)
        self.translation_cart = np.asarray(translation_cart, dtype=float)


class OperationTransport:
    def __init__(
        self,
        operation: SymmetryOperation,
        atom_mappings: list[int],
        image_shifts: list[np.ndarray],
        orbital_blocks: dict[int, np.ndarray],
        max_mapping_residual: float,
        *,
        antiunitary: bool = False,
    ):
        self.operation = operation
        self.atom_mappings = [int(v) for v in atom_mappings]
        self.image_shifts = [np.asarray(v, dtype=int) for v in image_shifts]
        self.orbital_blocks = dict(orbital_blocks)
        self.max_mapping_residual = float(max_mapping_residual)
        self.antiunitary = bool(antiunitary)
        self.orbital_blocks_dagger = {key: value.conj().T for key, value in self.orbital_blocks.items()}
        rotation_int = np.rint(self.operation.rotation_frac).astype(int)
        if not np.allclose(rotation_int, self.operation.rotation_frac, atol=1.0e-8):
            raise ValueError("spglib rotation matrix is not integer-valued in fractional coordinates.")
        self.rotation_frac_int = rotation_int


class SymmetryActionCache:
    def __init__(self, transports: list[OperationTransport]):
        self.transports = list(transports)
        self._actions: dict[AtomBlockKey, tuple[RawSymmetryAction, ...]] = {}
        self._canonical_actions: dict[AtomBlockKey, tuple[CanonicalSymmetryAction, ...]] = {}

    def actions_for(self, key: AtomBlockKey) -> tuple[RawSymmetryAction, ...]:
        cached = self._actions.get(key)
        if cached is not None:
            return cached
        rvec, atom_i, atom_j = key
        r_arr = np.asarray(rvec, dtype=int)
        actions = []
        for transport in self.transports:
            target_i = transport.atom_mappings[atom_i]
            target_j = transport.atom_mappings[atom_j]
            new_r = tuple(
                int(v)
                for v in (
                    transport.rotation_frac_int @ r_arr
                    + transport.image_shifts[atom_j]
                    - transport.image_shifts[atom_i]
                ).tolist()
            )
            actions.append(
                (
                    new_r,
                    target_i,
                    target_j,
                    transport.orbital_blocks[atom_i],
                    transport.orbital_blocks_dagger[atom_j],
                    transport.antiunitary,
                )
            )
        cached = tuple(actions)
        self._actions[key] = cached
        return cached

    def canonical_actions_for(self, key: AtomBlockKey) -> tuple[CanonicalSymmetryAction, ...]:
        cached = self._canonical_actions.get(key)
        if cached is not None:
            return cached
        actions = []
        for new_r, target_i, target_j, left, right_h, antiunitary in self.actions_for(key):
            target_key = (new_r, target_i, target_j)
            canon = _canonical_key(target_key)
            actions.append((canon, canon != target_key, left, right_h, antiunitary))
        cached = tuple(actions)
        self._canonical_actions[key] = cached
        return cached


class SparseMatrixSet:
    def __init__(self, label: str, nwann: int, nrpt: int, blocks: dict[tuple[int, int, int], scipy.sparse.csr_matrix]):
        self.label = str(label)
        self.nwann = int(nwann)
        self.nrpt = int(nrpt)
        self.blocks = dict(blocks)


class AtomBlockMatrixSet:
    def __init__(
        self,
        label: str,
        nwann: int,
        nrpt: int,
        input_nonzero: int,
        input_r_points: int,
        atom_blocks: dict[tuple[tuple[int, int, int], int, int], np.ndarray],
    ):
        self.label = str(label)
        self.nwann = int(nwann)
        self.nrpt = int(nrpt)
        self.input_nonzero = int(input_nonzero)
        self.input_r_points = int(input_r_points)
        self.atom_blocks = dict(atom_blocks)


def _strip_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


def _first_value_for_key(lines: list[str], key: str) -> str | None:
    key_l = key.lower()
    for line in lines:
        text = _strip_comment(line)
        if not text:
            continue
        parts = text.split()
        if parts and parts[0].lower() == key_l and len(parts) >= 2:
            return parts[1]
    return None


def _section(lines: list[str], name: str) -> list[str]:
    start = f"<{name}".lower()
    end = name.lower() + ">"
    out: list[str] = []
    active = False
    for line in lines:
        text = _strip_comment(line)
        low = text.lower()
        if not active and low.startswith(start):
            active = True
            continue
        if active:
            if low.startswith(end):
                return out
            if text:
                out.append(text)
    return out


def _parse_orbital_shells(orbital_spec: str) -> dict[str, int]:
    import re

    text = str(orbital_spec)
    if "-" in text:
        text = text.split("-", 1)[1]
    shells: dict[str, int] = {}
    for shell, count in re.findall(r"([spdf])(\d+)", text):
        shells[shell] = int(count)
    if not shells:
        raise ValueError(f"Could not parse orbital specification {orbital_spec!r}")
    return shells


def orbital_count(orbital_spec: str) -> int:
    dims = {"s": 1, "p": 3, "d": 5, "f": 7}
    return int(sum(dims[shell] * count for shell, count in _parse_orbital_shells(orbital_spec).items()))


def orbital_rotation_block(orbital_spec: str, rotation_cart: np.ndarray) -> np.ndarray:
    shells = _parse_orbital_shells(orbital_spec)
    mapping = {
        "s": get_any_rot_orb_twostep("s", rotation_cart),
        "p": get_any_rot_orb_twostep("p", rotation_cart),
        "d": get_any_rot_orb_twostep("d", rotation_cart),
        "f": get_any_rot_orb_twostep("f", rotation_cart),
    }
    return np.asarray(direct_sum(*generate_direct_sum_params(shells, mapping)), dtype=np.complex128)


def time_reversal_spin_major_block(orbital_count: int) -> np.ndarray:
    norb = int(orbital_count)
    if norb < 1:
        raise ValueError("orbital_count must be positive.")
    out = np.zeros((2 * norb, 2 * norb), dtype=np.complex128)
    eye = np.eye(norb, dtype=np.complex128)
    out[:norb, norb:] = -eye
    out[norb:, :norb] = eye
    return out


def _unit_scale(unit: str | None) -> float:
    if unit is None:
        return 1.0
    text = str(unit).strip().lower()
    if text in {"ang", "angstrom", "angstroms"}:
        return 1.0
    if text in {"au", "bohr"}:
        return BOHR_TO_ANG
    return 1.0


def _fractional_from_cart(lattice: np.ndarray, cart: np.ndarray) -> np.ndarray:
    return np.linalg.solve(np.asarray(lattice, dtype=float).T, np.asarray(cart, dtype=float).T).T


def _openmx_nc_magnetic_moment(values: list[str], spinful: bool) -> np.ndarray | None:
    if not spinful or len(values) < 7:
        return None
    try:
        up = float(values[5])
        down = float(values[6])
    except ValueError:
        return None
    magnitude = up - down
    if len(values) >= 9:
        try:
            theta = math.radians(float(values[7]))
            phi = math.radians(float(values[8]))
        except ValueError:
            theta = 0.0
            phi = 0.0
    else:
        theta = 0.0
        phi = 0.0
    direction = np.asarray(
        [
            math.sin(theta) * math.cos(phi),
            math.sin(theta) * math.sin(phi),
            math.cos(theta),
        ],
        dtype=float,
    )
    return float(magnitude) * direction


def parse_openmx_structure(path: str | Path) -> OpenMXStructure:
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    spin = (_first_value_for_key(lines, "scf.SpinPolarization") or "off").lower()
    soc = (_first_value_for_key(lines, "scf.SpinOrbit.Coupling") or "off").lower()
    if spin == "on" and soc != "on":
        spin_mode = "collinear"
    elif spin == "nc" or soc == "on":
        spin_mode = "spinor"
    else:
        spin_mode = "none"
    spinful = spin_mode != "none"

    lattice_scale = _unit_scale(_first_value_for_key(lines, "Atoms.UnitVectors.Unit"))
    lattice_rows = []
    for row in _section(lines, "Atoms.UnitVectors"):
        values = row.split()
        if len(values) < 3:
            continue
        lattice_rows.append([float(values[0]) * lattice_scale, float(values[1]) * lattice_scale, float(values[2]) * lattice_scale])
    if len(lattice_rows) != 3:
        raise ValueError(f"Could not parse three Atoms.UnitVectors rows from {path}")
    lattice = np.asarray(lattice_rows, dtype=float)

    species_basis = {}
    for row in _section(lines, "Definition.of.Atomic.Species"):
        values = row.split()
        if len(values) >= 2:
            species_basis[values[0]] = values[1]
    if not species_basis:
        raise ValueError(f"Could not parse Definition.of.Atomic.Species from {path}")

    coord_unit = (_first_value_for_key(lines, "Atoms.SpeciesAndCoordinates.Unit") or "Ang").lower()
    coord_scale = _unit_scale(coord_unit)
    atoms: list[AtomRecord] = []
    for row in _section(lines, "Atoms.SpeciesAndCoordinates"):
        values = row.split()
        if len(values) < 5:
            continue
        index = int(values[0])
        species = values[1]
        if species not in species_basis:
            raise ValueError(f"Species {species!r} has no orbital basis definition.")
        raw = np.asarray([float(values[2]), float(values[3]), float(values[4])], dtype=float)
        if coord_unit.startswith("f"):
            frac = raw
            cart = lattice.T @ frac
        else:
            cart = raw * coord_scale
            frac = _fractional_from_cart(lattice, cart.reshape(1, 3))[0]
        atoms.append(
            AtomRecord(
                index=index,
                species=species,
                frac=frac,
                cart=cart,
                orbital_spec=species_basis[species],
                magnetic_moment=_openmx_nc_magnetic_moment(values, spinful=spinful),
            )
        )
    if not atoms:
        raise ValueError(f"Could not parse Atoms.SpeciesAndCoordinates from {path}")

    expected_atoms = _first_value_for_key(lines, "Atoms.Number")
    if expected_atoms is not None and int(expected_atoms) != len(atoms):
        raise ValueError(f"Atoms.Number={expected_atoms} but parsed {len(atoms)} atoms.")
    return OpenMXStructure(path, lattice, atoms, species_basis, spinful=spinful, spin_mode=spin_mode)


def _species_numbers(species: list[str]) -> np.ndarray:
    try:
        from ase.data import atomic_numbers
    except Exception:
        atomic_numbers = {}
    assigned: dict[str, int] = {}
    next_unknown = 1000
    numbers = []
    for item in species:
        symbol = str(item)
        number = int(atomic_numbers.get(symbol, 0)) if atomic_numbers else 0
        if number == 0:
            if symbol not in assigned:
                assigned[symbol] = next_unknown
                next_unknown += 1
            number = assigned[symbol]
        numbers.append(number)
    return np.asarray(numbers, dtype=int)


def build_symmetry_operations(structure: OpenMXStructure, symprec: float) -> list[SymmetryOperation]:
    import spglib

    positions_frac = np.vstack([atom.frac for atom in structure.atoms])
    numbers = _species_numbers([atom.species for atom in structure.atoms])
    symmetry = spglib.get_symmetry((structure.lattice, positions_frac, numbers), symprec=float(symprec))
    if symmetry is None:
        raise RuntimeError(f"spglib.get_symmetry failed with symprec={symprec}.")
    lattice_inv_t = np.linalg.inv(structure.lattice.T)
    operations = []
    for index, (rotation_frac, translation_frac) in enumerate(zip(symmetry["rotations"], symmetry["translations"])):
        rotation_frac = np.asarray(rotation_frac, dtype=float)
        translation_frac = np.asarray(translation_frac, dtype=float)
        rotation_cart = structure.lattice.T @ rotation_frac @ lattice_inv_t
        translation_cart = structure.lattice.T @ translation_frac
        operations.append(SymmetryOperation(index, rotation_frac, translation_frac, rotation_cart, translation_cart))
    return operations


def build_operation_transports(
    structure: OpenMXStructure,
    operations: list[SymmetryOperation],
    atom_tol: float = 5.0e-6,
) -> list[OperationTransport]:
    for atom_index, atom in enumerate(structure.atoms):
        expected = structure.species_basis.get(atom.species)
        if expected is not None and atom.orbital_spec != expected:
            raise ValueError(
                "Orbital block mismatch for atom "
                f"{atom_index}: species {atom.species!r} expects {expected!r}, got {atom.orbital_spec!r}"
            )
    positions_frac = np.vstack([atom.frac for atom in structure.atoms])
    shifts = np.asarray([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=float)
    compatible_groups = {}
    for atom in structure.atoms:
        key = (atom.species, atom.orbital_count, atom.orbital_spec)
        if key in compatible_groups:
            continue
        target_indices = [
            idx
            for idx, other in enumerate(structure.atoms)
            if other.species == atom.species
            and other.orbital_count == atom.orbital_count
            and other.orbital_spec == atom.orbital_spec
        ]
        if not target_indices:
            raise ValueError(f"No target atoms compatible with species/basis group {key}.")
        target_positions = positions_frac[target_indices]
        images = np.vstack([target_positions + shift for shift in shifts])
        compatible_groups[key] = {
            "target_indices": np.asarray(target_indices, dtype=int),
            "images": images,
            "image_targets": np.concatenate([np.asarray(target_indices, dtype=int) for _ in shifts]),
            "tiled_shifts": np.vstack([np.repeat(shift.reshape(1, 3), len(target_indices), axis=0) for shift in shifts]).astype(int),
            "tree": cKDTree(images),
        }
    transports: list[OperationTransport] = []
    for operation in operations:
        atom_mappings = [-1] * len(structure.atoms)
        image_shifts = [np.zeros(3, dtype=int) for _ in structure.atoms]
        max_residual = 0.0
        used_targets: set[int] = set()
        for source_index, atom in enumerate(structure.atoms):
            group = compatible_groups[(atom.species, atom.orbital_count, atom.orbital_spec)]
            images = group["images"]
            transformed = atom.frac @ operation.rotation_frac.T + operation.translation_frac
            distances, matches = group["tree"].query(transformed, k=min(images.shape[0], max(8, len(group["target_indices"]))))
            distances = np.atleast_1d(distances)
            matches = np.atleast_1d(matches)
            picked = None
            for distance, match in zip(distances, matches):
                match = int(match)
                if match >= images.shape[0] or float(distance) > float(atom_tol):
                    continue
                target_index = int(group["image_targets"][match])
                if target_index in used_targets:
                    continue
                picked = (target_index, group["tiled_shifts"][match], float(distance))
                break
            if picked is None:
                raise ValueError(
                    f"Could not map atom {source_index} under spglib operation {operation.index}; "
                    f"increase --symprec/atom tolerance only if the structure is intentionally approximate."
                )
            target_index, shift, residual = picked
            target_atom = structure.atoms[target_index]
            if atom.orbital_spec != target_atom.orbital_spec:
                raise ValueError(
                    "Orbital block mismatch for mapped atoms "
                    f"{source_index}->{target_index}: {atom.orbital_spec!r} vs {target_atom.orbital_spec!r}"
                )
            atom_mappings[source_index] = target_index
            image_shifts[source_index] = shift
            used_targets.add(target_index)
            max_residual = max(max_residual, residual)

        block_cache: dict[str, np.ndarray] = {}
        orbital_blocks = {}
        for source_index, atom in enumerate(structure.atoms):
            block = block_cache.get(atom.orbital_spec)
            if block is None:
                block = orbital_rotation_block(atom.orbital_spec, operation.rotation_cart)
                if structure.spin_mode == "spinor":
                    block = np.kron(spin_reps(operation.rotation_cart), block)
                elif structure.spin_mode == "collinear":
                    block = np.kron(np.eye(2, dtype=np.complex128), block)
                block_cache[atom.orbital_spec] = block
            orbital_blocks[source_index] = block
        transports.append(OperationTransport(operation, atom_mappings, image_shifts, orbital_blocks, max_residual))
    return transports


def _significant_magnetic_moments(structure: OpenMXStructure, magmom_tol: float) -> bool:
    threshold = float(magmom_tol)
    return any(atom.magnetic_moment is not None and np.linalg.norm(atom.magnetic_moment) > threshold for atom in structure.atoms)


def _magnetic_operation_error(
    structure: OpenMXStructure,
    transport: OperationTransport,
    *,
    time_reversal_sign: float,
    magmom_tol: float,
) -> float:
    max_error = 0.0
    threshold = float(magmom_tol)
    det = float(np.linalg.det(transport.operation.rotation_cart))
    axial_rotation = det * transport.operation.rotation_cart
    for source_index, target_index in enumerate(transport.atom_mappings):
        source_moment = structure.atoms[source_index].magnetic_moment
        target_moment = structure.atoms[target_index].magnetic_moment
        if source_moment is None and target_moment is None:
            continue
        source = np.zeros(3, dtype=float) if source_moment is None else np.asarray(source_moment, dtype=float)
        target = np.zeros(3, dtype=float) if target_moment is None else np.asarray(target_moment, dtype=float)
        if np.linalg.norm(source) <= threshold and np.linalg.norm(target) <= threshold:
            continue
        transformed = float(time_reversal_sign) * axial_rotation @ source
        max_error = max(max_error, float(np.linalg.norm(transformed - target)))
    return max_error


def _as_antiunitary_transport(structure: OpenMXStructure, transport: OperationTransport) -> OperationTransport:
    if not structure.spinful:
        raise ValueError("Antiunitary magnetic operations require a spinful collinear or NC/SOC basis.")
    anti_blocks: dict[int, np.ndarray] = {}
    for atom_index, block in transport.orbital_blocks.items():
        atom = structure.atoms[atom_index]
        anti_blocks[atom_index] = block @ time_reversal_spin_major_block(atom.orbital_count)
    return OperationTransport(
        transport.operation,
        transport.atom_mappings,
        transport.image_shifts,
        anti_blocks,
        transport.max_mapping_residual,
        antiunitary=True,
    )


def filter_magnetic_unitary_operations(
    structure: OpenMXStructure,
    operations: list[SymmetryOperation],
    transports: list[OperationTransport],
    *,
    magmom_tol: float = 1.0e-5,
) -> tuple[list[SymmetryOperation], list[OperationTransport], dict[str, object]]:
    if len(operations) != len(transports):
        raise ValueError(f"Operation count {len(operations)} does not match transport count {len(transports)}.")
    has_moments = _significant_magnetic_moments(structure, magmom_tol)
    if not has_moments:
        return list(operations), list(transports), {
            "mode": "unitary",
            "applied": False,
            "reason": "no_significant_magnetic_moments",
            "magmom_tol": float(magmom_tol),
            "input_operation_count": len(operations),
            "kept_operation_count": len(operations),
            "filtered_operation_indices": [],
            "primed_candidate_indices": [],
            "operation_errors": [],
        }

    kept_operations: list[SymmetryOperation] = []
    kept_transports: list[OperationTransport] = []
    filtered_indices: list[int] = []
    primed_candidate_indices: list[int] = []
    operation_errors: list[dict[str, float | int | bool]] = []
    for operation, transport in zip(operations, transports):
        unitary_error = _magnetic_operation_error(structure, transport, time_reversal_sign=1.0, magmom_tol=magmom_tol)
        primed_error = _magnetic_operation_error(structure, transport, time_reversal_sign=-1.0, magmom_tol=magmom_tol)
        keep = unitary_error <= float(magmom_tol)
        if keep:
            kept_operations.append(operation)
            kept_transports.append(transport)
        else:
            filtered_indices.append(int(operation.index))
            if primed_error <= float(magmom_tol):
                primed_candidate_indices.append(int(operation.index))
        operation_errors.append(
            {
                "index": int(operation.index),
                "unitary_error": float(unitary_error),
                "primed_error": float(primed_error),
                "kept_unitary": bool(keep),
                "primed_candidate": bool(primed_error <= float(magmom_tol)),
            }
        )
    if not kept_operations:
        raise ValueError("Magnetic unitary filtering removed all symmetry operations; check magnetic moments or use --magnetic-symmetry off.")
    return kept_operations, kept_transports, {
        "mode": "unitary",
        "applied": True,
        "reason": "magnetic_moments_present",
        "magmom_tol": float(magmom_tol),
        "input_operation_count": len(operations),
        "kept_operation_count": len(kept_operations),
        "filtered_operation_indices": filtered_indices,
        "primed_candidate_indices": primed_candidate_indices,
        "operation_errors": operation_errors,
    }


def filter_magnetic_operations(
    structure: OpenMXStructure,
    operations: list[SymmetryOperation],
    transports: list[OperationTransport],
    *,
    magmom_tol: float = 1.0e-5,
) -> tuple[list[SymmetryOperation], list[OperationTransport], dict[str, object]]:
    if len(operations) != len(transports):
        raise ValueError(f"Operation count {len(operations)} does not match transport count {len(transports)}.")
    has_moments = _significant_magnetic_moments(structure, magmom_tol)
    if not has_moments:
        return list(operations), list(transports), {
            "mode": "magnetic",
            "applied": False,
            "reason": "no_significant_magnetic_moments",
            "magmom_tol": float(magmom_tol),
            "input_operation_count": len(operations),
            "kept_operation_count": len(operations),
            "kept_unitary_operation_indices": [int(op.index) for op in operations],
            "kept_primed_operation_indices": [],
            "filtered_operation_indices": [],
            "primed_candidate_indices": [],
            "operation_errors": [],
        }

    kept_operations: list[SymmetryOperation] = []
    kept_transports: list[OperationTransport] = []
    kept_unitary_indices: list[int] = []
    kept_primed_indices: list[int] = []
    filtered_indices: list[int] = []
    operation_errors: list[dict[str, float | int | bool | str]] = []
    for operation, transport in zip(operations, transports):
        unitary_error = _magnetic_operation_error(structure, transport, time_reversal_sign=1.0, magmom_tol=magmom_tol)
        primed_error = _magnetic_operation_error(structure, transport, time_reversal_sign=-1.0, magmom_tol=magmom_tol)
        if unitary_error <= float(magmom_tol):
            kept_operations.append(operation)
            kept_transports.append(transport)
            kept_unitary_indices.append(int(operation.index))
            kept_as = "unitary"
        elif primed_error <= float(magmom_tol):
            kept_operations.append(operation)
            kept_transports.append(_as_antiunitary_transport(structure, transport))
            kept_primed_indices.append(int(operation.index))
            kept_as = "primed"
        else:
            filtered_indices.append(int(operation.index))
            kept_as = "filtered"
        operation_errors.append(
            {
                "index": int(operation.index),
                "unitary_error": float(unitary_error),
                "primed_error": float(primed_error),
                "kept_unitary": kept_as == "unitary",
                "kept_primed": kept_as == "primed",
                "kept_as": kept_as,
            }
        )
    if not kept_operations:
        raise ValueError("Magnetic filtering removed all symmetry operations; check magnetic moments or use --magnetic-symmetry off.")
    return kept_operations, kept_transports, {
        "mode": "magnetic",
        "applied": True,
        "reason": "magnetic_moments_present",
        "magmom_tol": float(magmom_tol),
        "input_operation_count": len(operations),
        "kept_operation_count": len(kept_operations),
        "kept_unitary_operation_indices": kept_unitary_indices,
        "kept_primed_operation_indices": kept_primed_indices,
        "filtered_operation_indices": filtered_indices,
        "primed_candidate_indices": kept_primed_indices,
        "operation_errors": operation_errors,
    }


def apply_magnetic_symmetry_filter(
    structure: OpenMXStructure,
    operations: list[SymmetryOperation],
    transports: list[OperationTransport],
    *,
    mode: str,
    magmom_tol: float,
) -> tuple[list[SymmetryOperation], list[OperationTransport], dict[str, object]]:
    if mode not in {"auto", "off", "unitary", "magnetic"}:
        raise ValueError("--magnetic-symmetry must be auto, off, unitary, or magnetic.")
    if mode == "off":
        return list(operations), list(transports), {
            "mode": "off",
            "applied": False,
            "reason": "disabled",
            "magmom_tol": float(magmom_tol),
            "input_operation_count": len(operations),
            "kept_operation_count": len(operations),
            "filtered_operation_indices": [],
            "primed_candidate_indices": [],
        }
    if mode == "auto" and not _significant_magnetic_moments(structure, magmom_tol):
        return list(operations), list(transports), {
            "mode": "auto",
            "applied": False,
            "reason": "no_significant_magnetic_moments",
            "magmom_tol": float(magmom_tol),
            "input_operation_count": len(operations),
            "kept_operation_count": len(operations),
            "filtered_operation_indices": [],
            "primed_candidate_indices": [],
        }
    if mode in {"auto", "magnetic"}:
        kept_operations, kept_transports, report = filter_magnetic_operations(
            structure, operations, transports, magmom_tol=magmom_tol
        )
        report["mode"] = "auto-magnetic" if mode == "auto" else "magnetic"
        return kept_operations, kept_transports, report
    kept_operations, kept_transports, report = filter_magnetic_unitary_operations(
        structure, operations, transports, magmom_tol=magmom_tol
    )
    report["mode"] = "auto-unitary" if mode == "auto" else "unitary"
    return kept_operations, kept_transports, report


def _real_vector(values: np.ndarray) -> list[float]:
    return [float(item) for item in np.asarray(values, dtype=float).reshape(-1)]


def _real_matrix(values: np.ndarray) -> list[list[float]]:
    return [[float(item) for item in row] for row in np.asarray(values, dtype=float)]


def _int_vector(values: np.ndarray | list[int]) -> list[int]:
    return [int(item) for item in np.asarray(values, dtype=int).reshape(-1)]


def _complex_matrix(values: np.ndarray) -> list[list[list[float]]]:
    matrix = np.asarray(values, dtype=np.complex128)
    return [
        [[float(value.real), float(value.imag)] for value in row]
        for row in matrix
    ]


def symmetry_transports_payload(
    structure: OpenMXStructure,
    operations: list[SymmetryOperation],
    transports: list[OperationTransport],
    *,
    symprec: float,
    magnetic_report: dict[str, object] | None = None,
) -> dict[str, object]:
    if len(operations) != len(transports):
        raise ValueError(f"Operation count {len(operations)} does not match transport count {len(transports)}.")
    payload = {
        "schema": "openmx_hs_symmetry_transports/v1",
        "spglib_symprec": float(symprec),
        "spglib_operation_count": len(operations),
        "max_atom_mapping_residual": max((item.max_mapping_residual for item in transports), default=0.0),
        "spinful": bool(structure.spinful),
        "spin_mode": structure.spin_mode,
        "natoms": len(structure.atoms),
        "nwann": structure.nwann,
        "nwann_spinless": structure.nwann_spinless,
        "lattice": _real_matrix(structure.lattice),
        "atom_orbital_counts": [int(atom.orbital_count) for atom in structure.atoms],
        "atom_offsets": [int(atom.offset) for atom in structure.atoms],
        "atoms": [
            {
                "index": int(atom.index),
                "species": atom.species,
                "orbital_spec": atom.orbital_spec,
                "orbital_count": int(atom.orbital_count),
                "frac": _real_vector(atom.frac),
                "cart": _real_vector(atom.cart),
                "magnetic_moment": None if atom.magnetic_moment is None else _real_vector(atom.magnetic_moment),
            }
            for atom in structure.atoms
        ],
        "operations": [
            {
                "index": int(operation.index),
                "antiunitary": bool(transport.antiunitary),
                "rotation_frac": _real_matrix(operation.rotation_frac),
                "translation_frac": _real_vector(operation.translation_frac),
                "rotation_cart": _real_matrix(operation.rotation_cart),
                "translation_cart": _real_vector(operation.translation_cart),
                "atom_mappings": [int(value) for value in transport.atom_mappings],
                "image_shifts": [_int_vector(value) for value in transport.image_shifts],
                "orbital_blocks": {
                    str(atom_index): _complex_matrix(transport.orbital_blocks[atom_index])
                    for atom_index in sorted(transport.orbital_blocks)
                },
                "max_mapping_residual": float(transport.max_mapping_residual),
            }
            for operation, transport in zip(operations, transports)
        ],
    }
    if magnetic_report is not None:
        payload["magnetic_symmetry"] = magnetic_report
    return payload


def write_symmetry_transports_json(
    path: str | Path,
    structure: OpenMXStructure,
    operations: list[SymmetryOperation],
    transports: list[OperationTransport],
    *,
    symprec: float,
    magnetic_report: dict[str, object] | None = None,
) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = symmetry_transports_payload(structure, operations, transports, symprec=symprec, magnetic_report=magnetic_report)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _first_int(line: str) -> int:
    return int(line.strip().split()[0])


def read_openmx_sparse(path: str | Path) -> SparseMatrixSet:
    path = Path(path)
    if path.suffix == ".npz":
        return read_openmx_sparse_npz(path)
    with path.open("r", encoding="utf-8") as handle:
        header = [handle.readline() for _ in range(4)]
        if any(line == "" for line in header):
            raise ValueError(f"{path} is too short to be an OpenMX sparse matrix file.")
        label = "H" if "H" in header[0] else "S"
        nonzero = _first_int(header[1])
        nwann = _first_int(header[2])
        nrpt = _first_int(header[3])
        chunks: dict[tuple[int, int, int], tuple[list[int], list[int], list[complex]]] = {}
        count = 0
        for line in handle:
            parts = line.split()
            if len(parts) != 7:
                continue
            rx, ry, rz = (int(parts[0]), int(parts[1]), int(parts[2]))
            row = int(parts[3]) - 1
            col = int(parts[4]) - 1
            val = float(parts[5]) + 1.0j * float(parts[6])
            rows, cols, vals = chunks.setdefault((rx, ry, rz), ([], [], []))
            rows.append(row)
            cols.append(col)
            vals.append(val)
            count += 1
        if count != nonzero:
            raise ValueError(f"{path} header declares {nonzero} nonzeros, parsed {count}.")
    blocks = {
        rvec: scipy.sparse.coo_matrix((vals, (rows, cols)), shape=(nwann, nwann), dtype=np.complex128).tocsr()
        for rvec, (rows, cols, vals) in chunks.items()
    }
    return SparseMatrixSet(label=label, nwann=nwann, nrpt=nrpt, blocks=blocks)


def read_openmx_sparse_npz(path: str | Path, *, nwann: int | None = None, label: str | None = None) -> SparseMatrixSet:
    path = Path(path)
    label = label or ("H" if "H" in path.name else "S")
    chunks: dict[tuple[int, int, int], dict[str, np.ndarray]] = {}
    with np.load(path, allow_pickle=False) as archive:
        for key in archive.files:
            key_tuple_text, attr = key.rsplit("_", 1)
            if attr not in {"row", "col", "val"}:
                raise ValueError(f"{path} contains unsupported key {key!r}; expected '<rvec>_row/col/val'.")
            rvec = tuple(int(part.strip()) for part in key_tuple_text.strip("()").split(","))
            if len(rvec) != 3:
                raise ValueError(f"{path} contains invalid R vector key {key!r}.")
            chunks.setdefault(rvec, {})[attr] = archive[key]
    if not chunks:
        inferred_nwann = int(nwann or 0)
    elif nwann is None:
        max_index = max(
            int(np.max(values[attr])) if values.get(attr) is not None and values[attr].size else -1
            for values in chunks.values()
            for attr in ("row", "col")
        )
        inferred_nwann = max_index + 1
    else:
        inferred_nwann = int(nwann)
    blocks = {}
    for rvec, values in chunks.items():
        missing = {"row", "col", "val"} - set(values)
        if missing:
            raise ValueError(f"{path} is missing {sorted(missing)} arrays for R={rvec}.")
        rows = np.asarray(values["row"], dtype=np.int32)
        cols = np.asarray(values["col"], dtype=np.int32)
        vals = np.asarray(values["val"], dtype=np.complex128)
        blocks[rvec] = scipy.sparse.coo_matrix((vals, (rows, cols)), shape=(inferred_nwann, inferred_nwann), dtype=np.complex128).tocsr()
    return SparseMatrixSet(label=label, nwann=inferred_nwann, nrpt=len(blocks), blocks=blocks)


def _orbital_to_atom(structure: OpenMXStructure) -> tuple[np.ndarray, np.ndarray]:
    atom_for_orb = np.empty(structure.nwann_spinless, dtype=int)
    local_for_orb = np.empty(structure.nwann_spinless, dtype=int)
    for atom_index, atom in enumerate(structure.atoms):
        start = atom.offset
        stop = start + atom.orbital_count
        atom_for_orb[start:stop] = atom_index
        local_for_orb[start:stop] = np.arange(atom.orbital_count, dtype=int)
    return atom_for_orb, local_for_orb


def _split_spin_orbital_index(index: int, structure: OpenMXStructure) -> tuple[int, int]:
    if not structure.spinful:
        return 0, int(index)
    base = structure.nwann_spinless
    spin = int(index) // base
    orbital = int(index) % base
    if spin not in (0, 1):
        raise ValueError(f"Spinful orbital index {index} is outside the expected 2*{base} basis.")
    return spin, orbital


def _global_orbital_to_atom_local(structure: OpenMXStructure) -> tuple[np.ndarray, np.ndarray]:
    atom_for_spinless, local_for_spinless = _orbital_to_atom(structure)
    if not structure.spinful:
        return atom_for_spinless.astype(np.int32, copy=False), local_for_spinless.astype(np.int32, copy=False)
    base = structure.nwann_spinless
    atom_for_global = np.empty(structure.nwann, dtype=np.int32)
    local_for_global = np.empty(structure.nwann, dtype=np.int32)
    atom_for_spinless = atom_for_spinless.astype(np.int32, copy=False)
    local_for_spinless = local_for_spinless.astype(np.int32, copy=False)
    atom_counts = np.asarray([atom.orbital_count for atom in structure.atoms], dtype=np.int32)
    atom_for_global[:base] = atom_for_spinless
    atom_for_global[base:] = atom_for_spinless
    local_for_global[:base] = local_for_spinless
    local_for_global[base:] = local_for_spinless + atom_counts[atom_for_spinless]
    return atom_for_global, local_for_global


def _global_orbital_index(atom: AtomRecord, local_spin_major: int, structure: OpenMXStructure) -> int:
    if not structure.spinful:
        return atom.offset + int(local_spin_major)
    spin = int(local_spin_major) // atom.orbital_count
    local = int(local_spin_major) % atom.orbital_count
    if spin not in (0, 1):
        raise ValueError(f"Local spinor orbital index {local_spin_major} is outside atom block size {2 * atom.orbital_count}.")
    return spin * structure.nwann_spinless + atom.offset + local


def _atom_global_indices(atom: AtomRecord, structure: OpenMXStructure) -> np.ndarray:
    if not structure.spinful:
        return atom.offset + np.arange(atom.orbital_count, dtype=int)
    up = atom.offset + np.arange(atom.orbital_count, dtype=int)
    down = structure.nwann_spinless + atom.offset + np.arange(atom.orbital_count, dtype=int)
    return np.concatenate([up, down])


def _matrix_to_atom_blocks(
    matrix_set: SparseMatrixSet,
    structure: OpenMXStructure,
) -> dict[tuple[tuple[int, int, int], int, int], np.ndarray]:
    atom_for_global, local_for_global = _global_orbital_to_atom_local(structure)
    spin_factor = 2 if structure.spinful else 1
    block_shapes = [
        (spin_factor * atom_i.orbital_count, spin_factor * atom_j.orbital_count)
        for atom_i in structure.atoms
        for atom_j in structure.atoms
    ]
    natoms = len(structure.atoms)
    blocks: dict[tuple[tuple[int, int, int], int, int], np.ndarray] = {}
    for rvec, matrix in matrix_set.blocks.items():
        coo = matrix.tocoo()
        for row, col, val in zip(coo.row, coo.col, coo.data):
            atom_i = int(atom_for_global[int(row)])
            atom_j = int(atom_for_global[int(col)])
            key = (rvec, atom_i, atom_j)
            block = blocks.get(key)
            if block is None:
                block = np.zeros(block_shapes[atom_i * natoms + atom_j], dtype=np.complex128)
                blocks[key] = block
            local_row = int(local_for_global[int(row)])
            local_col = int(local_for_global[int(col)])
            block[local_row, local_col] += val
    return blocks


def _atom_blocks_from_sparse_triplet_arrays(
    rvecs: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    vals: np.ndarray,
    structure: OpenMXStructure,
) -> dict[tuple[tuple[int, int, int], int, int], np.ndarray]:
    rvecs = np.asarray(rvecs, dtype=np.int32)
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    vals = np.asarray(vals, dtype=np.complex128)
    if rows.size == 0:
        return {}
    atom_for_global, local_for_global = _global_orbital_to_atom_local(structure)
    atom_i = atom_for_global[rows]
    atom_j = atom_for_global[cols]
    local_rows = local_for_global[rows]
    local_cols = local_for_global[cols]
    order = np.lexsort((atom_j, atom_i, rvecs[:, 2], rvecs[:, 1], rvecs[:, 0]))
    rvecs = rvecs[order]
    atom_i = atom_i[order]
    atom_j = atom_j[order]
    local_rows = local_rows[order]
    local_cols = local_cols[order]
    vals = vals[order]
    starts_mask = np.empty(rows.size, dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = (
        (rvecs[1:, 0] != rvecs[:-1, 0])
        | (rvecs[1:, 1] != rvecs[:-1, 1])
        | (rvecs[1:, 2] != rvecs[:-1, 2])
        | (atom_i[1:] != atom_i[:-1])
        | (atom_j[1:] != atom_j[:-1])
    )
    starts = np.flatnonzero(starts_mask)
    stops = np.concatenate([starts[1:], np.asarray([rows.size], dtype=np.int64)])
    spin_factor = 2 if structure.spinful else 1
    natoms = len(structure.atoms)
    block_shapes = [
        (spin_factor * atom_a.orbital_count, spin_factor * atom_b.orbital_count)
        for atom_a in structure.atoms
        for atom_b in structure.atoms
    ]
    atom_blocks: dict[tuple[tuple[int, int, int], int, int], np.ndarray] = {}
    for start, stop in zip(starts, stops):
        ai = int(atom_i[start])
        aj = int(atom_j[start])
        key = ((int(rvecs[start, 0]), int(rvecs[start, 1]), int(rvecs[start, 2])), ai, aj)
        block = np.zeros(block_shapes[ai * natoms + aj], dtype=np.complex128)
        np.add.at(block, (local_rows[start:stop], local_cols[start:stop]), vals[start:stop])
        atom_blocks[key] = block
    return atom_blocks


def _parse_sparse_entry(line: str) -> tuple[tuple[int, int, int], int, int, complex] | None:
    # OpenMX sparse files are fixed-width in normal output. Slicing avoids the
    # allocation-heavy split path for tens of millions of matrix elements.
    if len(line) >= 59:
        try:
            return (
                (int(line[0:5]), int(line[5:10]), int(line[10:15])),
                int(line[15:21]) - 1,
                int(line[21:27]) - 1,
                float(line[27:43]) + 1.0j * float(line[43:59]),
            )
        except ValueError:
            pass
    parts = line.split()
    if len(parts) != 7:
        return None
    return (
        (int(parts[0]), int(parts[1]), int(parts[2])),
        int(parts[3]) - 1,
        int(parts[4]) - 1,
        float(parts[5]) + 1.0j * float(parts[6]),
    )


def read_openmx_atom_blocks(
    path: str | Path,
    structure: OpenMXStructure,
    *,
    text_parser: str = "line",
) -> AtomBlockMatrixSet:
    path = Path(path)
    if path.suffix == ".npz":
        matrix_set = read_openmx_sparse_npz(path, nwann=structure.nwann)
        return AtomBlockMatrixSet(
            label=matrix_set.label,
            nwann=matrix_set.nwann,
            nrpt=matrix_set.nrpt,
            input_nonzero=int(sum(block.nnz for block in matrix_set.blocks.values())),
            input_r_points=len(matrix_set.blocks),
            atom_blocks=_matrix_to_atom_blocks(matrix_set, structure),
        )
    if text_parser not in {"line", "loadtxt"}:
        raise ValueError("--text-parser must be 'line' or 'loadtxt'.")
    if text_parser == "loadtxt":
        return _read_openmx_atom_blocks_loadtxt(path, structure)
    atom_for_global, local_for_global = _global_orbital_to_atom_local(structure)
    spin_factor = 2 if structure.spinful else 1
    natoms = len(structure.atoms)
    block_shapes = [
        (spin_factor * atom_i.orbital_count, spin_factor * atom_j.orbital_count)
        for atom_i in structure.atoms
        for atom_j in structure.atoms
    ]
    atom_blocks: dict[tuple[tuple[int, int, int], int, int], np.ndarray] = {}
    seen_rvecs: set[tuple[int, int, int]] = set()
    last_key: tuple[tuple[int, int, int], int, int] | None = None
    last_block: np.ndarray | None = None

    with path.open("r", encoding="utf-8") as handle:
        header = [handle.readline() for _ in range(4)]
        if any(line == "" for line in header):
            raise ValueError(f"{path} is too short to be an OpenMX sparse matrix file.")
        label = "H" if "H" in header[0] else "S"
        nonzero = _first_int(header[1])
        nwann = _first_int(header[2])
        nrpt = _first_int(header[3])
        if nwann != structure.nwann:
            raise ValueError(f"{label}.dat has {nwann} orbitals but openmx structure has {structure.nwann}.")

        count = 0
        for line in handle:
            parsed = _parse_sparse_entry(line)
            if parsed is None:
                continue
            rvec, row, col, val = parsed
            atom_i = int(atom_for_global[row])
            atom_j = int(atom_for_global[col])
            key = (rvec, atom_i, atom_j)
            if key == last_key:
                block = last_block
            else:
                block = atom_blocks.get(key)
                if block is None:
                    block = np.zeros(block_shapes[atom_i * natoms + atom_j], dtype=np.complex128)
                    atom_blocks[key] = block
                last_key = key
                last_block = block
            local_row = int(local_for_global[row])
            local_col = int(local_for_global[col])
            block[local_row, local_col] += val
            seen_rvecs.add(rvec)
            count += 1
        if count != nonzero:
            raise ValueError(f"{path} header declares {nonzero} nonzeros, parsed {count}.")

    return AtomBlockMatrixSet(
        label=label,
        nwann=nwann,
        nrpt=nrpt,
        input_nonzero=count,
        input_r_points=len(seen_rvecs),
        atom_blocks=atom_blocks,
    )


def _read_openmx_atom_blocks_loadtxt(path: Path, structure: OpenMXStructure) -> AtomBlockMatrixSet:
    with path.open("r", encoding="utf-8") as handle:
        header = [handle.readline() for _ in range(4)]
    if any(line == "" for line in header):
        raise ValueError(f"{path} is too short to be an OpenMX sparse matrix file.")
    label = "H" if "H" in header[0] else "S"
    nonzero = _first_int(header[1])
    nwann = _first_int(header[2])
    nrpt = _first_int(header[3])
    if nwann != structure.nwann:
        raise ValueError(f"{label}.dat has {nwann} orbitals but openmx structure has {structure.nwann}.")
    if nonzero == 0:
        data = np.empty((0, 7), dtype=np.float64)
    else:
        data = np.loadtxt(path, dtype=np.float64, skiprows=4, ndmin=2)
    if data.shape[1] != 7:
        raise ValueError(f"{path} sparse text must have seven columns, got shape {data.shape}.")
    if int(data.shape[0]) != nonzero:
        raise ValueError(f"{path} header declares {nonzero} nonzeros, parsed {data.shape[0]}.")
    rvecs = data[:, 0:3].astype(np.int32, copy=False)
    rows = data[:, 3].astype(np.int64, copy=False) - 1
    cols = data[:, 4].astype(np.int64, copy=False) - 1
    vals = data[:, 5].astype(np.float64, copy=False) + 1.0j * data[:, 6].astype(np.float64, copy=False)
    return AtomBlockMatrixSet(
        label=label,
        nwann=nwann,
        nrpt=nrpt,
        input_nonzero=nonzero,
        input_r_points=len({tuple(int(v) for v in row) for row in rvecs}),
        atom_blocks=_atom_blocks_from_sparse_triplet_arrays(rvecs, rows, cols, vals, structure),
    )


def _round_rvec(vector: np.ndarray, *, atol: float = 1.0e-6) -> tuple[int, int, int]:
    rounded = np.rint(np.asarray(vector, dtype=float))
    if float(np.linalg.norm(np.asarray(vector, dtype=float) - rounded)) > float(atol):
        raise ValueError(f"Symmetry operation produced non-integer R vector {vector}.")
    return tuple(int(v) for v in rounded.tolist())


def _mate_key(key: tuple[tuple[int, int, int], int, int]) -> tuple[tuple[int, int, int], int, int]:
    rvec, atom_i, atom_j = key
    return ((-rvec[0], -rvec[1], -rvec[2]), atom_j, atom_i)


def _key_sort_tuple(key: tuple[tuple[int, int, int], int, int]) -> tuple[int, int, int, int, int]:
    rvec, atom_i, atom_j = key
    return (int(rvec[0]), int(rvec[1]), int(rvec[2]), int(atom_i), int(atom_j))


def _canonical_key(key: tuple[tuple[int, int, int], int, int]) -> tuple[tuple[int, int, int], int, int]:
    mate = _mate_key(key)
    return key if _key_sort_tuple(key) <= _key_sort_tuple(mate) else mate


def canonicalize_hermitian_blocks(
    atom_blocks: dict[AtomBlockKey, np.ndarray],
) -> dict[AtomBlockKey, np.ndarray]:
    canonical: dict[AtomBlockKey, np.ndarray] = {}
    for key in sorted(atom_blocks, key=_key_sort_tuple):
        canon = _canonical_key(key)
        if canon in canonical:
            continue
        block = atom_blocks[key] if canon == key else atom_blocks[key].conj().T
        canonical[canon] = block
    return canonical


def hermitize_canonical_atom_blocks(atom_blocks: dict[AtomBlockKey, np.ndarray]) -> dict[AtomBlockKey, np.ndarray]:
    canonical: dict[AtomBlockKey, np.ndarray] = {}
    for key, block in atom_blocks.items():
        mate_key = _mate_key(key)
        if key == mate_key:
            canonical[key] = 0.5 * (block + block.conj().T)
            continue
        canon = _canonical_key(key)
        contribution = block if canon == key else block.conj().T
        accum = canonical.get(canon)
        if accum is None:
            canonical[canon] = 0.5 * contribution.copy()
        else:
            accum += 0.5 * contribution
    return {key: canonical[key] for key in sorted(canonical, key=_key_sort_tuple)}


def expand_hermitian_blocks(
    canonical_blocks: dict[AtomBlockKey, np.ndarray],
) -> dict[AtomBlockKey, np.ndarray]:
    out: dict[AtomBlockKey, np.ndarray] = {}
    for key, block in canonical_blocks.items():
        out[key] = block
        mate = _mate_key(key)
        if mate != key:
            out[mate] = block.conj().T
    return out


def transform_atom_blocks(
    atom_blocks: dict[AtomBlockKey, np.ndarray],
    transports: list[OperationTransport],
    action_cache: SymmetryActionCache | None = None,
) -> dict[AtomBlockKey, np.ndarray]:
    if action_cache is None:
        action_cache = SymmetryActionCache(transports)
    transformed: dict[AtomBlockKey, np.ndarray] = {}
    for key, block in atom_blocks.items():
        for new_r, target_i, target_j, left, right_h, antiunitary in action_cache.actions_for(key):
            source_block = block.conj() if antiunitary else block
            new_block = left @ source_block @ right_h
            target_key = (new_r, target_i, target_j)
            accum = transformed.get(target_key)
            if accum is None:
                accum = np.zeros_like(new_block, dtype=np.complex128)
                transformed[target_key] = accum
            accum += new_block
    return transformed


def symmetrize_atom_blocks(
    atom_blocks: dict[AtomBlockKey, np.ndarray],
    transports: list[OperationTransport],
    action_cache: SymmetryActionCache | None = None,
) -> dict[AtomBlockKey, np.ndarray]:
    transformed = transform_atom_blocks(atom_blocks, transports, action_cache=action_cache)
    scale = 1.0 / float(len(transports))
    return {key: value * scale for key, value in transformed.items()}


def transform_canonical_atom_blocks(
    canonical_blocks: dict[AtomBlockKey, np.ndarray],
    transports: list[OperationTransport],
    action_cache: SymmetryActionCache | None = None,
    block_workers: int = 1,
) -> dict[AtomBlockKey, np.ndarray]:
    if action_cache is None:
        action_cache = SymmetryActionCache(transports)
    block_workers = int(block_workers)
    items = sorted(canonical_blocks.items(), key=lambda item: _key_sort_tuple(item[0]))
    actions_by_key = _prepare_canonical_actions_by_key(items, action_cache)
    if block_workers <= 1 or len(canonical_blocks) < 2:
        return _transform_canonical_atom_block_items(items, actions_by_key)
    chunks = _chunk_atom_block_items(items, max_chunks=block_workers)
    transformed: dict[AtomBlockKey, np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=block_workers) as executor:
        futures = [
            executor.submit(_transform_canonical_atom_block_items, chunk, actions_by_key)
            for chunk in chunks
        ]
        for future in futures:
            partial = future.result()
            for key in sorted(partial, key=_key_sort_tuple):
                accum = transformed.get(key)
                if accum is None:
                    transformed[key] = partial[key].copy()
                else:
                    accum += partial[key]
    return {key: transformed[key] for key in sorted(transformed, key=_key_sort_tuple)}


def _prepare_canonical_actions_by_key(
    items: list[AtomBlockItem],
    action_cache: SymmetryActionCache,
) -> dict[AtomBlockKey, tuple[CanonicalSymmetryAction, ...]]:
    return {key: action_cache.canonical_actions_for(key) for key, _block in items}


def _chunk_atom_block_items(
    items: list[AtomBlockItem],
    max_chunks: int,
) -> list[list[AtomBlockItem]]:
    max_chunks = max(1, min(int(max_chunks), len(items)))
    total_work = sum(max(1, int(block.size)) for _key, block in items)
    target_work = max(1, math.ceil(total_work / max_chunks))
    chunks: list[list[AtomBlockItem]] = []
    current: list[AtomBlockItem] = []
    current_work = 0
    for item in items:
        block_work = max(1, int(item[1].size))
        if current and current_work + block_work > target_work and len(chunks) + 1 < max_chunks:
            chunks.append(current)
            current = []
            current_work = 0
        current.append(item)
        current_work += block_work
    if current:
        chunks.append(current)
    return chunks


def _transform_canonical_atom_block_items(
    items: list[AtomBlockItem],
    actions_by_key: dict[AtomBlockKey, tuple[CanonicalSymmetryAction, ...]],
) -> dict[AtomBlockKey, np.ndarray]:
    transformed: dict[AtomBlockKey, np.ndarray] = {}
    for key, block in items:
        for canon, needs_adjoint, left, right_h, antiunitary in actions_by_key[key]:
            source_block = block.conj() if antiunitary else block
            new_block = left @ source_block @ right_h
            accum_block = new_block.conj().T if needs_adjoint else new_block
            accum = transformed.get(canon)
            if accum is None:
                accum = np.zeros_like(accum_block, dtype=np.complex128)
                transformed[canon] = accum
            accum += accum_block
    return transformed


def symmetrize_canonical_atom_blocks(
    canonical_blocks: dict[tuple[tuple[int, int, int], int, int], np.ndarray],
    transports: list[OperationTransport],
    action_cache: SymmetryActionCache | None = None,
    block_workers: int = 1,
) -> dict[tuple[tuple[int, int, int], int, int], np.ndarray]:
    transformed = transform_canonical_atom_blocks(
        canonical_blocks,
        transports,
        action_cache=action_cache,
        block_workers=block_workers,
    )
    scale = 1.0 / float(len(transports))
    return {key: value * scale for key, value in transformed.items()}


def _blas_thread_context(blas_threads: int | None):
    if blas_threads is None or int(blas_threads) <= 0:
        return nullcontext()
    try:
        from threadpoolctl import threadpool_limits
    except Exception:
        return nullcontext()
    return threadpool_limits(limits=int(blas_threads))


def hermitize_atom_blocks(
    atom_blocks: dict[tuple[tuple[int, int, int], int, int], np.ndarray],
) -> dict[tuple[tuple[int, int, int], int, int], np.ndarray]:
    keys = set(atom_blocks)
    keys.update(_mate_key(key) for key in atom_blocks)
    out: dict[tuple[tuple[int, int, int], int, int], np.ndarray] = {}
    visited: set[tuple[tuple[int, int, int], int, int]] = set()
    for key in sorted(keys, key=_key_sort_tuple):
        if key in visited:
            continue
        mate_key = _mate_key(key)
        block = atom_blocks.get(key)
        mate = atom_blocks.get(mate_key)
        visited.add(key)
        visited.add(mate_key)
        if key == mate_key:
            if block is None:
                continue
            out[key] = 0.5 * (block + block.conj().T)
            continue
        if block is None and mate is None:
            continue
        if block is None:
            averaged = 0.5 * mate.conj().T
        elif mate is None:
            averaged = 0.5 * block
        else:
            averaged = 0.5 * (block + mate.conj().T)
        out[key] = averaged
        out[mate_key] = averaged.conj().T
    return out


def _norm_blocks(atom_blocks: dict[tuple[tuple[int, int, int], int, int], np.ndarray]) -> float:
    total = 0.0
    for block in atom_blocks.values():
        total += float(np.sum(np.abs(block) ** 2))
    return math.sqrt(total)


def _difference_norm(
    left: dict[tuple[tuple[int, int, int], int, int], np.ndarray],
    right: dict[tuple[tuple[int, int, int], int, int], np.ndarray],
) -> float:
    total = 0.0
    for key in set(left).union(right):
        l_block = left.get(key)
        r_block = right.get(key)
        if l_block is None:
            diff = r_block
        elif r_block is None:
            diff = l_block
        else:
            diff = l_block - r_block
        total += float(np.sum(np.abs(diff) ** 2))
    return math.sqrt(total)


def relative_difference(
    left: dict[tuple[tuple[int, int, int], int, int], np.ndarray],
    right: dict[tuple[tuple[int, int, int], int, int], np.ndarray],
) -> float:
    denom = _norm_blocks(right)
    diff = _difference_norm(left, right)
    if denom == 0.0:
        return 0.0 if diff == 0.0 else float("inf")
    return diff / denom


def symmetry_covariance_residual(
    atom_blocks: dict[tuple[tuple[int, int, int], int, int], np.ndarray],
    transports: list[OperationTransport],
    action_cache: SymmetryActionCache | None = None,
) -> float:
    transformed = transform_atom_blocks(atom_blocks, transports, action_cache=action_cache)
    averaged_transform = {key: value / float(len(transports)) for key, value in transformed.items()}
    return relative_difference(averaged_transform, atom_blocks)


def hermiticity_residual(atom_blocks: dict[tuple[tuple[int, int, int], int, int], np.ndarray]) -> float:
    mate_blocks = {
        ((-rvec[0], -rvec[1], -rvec[2]), atom_j, atom_i): block.conj().T
        for (rvec, atom_i, atom_j), block in atom_blocks.items()
    }
    return relative_difference(mate_blocks, atom_blocks)


def max_abs_change(
    before: dict[tuple[tuple[int, int, int], int, int], np.ndarray],
    after: dict[tuple[tuple[int, int, int], int, int], np.ndarray],
) -> float:
    result = 0.0
    for key in set(before).union(after):
        left = before.get(key)
        right = after.get(key)
        if left is None:
            diff = right
        elif right is None:
            diff = left
        else:
            diff = left - right
        if diff is not None and diff.size:
            result = max(result, float(np.max(np.abs(diff))))
    return result


def _prune_and_count_atom_blocks(
    atom_blocks: dict[AtomBlockKey, np.ndarray],
    cutoff: float,
) -> tuple[dict[AtomBlockKey, np.ndarray], int]:
    pruned: dict[AtomBlockKey, np.ndarray] = {}
    nonzero = 0
    threshold = float(cutoff)
    for key, block in atom_blocks.items():
        count = int(np.count_nonzero(np.abs(block) > threshold))
        if count:
            pruned[key] = block
            nonzero += count
    return pruned, nonzero


def _symmetrize_atom_blocks_with_report(
    label: str,
    input_r_points: int,
    input_nonzero: int,
    atom_blocks: dict[tuple[tuple[int, int, int], int, int], np.ndarray],
    structure: OpenMXStructure,
    transports: list[OperationTransport],
    cutoff: float,
    action_cache: SymmetryActionCache | None = None,
    block_workers: int = 1,
    diagnostics: str = "basic",
) -> tuple[dict[tuple[tuple[int, int, int], int, int], np.ndarray], dict[str, float | int | None]]:
    if action_cache is None:
        action_cache = SymmetryActionCache(transports)
    if diagnostics not in {"basic", "full"}:
        raise ValueError("--diagnostics must be 'basic' or 'full'.")
    timing: dict[str, float] = {}
    t0 = time.perf_counter()
    sym_blocks = hermitize_canonical_atom_blocks(atom_blocks)
    timing["hermitize_canonical_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(2):
        sym_blocks = symmetrize_canonical_atom_blocks(
            sym_blocks,
            transports,
            action_cache=action_cache,
            block_workers=block_workers,
        )
    timing["project_twice_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    sym_blocks_full = expand_hermitian_blocks(sym_blocks)
    pruned, output_nonzero = _prune_and_count_atom_blocks(sym_blocks_full, cutoff)
    timing["expand_prune_s"] = time.perf_counter() - t0
    report = {
        "input_r_points": int(input_r_points),
        "input_nonzero": int(input_nonzero),
        "output_r_points": len({key[0] for key in pruned}),
        "output_nonzero": int(output_nonzero),
    }
    t0 = time.perf_counter()
    if diagnostics == "full":
        report.update(
            {
                "hermiticity_residual_before": hermiticity_residual(atom_blocks),
                "hermiticity_residual_after": hermiticity_residual(pruned),
                "symmetry_covariance_residual_before": symmetry_covariance_residual(
                    atom_blocks, transports, action_cache=action_cache
                ),
                "symmetry_covariance_residual_after": symmetry_covariance_residual(pruned, transports, action_cache=action_cache),
                "max_absolute_change": max_abs_change(atom_blocks, pruned),
            }
        )
    else:
        report.update(
            {
                "hermiticity_residual_before": None,
                "hermiticity_residual_after": 0.0,
                "symmetry_covariance_residual_before": None,
                "symmetry_covariance_residual_after": None,
                "max_absolute_change": None,
            }
        )
    timing["diagnostics_s"] = time.perf_counter() - t0
    timing["symmetrize_compute_s"] = (
        timing["hermitize_canonical_s"]
        + timing["project_twice_s"]
        + timing["expand_prune_s"]
        + timing["diagnostics_s"]
    )
    report["timing_s"] = timing
    return pruned, report


def symmetrize_matrix_set(
    matrix_set: SparseMatrixSet,
    structure: OpenMXStructure,
    transports: list[OperationTransport],
    cutoff: float,
    action_cache: SymmetryActionCache | None = None,
    block_workers: int = 1,
    diagnostics: str = "basic",
) -> tuple[dict[tuple[tuple[int, int, int], int, int], np.ndarray], dict[str, float | int | None]]:
    if matrix_set.nwann != structure.nwann:
        raise ValueError(f"{matrix_set.label}.dat has {matrix_set.nwann} orbitals but openmx structure has {structure.nwann}.")
    atom_blocks = _matrix_to_atom_blocks(matrix_set, structure)
    return _symmetrize_atom_blocks_with_report(
        matrix_set.label,
        len(matrix_set.blocks),
        int(sum(block.nnz for block in matrix_set.blocks.values())),
        atom_blocks,
        structure,
        transports,
        cutoff,
        action_cache=action_cache,
        block_workers=block_workers,
        diagnostics=diagnostics,
    )


def symmetrize_atom_block_matrix_set(
    matrix_set: AtomBlockMatrixSet,
    structure: OpenMXStructure,
    transports: list[OperationTransport],
    cutoff: float,
    action_cache: SymmetryActionCache | None = None,
    block_workers: int = 1,
    diagnostics: str = "basic",
) -> tuple[dict[tuple[tuple[int, int, int], int, int], np.ndarray], dict[str, float | int | None]]:
    if matrix_set.nwann != structure.nwann:
        raise ValueError(f"{matrix_set.label}.dat has {matrix_set.nwann} orbitals but openmx structure has {structure.nwann}.")
    return _symmetrize_atom_blocks_with_report(
        matrix_set.label,
        matrix_set.input_r_points,
        matrix_set.input_nonzero,
        matrix_set.atom_blocks,
        structure,
        transports,
        cutoff,
        action_cache=action_cache,
        block_workers=block_workers,
        diagnostics=diagnostics,
    )


def _count_output_nonzeros(atom_blocks: dict[tuple[tuple[int, int, int], int, int], np.ndarray], cutoff: float) -> int:
    return int(sum(np.count_nonzero(np.abs(block) > float(cutoff)) for block in atom_blocks.values()))


def _write_triplet_text_chunk(
    handle,
    rvec: tuple[int, int, int],
    rows: np.ndarray,
    cols: np.ndarray,
    vals: np.ndarray,
    *,
    chunk_size: int = 1_000_000,
) -> None:
    total = int(rows.size)
    fmt = "%5.0f%5.0f%5.0f%6.0f%6.0f%16.8f%16.8f"
    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        size = stop - start
        data = np.empty((size, 7), dtype=np.float64)
        data[:, 0] = rvec[0]
        data[:, 1] = rvec[1]
        data[:, 2] = rvec[2]
        data[:, 3] = rows[start:stop].astype(np.float64, copy=False) + 1.0
        data[:, 4] = cols[start:stop].astype(np.float64, copy=False) + 1.0
        data[:, 5] = vals[start:stop].real
        data[:, 6] = vals[start:stop].imag
        np.savetxt(handle, data, fmt=fmt)


def write_openmx_sparse_triplets(
    path: str | Path,
    label: str,
    structure: OpenMXStructure,
    triplets: SparseTriplets,
) -> None:
    path = Path(path)
    rvecs = sorted(triplets)
    nonzero = int(sum(rows.size for rows, _cols, _vals in triplets.values()))
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f" ! Sparse format of {label}\n")
        handle.write(f" {nonzero} ! Number of non-zeros lines of {label}mnR\n")
        handle.write(f" {structure.nwann} ! Number of orbitals\n")
        handle.write(f" {len(rvecs)} ! Number of R points\n")
        for rvec in rvecs:
            rows, cols, vals = triplets[rvec]
            _write_triplet_text_chunk(handle, rvec, rows, cols, vals)


def write_openmx_sparse(
    path: str | Path,
    label: str,
    structure: OpenMXStructure,
    atom_blocks: dict[tuple[tuple[int, int, int], int, int], np.ndarray],
    cutoff: float,
) -> None:
    path = Path(path)
    rvecs = sorted({key[0] for key in atom_blocks})
    nonzero = _count_output_nonzeros(atom_blocks, cutoff)
    global_indices = [_atom_global_indices(atom, structure) for atom in structure.atoms]
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f" ! Sparse format of {label}\n")
        handle.write(f" {nonzero} ! Number of non-zeros lines of {label}mnR\n")
        handle.write(f" {structure.nwann} ! Number of orbitals\n")
        handle.write(f" {len(rvecs)} ! Number of R points\n")
        sorted_items = sorted(atom_blocks.items(), key=lambda item: (item[0][0], item[0][1], item[0][2]))
        for (rvec, atom_i, atom_j), block in sorted_items:
            rows, cols = np.nonzero(np.abs(block) > float(cutoff))
            order = np.lexsort((cols, rows))
            row_indices = global_indices[atom_i][rows[order]] + 1
            col_indices = global_indices[atom_j][cols[order]] + 1
            local_rows = rows[order]
            local_cols = cols[order]
            lines = []
            for row_index, col_index, local_row, local_col in zip(row_indices, col_indices, local_rows, local_cols):
                value = block[int(local_row), int(local_col)]
                lines.append(
                    f"{rvec[0]:5d}{rvec[1]:5d}{rvec[2]:5d}"
                    f"{int(row_index):6d}{int(col_index):6d}"
                    f"{value.real:16.8f}{value.imag:16.8f}\n"
                )
            handle.writelines(lines)


def _atom_block_sparse_triplets(
    structure: OpenMXStructure,
    atom_blocks: dict[AtomBlockKey, np.ndarray],
    cutoff: float,
) -> SparseTriplets:
    global_indices = [_atom_global_indices(atom, structure) for atom in structure.atoms]
    grouped: dict[tuple[int, int, int], list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for (rvec, atom_i, atom_j), block in sorted(atom_blocks.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        rows, cols = np.nonzero(np.abs(block) > float(cutoff))
        if rows.size == 0:
            continue
        order = np.lexsort((cols, rows))
        grouped.setdefault(rvec, []).append(
            (
                global_indices[atom_i][rows[order]].astype(np.int32, copy=False),
                global_indices[atom_j][cols[order]].astype(np.int32, copy=False),
                block[rows[order], cols[order]].astype(np.complex128, copy=False),
            )
        )
    triplets = {}
    for rvec in sorted(grouped):
        pieces = grouped[rvec]
        triplets[rvec] = (
            np.concatenate([piece[0] for piece in pieces]).astype(np.int32, copy=False),
            np.concatenate([piece[1] for piece in pieces]).astype(np.int32, copy=False),
            np.concatenate([piece[2] for piece in pieces]).astype(np.complex128, copy=False),
        )
    return triplets


def write_openmx_sparse_npz(
    path: str | Path,
    structure: OpenMXStructure,
    atom_blocks: dict[AtomBlockKey, np.ndarray],
    cutoff: float,
) -> None:
    write_openmx_sparse_npz_triplets(path, _atom_block_sparse_triplets(structure, atom_blocks, cutoff))


def write_openmx_sparse_npz_triplets(
    path: str | Path,
    triplets: SparseTriplets,
) -> None:
    storable = {}
    for rvec, (rows, cols, vals) in triplets.items():
        storable[f"{rvec}_row"] = rows
        storable[f"{rvec}_col"] = cols
        storable[f"{rvec}_val"] = vals
    np.savez(Path(path), **storable)


def _resolve_structure_file(input_dir: Path) -> Path:
    rigid = input_dir / "openmx.dat_rigid"
    if rigid.is_file():
        return rigid
    regular = input_dir / "openmx.dat"
    if regular.is_file():
        return regular
    raise FileNotFoundError(f"Neither openmx.dat_rigid nor openmx.dat exists under {input_dir}.")


def _symmetrize_and_write_matrix(
    label: str,
    matrix_file: Path,
    output_file: Path,
    structure: OpenMXStructure,
    transports: list[OperationTransport],
    cutoff: float,
    block_workers: int,
    write_npz_cache: bool,
    diagnostics: str,
    blas_threads: int | None,
    text_parser: str,
) -> tuple[str, dict[str, float | int | None]]:
    action_cache = SymmetryActionCache(transports)
    timing: dict[str, float] = {}
    t0 = time.perf_counter()
    matrix_set = read_openmx_atom_blocks(matrix_file, structure, text_parser=text_parser)
    timing["read_input_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    with _blas_thread_context(blas_threads):
        blocks, matrix_report = symmetrize_atom_block_matrix_set(
            matrix_set,
            structure,
            transports,
            cutoff=cutoff,
            action_cache=action_cache,
            block_workers=block_workers,
            diagnostics=diagnostics,
        )
    timing["symmetrize_total_s"] = time.perf_counter() - t0
    timing.update({f"sym_{key}": float(value) for key, value in matrix_report.get("timing_s", {}).items()})
    if write_npz_cache:
        t0 = time.perf_counter()
        triplets = _atom_block_sparse_triplets(structure, blocks, cutoff=cutoff)
        timing["collect_triplets_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        write_openmx_sparse_triplets(output_file, label, structure, triplets)
        timing["write_dat_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        write_openmx_sparse_npz_triplets(output_file.with_suffix(".npz"), triplets)
        timing["write_npz_s"] = time.perf_counter() - t0
    else:
        timing["collect_triplets_s"] = 0.0
        t0 = time.perf_counter()
        write_openmx_sparse(output_file, label, structure, blocks, cutoff=cutoff)
        timing["write_dat_s"] = time.perf_counter() - t0
        timing["write_npz_s"] = 0.0
    matrix_report["timing_s"] = {**matrix_report.get("timing_s", {}), **timing}
    return label, matrix_report


def run(args: argparse.Namespace) -> dict[str, object]:
    run_start = time.perf_counter()
    timing: dict[str, float] = {}
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    structure_file = _resolve_structure_file(input_dir)
    matrix_workers = int(args.matrix_workers)
    if matrix_workers < 1:
        raise ValueError("--matrix-workers must be >= 1.")
    if matrix_workers > 2:
        raise ValueError("--matrix-workers must be <= 2 because there are only H and S matrix jobs.")
    block_workers = int(args.block_workers)
    if block_workers < 1:
        raise ValueError("--block-workers must be >= 1.")
    diagnostics = str(args.diagnostics)
    if diagnostics not in {"basic", "full"}:
        raise ValueError("--diagnostics must be 'basic' or 'full'.")
    text_parser = str(args.text_parser)
    if text_parser not in {"line", "loadtxt"}:
        raise ValueError("--text-parser must be 'line' or 'loadtxt'.")
    blas_threads = None if args.blas_threads is None else int(args.blas_threads)
    if blas_threads is not None and blas_threads < 1:
        raise ValueError("--blas-threads must be >= 1 when provided.")
    max_worker_cores = None if args.max_worker_cores is None else int(args.max_worker_cores)
    if max_worker_cores is not None:
        if max_worker_cores < 1:
            raise ValueError("--max-worker-cores must be >= 1 when provided.")
        effective_workers = matrix_workers * block_workers * int(blas_threads or 1)
        if effective_workers > max_worker_cores:
            raise ValueError(
                f"Requested matrix_workers*block_workers*blas_threads={effective_workers}, "
                f"above --max-worker-cores={max_worker_cores}. Reduce --matrix-workers, "
                "--block-workers, or --blas-threads."
            )

    t0 = time.perf_counter()
    structure = parse_openmx_structure(structure_file)
    timing["parse_openmx_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    operations = build_symmetry_operations(structure, symprec=args.symprec)
    timing["spglib_get_symmetry_s"] = time.perf_counter() - t0
    raw_operation_count = len(operations)
    t0 = time.perf_counter()
    transports = build_operation_transports(structure, operations, atom_tol=max(float(args.symprec), 5.0e-6))
    timing["transport_build_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    operations, transports, magnetic_report = apply_magnetic_symmetry_filter(
        structure,
        operations,
        transports,
        mode=str(args.magnetic_symmetry),
        magmom_tol=float(args.magmom_tol),
    )
    timing["magnetic_filter_s"] = time.perf_counter() - t0
    report: dict[str, object] = {
        "input_dir": str(input_dir),
        "structure_file": str(structure_file),
        "spglib_symprec": float(args.symprec),
        "spglib_operation_count": raw_operation_count,
        "operation_count": len(operations),
        "magnetic_symmetry": magnetic_report,
        "max_atom_mapping_residual": max((item.max_mapping_residual for item in transports), default=0.0),
        "nwann": structure.nwann,
        "nwann_spinless": structure.nwann_spinless,
        "natoms": len(structure.atoms),
        "spinful": structure.spinful,
        "spin_mode": structure.spin_mode,
        "matrix_workers": matrix_workers,
        "block_workers": block_workers,
        "diagnostics": diagnostics,
        "text_parser": text_parser,
        "blas_threads": blas_threads,
        "max_worker_cores": max_worker_cores,
        "write_npz_cache": bool(args.write_npz_cache),
        "timing_s": timing,
    }
    if args.export_transports_json:
        transport_json = Path(args.export_transports_json).expanduser()
        t0 = time.perf_counter()
        write_symmetry_transports_json(
            transport_json,
            structure,
            operations,
            transports,
            symprec=args.symprec,
            magnetic_report=magnetic_report,
        )
        timing["write_transports_json_s"] = time.perf_counter() - t0
        report["symmetry_transports_json"] = str(transport_json.resolve())
    else:
        timing["write_transports_json_s"] = 0.0

    if args.dry_run:
        timing["total_wall_s"] = time.perf_counter() - run_start
        print(json.dumps(report, indent=2, sort_keys=True))
        return report

    h_file = input_dir / args.h_file
    s_file = input_dir / args.s_file
    if not h_file.is_file():
        raise FileNotFoundError(h_file)
    if not s_file.is_file():
        raise FileNotFoundError(s_file)

    if output_dir.exists():
        if not args.force:
            raise FileExistsError(f"{output_dir} already exists; use --force to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    matrix_jobs = [
        ("H", h_file, output_dir / "H_sym.dat"),
        ("S", s_file, output_dir / "S_sym.dat"),
    ]
    if matrix_workers == 1:
        action_cache = SymmetryActionCache(transports)
        for label, matrix_file, output_file in matrix_jobs:
            matrix_timing: dict[str, float] = {}
            t0 = time.perf_counter()
            matrix_set = read_openmx_atom_blocks(matrix_file, structure, text_parser=text_parser)
            matrix_timing["read_input_s"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            with _blas_thread_context(blas_threads):
                blocks, matrix_report = symmetrize_atom_block_matrix_set(
                    matrix_set,
                    structure,
                    transports,
                    cutoff=args.cutoff,
                    action_cache=action_cache,
                    block_workers=block_workers,
                    diagnostics=diagnostics,
                )
            matrix_timing["symmetrize_total_s"] = time.perf_counter() - t0
            matrix_timing.update({f"sym_{key}": float(value) for key, value in matrix_report.get("timing_s", {}).items()})
            if args.write_npz_cache:
                t0 = time.perf_counter()
                triplets = _atom_block_sparse_triplets(structure, blocks, cutoff=args.cutoff)
                matrix_timing["collect_triplets_s"] = time.perf_counter() - t0
                t0 = time.perf_counter()
                write_openmx_sparse_triplets(output_file, label, structure, triplets)
                matrix_timing["write_dat_s"] = time.perf_counter() - t0
                t0 = time.perf_counter()
                write_openmx_sparse_npz_triplets(output_file.with_suffix(".npz"), triplets)
                matrix_timing["write_npz_s"] = time.perf_counter() - t0
            else:
                matrix_timing["collect_triplets_s"] = 0.0
                t0 = time.perf_counter()
                write_openmx_sparse(output_file, label, structure, blocks, cutoff=args.cutoff)
                matrix_timing["write_dat_s"] = time.perf_counter() - t0
                matrix_timing["write_npz_s"] = 0.0
            matrix_report["timing_s"] = {**matrix_report.get("timing_s", {}), **matrix_timing}
            report[label] = matrix_report
    else:
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=min(matrix_workers, len(matrix_jobs))) as executor:
            futures = [
                executor.submit(
                    _symmetrize_and_write_matrix,
                    label,
                    matrix_file,
                    output_file,
                    structure,
                    transports,
                    args.cutoff,
                    block_workers,
                    bool(args.write_npz_cache),
                    diagnostics,
                    blas_threads,
                    text_parser,
                )
                for label, matrix_file, output_file in matrix_jobs
            ]
            for future in as_completed(futures):
                label, matrix_report = future.result()
                report[label] = matrix_report
        timing["matrix_workers_wall_s"] = time.perf_counter() - t0
    timing["total_wall_s"] = time.perf_counter() - run_start
    (output_dir / "symmetrization_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {output_dir / 'H_sym.dat'}")
    print(f"Wrote {output_dir / 'S_sym.dat'}")
    if args.write_npz_cache:
        print(f"Wrote {output_dir / 'H_sym.npz'}")
        print(f"Wrote {output_dir / 'S_sym.npz'}")
    print(f"Wrote {output_dir / 'symmetrization_report.json'}")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experimental OpenMX H.dat/S.dat space-group symmetrizer for NSOC and NC/SOC spinor outputs.")
    parser.add_argument("--input-dir", required=True, help="Directory containing openmx.dat[_rigid], H.dat, and S.dat.")
    parser.add_argument("--output-dir", required=True, help="Output directory for H_sym.dat, S_sym.dat, and report JSON.")
    parser.add_argument("--h-file", default="H.dat", help="Hamiltonian filename under input-dir.")
    parser.add_argument("--s-file", default="S.dat", help="Overlap filename under input-dir.")
    parser.add_argument("--symprec", type=float, default=1.0e-3, help="spglib symmetry precision.")
    parser.add_argument(
        "--magnetic-symmetry",
        choices=("auto", "off", "unitary", "magnetic"),
        default="auto",
        help=(
            "Magnetic operation filter. 'auto' keeps ordinary behavior for non-magnetic inputs and, "
            "when OpenMX collinear or NC/SOC magnetic moments are present, keeps unitary operations and primed "
            "antiunitary operations that preserve axial magnetic moments."
        ),
    )
    parser.add_argument(
        "--magmom-tol",
        type=float,
        default=1.0e-5,
        help="Tolerance in Bohr magneton-like OpenMX occupation units for magnetic operation filtering.",
    )
    parser.add_argument("--cutoff", type=float, default=1.0e-7, help="Drop output matrix elements with abs(value) <= cutoff.")
    parser.add_argument("--matrix-workers", type=int, default=1, help="Independent worker processes for H/S matrices; use 2 to process H.dat and S.dat concurrently.")
    parser.add_argument("--block-workers", type=int, default=1, help="Thread workers per matrix for canonical atom-block symmetry transforms.")
    parser.add_argument(
        "--diagnostics",
        choices=("basic", "full"),
        default="basic",
        help="Diagnostic cost level. 'full' computes full covariance/max-change residuals; 'basic' skips those expensive passes.",
    )
    parser.add_argument(
        "--text-parser",
        choices=("line", "loadtxt"),
        default="line",
        help="Sparse DAT parser. 'line' is low-memory; 'loadtxt' is faster but uses more memory.",
    )
    parser.add_argument(
        "--blas-threads",
        type=int,
        default=1,
        help="Per-process BLAS/OpenMP threadpool limit applied inside the symmetrizer; use with block workers to avoid oversubscription.",
    )
    parser.add_argument(
        "--max-worker-cores",
        type=int,
        help="Reject runs whose matrix_workers*block_workers*blas_threads exceeds this budget.",
    )
    parser.add_argument("--write-npz-cache", action="store_true", help="Also write TAPW-compatible H_sym.npz and S_sym.npz sparse cache sidecars.")
    parser.add_argument("--export-transports-json", help="Write C/C++ native symmetrizer transport JSON after spglib analysis.")
    parser.add_argument("--force", action="store_true", help="Replace output-dir if it already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Parse structure and symmetry operations without writing outputs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except Exception as exc:
        parser.exit(1, f"openmx_symmetrize_hs: error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
