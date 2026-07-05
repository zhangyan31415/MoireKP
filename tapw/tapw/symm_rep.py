"""CLI entry point for TAPW symmetry-representation post-processing."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import canonical_profile_name, canonical_qshell_name
from .config import Config
from .io.hr import HrSparseHandler
from .io.structure import OpenMXFile, StructureProcessorSpglib
from .reporting import TapwReporter
from .workflows.band import BandStructureCalculator
from .workflows.symm_rep import run_symm_rep, run_symm_rep_from_point_sources, select_band_indices


@dataclass(frozen=True)
class ConfigSymmRepRequest:
    config_path: Path
    symmetry_dir: Path
    output_dir: Path
    fermi_energy: float
    valence_count: int
    conduction_count: int
    degeneracy_tol: float
    points: dict[str, tuple[float, float, float]]


def _point_tuple(value: Any, *, label: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) not in {2, 3}:
        raise ValueError(f"symmetry.representation.points.{label} must be a 2- or 3-entry fractional coordinate.")
    coords = tuple(float(item) for item in value)
    if len(coords) == 2:
        return coords[0], coords[1], 0.0
    return coords


def _parse_representation_points(raw_points: Any) -> dict[str, tuple[float, float, float]]:
    if not isinstance(raw_points, dict) or not raw_points:
        raise ValueError("symmetry.representation.points must be a non-empty mapping like {Gamma: [0, 0]}.")
    return {str(label): _point_tuple(value, label=str(label)) for label, value in raw_points.items()}


def resolve_config_request(config_path: Path) -> ConfigSymmRepRequest:
    config_path = Path(config_path).expanduser().resolve()
    config = Config.from_yaml(str(config_path))
    symmetry = dict(getattr(config, "symmetry", {}) or {})
    representation = dict(symmetry.get("representation", {}) or {})
    if not representation:
        raise ValueError("Config is missing symmetry.representation.")
    if "efermi" not in symmetry:
        raise ValueError("symmetry.efermi is required for tapw symm-rep config mode.")

    profile = canonical_profile_name(
        getattr(config.compute, "valley_flag", symmetry.get("valley", "unknown")),
        spin=getattr(config.twist, "spin", True),
    )
    q_shell = canonical_qshell_name(symmetry.get("q_shell", getattr(config.compute, "n_g")))
    run_root = Path(config.paths.output_dir) / profile / q_shell
    symmetry_dir = run_root / "symmetry"
    packed_rawh = symmetry_dir / "representations.npz"
    if not packed_rawh.is_file():
        raise FileNotFoundError(
            f"Missing raw-H symmetry representations: {packed_rawh}\n"
            f"Run `tapw symm -c {config_path}` first."
        )

    output_dir = Path(representation.get("output_dir", "symm_rep"))
    if not output_dir.is_absolute():
        output_dir = run_root / output_dir

    return ConfigSymmRepRequest(
        config_path=config_path,
        symmetry_dir=symmetry_dir,
        output_dir=output_dir,
        fermi_energy=float(symmetry["efermi"]),
        valence_count=int(representation.get("valence_count", 20)),
        conduction_count=int(representation.get("conduction_count", 20)),
        degeneracy_tol=float(representation.get("degeneracy_tol", 2.0e-3)),
        points=_parse_representation_points(representation.get("points")),
    )


def _load_hr(path: str, processor: StructureProcessorSpglib, *, tapw: bool):
    if str(path).endswith(".npz"):
        handler = HrSparseHandler(
            file_name="",
            npz_file_name=path,
            A=processor.transformed_index_matrix if tapw else None,
            read_from_npz=True,
        )
    elif str(path).endswith(".dat"):
        handler = HrSparseHandler(
            file_name=path,
            npz_file_name="",
            A=processor.transformed_index_matrix if tapw else None,
            read_from_npz=False,
        )
    else:
        raise ValueError(f"H/S file suffix must be .npz or .dat, got {path!r}")
    return handler.get_hr_sparse()


def _build_point_calculator(config: Config) -> BandStructureCalculator:
    logger = logging.getLogger(__name__ + ".config")
    reporter = TapwReporter(logger)
    structure = OpenMXFile(
        file_path=str(Path(config.paths.input_file)),
        twist_index=config.twist.twist_index_m,
        spin=config.twist.spin,
    )
    processor = StructureProcessorSpglib(
        input_data=structure.sorted_species_coordinates,
        num_layers=config.twist.num_layers,
        twist_layer=config.twist.twist_layer,
        layer_eps=config.cluster.layer_eps,
        layer_min_samples=config.cluster.layer_min_samples,
        sublayer_eps=config.cluster.sublayer_eps,
        sublayer_min_samples=config.cluster.sublayer_min_samples,
        atom_eps=config.cluster.atom_eps,
        atom_min_samples=config.cluster.atom_min_samples,
        period=config.cluster.period,
        k_max=config.cluster.k_max,
        spin=structure.spin,
        twist_index=config.twist.twist_index_m,
        Tmat=structure.Tmat,
        reciprocal_Tmat=structure.reciprocal_Tmat,
        reporter=reporter,
    )
    if config.compute.TAPW:
        processor.process()
    hr = _load_hr(config.paths.H_file, processor, tapw=bool(config.compute.TAPW))
    if bool(getattr(config.compute, "orthogonal_basis", False)):
        sr = None
    else:
        if not config.paths.S_file:
            raise ValueError("Non-orthogonal TAPW symm-rep config mode requires paths.S_file.")
        sr = _load_hr(config.paths.S_file, processor, tapw=bool(config.compute.TAPW))
    return BandStructureCalculator(
        hr_supercell=hr,
        sr_supercell=sr,
        structure=processor,
        config=config.compute,
        kpath_config=None,
        reporter=reporter,
    )


def calculate_config_point_sources(request: ConfigSymmRepRequest) -> dict[str, dict[str, Any]]:
    config = Config.from_yaml(str(request.config_path))
    config.compute.mode = "band"
    config.compute.efermi = request.fermi_energy
    config.compute.eig_vec_cal = True
    config.compute.hamk_save = False
    initial_num_bands = max(
        int(getattr(config.compute, "num_bands_cal", 0) or 0),
        int(request.valence_count + request.conduction_count + 8),
    )
    config.compute.num_bands_cal = initial_num_bands
    calculator = _build_point_calculator(config)
    tapw_parameters = getattr(calculator, "TAPW_parameters", None)
    sewing_context = None
    if tapw_parameters is not None:
        g1 = np.asarray(getattr(tapw_parameters, "g_vec_list_K1", np.zeros((0, 2))), dtype=float)
        g2 = np.asarray(getattr(tapw_parameters, "g_vec_list_K2", np.zeros((0, 2))), dtype=float)
        reciprocal = np.vstack(
            [
                np.asarray(getattr(calculator.structure, "reciprocal_Tmat", np.eye(3))[0][:2], dtype=float),
                np.asarray(getattr(calculator.structure, "reciprocal_Tmat", np.eye(3))[1][:2], dtype=float),
            ]
        )
        sewing_context = {
            "g_vectors_by_group": [g1, g2],
            "reciprocal_basis": reciprocal,
            "spin_blocks": 2 if bool(getattr(config.twist, "spin", True)) else 1,
        }
    point_sources: dict[str, dict[str, Any]] = {}
    for point_index, (label, coords) in enumerate(request.points.items()):
        eig, vec, selections = _solve_config_point_with_requested_window(
            calculator,
            config.compute,
            np.asarray(coords, dtype=float),
            point_index,
            label=label,
            initial_num_bands=initial_num_bands,
            fermi_energy=request.fermi_energy,
            valence_count=request.valence_count,
            conduction_count=request.conduction_count,
        )
        sectors: dict[str, dict[str, Any]] = {}
        for sector, indices in selections.items():
            indices = np.asarray(indices, dtype=int)
            if len(indices) == 0:
                continue
            sectors[sector] = {
                "energies": eig[indices],
                "vectors": vec[:, indices],
                "band_indices": indices,
            }
        point_sources[label] = {
            "source_kind": "computed_config_points",
            "source_index": 0,
            "source_count": 1,
            "coords": tuple(float(x) for x in coords),
            "sectors": sectors,
        }
        if sewing_context is not None:
            point_sources[label]["sewing_context"] = sewing_context
    return point_sources


def _solve_config_point_with_requested_window(
    calculator: BandStructureCalculator,
    compute_config: Any,
    coords: np.ndarray,
    point_index: int,
    *,
    label: str,
    initial_num_bands: int,
    fermi_energy: float,
    valence_count: int,
    conduction_count: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    requested_valence = max(0, int(valence_count))
    requested_conduction = max(0, int(conduction_count))
    requested_total = requested_valence + requested_conduction
    num_bands = max(1, int(initial_num_bands))
    max_num_bands: int | None = None

    while True:
        compute_config.num_bands_cal = num_bands
        eig, vec, _hamk, _samk = calculator.calculate_band_01(coords, point_index)
        if vec is None:
            raise RuntimeError("TAPW point solve did not return eigenvectors; eig_vec_cal must be true.")
        eig = np.asarray(eig, dtype=float)
        vec = np.asarray(vec, dtype=np.complex128)
        selections = select_band_indices(
            eig,
            fermi_energy,
            valence_count=requested_valence,
            conduction_count=requested_conduction,
        )
        have_valence = len(selections["valence"])
        have_conduction = len(selections["conduction"])
        if have_valence >= requested_valence and have_conduction >= requested_conduction:
            return eig, vec, selections

        if max_num_bands is None:
            basis_dim = int(vec.shape[0])
            if bool(getattr(compute_config, "eigsh_cal", True)):
                max_num_bands = max(1, basis_dim - 2)
            else:
                max_num_bands = basis_dim
        if num_bands >= max_num_bands:
            missing = []
            if have_valence < requested_valence:
                missing.append(f"valence {have_valence}/{requested_valence}")
            if have_conduction < requested_conduction:
                missing.append(f"conduction {have_conduction}/{requested_conduction}")
            raise RuntimeError(
                f"Point {label} did not provide enough states around efermi={fermi_energy:.12g} "
                f"after solving {num_bands} bands: {', '.join(missing)}."
            )

        growth = max(requested_total + 8, num_bands)
        next_num_bands = min(max_num_bands, num_bands + growth)
        if next_num_bands <= num_bands:
            raise RuntimeError(f"Could not expand TAPW symm-rep solve window for point {label}.")
        num_bands = next_num_bands


def run_configured_symm_rep(
    config_path: Path,
    *,
    valence_count: int | None = None,
    conduction_count: int | None = None,
    degeneracy_tol: float | None = None,
    overwrite: bool = False,
) -> Path:
    request = resolve_config_request(config_path)
    if valence_count is not None or conduction_count is not None or degeneracy_tol is not None:
        request = replace(
            request,
            valence_count=request.valence_count if valence_count is None else int(valence_count),
            conduction_count=request.conduction_count if conduction_count is None else int(conduction_count),
            degeneracy_tol=request.degeneracy_tol if degeneracy_tol is None else float(degeneracy_tol),
        )
    result = run_symm_rep_from_point_sources(
        point_sources=calculate_config_point_sources(request),
        symmetry_dir=request.symmetry_dir,
        output_dir=request.output_dir,
        fermi_energy=request.fermi_energy,
        degeneracy_tol=request.degeneracy_tol,
        overwrite=overwrite,
        source_label=f"config:{request.config_path}",
    )
    return result.output_dir




def build_parser(prog: str = "tapw symm-rep") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Project saved TAPW raw-H symmetry actions into high-symmetry band subspaces.",
    )
    parser.add_argument("-c", "--config", type=Path, help="Release TAPW config with symmetry.representation settings.")
    parser.add_argument("--band-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--symmetry-dir",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--output-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--valence-count", type=int, default=None, help="Number of valence states to report per point.")
    parser.add_argument(
        "--conduction-count",
        type=int,
        default=None,
        help="Number of conduction states to report per point.",
    )
    parser.add_argument(
        "--degeneracy-tol",
        type=float,
        default=None,
        help="Energy tolerance for automatic degeneracy grouping.",
    )
    parser.add_argument(
        "--fermi-energy",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--hamiltonian-index",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--points",
        nargs="+",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing non-empty output directory.")
    return parser


def main(argv=None, *, prog: str = "tapw symm-rep") -> int:
    args = build_parser(prog=prog).parse_args(argv)
    if args.config is not None:
        output_dir = run_configured_symm_rep(
            args.config,
            valence_count=args.valence_count,
            conduction_count=args.conduction_count,
            degeneracy_tol=args.degeneracy_tol,
            overwrite=args.overwrite,
        )
        print(f"Wrote TAPW symmetry representations to {output_dir}")
        return 0
    if args.band_dir is None or args.symmetry_dir is None or args.output_dir is None:
        raise SystemExit("tapw symm-rep requires either -c/--config or --band-dir, --symmetry-dir, and --output-dir.")
    result = run_symm_rep(
        band_dir=args.band_dir,
        symmetry_dir=args.symmetry_dir,
        output_dir=args.output_dir,
        valence_count=20 if args.valence_count is None else args.valence_count,
        conduction_count=20 if args.conduction_count is None else args.conduction_count,
        degeneracy_tol=2.0e-3 if args.degeneracy_tol is None else args.degeneracy_tol,
        fermi_energy=args.fermi_energy,
        hamiltonian_index=args.hamiltonian_index,
        points=args.points,
        overwrite=args.overwrite,
    )
    print(f"Wrote TAPW symmetry representations to {result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
