"""CLI entry point for TAPW symmetry-representation post-processing."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass, replace
import logging
from pathlib import Path
import time
from typing import Any

from joblib import Parallel, delayed
import numpy as np

from .artifacts import canonical_profile_name, canonical_qshell_name
from .config import Config
from .io.hr import HrSparseHandler
from .io.structure import OpenMXFile, StructureProcessorSpglib
from .reporting import TapwReporter
from .workflows.band import BandStructureCalculator, _maybe_pin_current_worker, _set_thread_limits
from .workflows.symm_rep import _prepare_output_dir, run_symm_rep, run_symm_rep_from_point_sources, select_band_indices

ProgressCallback = Callable[[str], None]
TimingBreakdown = dict[str, Any]


@dataclass(frozen=True)
class ConfigSymmRepRequest:
    config_path: Path
    symmetry_dir: Path
    output_dir: Path
    fermi_energy: float
    num_bands: int
    valence_count: int
    conduction_count: int
    degeneracy_tol: float
    points: dict[str, tuple[float, float, float]]
    cache_dir: Path | None = None
    cache_enabled: bool = True


_HIGH_SYMMETRY_CACHE_SCHEMA = "tapw.symm_rep.high_symmetry_wavefunctions.v2"
_HIGH_SYMMETRY_CACHE_FILE = "high_symmetry_wavefunctions.npz"


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
    cache_dir = Path(representation.get("cache_dir", "symm_rep_cache"))
    if not cache_dir.is_absolute():
        cache_dir = run_root / cache_dir

    return ConfigSymmRepRequest(
        config_path=config_path,
        symmetry_dir=symmetry_dir,
        output_dir=output_dir,
        fermi_energy=float(symmetry["efermi"]),
        num_bands=int(representation.get("num_bands", 50)),
        valence_count=int(representation.get("valence_count", 20)),
        conduction_count=int(representation.get("conduction_count", 20)),
        degeneracy_tol=float(representation.get("degeneracy_tol", 2.0e-3)),
        points=_parse_representation_points(representation.get("points")),
        cache_dir=cache_dir,
        cache_enabled=bool(representation.get("cache", True)),
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


def _integer_lattice_residual(vectors: np.ndarray, basis: np.ndarray) -> float:
    vectors = np.asarray(vectors, dtype=float)
    basis = np.asarray(basis, dtype=float).reshape(2, 2)
    if vectors.size == 0:
        return 0.0
    try:
        coeffs = np.linalg.solve(basis.T, vectors[:, :2].T).T
    except np.linalg.LinAlgError:
        return float("inf")
    rounded = np.rint(coeffs)
    return float(np.linalg.norm(vectors[:, :2] - rounded @ basis, axis=1).max())


def _candidate_reciprocal_vectors(groups: list[np.ndarray], *, max_norm: float) -> np.ndarray:
    vectors = [np.asarray(group, dtype=float)[:, :2] for group in groups if len(group) > 0]
    if not vectors:
        return np.empty((0, 2), dtype=float)
    points = np.vstack(vectors)
    candidates: list[np.ndarray] = []
    for vector in points:
        if 1.0e-12 < float(np.linalg.norm(vector)) <= max_norm:
            candidates.append(np.asarray(vector, dtype=float))
    for source in points:
        deltas = points - source
        norms = np.linalg.norm(deltas, axis=1)
        for delta in deltas[(norms > 1.0e-12) & (norms <= max_norm)]:
            candidates.append(np.asarray(delta, dtype=float))
    if not candidates:
        return np.empty((0, 2), dtype=float)
    return np.unique(np.round(np.vstack(candidates), decimals=12), axis=0)


def _snap_reciprocal_basis_to_g_vectors(
    reference_basis: np.ndarray,
    groups: list[np.ndarray],
    *,
    relative_tol: float = 5.0e-3,
    residual_tol: float = 1.0e-7,
) -> np.ndarray:
    """Snap the reciprocal basis rows to the actual finite-Q grid while preserving row directions."""
    reference = np.asarray(reference_basis, dtype=float).reshape(2, 2)
    group_vectors = [np.asarray(group, dtype=float)[:, :2] for group in groups if len(group) > 0]
    if not group_vectors:
        return reference
    all_vectors = np.vstack(group_vectors)
    row_norms = np.linalg.norm(reference, axis=1)
    max_norm = max(float(np.max(row_norms)) * 2.5, 1.0e-12)
    candidates = _candidate_reciprocal_vectors(group_vectors, max_norm=max_norm)
    if len(candidates) == 0:
        return reference

    snapped_rows: list[np.ndarray] = []
    for row, row_norm in zip(reference, row_norms):
        if float(row_norm) <= 1.0e-12:
            return reference
        distances = np.linalg.norm(candidates - row, axis=1)
        index = int(np.argmin(distances))
        if float(distances[index]) > max(residual_tol, relative_tol * float(row_norm)):
            return reference
        snapped_rows.append(np.asarray(candidates[index], dtype=float))

    snapped = np.vstack(snapped_rows)
    if abs(float(np.linalg.det(snapped))) <= 1.0e-14:
        return reference

    reference_residual = _integer_lattice_residual(all_vectors, reference)
    snapped_residual = _integer_lattice_residual(all_vectors, snapped)
    if snapped_residual <= residual_tol and snapped_residual < max(reference_residual, residual_tol):
        return snapped
    return reference


def _request_cache_path(request: ConfigSymmRepRequest) -> Path | None:
    if not request.cache_enabled:
        return None
    cache_dir = request.cache_dir if request.cache_dir is not None else request.output_dir.parent / "symm_rep_cache"
    return Path(cache_dir) / _HIGH_SYMMETRY_CACHE_FILE


def _request_point_labels(request: ConfigSymmRepRequest) -> list[str]:
    return [str(label) for label in request.points]


def _request_point_coords(request: ConfigSymmRepRequest) -> np.ndarray:
    return np.asarray([request.points[label] for label in _request_point_labels(request)], dtype=float)


def _select_sectors_for_point(
    *,
    label: str,
    coords: tuple[float, float, float],
    eig: np.ndarray,
    vec: np.ndarray,
    request: ConfigSymmRepRequest,
    sewing_context: dict[str, Any] | None,
    source_kind: str,
) -> dict[str, Any]:
    eig = np.asarray(eig, dtype=float)
    vec = np.asarray(vec, dtype=np.complex128)
    selections = select_band_indices(
        eig,
        request.fermi_energy,
        valence_count=request.valence_count,
        conduction_count=request.conduction_count,
    )
    have_valence = len(selections["valence"])
    have_conduction = len(selections["conduction"])
    if have_valence < request.valence_count or have_conduction < request.conduction_count:
        missing = []
        if have_valence < request.valence_count:
            missing.append(f"valence {have_valence}/{request.valence_count}")
        if have_conduction < request.conduction_count:
            missing.append(f"conduction {have_conduction}/{request.conduction_count}")
        raise RuntimeError(
            f"Point {label} did not provide enough states around efermi={request.fermi_energy:.12g} "
            f"after solving symmetry.representation.num_bands={request.num_bands}: {', '.join(missing)}. "
            f"Increase symmetry.representation.num_bands to a value larger than {request.num_bands}."
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
    point_source: dict[str, Any] = {
        "source_kind": source_kind,
        "source_index": 0,
        "source_count": 1,
        "coords": tuple(float(x) for x in coords),
        "sectors": sectors,
    }
    if sewing_context is not None:
        point_source["sewing_context"] = sewing_context
    return point_source


def _sewing_context_from_calculator(calculator: BandStructureCalculator, config: Config) -> dict[str, Any] | None:
    tapw_parameters = getattr(calculator, "TAPW_parameters", None)
    if tapw_parameters is None:
        return None
    g1 = np.asarray(getattr(tapw_parameters, "g_vec_list_K1", np.zeros((0, 2))), dtype=float)
    g2 = np.asarray(getattr(tapw_parameters, "g_vec_list_K2", np.zeros((0, 2))), dtype=float)
    reciprocal = np.vstack(
        [
            np.asarray(getattr(calculator.structure, "reciprocal_Tmat", np.eye(3))[0][:2], dtype=float),
            np.asarray(getattr(calculator.structure, "reciprocal_Tmat", np.eye(3))[1][:2], dtype=float),
        ]
    )
    reciprocal = _snap_reciprocal_basis_to_g_vectors(reciprocal, [g1, g2])
    return {
        "g_vectors_by_group": [g1, g2],
        "reciprocal_basis": reciprocal,
        "spin_blocks": 2 if bool(getattr(config.twist, "spin", True)) else 1,
    }


def _cache_sewing_context(payload: Any) -> dict[str, Any] | None:
    files = set(getattr(payload, "files", []))
    if "sewing_group_count" not in files:
        return None
    group_count = int(np.asarray(payload["sewing_group_count"]).item())
    groups = []
    for index in range(group_count):
        key = f"sewing_g_vectors_group{index}"
        if key not in files:
            return None
        groups.append(np.asarray(payload[key], dtype=float))
    if "sewing_reciprocal_basis" not in files or "sewing_spin_blocks" not in files:
        return None
    return {
        "g_vectors_by_group": groups,
        "reciprocal_basis": np.asarray(payload["sewing_reciprocal_basis"], dtype=float),
        "spin_blocks": int(np.asarray(payload["sewing_spin_blocks"]).item()),
    }


def _load_high_symmetry_point_cache(
    request: ConfigSymmRepRequest,
    *,
    progress: ProgressCallback | None = None,
    timings: TimingBreakdown | None = None,
) -> dict[str, dict[str, Any]] | None:
    cache_path = _request_cache_path(request)
    if cache_path is None or not cache_path.is_file():
        return None
    started = time.perf_counter()
    try:
        with np.load(cache_path, allow_pickle=False) as payload:
            schema = str(np.asarray(payload["schema"]).item())
            if schema != _HIGH_SYMMETRY_CACHE_SCHEMA:
                return None
            cached_fermi = float(np.asarray(payload["fermi_energy"]).item())
            if not np.isclose(cached_fermi, request.fermi_energy, rtol=0.0, atol=1.0e-10):
                return None
            cached_num_bands = int(np.asarray(payload["num_bands"]).item())
            if cached_num_bands < int(request.num_bands):
                return None
            labels = [str(item) for item in np.asarray(payload["points"]).tolist()]
            if labels != _request_point_labels(request):
                return None
            if not np.allclose(np.asarray(payload["point_coords"], dtype=float), _request_point_coords(request), atol=1.0e-12):
                return None
            sewing_context = _cache_sewing_context(payload)
            point_sources: dict[str, dict[str, Any]] = {}
            for label, coords in request.points.items():
                eig_key = f"{label}_energies"
                vec_key = f"{label}_eigenvectors"
                if eig_key not in payload.files or vec_key not in payload.files:
                    return None
                point_sources[label] = _select_sectors_for_point(
                    label=label,
                    coords=coords,
                    eig=np.asarray(payload[eig_key], dtype=float),
                    vec=np.asarray(payload[vec_key], dtype=np.complex128),
                    request=request,
                    sewing_context=sewing_context,
                    source_kind="computed_config_points",
                )
    except (OSError, KeyError, ValueError):
        return None
    finally:
        if timings is not None:
            timings["cache_read"] = timings.get("cache_read", 0.0) + (time.perf_counter() - started)
    if progress is not None:
        progress(f"Using cached high-symmetry wavefunctions: {cache_path}")
    return point_sources


def _write_high_symmetry_point_cache(
    request: ConfigSymmRepRequest,
    solved_points: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    sewing_context: dict[str, Any] | None,
    timings: TimingBreakdown | None = None,
) -> None:
    cache_path = _request_cache_path(request)
    if cache_path is None:
        return
    started = time.perf_counter()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema": np.array(_HIGH_SYMMETRY_CACHE_SCHEMA, dtype=str),
        "fermi_energy": np.array(float(request.fermi_energy), dtype=float),
        "num_bands": np.array(int(request.num_bands), dtype=int),
        "points": np.array(_request_point_labels(request), dtype=str),
        "point_coords": _request_point_coords(request),
    }
    for label, (eig, vec) in solved_points.items():
        payload[f"{label}_energies"] = np.asarray(eig, dtype=float)
        payload[f"{label}_eigenvectors"] = np.asarray(vec, dtype=np.complex128)
    if sewing_context is not None:
        groups = [np.asarray(group, dtype=float) for group in sewing_context.get("g_vectors_by_group", [])]
        payload["sewing_group_count"] = np.array(len(groups), dtype=int)
        for index, group in enumerate(groups):
            payload[f"sewing_g_vectors_group{index}"] = group
        payload["sewing_reciprocal_basis"] = np.asarray(sewing_context["reciprocal_basis"], dtype=float)
        payload["sewing_spin_blocks"] = np.array(int(sewing_context["spin_blocks"]), dtype=int)
    np.savez(cache_path, **payload)
    if timings is not None:
        timings["cache_write"] = timings.get("cache_write", 0.0) + (time.perf_counter() - started)


def _solve_config_point_loky_worker(
    *,
    config_path: str,
    fermi_energy: float,
    num_bands: int,
    label: str,
    coords: tuple[float, float, float],
    point_index: int,
    blas_threads: int,
    worker_count: int,
) -> dict[str, Any]:
    _maybe_pin_current_worker(blas_threads, worker_count)
    _set_thread_limits(blas_threads)
    config = Config.from_yaml(config_path)
    config.compute.mode = "band"
    config.compute.efermi = float(fermi_energy)
    config.compute.eig_vec_cal = True
    config.compute.hamk_save = False
    config.compute.num_bands_cal = max(1, int(num_bands))
    config.compute.num_processes = 1

    setup_started = time.perf_counter()
    calculator = _build_point_calculator(config)
    setup_elapsed = time.perf_counter() - setup_started
    sewing_context = _sewing_context_from_calculator(calculator, config)

    solve_started = time.perf_counter()
    eig, vec, _hamk, _samk = calculator.calculate_band_01(np.asarray(coords, dtype=float), int(point_index))
    solve_elapsed = time.perf_counter() - solve_started
    if vec is None:
        raise RuntimeError(f"TAPW point solve for {label} did not return eigenvectors; eig_vec_cal must be true.")
    return {
        "label": label,
        "coords": tuple(float(x) for x in coords),
        "eig": np.asarray(eig, dtype=float),
        "vec": np.asarray(vec, dtype=np.complex128),
        "sewing_context": sewing_context,
        "setup_seconds": setup_elapsed,
        "solve_seconds": solve_elapsed,
        "bands": int(len(eig)),
    }


def _run_loky_point_workers(
    *,
    request: ConfigSymmRepRequest,
    workers: int,
    blas_threads: int,
) -> list[dict[str, Any]]:
    items = list(request.points.items())
    return Parallel(n_jobs=int(workers), backend="loky")(
        delayed(_solve_config_point_loky_worker)(
            config_path=str(request.config_path),
            fermi_energy=float(request.fermi_energy),
            num_bands=int(request.num_bands),
            label=str(label),
            coords=tuple(float(x) for x in coords),
            point_index=index,
            blas_threads=int(blas_threads),
            worker_count=int(workers),
        )
        for index, (label, coords) in enumerate(items)
    )


def calculate_config_point_sources(
    request: ConfigSymmRepRequest,
    *,
    progress: ProgressCallback | None = None,
    timings: TimingBreakdown | None = None,
) -> dict[str, dict[str, Any]]:
    cached = _load_high_symmetry_point_cache(request, progress=progress, timings=timings)
    if cached is not None:
        return cached

    config = Config.from_yaml(str(request.config_path))
    config.compute.mode = "band"
    config.compute.efermi = request.fermi_energy
    config.compute.eig_vec_cal = True
    config.compute.hamk_save = False
    initial_num_bands = max(1, int(request.num_bands))
    config.compute.num_bands_cal = initial_num_bands
    total_points = len(request.points)
    point_sources: dict[str, dict[str, Any]] = {}
    solved_points: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    requested_workers = int(getattr(config.compute, "num_processes", 1))
    workers = min(requested_workers, total_points)
    parallel_impl = str(getattr(config.compute, "parallel_impl", "joblib"))
    parallel_backend = str(getattr(config.compute, "parallel_backend", "loky"))
    use_parallel_points = total_points > 1 and requested_workers > 1
    use_fresh_loky_workers = use_parallel_points and parallel_impl == "joblib" and parallel_backend == "loky"
    if use_fresh_loky_workers:
        if progress is not None:
            progress(
                "Solving {count} points with fresh joblib/loky workers: workers={workers}, num_bands={bands}.".format(
                    count=total_points,
                    workers=workers,
                    bands=initial_num_bands,
                )
            )
        started = time.perf_counter()
        worker_results = _run_loky_point_workers(
            request=request,
            workers=workers,
            blas_threads=int(getattr(config.compute, "blas_threads", 1)),
        )
        elapsed = time.perf_counter() - started
        setup_sum = sum(float(item.get("setup_seconds", 0.0)) for item in worker_results)
        solve_sum = sum(float(item.get("solve_seconds", 0.0)) for item in worker_results)
        labels = [str(item["label"]) for item in worker_results]
        if timings is not None:
            timings["diagonalize_parallel"] = {
                "seconds": elapsed,
                "points": labels,
                "bands": int(max((int(item.get("bands", 0)) for item in worker_results), default=0)),
                "workers": int(workers),
                "worker_setup_sum": setup_sum,
                "worker_solve_sum": solve_sum,
            }
        if progress is not None:
            progress(f"Finished fresh joblib/loky point solve: {total_points} points in {elapsed:.1f}s.")
        sewing_context = next((item.get("sewing_context") for item in worker_results if item.get("sewing_context") is not None), None)
        for item in worker_results:
            label = str(item["label"])
            coords = tuple(float(x) for x in item["coords"])
            eig = np.asarray(item["eig"], dtype=float)
            vec = np.asarray(item["vec"], dtype=np.complex128)
            solved_points[label] = (eig, vec)
            point_sources[label] = _select_sectors_for_point(
                label=label,
                coords=coords,
                eig=eig,
                vec=vec,
                request=request,
                sewing_context=sewing_context,
                source_kind="computed_config_points",
            )
            if progress is not None:
                sectors = point_sources[label]["sectors"]
                progress(
                    "Prepared point {label}: selected valence={valence}, conduction={conduction}.".format(
                        label=label,
                        valence=len(sectors.get("valence", {}).get("band_indices", [])),
                        conduction=len(sectors.get("conduction", {}).get("band_indices", [])),
                    )
                )
        _write_high_symmetry_point_cache(request, solved_points, sewing_context=sewing_context, timings=timings)
        return point_sources

    if progress is not None:
        progress(f"Building TAPW point solver with initial num_bands={initial_num_bands}.")
    setup_started = time.perf_counter()
    calculator = _build_point_calculator(config)
    setup_elapsed = time.perf_counter() - setup_started
    if timings is not None:
        timings["build_point_solver"] = setup_elapsed
    if progress is not None:
        progress(f"Built TAPW point solver in {setup_elapsed:.1f}s.")
    sewing_context = _sewing_context_from_calculator(calculator, config)
    if use_parallel_points:
        labels = _request_point_labels(request)
        coords_array = _request_point_coords(request)
        config.compute.num_processes = workers
        if progress is not None:
            progress(
                "Solving {count} points with parallel k-point solver: impl={impl}, backend={backend}, workers={workers}.".format(
                    count=total_points,
                    impl=parallel_impl,
                    backend=parallel_backend,
                    workers=workers,
                )
            )
        started = time.perf_counter()
        calculator.parallel_calculate_band_01(
            coords_array,
            kpoint_indices=list(range(total_points)),
            nk_total=total_points,
            out_dir=str(request.cache_dir or request.output_dir),
            suffix="symm_rep",
        )
        elapsed = time.perf_counter() - started
        eig_all = np.asarray(calculator.result.get("eig"), dtype=float)
        vec_result = calculator.result.get("vec")
        if vec_result is None:
            raise RuntimeError("TAPW point solve did not return eigenvectors; eig_vec_cal must be true.")
        vec_all = np.asarray(vec_result, dtype=np.complex128)
        if timings is not None:
            timings["diagonalize_parallel"] = {
                "seconds": elapsed,
                "points": labels,
                "bands": int(eig_all.shape[1]) if eig_all.ndim == 2 else 0,
                "workers": int(workers),
            }
        if progress is not None:
            progress(f"Finished parallel point solve: {total_points} points in {elapsed:.1f}s.")
        for point_index, label in enumerate(labels):
            eig = np.asarray(eig_all[point_index], dtype=float)
            vec = np.asarray(vec_all[point_index], dtype=np.complex128)
            coords = request.points[label]
            solved_points[label] = (eig, vec)
            point_sources[label] = _select_sectors_for_point(
                label=label,
                coords=coords,
                eig=eig,
                vec=vec,
                request=request,
                sewing_context=sewing_context,
                source_kind="computed_config_points",
            )
            if progress is not None:
                sectors = point_sources[label]["sectors"]
                progress(
                    "Prepared point {label}: selected valence={valence}, conduction={conduction}.".format(
                        label=label,
                        valence=len(sectors.get("valence", {}).get("band_indices", [])),
                        conduction=len(sectors.get("conduction", {}).get("band_indices", [])),
                    )
                )
        _write_high_symmetry_point_cache(request, solved_points, sewing_context=sewing_context, timings=timings)
        return point_sources

    for point_index, (label, coords) in enumerate(request.points.items()):
        point_started = time.perf_counter()
        if progress is not None:
            progress(
                "Solving point {index}/{total}: {label} at ({kx:.10g}, {ky:.10g}, {kz:.10g}).".format(
                    index=point_index + 1,
                    total=total_points,
                    label=label,
                    kx=coords[0],
                    ky=coords[1],
                    kz=coords[2],
                )
            )
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
            progress=progress,
            timings=timings,
        )
        solved_points[label] = (eig, vec)
        point_sources[label] = _select_sectors_for_point(
            label=label,
            coords=coords,
            eig=eig,
            vec=vec,
            request=request,
            sewing_context=sewing_context,
            source_kind="computed_config_points",
        )
        if progress is not None:
            point_elapsed = time.perf_counter() - point_started
            progress(
                "Finished point {label}: selected valence={valence}, conduction={conduction}; total point time={elapsed:.1f}s.".format(
                    label=label,
                    valence=len(selections["valence"]),
                    conduction=len(selections["conduction"]),
                    elapsed=point_elapsed,
                )
            )
        if timings is not None:
            timings.setdefault("point_total", []).append(
                {
                    "point": label,
                    "seconds": time.perf_counter() - point_started,
                    "valence": int(len(selections["valence"])),
                    "conduction": int(len(selections["conduction"])),
                }
            )
    _write_high_symmetry_point_cache(request, solved_points, sewing_context=sewing_context, timings=timings)
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
    progress: ProgressCallback | None = None,
    timings: TimingBreakdown | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    requested_valence = max(0, int(valence_count))
    requested_conduction = max(0, int(conduction_count))
    num_bands = max(1, int(initial_num_bands))

    compute_config.num_bands_cal = num_bands
    if progress is not None:
        progress(f"Diagonalizing {label} with num_bands={num_bands}.")
    started = time.perf_counter()
    eig, vec, _hamk, _samk = calculator.calculate_band_01(coords, point_index)
    elapsed = time.perf_counter() - started
    if vec is None:
        raise RuntimeError("TAPW point solve did not return eigenvectors; eig_vec_cal must be true.")
    eig = np.asarray(eig, dtype=float)
    vec = np.asarray(vec, dtype=np.complex128)
    if progress is not None:
        progress(f"Finished {label}: {len(eig)} bands in {elapsed:.1f}s.")
    if timings is not None:
        timings.setdefault("diagonalize_points", []).append(
            {"point": label, "seconds": elapsed, "bands": int(len(eig))}
        )
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

    missing = []
    if have_valence < requested_valence:
        missing.append(f"valence {have_valence}/{requested_valence}")
    if have_conduction < requested_conduction:
        missing.append(f"conduction {have_conduction}/{requested_conduction}")
    raise RuntimeError(
        f"Point {label} did not provide enough states around efermi={fermi_energy:.12g} "
        f"after solving symmetry.representation.num_bands={num_bands}: {', '.join(missing)}. "
        f"Increase symmetry.representation.num_bands to a value larger than {num_bands}."
    )


def run_configured_symm_rep(
    config_path: Path,
    *,
    valence_count: int | None = None,
    conduction_count: int | None = None,
    degeneracy_tol: float | None = None,
    overwrite: bool = True,
    progress: ProgressCallback | None = None,
    timings: TimingBreakdown | None = None,
) -> Path:
    resolve_started = time.perf_counter()
    request = resolve_config_request(config_path)
    if timings is not None:
        timings["resolve_config"] = time.perf_counter() - resolve_started
    if valence_count is not None or conduction_count is not None or degeneracy_tol is not None:
        request = replace(
            request,
            valence_count=request.valence_count if valence_count is None else int(valence_count),
            conduction_count=request.conduction_count if conduction_count is None else int(conduction_count),
            degeneracy_tol=request.degeneracy_tol if degeneracy_tol is None else float(degeneracy_tol),
        )
    prepare_started = time.perf_counter()
    _prepare_output_dir(request.output_dir, overwrite=overwrite)
    if timings is not None:
        timings["prepare_output"] = time.perf_counter() - prepare_started
    if progress is not None:
        progress(f"Resolved config: {request.config_path}")
        progress(f"Using raw-H representations: {request.symmetry_dir / 'representations.npz'}")
        progress(f"Writing symmetry representations to: {request.output_dir}")
        progress(
            f"Band window: num_bands={request.num_bands}, valence_count={request.valence_count}, "
            f"conduction_count={request.conduction_count}, degeneracy_tol={request.degeneracy_tol:g}"
        )
        progress(f"Computing high-symmetry point wavefunctions for: {', '.join(request.points)}")
    points_started = time.perf_counter()
    point_sources = calculate_config_point_sources(request, progress=progress, timings=timings)
    if timings is not None:
        timings["point_source_total"] = time.perf_counter() - points_started
    if progress is not None:
        progress("Projecting raw-H actions into band subspaces.")
    projection_started = time.perf_counter()
    result = run_symm_rep_from_point_sources(
        point_sources=point_sources,
        symmetry_dir=request.symmetry_dir,
        output_dir=request.output_dir,
        fermi_energy=request.fermi_energy,
        degeneracy_tol=request.degeneracy_tol,
        overwrite=overwrite,
        source_label=f"config:{request.config_path}",
    )
    if timings is not None:
        timings["projection_write"] = time.perf_counter() - projection_started
    if progress is not None:
        progress(f"Finished: {result.output_dir}")
    return result.output_dir


_CLI_RULE = "=" * 72
_CLI_SUBRULE = "-" * 72


def _print_section(title: str) -> None:
    print(_CLI_RULE, flush=True)
    print(f"[tapw symm-rep] {title}", flush=True)
    print(_CLI_SUBRULE, flush=True)


def _print_progress(message: str) -> None:
    print(f"  {message}", flush=True)


def _timing_value(timings: TimingBreakdown | None, key: str) -> float:
    if timings is None:
        return 0.0
    try:
        return float(timings.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _print_timing_breakdown(*, elapsed: float, timings: TimingBreakdown | None) -> None:
    min_timing_seconds = 0.05
    print("  time breakdown   :", flush=True)
    if timings:
        setup = (
            _timing_value(timings, "resolve_config")
            + _timing_value(timings, "prepare_output")
            + _timing_value(timings, "build_point_solver")
        )
        setup_outside_point_solve = _timing_value(timings, "resolve_config") + _timing_value(timings, "prepare_output")
        point_solver_setup = _timing_value(timings, "build_point_solver")
        parallel_diag = timings.get("diagonalize_parallel") or {}
        parallel_diag_seconds = float(parallel_diag.get("seconds", 0.0) or 0.0)
        diagonalize = parallel_diag_seconds + sum(
            float(item.get("seconds", 0.0)) for item in timings.get("diagonalize_points", [])
        )
        point_total = _timing_value(timings, "point_source_total")
        projection = _timing_value(timings, "projection_write")
        postprocess = _timing_value(timings, "postprocess")
        cache_read = _timing_value(timings, "cache_read")
        cache_write = _timing_value(timings, "cache_write")
        hidden_time = 0.0
        if setup >= min_timing_seconds:
            print(f"    setup          : {setup:.1f}s", flush=True)
        else:
            hidden_time += setup
        if cache_read >= min_timing_seconds:
            print(f"    cache read     : {cache_read:.1f}s", flush=True)
        else:
            hidden_time += cache_read
        if diagonalize >= min_timing_seconds:
            print(f"    diagonalize    : {diagonalize:.1f}s", flush=True)
            if parallel_diag_seconds >= min_timing_seconds:
                points = ", ".join(str(point) for point in parallel_diag.get("points", []))
                print(
                    "      parallel     {seconds:>8.1f}s  workers={workers}  bands={bands}  points={points}".format(
                        seconds=parallel_diag_seconds,
                        workers=int(parallel_diag.get("workers", 0)),
                        bands=int(parallel_diag.get("bands", 0)),
                        points=points,
                    ),
                    flush=True,
                )
                worker_setup_sum = float(parallel_diag.get("worker_setup_sum", 0.0) or 0.0)
                worker_solve_sum = float(parallel_diag.get("worker_solve_sum", 0.0) or 0.0)
                if worker_setup_sum >= min_timing_seconds:
                    print(f"      worker setup sum: {worker_setup_sum:.1f}s", flush=True)
                if worker_solve_sum >= min_timing_seconds:
                    print(f"      worker solve sum : {worker_solve_sum:.1f}s", flush=True)
            for item in timings.get("diagonalize_points", []):
                seconds = float(item.get("seconds", 0.0))
                if seconds >= min_timing_seconds:
                    print(
                        "      {point:<12} {seconds:>8.1f}s  bands={bands}".format(
                            point=str(item.get("point", "")),
                            seconds=seconds,
                            bands=int(item.get("bands", 0)),
                        ),
                        flush=True,
                    )
        else:
            hidden_time += diagonalize
        point_prep = max(0.0, point_total - point_solver_setup - diagonalize - cache_read - cache_write)
        if point_prep >= min_timing_seconds:
            print(f"    point prep      : {point_prep:.1f}s", flush=True)
        else:
            hidden_time += point_prep
        if projection >= min_timing_seconds:
            print(f"    project/write   : {projection:.1f}s", flush=True)
        else:
            hidden_time += projection
        if cache_write >= min_timing_seconds:
            print(f"    cache write    : {cache_write:.1f}s", flush=True)
        else:
            hidden_time += cache_write
        if postprocess >= min_timing_seconds:
            print(f"    postprocess     : {postprocess:.1f}s", flush=True)
        else:
            hidden_time += postprocess
        accounted = setup_outside_point_solve + point_total + projection + postprocess
        unaccounted = max(0.0, float(elapsed) - accounted)
        hidden_time += unaccounted
        if hidden_time >= min_timing_seconds:
            print(f"    other           : {hidden_time:.1f}s", flush=True)
    print(f"    total           : {elapsed:.1f}s", flush=True)


def _print_output_summary(output_dir: Path, *, elapsed: float, timings: TimingBreakdown | None = None) -> None:
    _print_section("Complete")
    print(f"  output directory : {output_dir}", flush=True)
    print(f"  summary          : {output_dir / 'summary.md'}", flush=True)
    print(f"  band reps        : {output_dir / 'band_representations.npz'}", flush=True)
    print(f"  characters       : {output_dir / 'characters.csv'}", flush=True)
    _print_timing_breakdown(elapsed=elapsed, timings=timings)




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
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        default=True,
        help="Replace an existing non-empty output directory (default for tapw symm-rep).",
    )
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Fail if the output directory already exists and is not empty.",
    )
    return parser


def main(argv=None, *, prog: str = "tapw symm-rep") -> int:
    args = build_parser(prog=prog).parse_args(argv)
    if args.config is not None:
        started = time.perf_counter()
        timings: TimingBreakdown = {}
        _print_section("Start")
        output_dir = run_configured_symm_rep(
            args.config,
            valence_count=args.valence_count,
            conduction_count=args.conduction_count,
            degeneracy_tol=args.degeneracy_tol,
            overwrite=args.overwrite,
            progress=_print_progress,
            timings=timings,
        )
        _print_output_summary(output_dir, elapsed=time.perf_counter() - started, timings=timings)
        return 0
    if args.band_dir is None or args.symmetry_dir is None or args.output_dir is None:
        raise SystemExit("tapw symm-rep requires either -c/--config or --band-dir, --symmetry-dir, and --output-dir.")
    started = time.perf_counter()
    timings = {}
    _print_section("Start")
    _print_progress(f"Reading band data from: {args.band_dir}")
    _print_progress(f"Reading raw-H representations from: {args.symmetry_dir}")
    _print_progress(f"Writing symmetry representations to: {args.output_dir}")
    postprocess_started = time.perf_counter()
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
    timings["postprocess"] = time.perf_counter() - postprocess_started
    _print_output_summary(result.output_dir, elapsed=time.perf_counter() - started, timings=timings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
