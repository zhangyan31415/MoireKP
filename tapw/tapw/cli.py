#!/usr/bin/env python3
import argparse
import logging
import sys
import time
from pathlib import Path
import os
import shutil
from typing import Optional, Tuple

from .config import Config
from .workflows.band import BandStructureCalculator
from .io.structure import OpenMXFile, StructureProcessorSpglib
from .io.kpath import KPathGenerator
from .io.hr import HrSparseHandler
from .workflows.symmetry import SymmetryAnalysisRunner, resolve_requested_symmetrization_operations

VALLEY_LABELS = dict(BandStructureCalculator.VALLEY_MAP)


def _mpi_world_rank_size() -> Tuple[int, int]:
    # Best-effort: works both under mpiexec/srun and in normal (non-MPI) runs.
    try:
        from mpi4py import MPI  # type: ignore

        comm = MPI.COMM_WORLD
        return int(comm.Get_rank()), int(comm.Get_size())
    except Exception:
        # Fallback to common env vars set by MPI launchers.
        rank = int(os.environ.get("PMI_RANK") or os.environ.get("OMPI_COMM_WORLD_RANK") or "0")
        size = int(os.environ.get("PMI_SIZE") or os.environ.get("OMPI_COMM_WORLD_SIZE") or "1")
        return rank, size

def setup_logging(log_file: str = None):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )
    return logging.getLogger(__name__)

def build_calc_parser(prog: str = None, *, fixed_mode: str | None = None):
    """Build the TAPW calculation parser used by both tapw calc and tapw-calc."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description='Twisted Material Band Structure Calculator',
    )
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--twist-index', type=int,
                       help='Twist index (overrides config file)')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory (overrides config file)')
    parser.add_argument('--valleys', type=int, nargs='+',
                       help='List of valleys to calculate (overrides config file)')
    if fixed_mode is None:
        parser.add_argument('--mode', choices=['band', 'chern', 'symmetry'],
                           help='Calculation mode (overrides config file)')
    parser.add_argument('--n_g', type=int,
                       help='Harmonic of G vectors (overrides config file)')
    parser.add_argument('--num_chern', type=int,
                       help='Number of k-points for Chern number calculation (overrides config file)')
    parser.add_argument('--num_k1', type=int,
                       help='Number of fractional-grid points along kappa1 for Chern mode (overrides config file)')
    parser.add_argument('--num_k2', type=int,
                       help='Number of fractional-grid points along kappa2 for Chern mode (overrides config file)')
    parser.add_argument('--num_processes', type=int,
                       help='Number of processes (overrides config file)')
    parser.add_argument('--blas_threads', type=int,
                       help='BLAS/OpenMP threads per worker (overrides config file)')
    parser.add_argument('--parallel_impl', choices=['joblib', 'mp'],
                       help='Parallel implementation for k-point loop (overrides config file)')
    parser.add_argument('--parallel_backend', choices=['loky', 'multiprocessing'],
                       help='Joblib backend when parallel_impl=joblib (overrides config file)')
    parser.add_argument('--vec_store', choices=['memory', 'memmap'],
                       help='Where to store eigenvectors (overrides config file)')
    parser.add_argument('--memmap_dir', type=str,
                       help='Optional directory for memmap outputs (overrides config file)')
    parser.add_argument('--kpoint_chunk_id', type=int,
                       help='0-based chunk id for job-array sharding (overrides config file)')
    parser.add_argument('--kpoint_chunk_count', type=int,
                       help='Total chunk count for job-array sharding (overrides config file)')
    parser.add_argument('--developer-outputs', action='store_true',
                       help='Write developer-only symmetry intermediate matrices under diagnostics/')
    return parser


def parse_args(argv=None):
    """Parse command line arguments"""
    return build_calc_parser().parse_args(argv)


def can_reuse_m_valley_c3_band_outputs(compute_cfg) -> bool:
    # Disabled by default. Real ZnI2 M-valley C3-only outputs show that the saved band
    # files for M1/M2/M3 are not identical, so direct file reuse is scientifically unsafe.
    return False


def _source_hs_input_is_symmetrized(config_or_compute_cfg) -> bool:
    paths = getattr(config_or_compute_cfg, "paths", None)
    if paths is None:
        return False
    h_file = os.path.basename(str(getattr(paths, "H_file", "") or ""))
    s_file = os.path.basename(str(getattr(paths, "S_file", "") or ""))
    return h_file in {"H_symm.npz", "H_symm.dat"} or s_file in {"S_symm.npz", "S_symm.dat"}


def resolve_qshell_dir_name(config_or_compute_cfg, calculator) -> str:
    compute_cfg = getattr(config_or_compute_cfg, "compute", config_or_compute_cfg)
    if not bool(getattr(compute_cfg, "TAPW", True)):
        return "direct"
    qshell_name = f"Q_shell_{compute_cfg.n_g}"
    uses_symm = bool(getattr(calculator, "uses_hamiltonian_symmetrization", False)) or _source_hs_input_is_symmetrized(config_or_compute_cfg)
    if uses_symm:
        return qshell_name + "_symm"
    return qshell_name


def calculation_targets(compute_cfg) -> list:
    if not bool(getattr(compute_cfg, "TAPW", True)):
        return [None]
    return list(getattr(compute_cfg, "valleys", []))


def set_compute_valley(compute_cfg, valley: int) -> None:
    if hasattr(compute_cfg, "set_valley"):
        compute_cfg.set_valley(valley)
        return
    compute_cfg.valley = valley
    compute_cfg.valley_flag = _valley_label_for_target(valley)


def hamiltonian_symmetrization_requested(compute_cfg) -> bool:
    return getattr(compute_cfg, "symmetrize_hamiltonian", False) is not False


def _valley_label_for_target(target) -> str:
    return VALLEY_LABELS.get(target, str(target))


def _run_symmetry_analysis(config, processor, hr, sr, logger):
    payload = SymmetryAnalysisRunner(
        config=config,
        structure=processor,
        hr_supercell=hr,
        sr_supercell=sr,
        logger=logger,
    ).run()
    logger.info("Completed symmetry analysis")
    return payload


def _resolve_hamiltonian_symmetrization_from_summary(config, payload) -> None:
    if not hamiltonian_symmetrization_requested(config.compute):
        return
    if not isinstance(payload, dict):
        raise ValueError("Hamiltonian symmetrization requires a symmetry-analysis payload.")
    summary = payload.get("summary", {})
    request = getattr(config.compute, "symmetrize_hamiltonian", False)
    resolved = {}
    for target in calculation_targets(config.compute):
        if target is None:
            continue
        valley_label = _valley_label_for_target(target)
        resolved[valley_label] = resolve_requested_symmetrization_operations(summary, valley_label, request)
    config.compute.resolved_hamiltonian_symmetry_operations_by_valley = resolved


def shutdown_parallel_runtime(*, wait: bool = True, kill_workers: bool = False) -> None:
    try:
        import joblib.externals.loky.reusable_executor as reusable_executor
    except Exception:
        return

    executor = getattr(reusable_executor, "_executor", None)
    if executor is None:
        return

    try:
        terminate = getattr(executor, "terminate", None)
        if callable(terminate):
            terminate(kill_workers=kill_workers)
        else:
            executor.shutdown(wait=wait, kill_workers=kill_workers)
    finally:
        if getattr(reusable_executor, "_executor", None) is executor:
            reusable_executor._executor = None
            reusable_executor._executor_kwargs = None


def _coerce_exit_code(code: Optional[int]) -> int:
    return 0 if code is None else int(code)


def exit_cli(code: Optional[int]) -> None:
    shutdown_parallel_runtime()
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    raise SystemExit(_coerce_exit_code(code))


def finish_calculation_process(code: Optional[int]) -> None:
    shutdown_parallel_runtime(wait=False, kill_workers=True)
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_coerce_exit_code(code))


def copy_reused_m_valley_band_outputs(qshell_path: Path, source_valley_flag: str, target_valley_flag: str) -> None:
    band_dir = Path(qshell_path) / "band"
    band_dir.mkdir(exist_ok=True)
    source_files = sorted(band_dir.glob(f"*_{source_valley_flag}_valley*.txt"))
    if not source_files:
        raise FileNotFoundError(
            f"No reusable band files found in {band_dir} for source valley flag {source_valley_flag}"
        )
    for source in source_files:
        target_name = source.name.replace(
            f"_{source_valley_flag}_valley",
            f"_{target_valley_flag}_valley",
        )
        shutil.copy2(source, band_dir / target_name)

def run_calc(args):
    """Main program"""
    config = Config.from_yaml(args.config)

    mpi_rank, mpi_size = _mpi_world_rank_size()
    
    # Override config with command line arguments
    if args.twist_index is not None:
        config.twist.twist_index_m = args.twist_index
    if args.output_dir:
        config.paths.output_dir = args.output_dir
    if args.valleys:
        config.compute.valleys = args.valleys
        set_compute_valley(config.compute, args.valleys[0])
    if args.mode:
        config.compute.mode = args.mode
    if args.n_g is not None:
        config.compute.n_g = args.n_g
    if args.num_chern is not None:
        config.compute.num_chern = args.num_chern
    if args.num_k1 is not None:
        config.compute.num_k1 = args.num_k1
    if args.num_k2 is not None:
        config.compute.num_k2 = args.num_k2
    if args.num_processes is not None:
        config.compute.num_processes = args.num_processes
    if args.blas_threads is not None:
        config.compute.blas_threads = args.blas_threads
    if args.parallel_impl:
        config.compute.parallel_impl = args.parallel_impl
    if args.parallel_backend:
        config.compute.parallel_backend = args.parallel_backend
    if args.vec_store:
        config.compute.vec_store = args.vec_store
    if args.memmap_dir:
        config.compute.memmap_dir = args.memmap_dir
    if args.kpoint_chunk_id is not None:
        config.compute.kpoint_chunk_id = args.kpoint_chunk_id
    if args.kpoint_chunk_count is not None:
        config.compute.kpoint_chunk_count = args.kpoint_chunk_count
    if args.developer_outputs:
        config.symmetry_analysis.developer_outputs = True

    # Re-check constraints after applying CLI overrides.
    config.validate()

    # Setup logging
    os.makedirs(config.paths.output_dir + "/logs", exist_ok=True)
    log_suffix = f"_rank{mpi_rank}" if mpi_size > 1 else ""
    log_file = Path(config.paths.output_dir + "/logs") / f"run_{time.strftime('%Y%m%d_%H%M%S')}{log_suffix}.log"
    logger = setup_logging(str(log_file))
    logger.info("Starting calculation with configuration:")
    logger.info(f"Twist index: {config.twist.twist_index_m}")
    logger.info(f"Output directory: {config.paths.output_dir}")
    logger.info(f"Calculation mode: {config.compute.mode}")
    if config.compute.TAPW:
        logger.info(f"Valleys to calculate: {config.compute.valleys}")
        logger.info(f"Harmonic of G vectors: {config.compute.n_g}")
    else:
        logger.info("Basis: non-TAPW full-space generalized eigenproblem")
    if config.compute.mode == "chern":
        num_k1, num_k2 = config.compute.get_chern_grid_shape()
        logger.info(f"Number of k-points for Chern number calculation: {num_k1}x{num_k2}")
        logger.info(f"Number of processes: {config.compute.num_processes}")
    try:
        # Initialize structure
        structure = OpenMXFile(
            file_path=str(Path(config.paths.input_file)),
            twist_index=config.twist.twist_index_m,
            spin=config.twist.spin
        )
        structure.display_properties()
        
        # Process structure (spglib-based atom typing via basis_id + sublayer)
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
        )
        if config.compute.TAPW:
            processor.process()

            # # Plot clustering results
            os.makedirs(config.paths.output_dir + "/lattice", exist_ok=True)
            processor.plot_clusters_loc(save=True, save_path=config.paths.output_dir + "/lattice")
            processor.plot_clusters_phase(save=True, save_path=config.paths.output_dir + "/lattice")

        # Handle Hamiltonian
        H_file = config.paths.H_file
        if H_file.endswith('.npz'):
            H_handler = HrSparseHandler(
                file_name='',
                npz_file_name=H_file,
                A=processor.transformed_index_matrix if config.compute.TAPW else None,
                read_from_npz=True
            )
        elif H_file.endswith('.dat'):
            H_handler = HrSparseHandler(
                file_name=H_file,
                npz_file_name='',
                A=processor.transformed_index_matrix if config.compute.TAPW else None,
                read_from_npz=False
            )
        else:
            raise ValueError(f"H_file后缀必须为.npz或.dat，当前为: {H_file}")
        hr = H_handler.get_hr_sparse()

        # Symmetry analysis is raw-H-only; avoid reading S.dat on this path.
        if config.compute.mode == "symmetry":
            sr = None
        elif config.compute.orthogonal_basis:
            sr = None
        else:
            S_file = config.paths.S_file
            if S_file is not None:
                if S_file.endswith('.npz'):
                    S_handler = HrSparseHandler(
                        file_name='',
                        npz_file_name=S_file,
                        A=processor.transformed_index_matrix if config.compute.TAPW else None,
                        read_from_npz=True
                    )
                    sr = S_handler.get_hr_sparse()
                elif S_file.endswith('.dat'):
                    S_handler = HrSparseHandler(
                        file_name=S_file,
                        npz_file_name='',
                        A=processor.transformed_index_matrix if config.compute.TAPW else None,
                        read_from_npz=False
                    )
                    sr = S_handler.get_hr_sparse()
                else:
                    raise ValueError(f"S_file后缀必须为.npz或.dat，当前为: {S_file}")
            else:
                raise ValueError("S_file is None, please check the config.yaml")

        # Initialize k-path if needed
        kpath_config = None
        if config.compute.mode == "band":
            # Avoid multiple MPI ranks clobbering the same KPATH.out.
            if mpi_size > 1 and mpi_rank != 0:
                config.paths.kpath_out = str(config.paths.kpath_out) + f".rank{mpi_rank}"
            kpath_config = KPathGenerator(structure.Tmat)
            kpath_config.read_and_generate_kpath(config.paths.kpath_in, config.paths.kpath_out)

        if config.compute.mode == "symmetry":
            _run_symmetry_analysis(config, processor, hr, sr, logger)
            shutdown_parallel_runtime(wait=False, kill_workers=True)
            logging.shutdown()
            return 0

        if hamiltonian_symmetrization_requested(config.compute):
            payload = _run_symmetry_analysis(config, processor, hr, sr, logger)
            _resolve_hamiltonian_symmetrization_from_summary(config, payload)

        reuse_m_valley_band_outputs = can_reuse_m_valley_c3_band_outputs(config.compute)
        reused_reference_valley_flag = None
        reusable_m_valley_calculator = None

        # TAPW projects around configured valleys. non-TAPW solves the full-space
        # generalized problem once; valley and n_g are not part of that basis.
        for target in calculation_targets(config.compute):
            if target is None:
                logger.info("Starting non-TAPW direct calculation")
            else:
                logger.info(f"Starting calculation for valley {target}")
                set_compute_valley(config.compute, target)
            
            if reusable_m_valley_calculator is not None:
                reusable_m_valley_calculator.switch_m_valley(target)
                calculator = reusable_m_valley_calculator
                logger.info(
                    "Reusing initialized M-valley symmetry state for valley %s",
                    target,
                )
            else:
                calculator = BandStructureCalculator(
                    hr_supercell=hr,
                    sr_supercell=sr,
                    structure=processor,
                    config=config.compute,
                    kpath_config=kpath_config
                )
                if getattr(calculator, "use_M_valley_threefold_symm", False):
                    reusable_m_valley_calculator = calculator
            
            out_path = Path(config.paths.output_dir) / resolve_qshell_dir_name(config, calculator)
            out_path.mkdir(exist_ok=True)
            
            if reuse_m_valley_band_outputs and reused_reference_valley_flag is not None:
                calculator.save_band_static_metadata(str(out_path))
                copy_reused_m_valley_band_outputs(
                    out_path,
                    source_valley_flag=reused_reference_valley_flag,
                    target_valley_flag=calculator.valley_flag,
                )
                logger.info(
                    "Reused M-valley C3-only band outputs for valley %s from reference valley flag %s",
                    target,
                    reused_reference_valley_flag,
                )
            else:
                calculator.run_calculation(str(out_path))
                if target is None:
                    logger.info("Completed non-TAPW direct calculation")
                else:
                    logger.info(f"Completed calculation for valley {target}")
                if reuse_m_valley_band_outputs and getattr(calculator, "use_M_valley_threefold_symm", False):
                    reused_reference_valley_flag = calculator.valley_flag

        if getattr(config.symmetry_analysis, "enable", False) and not hamiltonian_symmetrization_requested(config.compute):
            _run_symmetry_analysis(config, processor, hr, sr, logger)

    except Exception as e:
        logger.error(f"Error during calculation: {str(e)}", exc_info=True)
        shutdown_parallel_runtime(wait=False, kill_workers=True)
        logging.shutdown()
        sys.stdout.flush()
        sys.stderr.flush()
        return 1

    logger.info("Calculation completed successfully")
    shutdown_parallel_runtime(wait=False, kill_workers=True)
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    return 0


def main_calc(argv=None, *, prog="tapw run", finalize: bool = True):
    """Run the TAPW calculator entry point."""
    code = _coerce_exit_code(run_calc(build_calc_parser(prog=prog).parse_args(argv)))
    if finalize:
        finish_calculation_process(code)
    return code


def main_chern(argv=None, *, prog="tapw chern", finalize: bool = True):
    """Run TAPW Chern calculation with mode fixed to `chern`."""
    args = build_calc_parser(prog=prog, fixed_mode="chern").parse_args(argv)
    args.mode = "chern"
    code = _coerce_exit_code(run_calc(args))
    if finalize:
        finish_calculation_process(code)
    return code


def build_main_parser():
    """Build the unified TAPW command dispatcher parser."""
    parser = argparse.ArgumentParser(
        prog="tapw",
        description="Unified TAPW command line interface",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="command")
    subparsers.add_parser("init", help="Generate TAPW configuration files")
    subparsers.add_parser("run", help="Run TAPW band, Chern, or symmetry calculations")
    subparsers.add_parser("chern", help="Run TAPW Chern calculation")
    subparsers.add_parser("plot", help="Plot TAPW band structures")
    subparsers.add_parser("topo", help="Post-process TAPW Chern/Wilson-loop outputs")
    subparsers.add_parser("postprocess-memmap", help="Finalize chunked TAPW memmap outputs")
    subparsers.add_parser("orbital", help="Analyze TAPW orbital weights")
    subparsers.add_parser("fatband", help="Plot orbital-weighted bands")
    return parser


def _main_orbital(argv):
    if argv and argv[0] == "analyze":
        from . import orbital_analysis_tool

        return orbital_analysis_tool.main(argv[1:], prog="tapw orbital")
    if argv and argv[0] == "plot":
        from . import plot_orbital_tool

        return plot_orbital_tool.main(argv[1:], prog="tapw fatband")
    from . import orbital_analysis_tool

    return orbital_analysis_tool.main(argv, prog="tapw orbital")


def main(argv=None):
    """Unified TAPW command dispatcher."""
    argv = list(sys.argv[1:] if argv is None else argv)
    program = Path(sys.argv[0]).name

    if program == "tapw-calc":
        return main_calc(argv, prog="tapw-calc")
    if program == "tapw-config":
        from . import config_generator

        return config_generator.main(argv)
    if program == "tapw-plot":
        from . import plot_band_01

        return plot_band_01.main(argv)
    if program == "tapw-chernpost":
        from . import chern_post

        return chern_post.main(argv)
    if program == "tapw-orbital":
        from . import orbital_analysis_tool

        return orbital_analysis_tool.main(argv)
    if program == "tapw-plot-orbital":
        from . import plot_orbital_tool

        return plot_orbital_tool.main(argv)

    if not argv:
        parser = build_main_parser()
        parser.print_help()
        return 0
    if argv[0] in ("-h", "--help"):
        build_main_parser().parse_args(argv)

    command, rest = argv[0], argv[1:]
    if command in {"init", "config"}:
        from . import config_generator

        return config_generator.main(rest, prog="tapw init")
    if command in {"run", "calc"}:
        return main_calc(rest, prog="tapw run")
    if command == "chern":
        return main_chern(rest, prog="tapw chern")
    if command == "plot":
        from . import plot_band_01

        return plot_band_01.main(rest, prog="tapw plot")
    if command in {"topo", "chern-post"}:
        from . import chern_post

        return chern_post.main(rest, prog="tapw topo")
    if command == "postprocess-memmap":
        from . import postprocess_memmap

        return postprocess_memmap.main(rest, prog="tapw postprocess-memmap")
    if command == "orbital":
        return _main_orbital(rest)
    if command == "fatband":
        from . import plot_orbital_tool

        return plot_orbital_tool.main(rest, prog="tapw fatband")

    build_main_parser().parse_args(argv)
    return None

if __name__ == "__main__":
    main()
