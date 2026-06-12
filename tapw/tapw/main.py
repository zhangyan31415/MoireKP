#!/usr/bin/env python3
import argparse
import logging
import sys
import time
from pathlib import Path
import os
import shutil
from typing import Tuple

from .config import Config
from .cal_ham_01 import BandStructureCalculator
from .read_pos_01 import OpenMXFile, StructureProcessorSpglib
from .read_kpath_01 import KPathGenerator
from .read_hr_01 import HrSparseHandler
from .tapw_slab import TAPWSlab
from .symmetry_analysis import SymmetryAnalysisRunner

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

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Twisted Material Band Structure Calculator')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--twist-index', type=int,
                       help='Twist index (overrides config file)')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory (overrides config file)')
    parser.add_argument('--valleys', type=int, nargs='+',
                       help='List of valleys to calculate (overrides config file)')
    parser.add_argument('--mode', choices=['band', 'chern', 'slab', 'symmetry'],
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
    return parser.parse_args()


def can_reuse_m_valley_c3_band_outputs(compute_cfg) -> bool:
    # Disabled by default. Real ZnI2 M-valley C3-only outputs show that the saved band
    # files for M1/M2/M3 are not identical, so direct file reuse is scientifically unsafe.
    return False


def resolve_qshell_dir_name(compute_cfg, calculator) -> str:
    qshell_name = f"Q_shell_{compute_cfg.n_g}"
    uses_symm = bool(
        getattr(calculator, "use_M_valley_threefold_symm", False)
        or getattr(calculator, "use_C3_H", False)
    )
    if uses_symm:
        return qshell_name + "_symm"
    return qshell_name


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

def main():
    """Main program"""
    # Parse arguments and load configuration
    args = parse_args()
    config = Config.from_yaml(args.config)

    mpi_rank, mpi_size = _mpi_world_rank_size()
    
    # Override config with command line arguments
    if args.twist_index is not None:
        config.twist.twist_index_m = args.twist_index
    if args.output_dir:
        config.paths.output_dir = args.output_dir
    if args.valleys:
        config.compute.valleys = args.valleys
        config.compute.valley = args.valleys[0]
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

    # Re-check constraints after applying CLI overrides.
    config.compute.validate()

    # Temporary compatibility mode for non-TAPW outputs:
    # keep legacy K1 / Q_shell_20 naming so existing scripts continue to work.
    if not config.compute.TAPW:
        original_valleys = list(config.compute.valleys)
        original_ng = int(config.compute.n_g)
        config.compute.valleys = [1]
        config.compute.valley = 1
        config.compute.n_g = 20
        if original_valleys != [1] or original_ng != 20:
            print(
                "[notapw] compatibility naming enabled: "
                f"forcing valleys={config.compute.valleys} and n_g={config.compute.n_g} "
                f"(from valleys={original_valleys}, n_g={original_ng})"
            )

    # Setup logging
    os.makedirs(config.paths.output_dir + "/logs", exist_ok=True)
    log_suffix = f"_rank{mpi_rank}" if mpi_size > 1 else ""
    log_file = Path(config.paths.output_dir + "/logs") / f"run_{time.strftime('%Y%m%d_%H%M%S')}{log_suffix}.log"
    logger = setup_logging(str(log_file))
    logger.info("Starting calculation with configuration:")
    logger.info(f"Twist index: {config.twist.twist_index_m}")
    logger.info(f"Output directory: {config.paths.output_dir}")
    logger.info(f"Valleys to calculate: {config.compute.valleys}")
    logger.info(f"Calculation mode: {config.compute.mode}")
    logger.info(f"Harmonic of G vectors: {config.compute.n_g}")
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
            logger.error(f"H_file后缀必须为.npz或.dat，当前为: {H_file}")
            sys.exit(1)
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
                    logger.error(f"S_file后缀必须为.npz或.dat，当前为: {S_file}")
                    sys.exit(1)
            else:
                logger.info("S_file is None, please check the config.yaml")
                sys.exit(1)

        # Initialize k-path if needed
        kpath_config = None
        if config.compute.mode == "band":
            # Avoid multiple MPI ranks clobbering the same KPATH.out.
            if mpi_size > 1 and mpi_rank != 0:
                config.paths.kpath_out = str(config.paths.kpath_out) + f".rank{mpi_rank}"
            kpath_config = KPathGenerator(structure.Tmat)
            kpath_config.read_and_generate_kpath(config.paths.kpath_in, config.paths.kpath_out)
        elif config.compute.mode == "slab":
            # For slab mode, k-path is handled internally by TAPWSlab
            pass

        if config.compute.mode == "symmetry":
            SymmetryAnalysisRunner(
                config=config,
                structure=processor,
                hr_supercell=hr,
                sr_supercell=sr,
                logger=logger,
            ).run()
            logger.info("Completed symmetry analysis")
            return

        reuse_m_valley_band_outputs = can_reuse_m_valley_c3_band_outputs(config.compute)
        reused_reference_valley_flag = None
        reusable_m_valley_calculator = None

        # Calculate for each valley
        for valley in config.compute.valleys:
            logger.info(f"Starting calculation for valley {valley}")
            config.compute.valley = valley
            
            # Choose calculator based on mode
            if config.compute.mode == "slab":
                
                calculator = TAPWSlab(
                    hr_supercell=hr,
                    sr_supercell=sr,
                    structure=processor,
                    config=config,  # Pass full config for slab parameters
                    kpath_config=kpath_config
                )
            else:
                if reusable_m_valley_calculator is not None:
                    reusable_m_valley_calculator.switch_m_valley(valley)
                    calculator = reusable_m_valley_calculator
                    logger.info(
                        "Reusing initialized M-valley symmetry state for valley %s",
                        valley,
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
            
            out_path = Path(config.paths.output_dir) / resolve_qshell_dir_name(config.compute, calculator)
            out_path.mkdir(exist_ok=True)
            
            if config.compute.mode == "slab":
                calculator.calculate_ribbon_bands(str(out_path))
                logger.info(f"Completed slab calculation for valley {valley}")
            else:
                if reuse_m_valley_band_outputs and reused_reference_valley_flag is not None:
                    calculator.save_band_static_metadata(str(out_path))
                    copy_reused_m_valley_band_outputs(
                        out_path,
                        source_valley_flag=reused_reference_valley_flag,
                        target_valley_flag=calculator.valley_flag,
                    )
                    logger.info(
                        "Reused M-valley C3-only band outputs for valley %s from reference valley flag %s",
                        valley,
                        reused_reference_valley_flag,
                    )
                else:
                    calculator.run_calculation(str(out_path))
                    logger.info(f"Completed calculation for valley {valley}")
                    if reuse_m_valley_band_outputs and getattr(calculator, "use_M_valley_threefold_symm", False):
                        reused_reference_valley_flag = calculator.valley_flag

        if getattr(config.symmetry_analysis, "enable", False):
            SymmetryAnalysisRunner(
                config=config,
                structure=processor,
                hr_supercell=hr,
                sr_supercell=sr,
                logger=logger,
            ).run()
            logger.info("Completed symmetry analysis")

    except Exception as e:
        logger.error(f"Error during calculation: {str(e)}", exc_info=True)
        sys.exit(1)

    logger.info("Calculation completed successfully")

if __name__ == "__main__":
    main() 
