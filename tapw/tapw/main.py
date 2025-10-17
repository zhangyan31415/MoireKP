#!/usr/bin/env python3
import argparse
import logging
import sys
import time
from pathlib import Path

from .config import Config
from .cal_ham_01 import BandStructureCalculator
from .read_pos_01 import LayeredLatticeAnalyzer, StructureProcessor,OpenMXFile
from .read_kpath_01 import KPathGenerator
from .read_hr_01 import HrSparseHandler
from .tapw_slab import TAPWSlab

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
    parser.add_argument('--mode', choices=['band', 'chern', 'slab'],
                       help='Calculation mode (overrides config file)')
    parser.add_argument('--n_g', type=int,
                       help='Harmonic of G vectors (overrides config file)')
    parser.add_argument('--num_chern', type=int,
                       help='Number of k-points for Chern number calculation (overrides config file)')
    parser.add_argument('--num_processes', type=int,
                       help='Number of processes (overrides config file)')
    return parser.parse_args()

def main():
    """Main program"""
    # Parse arguments and load configuration
    args = parse_args()
    config = Config.from_yaml(args.config)
    
    # Override config with command line arguments
    if args.twist_index:
        config.twist.twist_index_m = args.twist_index
    if args.output_dir:
        config.paths.output_dir = args.output_dir
    if args.valleys:
        config.compute.valleys = args.valleys
    if args.mode:
        config.compute.mode = args.mode
    if args.n_g:
        config.compute.n_g = args.n_g
    if args.num_chern:
        config.compute.num_chern = args.num_chern
    if args.num_processes:
        config.compute.num_processes = args.num_processes
    # Setup logging
    log_file = Path(config.paths.output_dir) / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logging(str(log_file))
    logger.info("Starting calculation with configuration:")
    logger.info(f"Twist index: {config.twist.twist_index_m}")
    logger.info(f"Output directory: {config.paths.output_dir}")
    logger.info(f"Valleys to calculate: {config.compute.valleys}")
    logger.info(f"Calculation mode: {config.compute.mode}")
    logger.info(f"Harmonic of G vectors: {config.compute.n_g}")
    if config.compute.mode == "chern":
        logger.info(f"Number of k-points for Chern number calculation: {config.compute.num_chern}x{config.compute.num_chern}")
        logger.info(f"Number of processes: {config.compute.num_processes}")
    try:
        # Initialize structure
        structure = OpenMXFile(
            file_path=str(Path(config.paths.input_file)),
            twist_index=config.twist.twist_index_m,
            spin=config.twist.spin
        )
        structure.display_properties()
        
        # Initialize lattice analyzer
        analyzer = LayeredLatticeAnalyzer(
            input_data=structure.sorted_species_coordinates,
            num_layers=config.twist.num_layers,
            type_structure=config.twist.type_structure,
            twist_layer=config.twist.twist_layer
        )
        # analyzer.process()

        if config.compute.TAPW:
            analyzer.process(TAPW=True)
            # Plot results
            analyzer.plot_lattice(save=True, save_path=config.paths.output_dir)
            analyzer.plot_nearest_vectors_phase(save=True, save_path=config.paths.output_dir)

            # Process structure
        processor = StructureProcessor(
            input_data=analyzer.input_data,
            num_layers=config.twist.num_layers,
            monolayer_reciprocal_list=analyzer.reciprocal_vectors,
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
            processor.plot_clusters_loc(save=True, save_path=config.paths.output_dir)
            processor.plot_clusters_phase(save=True, save_path=config.paths.output_dir)

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

        # Handle overlap matrix
        if config.compute.orthogonal_basis:
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
            kpath_config = KPathGenerator(structure.Tmat)
            kpath_config.read_and_generate_kpath(config.paths.kpath_in, config.paths.kpath_out)
        elif config.compute.mode == "slab":
            # For slab mode, k-path is handled internally by TAPWSlab
            pass

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
                calculator = BandStructureCalculator(
                    hr_supercell=hr,
                    sr_supercell=sr,
                    structure=processor,
                    config=config.compute,
                    kpath_config=kpath_config
                )
            
            out_path = Path(config.paths.output_dir) / f"Q_shell_{config.compute.n_g}"
            out_path.mkdir(exist_ok=True)
            
            if config.compute.mode == "slab":
                calculator.calculate_ribbon_bands(str(out_path))
                logger.info(f"Completed slab calculation for valley {valley}")
            else:
                calculator.run_calculation(str(out_path))
                logger.info(f"Completed calculation for valley {valley}")

    except Exception as e:
        logger.error(f"Error during calculation: {str(e)}", exc_info=True)
        sys.exit(1)

    logger.info("Calculation completed successfully")

if __name__ == "__main__":
    main() 