"""Use the archived K1 plane-wave sets for the outer and middle orientations."""
from pathlib import Path

import numpy as np
from tapw.cli import main
from tapw.workflows.band import TAPW_parameters


def archived_centers(parameters):
    if parameters.valley != 1 or parameters.n_g != 6:
        raise ValueError("The archived trilayer plane-wave sets are for K1 at Q shell 6.")
    source = Path(__file__).with_name("g_vectors.csv")
    vectors = np.loadtxt(source, delimiter=",", skiprows=1)
    outer = vectors[:, :2]
    middle = vectors[:, 2:]
    parameters.g_vec_list_K1, parameters.g_vec_list_K2 = outer, middle
    parameters.K1, parameters.K2 = outer.mean(axis=0), middle.mean(axis=0)


if __name__ == "__main__":
    TAPW_parameters.generate_g_vec_list = archived_centers
    main()
