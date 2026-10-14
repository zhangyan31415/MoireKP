"""Periodic plotting selects lattice origins by their distance to actual atoms."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from tapw.io.structure import LayeredLatticeAnalyzer


def _draw(method, atoms, vectors, cutoff, monkeypatch):
    analyzer = LayeredLatticeAnalyzer.__new__(LayeredLatticeAnalyzer)
    analyzer.df = pd.DataFrame({"layer": [0] * len(atoms), "x": np.array(atoms)[:, 0], "y": np.array(atoms)[:, 1]})
    analyzer.lattice_vectors = {0: np.asarray(vectors, dtype=float)}
    monkeypatch.setattr(plt, "show", lambda: None)
    getattr(analyzer, method)(layer=0, distance_threshold=cutoff)
    offsets = plt.gcf().axes[0].collections[1].get_offsets().copy()
    plt.close("all")
    return {tuple(np.round(item, 8)) for item in offsets}


@pytest.mark.parametrize("method", ["plot_layer_lattice", "plot_lattice"])
def test_explicit_distance_threshold_controls_selection(method, monkeypatch):
    atoms = [[0, 0]]
    vectors = [[2, 0], [0, 2]]
    assert _draw(method, atoms, vectors, 1.0, monkeypatch) == {(0.0, 0.0)}
    assert _draw(method, atoms, vectors, 2.0, monkeypatch) == {(0.0, 0.0)}
    assert len(_draw(method, atoms, vectors, 2.1, monkeypatch)) == 5


@pytest.mark.parametrize("method", ["plot_layer_lattice", "plot_lattice"])
def test_default_distance_cutoff_remains_three_point_five(method, monkeypatch):
    actual = _draw(method, [[0, 0]], [[2, 0], [0, 2]], None, monkeypatch)
    assert actual == {(2.0*i, 2.0*j) for i in range(-2, 3) for j in range(-2, 3) if np.hypot(2*i, 2*j) < 3.5}


@pytest.mark.parametrize("method", ["plot_layer_lattice", "plot_lattice"])
def test_enumeration_covers_displaced_atoms_and_skew_lattice(method, monkeypatch):
    vectors = np.array([[10., 0.], [9., 1.]])
    atom = 7*vectors[0] - 6*vectors[1]
    actual = _draw(method, [atom], vectors, 1.5, monkeypatch)
    expected = {tuple(i*vectors[0]+j*vectors[1]) for i in range(-20, 21) for j in range(-20, 21) if np.linalg.norm(i*vectors[0]+j*vectors[1]-atom) < 1.5}
    assert actual == expected
