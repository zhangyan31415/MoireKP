import numpy as np

from kp.blocks.downfold import DownfoldingOptions, downfold_from_projectors
from kp.cli import _downfold_method


def test_default_reduction_is_linearized_and_fixed_schur_remains_explicit():
    assert _downfold_method({}) == "linearized_lowdin"
    assert _downfold_method({"downfold_method": "fixed_schur"}) == "fixed_schur"
    assert _downfold_method({"method": "first_order"}) == "first_order"
    ham = np.array([[0.1, 0.4], [0.4, 2.0]])
    low = np.eye(2)[:, :1]
    high = np.eye(2)[:, 1:]
    result = downfold_from_projectors(ham, low, high, DownfoldingOptions(e_ref=0.0))
    assert result.method == "linearized_lowdin"
    np.testing.assert_allclose(result.heff, [[(0.1 - 0.08) / 1.04]])
