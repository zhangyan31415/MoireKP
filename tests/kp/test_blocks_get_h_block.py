from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class GetHBlockTests(unittest.TestCase):
    def test_non_gamma_alignment_uses_current_layer_reference(self) -> None:
        from kp.blocks import get_H_block

        ham = np.diag([1.0, 2.0, 1.0, 2.0]).astype(np.complex128)
        q_layers = [[np.array([[0.0, 0.0]])], [np.array([[0.0, 0.0]])]]

        _, h_vec_blocks, _, _ = get_H_block(
            ham,
            q_layers,
            [1, 1],
            [[2], [2]],
            [[0], [0]],
            [[[[0, 1.0]]], [[[0, -1.0]]]],
            spin="up",
            mode="M1",
        )

        block0 = np.asarray(h_vec_blocks[0], dtype=np.complex128)
        block1 = np.asarray(h_vec_blocks[1], dtype=np.complex128)

        np.testing.assert_allclose(block0[:, 0], np.array([1.0, 0.0]))
        np.testing.assert_allclose(block1[:, 0], np.array([-1.0, 0.0]))

    def test_spinful_non_gamma_projectors_use_model_basis_order(self) -> None:
        from kp.blocks.blocks import _assemble_projectors_from_block_eigenvectors

        vecs = np.empty(4, dtype=object)
        for idx in range(4):
            vecs[idx] = np.eye(2, dtype=np.complex128)
        idx_list = [
            np.array([0, 4]),  # layer 1, q0, two spinful low states
            np.array([1, 5]),  # layer 1, q1
            np.array([2, 6]),  # layer 2, q0
            np.array([3, 7]),  # layer 2, q1
        ]

        U_low, U_high = _assemble_projectors_from_block_eigenvectors(
            vecs,
            idx_list,
            [[0, 1], [0, 1]],
            include_high=False,
        )

        assert U_high is None
        assert U_low.shape == (8, 8)
        expected = np.zeros((8, 8), dtype=np.complex128)
        # ContinuumModelBuilder.get_global_index orders each sector as
        # orbital-major: all Q for orbital 0, then all Q for orbital 1.
        for row, col in [
            (0, 0),
            (1, 1),
            (4, 2),
            (5, 3),
            (2, 4),
            (3, 5),
            (6, 6),
            (7, 7),
        ]:
            expected[row, col] = 1.0
        np.testing.assert_allclose(U_low, expected)


if __name__ == "__main__":
    unittest.main()
