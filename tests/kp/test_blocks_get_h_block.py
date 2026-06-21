from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class GetHBlockTests(unittest.TestCase):
    def test_sparse_eigen_column_request_preserves_original_column_positions(self) -> None:
        from kp.blocks.blocks import _hermitian_eigh_columns

        eig, vec = _hermitian_eigh_columns(np.diag([1.0, 2.0, 3.0, 4.0]), [0, 3])

        np.testing.assert_allclose(eig, np.array([1.0, 0.0, 0.0, 4.0]))
        expected = np.zeros((4, 4), dtype=np.complex128)
        expected[0, 0] = 1.0
        expected[3, 3] = 1.0
        np.testing.assert_allclose(vec, expected)

    def test_extract_square_block_slices_contiguous_indices(self) -> None:
        from kp.blocks.blocks import _extract_square_block

        ham = np.arange(36).reshape(6, 6)

        contiguous = _extract_square_block(ham, np.array([2, 3, 4]))
        assert np.shares_memory(contiguous, ham)
        np.testing.assert_array_equal(contiguous, ham[2:5, 2:5])

        scattered = _extract_square_block(ham, np.array([0, 3, 5]))
        assert not np.shares_memory(scattered, ham)
        np.testing.assert_array_equal(scattered, ham[np.ix_([0, 3, 5], [0, 3, 5])])

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

    def test_gamma_energy_projector_uses_direct_fill_order(self) -> None:
        from kp.blocks.blocks import calculate_energy_lists

        vecs = np.empty(2, dtype=object)
        vecs[0] = np.eye(4, dtype=np.complex128)
        vecs[1] = np.eye(4, dtype=np.complex128)

        U_low, U_high = calculate_energy_lists(
            vecs,
            [[0, 1]],
            [[]],
            [[np.array([0, 1])], [np.array([0, 1])]],
            [[2], [2]],
            mode="gamma",
            include_high=False,
        )

        assert U_high is None
        assert U_low.shape == (8, 4)
        expected = np.zeros((8, 4), dtype=np.complex128)
        for row, col in [(0, 0), (1, 2), (2, 1), (3, 3)]:
            expected[row, col] = 1.0
        np.testing.assert_allclose(U_low, expected)

    def test_spinful_gamma_projector_preserves_group_q_layer_order(self) -> None:
        from kp.blocks.blocks import calculate_energy_lists

        vecs = np.empty(2, dtype=object)
        vecs[0] = np.eye(6, dtype=np.complex128)
        vecs[1] = np.eye(6, dtype=np.complex128)

        U_low, U_high = calculate_energy_lists(
            vecs,
            [[0, 1, 2, 3, 4, 5]],
            [[]],
            [[np.array([0, 1])], [np.array([0, 1])]],
            [[1], [1, 1]],
            mode="gamma",
            include_high=False,
        )

        assert U_high is None
        assert U_low.shape == (12, 12)
        expected = np.zeros((12, 12), dtype=np.complex128)
        for row, col in [
            (0, 0),   # spin up, group 0, q0
            (1, 1),   # spin up, group 0, q1
            (2, 2),   # spin up, group 1, q0 layer 0
            (4, 3),   # spin up, group 1, q1 layer 0
            (3, 4),   # spin up, group 1, q0 layer 1
            (5, 5),   # spin up, group 1, q1 layer 1
            (6, 6),   # spin down, group 0, q0
            (7, 7),   # spin down, group 0, q1
            (8, 8),   # spin down, group 1, q0 layer 0
            (10, 9),  # spin down, group 1, q1 layer 0
            (9, 10),  # spin down, group 1, q0 layer 1
            (11, 11), # spin down, group 1, q1 layer 1
        ]:
            expected[row, col] = 1.0
        np.testing.assert_allclose(U_low, expected)

    def test_gamma_nested_sector_projector_uses_one_same_q_row_space(self) -> None:
        from kp.blocks.blocks import _assemble_gamma_projectors_from_block_eigenvectors

        vecs = np.empty(2, dtype=object)
        vecs[0] = np.eye(6, dtype=np.complex128)
        vecs[1] = np.eye(6, dtype=np.complex128)
        idx_list = [
            np.array([0, 1, 2, 6, 7, 8]),
            np.array([3, 4, 5, 9, 10, 11]),
        ]

        U_low, U_high = _assemble_gamma_projectors_from_block_eigenvectors(
            vecs,
            idx_list,
            [[0, 1], [2, 3]],
            include_high=True,
        )

        assert U_low.shape == (12, 8)
        assert U_high is not None
        assert U_high.shape == (12, 4)
        expected_low = np.zeros((12, 8), dtype=np.complex128)
        for row, col in [
            (0, 0),
            (3, 1),
            (1, 2),
            (4, 3),
            (2, 4),
            (5, 5),
            (6, 6),
            (9, 7),
        ]:
            expected_low[row, col] = 1.0
        expected_high = np.zeros((12, 4), dtype=np.complex128)
        for row, col in [(7, 0), (8, 1), (10, 2), (11, 3)]:
            expected_high[row, col] = 1.0
        np.testing.assert_allclose(U_low, expected_low)
        np.testing.assert_allclose(U_high, expected_high)

    def test_project_heff_full_gamma_resolves_physical_layers_to_source_group_bands(self) -> None:
        from kp.blocks import blocks

        ham = np.eye(6, dtype=np.complex128)
        h_vec_blocks = np.empty(1, dtype=object)
        h_vec_blocks[0] = np.eye(3, dtype=np.complex128)
        captured = {}

        def fake_assemble(_vecs, _idx_list, nlow_state_list, **_kwargs):
            captured["nlow_state_list"] = nlow_state_list
            low_groups = [
                {
                    "rows": np.array([0, 1, 2], dtype=np.intp),
                    "cols": np.array([0, 1, 2, 3], dtype=np.intp),
                    "values": np.eye(3, 4, dtype=np.complex128),
                }
            ]
            return low_groups, None

        def fake_downfold(_ham, _low_groups, _high_groups, _options):
            return type("Result", (), {"heff": np.eye(4, dtype=np.complex128)})()

        with (
            patch.object(blocks, "get_H_block", return_value=(None, h_vec_blocks, None, None)),
            patch.object(blocks, "_assemble_gamma_projector_groups_from_block_eigenvectors", side_effect=fake_assemble),
            patch.object(blocks, "downfold_from_projector_groups", side_effect=fake_downfold),
        ):
            blocks.project_heff_full(
                ham,
                q_count=1,
                orb_per_layer0=1,
                num_layer_list=[1, 2],
                spin="up",
                Qlayer_list=[[np.array([[0.0, 0.0]])], [np.array([[0.0, 0.0]]), np.array([[0.0, 0.0]])]],
                num_orb_per_layer_list=[[1], [1, 1]],
                nlow_state_list=[[], [0, 1], [2, 3]],
                norb_fix_list=[[], [[[0, 1.0]], [[1, 1.0]]], [[[2, 1.0]], [[0, 1.0]]]],
                mode="gamma",
                downfold_method="first_order",
                second_order=False,
            )

        self.assertEqual(captured["nlow_state_list"], [[], [0, 1, 2, 3]])

    def test_project_heff_full_rejects_legacy_source_group_rows(self) -> None:
        from kp.blocks import blocks

        with self.assertRaisesRegex(ValueError, "physical-layer rows"):
            blocks.project_heff_full(
                np.eye(6, dtype=np.complex128),
                q_count=1,
                orb_per_layer0=1,
                num_layer_list=[1, 2],
                spin="up",
                Qlayer_list=[
                    [np.array([[0.0, 0.0]])],
                    [np.array([[0.0, 0.0]]), np.array([[0.0, 0.0]])],
                ],
                num_orb_per_layer_list=[[1], [1, 1]],
                nlow_state_list=[[], [0, 1]],
                norb_fix_list=[[], [[[0, 1.0]], [[1, 1.0]]]],
                mode="gamma",
                downfold_method="first_order",
                second_order=False,
            )

    def test_non_gamma_spin_up_project_heff_preserves_group_q_layer_order(self) -> None:
        from kp.blocks import blocks

        ham = np.diag([0.0, 0.0, 0.0, 10.0, 40.0, 20.0]).astype(np.complex128)

        _, heig, _ = blocks.project_heff_full(
            ham,
            q_count=2,
            orb_per_layer0=1,
            num_layer_list=[1, 2],
            spin="up",
            Qlayer_list=[
                [np.array([[0.0, 0.0], [1.0, 0.0]])],
                [np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([[0.0, 0.0], [1.0, 0.0]])],
            ],
            num_orb_per_layer_list=[[1], [1, 1]],
            nlow_state_list=[[], [], [0]],
            norb_fix_list=[[], [], [[[0, 1.0]]]],
            mode="K1",
            downfold_method="first_order",
            second_order=False,
        )

        np.testing.assert_allclose(heig, np.array([10.0, 20.0]))

    def test_project_heff_full_avoids_dead_projector_allocation_and_preserves_methods(self) -> None:
        from kp.blocks import blocks

        ham = np.diag([1.0, 2.0, 4.0, 5.0]).astype(np.complex128)
        h_vec_blocks = np.empty(1, dtype=object)
        h_vec_blocks[0] = np.eye(4, dtype=np.complex128)
        low_projector = np.eye(4, dtype=np.complex128)[:, :2]
        high_projector = np.eye(4, dtype=np.complex128)[:, 2:]

        def fake_calculate_energy_lists(*_args, include_high: bool, **_kwargs):
            return low_projector, high_projector if include_high else None

        original_zeros = blocks.np.zeros

        def guarded_zeros(shape, *args, **kwargs):
            if tuple(shape) == (4, 2):
                raise AssertionError("project_heff_full allocated dead U_low_full")
            return original_zeros(shape, *args, **kwargs)

        common_kwargs = dict(
            q_count=1,
            orb_per_layer0=2,
            num_layer_list=[1, 1],
            spin="up",
            Qlayer_list=[[np.array([[0.0, 0.0]])], [np.array([[0.0, 0.0]])]],
            num_orb_per_layer_list=[[2], [2]],
            nlow_state_list=[[0, 1], []],
            norb_fix_list=[[[[0, 1.0]], [[1, 1.0]]], []],
            mode="gamma",
        )

        with (
            patch.object(blocks, "get_H_block", return_value=(None, h_vec_blocks, None, None)),
            patch.object(blocks, "calculate_energy_lists", side_effect=fake_calculate_energy_lists),
            patch.object(blocks.np, "zeros", side_effect=guarded_zeros),
        ):
            first_order = blocks.project_heff_full(
                ham,
                downfold_method="first_order",
                second_order=False,
                **common_kwargs,
            )
            fixed_schur = blocks.project_heff_full(
                ham,
                downfold_method="fixed_schur",
                second_order=True,
                E_ref=0.0,
                **common_kwargs,
            )

        np.testing.assert_allclose(first_order[0], fixed_schur[0])
        np.testing.assert_allclose(first_order[0], np.diag([1.0, 2.0]))
        np.testing.assert_allclose(first_order[1], fixed_schur[1])

    def test_project_heff_full_first_order_requests_only_low_eigenvectors(self) -> None:
        from kp.blocks import blocks

        ham = np.diag([1.0, 2.0, 4.0, 5.0]).astype(np.complex128)
        real_eigh = blocks.scipy.linalg.eigh
        subset_requests = []

        def tracking_eigh(*args, **kwargs):
            subset_requests.append((np.asarray(args[0]).shape, kwargs.get("subset_by_index")))
            return real_eigh(*args, **kwargs)

        with patch.object(blocks.scipy.linalg, "eigh", side_effect=tracking_eigh):
            blocks.project_heff_full(
                ham,
                q_count=1,
                orb_per_layer0=2,
                num_layer_list=[1, 1],
                spin="up",
                Qlayer_list=[[np.array([[0.0, 0.0]])], [np.array([[0.0, 0.0]])]],
                num_orb_per_layer_list=[[2], [2]],
                nlow_state_list=[[0, 1], []],
                norb_fix_list=[[[[0, 1.0]], [[1, 1.0]]], []],
                mode="gamma",
                downfold_method="first_order",
                second_order=False,
            )

        block_requests = [request for shape, request in subset_requests if shape == (4, 4)]
        assert block_requests
        assert all(request == (0, 1) for request in block_requests)

if __name__ == "__main__":
    unittest.main()
