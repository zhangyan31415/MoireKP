from __future__ import annotations

import numpy as np
import pytest

from kp.symmetry.joint_exactification import (
    BlockRouteAction,
    JointExactificationError,
    SemilinearBlock,
    compose_semilinear,
    extract_block_route_action,
    inverse_semilinear,
    materialize_block_route_action,
)


def test_semilinear_composition_conjugates_only_the_inner_block() -> None:
    outer = SemilinearBlock(np.array([[1.0j]]), antiunitary=True)
    inner = SemilinearBlock(
        np.array([[np.exp(0.3j)]]),
        antiunitary=False,
    )

    product = compose_semilinear(outer, inner)

    np.testing.assert_allclose(product.matrix, [[1.0j * np.exp(-0.3j)]])
    assert product.antiunitary is True


def test_semilinear_inverse_is_a_two_sided_inverse() -> None:
    unitary = np.array([[0.0, 1.0j], [1.0, 0.0]], dtype=np.complex128)
    value = SemilinearBlock(unitary, antiunitary=True)

    inverse = inverse_semilinear(value)

    identity = np.eye(2, dtype=np.complex128)
    np.testing.assert_allclose(compose_semilinear(value, inverse).matrix, identity)
    np.testing.assert_allclose(compose_semilinear(inverse, value).matrix, identity)


def test_semilinear_block_owns_a_readonly_complex_copy() -> None:
    source = np.eye(2, dtype=np.float64)

    value = SemilinearBlock(source, antiunitary=False)
    source[0, 0] = 0.0

    assert value.matrix.dtype == np.dtype(np.complex128)
    assert value.matrix.flags.c_contiguous
    assert value.matrix.flags.writeable is False
    np.testing.assert_array_equal(value.matrix, np.eye(2, dtype=np.complex128))


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        (np.zeros((2, 3), dtype=np.complex128), "square"),
        (np.array([[np.nan]], dtype=np.complex128), "finite"),
        (np.diag([1.0, 2.0]).astype(np.complex128), "unitarity"),
    ],
)
def test_semilinear_block_rejects_invalid_matrices(
    matrix: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(JointExactificationError, match=message):
        SemilinearBlock(
            matrix,
            antiunitary=False,
            unitarity_certification_bound=1.0e-14,
        )


def test_semilinear_block_rejects_invalid_certification_bound() -> None:
    with pytest.raises(JointExactificationError, match="certification bound"):
        SemilinearBlock(
            np.eye(1, dtype=np.complex128),
            antiunitary=False,
            unitarity_certification_bound=-1.0,
        )


def test_block_route_action_validates_and_owns_route_blocks() -> None:
    first = np.eye(2, dtype=np.complex128)
    second = np.diag([1.0j, -1.0j]).astype(np.complex128)

    action = BlockRouteAction(
        name="g",
        antiunitary=False,
        fiber_permutation=(1, 0),
        fiber_dimensions=(2, 2),
        route_blocks=(first, second),
    )
    first[0, 0] = 0.0

    assert action.route_blocks[0].flags.writeable is False
    assert action.route_blocks[0].flags.c_contiguous
    np.testing.assert_array_equal(action.route_blocks[0], np.eye(2))


@pytest.mark.parametrize(
    ("permutation", "dimensions", "blocks", "message"),
    [
        ((0, 0), (1, 1), (np.eye(1), np.eye(1)), "permutation"),
        ((1, 0), (1, 2), (np.ones((2, 1)), np.ones((1, 2))), "dimension"),
        ((1, 0), (2, 2), (np.eye(2),), "one route block"),
        (
            (1, 0),
            (2, 2),
            (np.ones((1, 2)), np.ones((2, 1))),
            "shape",
        ),
        (
            (0,),
            (2,),
            (np.diag([1.0, 2.0]),),
            "unitarity",
        ),
    ],
)
def test_block_route_action_rejects_invalid_structure(
    permutation: tuple[int, ...],
    dimensions: tuple[int, ...],
    blocks: tuple[np.ndarray, ...],
    message: str,
) -> None:
    with pytest.raises(JointExactificationError, match=message):
        BlockRouteAction(
            name="g",
            antiunitary=False,
            fiber_permutation=permutation,
            fiber_dimensions=dimensions,
            route_blocks=blocks,
        )


def test_route_round_trip_preserves_small_internal_entries() -> None:
    eps = 5.0e-5
    block = np.array(
        [[np.sqrt(1.0 - eps**2), eps], [-eps, np.sqrt(1.0 - eps**2)]],
        dtype=np.complex128,
    )
    action = BlockRouteAction(
        name="g",
        antiunitary=False,
        fiber_permutation=(1, 0),
        fiber_dimensions=(2, 2),
        route_blocks=(block, block.conj().T),
    )

    dense = materialize_block_route_action(action)
    restored = extract_block_route_action(
        dense,
        name="g",
        antiunitary=False,
        fiber_indices=((0, 1), (2, 3)),
        fiber_permutation=(1, 0),
        off_route_bound=0.0,
    )

    assert restored.route_blocks[0][0, 1] == eps
    np.testing.assert_array_equal(materialize_block_route_action(restored), dense)
    route_support = dense != 0.0
    assert np.count_nonzero(route_support) == 8
    assert np.count_nonzero(dense[~route_support]) == 0


def test_route_round_trip_preserves_noncontiguous_fiber_layout() -> None:
    action = BlockRouteAction(
        name="g",
        antiunitary=True,
        fiber_permutation=(1, 0),
        fiber_dimensions=(2, 2),
        route_blocks=(
            np.array([[0.0, 1.0j], [1.0, 0.0]], dtype=np.complex128),
            np.array([[0.0, 1.0], [1.0j, 0.0]], dtype=np.complex128),
        ),
        fiber_indices=((0, 2), (1, 3)),
    )

    dense = materialize_block_route_action(action)
    restored = extract_block_route_action(
        dense,
        name="g",
        antiunitary=True,
        fiber_indices=((0, 2), (1, 3)),
        fiber_permutation=(1, 0),
        off_route_bound=0.0,
    )

    assert restored.fiber_indices == ((0, 2), (1, 3))
    np.testing.assert_array_equal(materialize_block_route_action(restored), dense)


def test_route_extraction_rejects_off_route_pollution_above_bound() -> None:
    dense = np.eye(2, dtype=np.complex128)
    dense[1, 0] = 2.0e-8

    with pytest.raises(JointExactificationError, match="off-route"):
        extract_block_route_action(
            dense,
            name="g",
            antiunitary=False,
            fiber_indices=((0,), (1,)),
            fiber_permutation=(0, 1),
            off_route_bound=1.0e-9,
        )


def test_route_extraction_rejects_incompatible_fiber_dimensions() -> None:
    with pytest.raises(JointExactificationError, match="dimension"):
        extract_block_route_action(
            np.eye(3, dtype=np.complex128),
            name="g",
            antiunitary=False,
            fiber_indices=((0,), (1, 2)),
            fiber_permutation=(1, 0),
            off_route_bound=0.0,
        )


@pytest.mark.parametrize(
    "fiber_indices",
    [
        ((0,), (0,)),
        ((0,), (2,)),
        ((0,), (1, 2)),
    ],
)
def test_route_extraction_requires_a_complete_disjoint_fiber_partition(
    fiber_indices: tuple[tuple[int, ...], ...],
) -> None:
    with pytest.raises(JointExactificationError, match="fiber indices"):
        extract_block_route_action(
            np.eye(2, dtype=np.complex128),
            name="g",
            antiunitary=False,
            fiber_indices=fiber_indices,
            fiber_permutation=tuple(range(len(fiber_indices))),
            off_route_bound=0.0,
        )
