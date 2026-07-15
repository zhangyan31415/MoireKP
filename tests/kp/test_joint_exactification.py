from __future__ import annotations

import numpy as np
import pytest

from kp.symmetry.joint_exactification import (
    BlockRouteAction,
    JointExactificationError,
    SemilinearBlock,
    compose_semilinear,
    inverse_semilinear,
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
