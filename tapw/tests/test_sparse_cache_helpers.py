import numpy as np
import scipy.sparse
from types import SimpleNamespace

import tapw.cal_ham_01 as cal_ham_01


def test_realspace_block_cache_reuses_preprocessed_metadata():
    calculator_cls = getattr(cal_ham_01, "BandStructureCalculator", None)
    assert calculator_cls is not None

    helper = getattr(calculator_cls, "_get_or_build_realspace_block_cache", None)
    assert helper is not None

    fake = calculator_cls.__new__(calculator_cls)
    fake._realspace_block_cache = {}
    fake.structure = SimpleNamespace(
        Tmat=np.array(
            [
                [2.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
                [0.0, 0.0, 4.0],
            ],
            dtype=float,
        )
    )

    hr = {
        (0, 0, 0): {
            "row": [0, 1],
            "col": [1, 0],
            "val": [1.0 + 2.0j, 3.0 - 1.0j],
        },
        (1, -1, 0): {
            "row": np.array([0, 0]),
            "col": np.array([0, 1]),
            "val": np.array([4.0j, 5.0], dtype=np.complex128),
        },
    }

    cache_a = helper(fake, hr)
    cache_b = helper(fake, hr)

    assert cache_a is cache_b
    assert cache_a.nnz_total == 4
    assert np.array_equal(cache_a.rows_template, np.array([0, 1, 0, 0], dtype=np.int64))
    assert np.array_equal(cache_a.cols_template, np.array([1, 0, 0, 1], dtype=np.int64))
    assert len(cache_a.blocks) == 2
    assert np.allclose(cache_a.blocks[1].rvec_cart, np.array([2.0, -3.0, 0.0], dtype=float))
    assert cache_a.blocks[1].data_slice == slice(2, 4)


def test_make_cached_projector_term_stores_projector_and_conjugate_transpose():
    calculator_cls = getattr(cal_ham_01, "BandStructureCalculator", None)
    assert calculator_cls is not None

    helper = getattr(calculator_cls, "_make_cached_projector_term", None)
    assert helper is not None

    fake = calculator_cls.__new__(calculator_cls)
    projector = scipy.sparse.csr_matrix(
        np.array(
            [
                [1.0 + 0.0j, 2.0j],
                [0.0, 3.0 - 1.0j],
            ],
            dtype=np.complex128,
        )
    )

    term = helper(fake, "p", np.eye(2, dtype=float), projector)

    assert term.label == "p"
    assert np.array_equal(term.linear_map_2d, np.eye(2, dtype=float))
    assert scipy.sparse.issparse(term.projector)
    assert scipy.sparse.issparse(term.projector_h)
    assert np.allclose(term.projector.toarray(), projector.toarray())
    assert np.allclose(term.projector_h.toarray(), projector.conj().T.toarray())


def test_getk_super_gauge_sparse_fast_path_matches_reference_with_duplicate_entries():
    calculator_cls = getattr(cal_ham_01, "BandStructureCalculator", None)
    assert calculator_cls is not None

    fake = calculator_cls.__new__(calculator_cls)
    fake.config = SimpleNamespace(fast_getk=True)
    fake.structure = SimpleNamespace(reciprocal_Tmat=np.eye(3), Tmat=np.eye(3))
    fake._sorted_wann = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.25, 0.5, 0.0],
        ],
        dtype=float,
    )
    fake._num_wann = 2
    fake._ef_onsite_orb = None
    fake._realspace_block_cache = {}

    hr = {
        (0, 0, 0): {
            "row": np.array([0, 0, 1], dtype=np.int64),
            "col": np.array([1, 1, 0], dtype=np.int64),
            "val": np.array([1.0 + 0.5j, -0.75 + 0.25j, 2.0 - 1.0j], dtype=np.complex128),
        },
        (1, 0, 0): {
            "row": np.array([0], dtype=np.int64),
            "col": np.array([1], dtype=np.int64),
            "val": np.array([0.5 - 0.125j], dtype=np.complex128),
        },
    }
    k = np.array([0.17, -0.09, 0.0], dtype=float)

    fast = calculator_cls.Getk_super_gauge_sparse(fake, hr, k, type="H")
    fake.config.fast_getk = False
    slow = calculator_cls.Getk_super_gauge_sparse(fake, hr, k, type="H")

    assert np.allclose(fast.toarray(), slow.toarray())


def test_block_cache_can_compress_raw_data_into_csr_with_duplicate_entries():
    calculator_cls = getattr(cal_ham_01, "BandStructureCalculator", None)
    assert calculator_cls is not None

    fake = calculator_cls.__new__(calculator_cls)
    fake._realspace_block_cache = {}
    fake.structure = SimpleNamespace(Tmat=np.eye(3))

    hr = {
        (0, 0, 0): {
            "row": np.array([0, 0, 1], dtype=np.int64),
            "col": np.array([1, 1, 0], dtype=np.int64),
            "val": np.array([1.0 + 0.5j, -0.75 + 0.25j, 2.0 - 1.0j], dtype=np.complex128),
        },
        (1, 0, 0): {
            "row": np.array([0], dtype=np.int64),
            "col": np.array([1], dtype=np.int64),
            "val": np.array([0.5 - 0.125j], dtype=np.complex128),
        },
    }

    cache = calculator_cls._get_or_build_realspace_block_cache(fake, hr)
    helper = getattr(calculator_cls, "_compress_raw_realspace_data_to_csr", None)
    assert helper is not None

    raw_data = np.concatenate(
        [np.asarray(block.values, dtype=np.complex128) for block in cache.blocks]
    )
    actual = helper(fake, cache, raw_data, num_wann=2)
    expected = scipy.sparse.coo_matrix(
        (raw_data, (cache.rows_template, cache.cols_template)),
        shape=(2, 2),
        dtype=np.complex128,
    ).tocsr()
    expected.sum_duplicates()

    assert np.allclose(actual.toarray(), expected.toarray())


def test_block_cache_can_compress_raw_data_into_csr_without_duplicates():
    calculator_cls = getattr(cal_ham_01, "BandStructureCalculator", None)
    assert calculator_cls is not None

    fake = calculator_cls.__new__(calculator_cls)
    fake._realspace_block_cache = {}
    fake.structure = SimpleNamespace(Tmat=np.eye(3))

    hr = {
        (0, 0, 0): {
            "row": np.array([0, 1], dtype=np.int64),
            "col": np.array([0, 1], dtype=np.int64),
            "val": np.array([2.0 + 0.0j, 3.0 - 1.0j], dtype=np.complex128),
        },
        (1, 0, 0): {
            "row": np.array([1], dtype=np.int64),
            "col": np.array([0], dtype=np.int64),
            "val": np.array([-4.0j], dtype=np.complex128),
        },
    }

    cache = calculator_cls._get_or_build_realspace_block_cache(fake, hr)
    helper = getattr(calculator_cls, "_compress_raw_realspace_data_to_csr", None)
    assert helper is not None

    raw_data = np.concatenate(
        [np.asarray(block.values, dtype=np.complex128) for block in cache.blocks]
    )
    actual = helper(fake, cache, raw_data, num_wann=2)
    expected = scipy.sparse.coo_matrix(
        (raw_data, (cache.rows_template, cache.cols_template)),
        shape=(2, 2),
        dtype=np.complex128,
    ).tocsr()
    expected.sum_duplicates()

    assert np.allclose(actual.toarray(), expected.toarray())
