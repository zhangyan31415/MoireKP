import numpy as np
import scipy.sparse
from types import SimpleNamespace
from pathlib import Path

import tapw.workflows.band as band_workflow
from tapw.io.hr import (
    HrSparseHandler,
    SPARSE_NPZ_METADATA_KEY,
    SPARSE_NPZ_SCHEMA,
)


def test_realspace_block_cache_reuses_preprocessed_metadata():
    calculator_cls = getattr(band_workflow, "BandStructureCalculator", None)
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
    calculator_cls = getattr(band_workflow, "BandStructureCalculator", None)
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
    calculator_cls = getattr(band_workflow, "BandStructureCalculator", None)
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
    calculator_cls = getattr(band_workflow, "BandStructureCalculator", None)
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
    calculator_cls = getattr(band_workflow, "BandStructureCalculator", None)
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


def _write_sparse_dat(path: Path, label: str, nwann: int, entries):
    lines = [
        f" ! Sparse format of {label}",
        f" {len(entries)} ! Number of non-zeros lines of {label}mnR",
        f" {nwann} ! Number of orbitals",
        " 1 ! Number of R points",
    ]
    for rx, ry, rz, row, col, value in entries:
        lines.append(f"{rx:5d}{ry:5d}{rz:5d}{row + 1:6d}{col + 1:6d}{value.real:16.8f}{value.imag:16.8f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_hr_sparse_handler_reads_openmx_symm_npz_with_same_basis_order_as_dat(tmp_path):
    dat_path = tmp_path / "H_symm.dat"
    npz_path = tmp_path / "H_symm.npz"
    entries = [
        (0, 0, 0, 0, 2, 1.0 + 0.5j),
        (0, 0, 0, 2, 1, -0.25 + 0.75j),
        (1, 0, 0, 1, 0, 2.0 - 0.1j),
    ]
    _write_sparse_dat(dat_path, "H", 3, entries)
    np.savez(
        npz_path,
        **{
            "(0, 0, 0)_row": np.array([0, 2], dtype=np.int32),
            "(0, 0, 0)_col": np.array([2, 1], dtype=np.int32),
            "(0, 0, 0)_val": np.array([1.0 + 0.5j, -0.25 + 0.75j], dtype=np.complex128),
            "(1, 0, 0)_row": np.array([1], dtype=np.int32),
            "(1, 0, 0)_col": np.array([0], dtype=np.int32),
            "(1, 0, 0)_val": np.array([2.0 - 0.1j], dtype=np.complex128),
        },
    )

    perm = scipy.sparse.csr_matrix(
        np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.complex128,
        )
    )
    dat_hr = HrSparseHandler(file_name=str(dat_path), npz_file_name="", A=perm, read_from_npz=False).get_hr_sparse()
    npz_hr = HrSparseHandler(file_name="", npz_file_name=str(npz_path), A=perm, read_from_npz=True).get_hr_sparse()

    assert set(npz_hr) == set(dat_hr)
    for key in dat_hr:
        assert np.array_equal(np.asarray(npz_hr[key]["row"]), np.asarray(dat_hr[key]["row"]))
        assert np.array_equal(np.asarray(npz_hr[key]["col"]), np.asarray(dat_hr[key]["col"]))
        assert np.allclose(np.asarray(npz_hr[key]["val"]), np.asarray(dat_hr[key]["val"]))


def test_hr_sparse_handler_keeps_legacy_npz_path_for_plain_h_npz(tmp_path):
    npz_path = tmp_path / "H.npz"
    np.savez(
        npz_path,
        **{
            "(0, 0, 0)_row": np.array([0, 2], dtype=np.int32),
            "(0, 0, 0)_col": np.array([2, 1], dtype=np.int32),
            "(0, 0, 0)_val": np.array([1.0 + 0.5j, -0.25 + 0.75j], dtype=np.complex128),
        },
    )
    perm = scipy.sparse.csr_matrix(
        np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.complex128,
        )
    )

    npz_hr = HrSparseHandler(file_name="", npz_file_name=str(npz_path), A=perm, read_from_npz=True).get_hr_sparse()

    assert np.array_equal(np.asarray(npz_hr[(0, 0, 0)]["row"]), np.array([0, 2], dtype=np.int64))
    assert np.array_equal(np.asarray(npz_hr[(0, 0, 0)]["col"]), np.array([2, 1], dtype=np.int64))
    assert np.allclose(
        np.asarray(npz_hr[(0, 0, 0)]["val"]),
        np.array([1.0 + 0.5j, -0.25 + 0.75j], dtype=np.complex128),
    )


def test_hr_sparse_writer_records_versioned_exact_basis_dimension(tmp_path):
    import json

    npz_path = tmp_path / "H.npz"
    handler = HrSparseHandler()
    handler.nwann = 4
    handler.hr_sparse = {
        (0, 0, 0): {
            "row": np.array([0], dtype=np.int32),
            "col": np.array([0], dtype=np.int32),
            "val": np.array([1.0 + 0.0j], dtype=np.complex128),
        }
    }

    handler.save_to_npz(npz_path)

    with np.load(npz_path, allow_pickle=False) as payload:
        metadata = json.loads(str(np.asarray(payload[SPARSE_NPZ_METADATA_KEY]).item()))
    assert metadata == {
        "basis_dimension": 4,
        "schema": SPARSE_NPZ_SCHEMA,
        "schema_version": 1,
    }
    loaded = HrSparseHandler(
        file_name="",
        npz_file_name=str(npz_path),
        read_from_npz=True,
    ).get_hr_sparse()
    assert set(loaded) == {(0, 0, 0)}


def test_symm_npz_legacy_zero_tail_uses_transform_input_dimension(tmp_path):
    npz_path = tmp_path / "H_symm.npz"
    np.savez(
        npz_path,
        **{
            "(0, 0, 0)_row": np.array([0], dtype=np.int32),
            "(0, 0, 0)_col": np.array([0], dtype=np.int32),
            "(0, 0, 0)_val": np.array([2.0 + 0.0j], dtype=np.complex128),
        },
    )
    transform = scipy.sparse.identity(4, dtype=np.complex128, format="csr")

    loaded = HrSparseHandler(
        file_name="",
        npz_file_name=str(npz_path),
        A=transform,
        read_from_npz=True,
    ).get_hr_sparse()

    assert np.array_equal(loaded[(0, 0, 0)]["row"], np.array([0]))
    assert np.array_equal(loaded[(0, 0, 0)]["col"], np.array([0]))
    assert np.allclose(loaded[(0, 0, 0)]["val"], np.array([2.0 + 0.0j]))
