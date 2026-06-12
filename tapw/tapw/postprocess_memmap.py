#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np

from .artifacts import array_output_filename, band_output_filename, raw_memmap_filename
from .workflows.band import open_memmap, _ensure_memmap_file
from .config import format_chern_grid_suffix, resolve_chern_grid_shape


VALLEY_MAP = {
    1: "K1",
    2: "K2",
    11: "K1_120",
    12: "K1_240",
    5: "Gamma",
    31: "M1",
    32: "M2",
    33: "M3",
    3: "M",
    41: "X",
    42: "Y",
}


def _find_first_above_energy(energies: np.ndarray, E: float) -> np.ndarray:
    idxs = []
    for band in energies:
        idx = np.where(band > E)[0]
        idxs.append(int(idx[0]) if idx.size else int(len(band)))
    return np.asarray(idxs, dtype=int)


def postprocess_memmap(
    root_dir: str,
    valley: int,
    mode: str,
    num_chern: Optional[int],
    num_k1: Optional[int],
    num_k2: Optional[int],
    efermi: float,
    band_type: str,
) -> None:
    root = Path(root_dir).resolve()
    memmap_dir = root / "memmap"

    if valley not in VALLEY_MAP:
        raise ValueError(f"Unknown valley={valley}. Allowed: {sorted(VALLEY_MAP)}")
    valley_flag = VALLEY_MAP[valley]

    if mode not in {"band", "chern"}:
        raise ValueError("mode must be 'band' or 'chern'")

    suffix = "_2d" if mode == "chern" else ""
    if mode == "chern":
        if num_chern is None:
            raise ValueError("num_chern is required when mode='chern'")
        num_k1, num_k2 = resolve_chern_grid_shape(num_chern, num_k1, num_k2)
        suffix = format_chern_grid_suffix(num_k1, num_k2)

    eig_path = memmap_dir / raw_memmap_filename("eig", valley_flag=valley_flag, suffix=suffix)
    vec_path = memmap_dir / raw_memmap_filename("vec", valley_flag=valley_flag, suffix=suffix)
    if not eig_path.exists():
        raise FileNotFoundError(f"Missing {eig_path}")
    if not vec_path.exists():
        raise FileNotFoundError(f"Missing {vec_path}")

    energies = np.asarray(open_memmap(str(eig_path), mode="r"))
    vec_raw = open_memmap(str(vec_path), mode="r")

    first_indices = _find_first_above_energy(energies, efermi)
    min_index = int(np.min(first_indices))
    max_index = int(np.max(first_indices))
    n_all = int(energies.shape[1])
    n_keep = int(n_all - (max_index - min_index))
    if n_keep <= 0:
        raise ValueError(f"No bands to keep after alignment (n_keep={n_keep}). Check efermi={efermi}")
    pivot_col = min_index

    filtered = np.empty((energies.shape[0], n_keep), dtype=energies.dtype)
    for k in range(energies.shape[0]):
        start = int(first_indices[k] - min_index)
        end = start + n_keep
        filtered[k] = energies[k, start:end]

    vbm = filtered[:, :pivot_col] if pivot_col > 0 else np.array([]).reshape(filtered.shape[0], 0)
    cbm = filtered[:, pivot_col:] if pivot_col < filtered.shape[1] else np.array([]).reshape(filtered.shape[0], 0)

    if mode == "chern":
        out_dir = root / "topo"
    else:
        out_dir = root / "band"
    out_dir.mkdir(parents=True, exist_ok=True)

    band_type_u = band_type.upper()
    want_vbm = band_type_u != "CBM"
    want_cbm = band_type_u != "VBM"

    if want_vbm and vbm.size:
        np.savetxt(
            out_dir / band_output_filename("VBM", valley_flag=valley_flag, suffix=suffix, tapw=True),
            vbm,
            fmt="%15.11f",
        )
    if want_cbm and cbm.size:
        np.savetxt(
            out_dir / band_output_filename("CBM", valley_flag=valley_flag, suffix=suffix, tapw=True),
            cbm,
            fmt="%15.11f",
        )

    # Wavefunctions: write requested blocks as memmap .npy to avoid huge RAM usage.
    vbm_mm = None
    cbm_mm = None
    if want_vbm and vbm.shape[1] > 0:
        vbm_out = out_dir / array_output_filename("vec_VBM", valley_flag=valley_flag, suffix=suffix, tapw=True)
        _ensure_memmap_file(str(vbm_out), np.complex128, (energies.shape[0], vec_raw.shape[1], vbm.shape[1]))
        vbm_mm = open_memmap(str(vbm_out), mode="r+")
    if want_cbm and cbm.shape[1] > 0:
        cbm_out = out_dir / array_output_filename("vec_CBM", valley_flag=valley_flag, suffix=suffix, tapw=True)
        _ensure_memmap_file(str(cbm_out), np.complex128, (energies.shape[0], vec_raw.shape[1], cbm.shape[1]))
        cbm_mm = open_memmap(str(cbm_out), mode="r+")

    for k in range(energies.shape[0]):
        start = int(first_indices[k] - min_index)
        end = start + n_keep
        if vbm_mm is not None and cbm_mm is not None:
            vec_keep = vec_raw[k, :, start:end]
            vbm_mm[k] = vec_keep[:, :pivot_col]
            cbm_mm[k] = vec_keep[:, pivot_col:]
        elif vbm_mm is not None:
            vbm_mm[k] = vec_raw[k, :, start : start + pivot_col]
        elif cbm_mm is not None:
            cbm_mm[k] = vec_raw[k, :, start + pivot_col : end]

    if vbm_mm is not None:
        vbm_mm.flush()
    if cbm_mm is not None:
        cbm_mm.flush()

    print(f"Postprocess finished: {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="Postprocess TAPW raw memmap outputs into vec_CBM/VBM files.")
    p.add_argument("--root-dir", required=True, help="Q_shell_<n_g> directory that contains memmap/")
    p.add_argument("--valley", type=int, required=True, help="Valley integer (e.g. 31/32/33/5)")
    p.add_argument("--mode", choices=["band", "chern"], required=True)
    p.add_argument("--num-chern", type=int, default=None, help="Required when --mode=chern")
    p.add_argument("--num-k1", type=int, default=None, help="Optional rectangular Chern grid size along kappa1")
    p.add_argument("--num-k2", type=int, default=None, help="Optional rectangular Chern grid size along kappa2")
    p.add_argument("--efermi", type=float, required=True)
    p.add_argument("--band-type", default="CBM", help="CBM, VBM, or anything else to write both")
    args = p.parse_args()

    postprocess_memmap(
        root_dir=args.root_dir,
        valley=args.valley,
        mode=args.mode,
        num_chern=args.num_chern,
        num_k1=args.num_k1,
        num_k2=args.num_k2,
        efermi=args.efermi,
        band_type=args.band_type,
    )


if __name__ == "__main__":
    main()
