"""Verify the exported models against their archived TAPW band windows."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent
EXAMPLES = ROOT.parent


def verify_overlaps(folder, selected_bands, selected_tapw, selected_model, report):
    archive = folder / "native_overlaps.npz"
    with np.load(archive, allow_pickle=False) as data:
        model = data["model_eV"]
        native = data["tapw_eV"]
        overlaps = data["overlap_matrix"]
        matching = data["matching_native_indices"]
        expected = data["overlap_squared"]
        tolerance = float(data["cluster_threshold_eV"])
    assert len(model) == len(native) == len(overlaps) == 61
    np.testing.assert_allclose(native[:, selected_bands], selected_tapw, atol=1e-8, rtol=0)
    np.testing.assert_allclose(model[:, selected_bands], selected_model, atol=1e-8, rtol=0)
    target = expected[:, selected_bands]
    np.testing.assert_allclose([target.mean(), target.min()],
        [report["mean_squared_overlap"], report["minimum_squared_overlap"]], atol=1e-8, rtol=0)
    for predicted, reference, cross, assignment, saved in zip(
        model, native, overlaps, matching, expected
    ):
        count = len(predicted)
        np.testing.assert_array_equal(np.sort(assignment), np.arange(count))
        groups = np.arange(count)
        for energies in (predicted, reference[assignment]):
            ordered = np.argsort(energies, kind="stable")
            for left, right in zip(ordered[:-1], ordered[1:]):
                if energies[right] - energies[left] <= tolerance:
                    groups[groups == groups[right]] = groups[left]
        paired = cross[assignment]
        reconstructed = np.empty(count)
        for group in np.unique(groups):
            members = np.flatnonzero(groups == group)
            block = paired[np.ix_(members, members)]
            reconstructed[members] = np.sum(abs(block) ** 2) / len(members)
        np.testing.assert_allclose(reconstructed, saved, atol=1e-7, rtol=0)


def verify(entry, report):
    folder = EXAMPLES / entry["case"] / "model" / entry["valley"]
    evaluator = folder / "evaluate.py"
    sys.path.insert(0, str(folder))
    try:
        spec = importlib.util.spec_from_file_location("released_model_evaluator", evaluator)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.load_model(folder)
        with np.load(folder / "reference_bands.npz", allow_pickle=False) as reference:
            kpoints = reference["k_cartesian_inv_angstrom"]
            tapw = reference["tapw_eV"]
            shift = reference["model_shift_eV"]
            rows = reference["evaluation_rows"]
            edge = str(reference["band_edge"])
        authority = folder / "model_data.npz"
        if entry.get("export_format") == "fixed_angle_gamma":
            authority = folder / "COEFFICIENTS.npz"
            with np.load(folder / "MATERIAL_CONTROLS.npz", allow_pickle=False) as archive:
                count = archive["controls"].size
        elif authority.is_file():
            with np.load(authority, allow_pickle=False) as archive:
                count = archive["fitted_coefficients"].size
        else:
            authority = folder / "polynomial.npz"
            count = np.load(folder / "coefficients.npy", allow_pickle=False).size
        assert hashlib.sha256(authority.read_bytes()).hexdigest() == entry["export_sha256"]
        assert count == entry["coefficients"] and len(rows) == entry["eval_points"]
        energies = model.bands(kpoints)
    finally:
        sys.path.pop(0)
    assert energies.shape == (len(kpoints), entry["dimension"])
    assert np.isfinite(energies).all()
    assert tapw.shape[1] == entry["band_count"]
    frozen = np.load(
        folder / entry.get("validation_eigvals_file", "eigvals.npy"), allow_pickle=False
    )
    np.testing.assert_allclose(energies, frozen, atol=1e-8, rtol=0)
    n = entry["band_count"]
    selected = energies[:, -n:] if edge == "valence" else energies[:, :n]
    errors = 1000 * (selected + shift - tapw)[rows]
    rms, maximum = np.sqrt(np.mean(errors**2)), np.max(np.abs(errors))
    np.testing.assert_allclose([rms, maximum], [entry["rms_meV"], entry["max_meV"]],
                               atol=1e-6, rtol=0)
    if report is not None:
        assert report["coefficients_sha256"] == entry["export_sha256"]
        np.testing.assert_allclose([rms, maximum], [report["rms_meV"], report["max_meV"]],
                                   atol=1e-6, rtol=0)
        window = slice(-n, None) if edge == "valence" else slice(0, n)
        verify_overlaps(folder, window, tapw, selected + shift, report)
    print(f"{entry['case']}/{entry['valley']}: RMS {rms:.3f} meV, max {maximum:.3f} meV")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", help="Verify only the named example directory")
    args = parser.parse_args()
    entries = json.loads((ROOT / "models.json").read_text())
    selected = [entry for entry in entries if args.case is None or entry["case"] == args.case]
    if not selected:
        parser.error(f"unknown case: {args.case}")
    reports = {r["model"]: r for r in json.loads((ROOT / "validation61.json").read_text())}
    assert len(reports) == 12 and reports.keys() <= {f"{e['case']}/{e['valley']}" for e in entries}
    for entry in selected:
        verify(entry, reports.get(f"{entry['case']}/{entry['valley']}"))


if __name__ == "__main__":
    main()
