from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


def test_kp_model_reports_model_results_and_validation_sections(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    from kp import cli
    from kp.model import export as export_mod
    from kp.model import pipeline

    cfg_path = tmp_path / "model.yaml"
    cfg_path.write_text("model: {}\n", encoding="utf-8")
    output_dir = tmp_path / "outputs" / "model"
    model_cfg = SimpleNamespace(
        path=cfg_path,
        output_dir=output_dir,
        n_orb=(1, 1),
        fit_indices=[0, 2],
        bM_diagnostics={"source": "project metadata"},
        symmetry_source_metadata={"operations": [{"name": "C3z"}, {"name": "C2T"}]},
    )
    moire_cfg = SimpleNamespace(
        Q_set1=[0, 1],
        Q_set2=[0, 1],
        n_orb1=1,
        n_orb2=1,
        kpoints=[0, 1, 2],
        intra_harmonics_map={"q0": 0},
        inter_harmonics_map={"q1": 1},
    )
    results = {
        "configured_model": model_cfg,
        "moire_config": moire_cfg,
        "runtime_s": 1.25,
        "band_plot": output_dir / "band_comparison.pdf",
        "comparison": {
            "rms_error": 2.5e-5,
            "max_abs_error": 7.5e-5,
            "reference": "current_heff_support_mask",
        },
    }
    monkeypatch.setattr(pipeline, "run_configured_model", lambda _config: results)
    monkeypatch.setattr(
        export_mod,
        "export_standalone_model",
        lambda *_args, **_kwargs: output_dir,
    )

    cli.main(["model", "-c", str(cfg_path)])

    out = capsys.readouterr().out
    assert "[kp model] Fit and export continuum model" in out
    assert "[kp model] Model" in out
    assert "  basis         dim=4, Q=(2, 2), n_orb=(1, 1)" in out
    assert "  symmetry      C3z, C2T" in out
    assert "[kp model] Results" in out
    assert f"  band plot  {output_dir / 'band_comparison.pdf'}" in out
    assert "[kp model] Validation" in out
    assert "  RMS error" in out
    assert "2.500000e-05 eV (0.025 meV)" in out
    assert "  Max error" in out
    assert "7.500000e-05 eV (0.075 meV)" in out
    assert f"  standalone export  {output_dir}" in out
    assert "[kp model] OK  Completed in" in out
    assert "[kp model]   basis:" not in out
