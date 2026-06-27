from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


def test_kp_show_alias_dispatches_plot(monkeypatch):
    from kp import cli

    calls = []
    monkeypatch.setattr(cli, "cmd_plot_from_config", lambda config: calls.append(config))

    cli.main(["show", "-c", "source.yaml"])

    assert calls == ["source.yaml"]


def test_kp_proj_alias_dispatches_project(monkeypatch):
    from kp import cli

    calls = []
    monkeypatch.setattr(cli, "cmd_project_from_config", lambda config, overrides: calls.append((config, overrides)))

    cli.main(["proj", "-c", "source.yaml", "--active-indices", "1,2"])

    assert calls[0][0] == "source.yaml"
    assert calls[0][1]["active_indices"] == "1,2"


def test_kp_fit_alias_dispatches_model_run(monkeypatch):
    from kp import cli
    from kp.model import export as export_mod
    from kp.model import pipeline

    calls = []
    model_cfg = SimpleNamespace(
        path="model.yaml",
        output_dir=Path("outputs/model/case"),
        n_orb=(1, 1),
        fit_indices=[0],
        bM_diagnostics={},
        symmetry_source_metadata={},
    )
    moire_cfg = SimpleNamespace(
        Q_set1=[0],
        Q_set2=[0],
        n_orb1=1,
        n_orb2=1,
        kpoints=[0],
        intra_harmonics_map={},
        inter_harmonics_map={},
    )
    monkeypatch.setattr(
        pipeline,
        "run_configured_model",
        lambda config: calls.append(config) or {"configured_model": model_cfg, "moire_config": moire_cfg},
    )
    monkeypatch.setattr(export_mod, "export_standalone_model", lambda *_args, **_kwargs: Path("standalone"))
    monkeypatch.setattr(cli, "_record_standalone_export", lambda *_args, **_kwargs: None)

    cli.main(["fit", "-c", "model.yaml"])

    assert calls == ["model.yaml"]


def test_kp_export_alias_dispatches_standalone_export(monkeypatch, tmp_path: Path):
    from kp import cli
    from kp.model import export as export_mod

    calls = []

    def fake_export(model_output_dir, output_dir, *, force=False, debug_files=False):
        calls.append((Path(model_output_dir), output_dir, force, debug_files))
        return tmp_path / "portable"

    monkeypatch.setattr(export_mod, "export_standalone_model", fake_export)

    cli.main(["export", "outputs/model/case", "exported/case", "--force"])

    assert calls == [(Path("outputs/model/case"), "exported/case", True, False)]


def test_kp_help_lists_short_aliases(capsys):
    from kp import cli

    try:
        cli.main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0

    out = capsys.readouterr().out
    assert "show" in out
    assert "proj" in out
    assert "fit" in out
    assert "export" in out
