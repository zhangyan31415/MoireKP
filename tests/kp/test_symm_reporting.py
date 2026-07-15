from __future__ import annotations

from pathlib import Path

import yaml


def test_kp_symm_reports_production_and_developer_outputs_separately(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    from kp import cli

    cfg_path = tmp_path / "case.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"symm": {"output_dir": "outputs/symmetry"}}),
        encoding="utf-8",
    )
    calls: list[tuple[str, bool | None]] = []

    def fake_run(config: str, *, developer_outputs: bool | None = None):
        calls.append((config, developer_outputs))
        return {
            "valley": "K1",
            "spin": "up",
            "low_dim": 4,
            "q_count": 2,
            "operations": [
                {
                    "name": "C3z",
                    "status": "exactified",
                    "matrix_file": "exactified_C3z.npy",
                    "matrix_kind": "continuum_internal_rep_exact",
                    "developer_outputs": {
                        "polar_matrix_file": "diagnostics/C3z_low_polar.npy",
                    },
                },
                {
                    "name": "C2T",
                    "status": "exactified",
                    "matrix_file": "exactified_C2T.npy",
                    "matrix_kind": "continuum_internal_rep_exact",
                },
            ],
        }

    monkeypatch.setattr(cli, "run_symmetry_projection_from_config", fake_run)

    cli.main(["symm", "-c", str(cfg_path), "--developer-outputs"])

    out = capsys.readouterr().out
    assert calls == [(str(cfg_path), True)]
    assert "[kp symm] Project and exactify symmetry representations" in out
    assert "[kp symm] Validation" in out
    assert "  C3z  OK  production exactified matrix: exactified_C3z.npy" in out
    assert "  C2T  OK  production exactified matrix: exactified_C2T.npy" in out
    assert "[kp symm] Developer diagnostics" in out
    assert "diagnostics/C3z_low_polar.npy" in out
    assert "[kp symm] Results" in out
    assert str(tmp_path / "outputs" / "symmetry" / "manifest.json") in out
    assert "[kp symm] OK  Completed in" in out
    for forbidden in ("C2x", "C2y", "C2yT", "mirror_x", "mirror_y"):
        assert forbidden not in out


def test_kp_symm_result_path_uses_canonical_case_default(tmp_path: Path) -> None:
    from kp import cli

    cfg_path = tmp_path / "case.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "case": {
                    "profile": "K1",
                    "q_shell": "q06",
                    "output_root": "outputs",
                },
                "symm": {},
            }
        ),
        encoding="utf-8",
    )

    assert cli._kp_symm_output_dir(cfg_path) == (
        tmp_path / "outputs" / "K1" / "q06" / "symmetry"
    ).resolve()
