import pytest


def test_unified_tapw_help_lists_subcommands(capsys):
    from tapw import cli

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--help"])

    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "init" in out
    assert "run" in out
    assert "symm" in out
    assert "symm-rep" in out
    assert "plot" in out
    assert "topo" in out
    assert "orbital" in out
    assert "fatband" in out
    assert "chern" not in out
    assert "final" not in out
    assert "postprocess-memmap" not in out
    assert "chern-post" not in out
    assert "\n    calc" not in out
    assert "\n    symmetry" not in out


def test_unified_tapw_dispatches_to_tool_mains(monkeypatch):
    from tapw import cli
    from tapw import config_generator
    from tapw import plot_band_01
    from tapw import symm_rep

    calls = []

    monkeypatch.setattr(config_generator, "main", lambda argv=None, **_kwargs: calls.append(("init", argv)))
    monkeypatch.setattr(plot_band_01, "main", lambda argv=None, **_kwargs: calls.append(("plot", argv)))
    monkeypatch.setattr(symm_rep, "main", lambda argv=None, **_kwargs: calls.append(("symm-rep", argv)))
    monkeypatch.setattr(cli, "run_calc", lambda args: calls.append(("topo", args.config, args.mode)) or 0)
    monkeypatch.setattr(cli, "finish_calculation_process", lambda code: code)

    cli.main(["init", "-o", "cfg"])
    cli.main(["plot", "--config", "bands.yaml"])
    cli.main(["symm-rep", "--band-dir", "band", "--symmetry-dir", "symmetry", "--output-dir", "rep"])
    cli.main(["topo", "--config", "config.yaml"])

    assert calls == [
        ("init", ["-o", "cfg"]),
        ("plot", ["--config", "bands.yaml"]),
        ("symm-rep", ["--band-dir", "band", "--symmetry-dir", "symmetry", "--output-dir", "rep"]),
        ("topo", "config.yaml", "chern"),
    ]


def test_unified_tapw_dispatches_calc_and_finalizes_at_cli_boundary(monkeypatch):
    from tapw import cli

    calls = []
    monkeypatch.setattr(cli, "run_calc", lambda args: calls.append(("run", args.config)) or 0)

    monkeypatch.setattr(
        cli,
        "finish_calculation_process",
        lambda code: calls.append(("finish", code)) or code,
    )

    code = cli.main(["run", "--config", "config.yaml"])

    assert code == 0
    assert calls == [("run", "config.yaml"), ("finish", 0)]


def test_tapw_short_symm_alias_dispatches_fixed_symmetry_mode(monkeypatch):
    from tapw import cli

    calls = []

    def fake_run_calc(args):
        calls.append((args.config, args.mode))
        return 0

    monkeypatch.setattr(cli, "run_calc", fake_run_calc)
    monkeypatch.setattr(cli, "finish_calculation_process", lambda code: code)

    code = cli.main(["symm", "-c", "tapw.yaml"])

    assert code == 0
    assert calls == [("tapw.yaml", "symmetry")]


def test_tapw_removed_top_level_commands_are_rejected():
    from tapw import cli

    for argv in (
        ["chern", "-c", "config.yaml"],
        ["calc", "-c", "config.yaml"],
        ["symmetry", "-c", "config.yaml"],
        ["chern-post", "-c", "config.yaml"],
        ["final", "--root-dir", "outputs/K1/q06"],
        ["postprocess-memmap", "--root-dir", "outputs/K1/q06"],
    ):
        with pytest.raises(SystemExit):
            cli.main(argv)


def test_unified_tapw_treats_none_calc_result_as_success(monkeypatch):
    from tapw import cli

    calls = []
    monkeypatch.setattr(cli, "run_calc", lambda args: calls.append(("run", args.config)) or None)

    monkeypatch.setattr(
        cli,
        "finish_calculation_process",
        lambda code: calls.append(("finish", code)) or code,
    )

    code = cli.main(["run", "--config", "config.yaml"])

    assert code == 0
    assert calls == [("run", "config.yaml"), ("finish", 0)]


def test_exit_cli_raises_system_exit_without_os_exit(monkeypatch):
    from tapw import cli

    calls = []
    monkeypatch.setattr(cli, "shutdown_parallel_runtime", lambda **kwargs: calls.append(("shutdown", kwargs)))
    monkeypatch.setattr(cli.logging, "shutdown", lambda: calls.append("logging"))
    monkeypatch.setattr(
        cli.os,
        "_exit",
        lambda code: (_ for _ in ()).throw(AssertionError("os._exit should not be used")),
    )

    with pytest.raises(SystemExit) as excinfo:
        cli.exit_cli(7)

    assert excinfo.value.code == 7
    assert calls == [("shutdown", {}), "logging"]


def test_shutdown_parallel_runtime_terminates_existing_memmapping_executor(monkeypatch):
    from tapw import cli
    import joblib.externals.loky.reusable_executor as reusable_executor

    calls = []

    class Executor:
        def terminate(self, *, kill_workers=False):
            calls.append(("terminate", kill_workers))

    monkeypatch.setattr(reusable_executor, "_executor", Executor())
    monkeypatch.setattr(reusable_executor, "_executor_kwargs", {"old": "kwargs"})

    cli.shutdown_parallel_runtime(kill_workers=True)

    assert calls == [("terminate", True)]
    assert reusable_executor._executor is None
    assert reusable_executor._executor_kwargs is None


def test_finish_calculation_process_shutdowns_runtime_and_returns_status(monkeypatch):
    from tapw import cli

    calls = []

    monkeypatch.setattr(cli, "shutdown_parallel_runtime", lambda **kwargs: calls.append(("shutdown", kwargs)))
    monkeypatch.setattr(cli.logging, "shutdown", lambda: calls.append("logging"))
    monkeypatch.setattr(cli.sys.stdout, "flush", lambda: calls.append("stdout"))
    monkeypatch.setattr(cli.sys.stderr, "flush", lambda: calls.append("stderr"))

    code = cli.finish_calculation_process(5)

    assert code == 5
    assert calls == [("shutdown", {"wait": False, "kill_workers": True}), "logging", "stdout", "stderr"]


def test_finish_calculation_process_treats_none_as_success(monkeypatch):
    from tapw import cli

    calls = []

    monkeypatch.setattr(cli, "shutdown_parallel_runtime", lambda **kwargs: calls.append(("shutdown", kwargs)))
    monkeypatch.setattr(cli.logging, "shutdown", lambda: calls.append("logging"))
    monkeypatch.setattr(cli.sys.stdout, "flush", lambda: calls.append("stdout"))
    monkeypatch.setattr(cli.sys.stderr, "flush", lambda: calls.append("stderr"))

    code = cli.finish_calculation_process(None)

    assert code == 0
    assert calls == [("shutdown", {"wait": False, "kill_workers": True}), "logging", "stdout", "stderr"]


def test_tapw_run_help_is_single_config_surface(capsys):
    from tapw import cli

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["run", "--help"])

    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "-c CONFIG" in out
    assert "--developer-outputs" not in out
    assert "--valleys" not in out
    assert "--n_g" not in out
    assert "--num_chern" not in out


def test_tapw_symm_rep_help_lists_release_config_inputs(capsys):
    from tapw import cli

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["symm-rep", "--help"])

    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "-c CONFIG" in out
    assert "--valence-count" not in out
    assert "--conduction-count" not in out
    assert "--degeneracy-tol" in out
    assert "--band-dir" not in out
    assert "--symmetry-dir" not in out
    assert "--output-dir" not in out
    assert "--hamiltonian-index" not in out
    assert "--points" not in out


@pytest.mark.parametrize("command", ["plot", "topo"])
def test_tapw_plot_and_topo_help_list_short_config_flag(command, capsys):
    from tapw import cli

    with pytest.raises(SystemExit) as excinfo:
        cli.main([command, "--help"])

    assert excinfo.value.code == 0
    assert "-c CONFIG" in capsys.readouterr().out


def test_unified_tapw_dispatches_orbital_commands(monkeypatch):
    from tapw import cli
    from tapw import orbital_analysis_tool
    from tapw import plot_orbital_tool

    calls = []
    monkeypatch.setattr(orbital_analysis_tool, "main", lambda argv=None, **_kwargs: calls.append(("orbital", argv)))
    monkeypatch.setattr(plot_orbital_tool, "main", lambda argv=None, **_kwargs: calls.append(("fatband", argv)))

    cli.main(["orbital", ".", "--config", "config.yaml"])
    cli.main(["fatband", ".", "--valley", "Gamma"])

    assert calls == [
        ("orbital", [".", "--config", "config.yaml"]),
        ("fatband", [".", "--valley", "Gamma"]),
    ]


def test_postprocess_memmap_is_not_a_public_tapw_command():
    from tapw import cli

    with pytest.raises(SystemExit):
        cli.main(["postprocess-memmap", "--root-dir", "outputs/K1/q06"])


def test_tapw_orbital_exits_nonzero_when_required_inputs_are_missing(tmp_path):
    from tapw import orbital_analysis_tool

    config_path = tmp_path / "config.yaml"
    config_path.write_text("paths: {}\ncompute: {}\n", encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        orbital_analysis_tool.main([str(tmp_path), "--config", str(config_path), "--quiet"])

    assert excinfo.value.code == 1


def test_legacy_tapw_subcommands_are_rejected():
    from tapw import cli

    for argv in (
        ["config", "-o", "cfg"],
        ["calc", "--config", "config.yaml"],
        ["chern-post", "--config", "config.yaml", "-b", "-1"],
        ["orbital", "analyze", ".", "--config", "config.yaml"],
        ["orbital", "plot", ".", "--valley", "Gamma"],
    ):
        with pytest.raises(SystemExit):
            cli.main(argv)


def test_legacy_console_script_names_are_rejected(monkeypatch):
    import sys

    from tapw import cli

    for alias in (
        "tapw-calc",
        "tapw-config",
        "tapw-plot",
        "tapw-chernpost",
        "tapw-orbital",
        "tapw-plot-orbital",
    ):
        monkeypatch.setattr(sys, "argv", [alias, "--help"])
        with pytest.raises(SystemExit):
            cli.main()


@pytest.mark.parametrize(
    "argv",
    [
        ["-b", "-1", "-2"],
        ["-b", "-1,-2"],
        ["-wb", "-1", "-2"],
        ["-wb", "-1,-2"],
    ],
)
def test_chern_post_parser_accepts_negative_band_lists(argv):
    from tapw import chern_post

    parser = chern_post.build_parser()
    args = parser.parse_args(argv)

    assert (args.band or args.wcc_bands) == [-1, -2]


def test_pyproject_defines_unified_tapw_script():
    import tomllib
    from pathlib import Path

    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["scripts"]["tapw"] == "tapw.cli:main"
    assert pyproject["project"]["scripts"] == {
        "kp": "kp.cli:main",
        "tapw": "tapw.cli:main",
    }
