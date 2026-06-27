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
    assert "plot" in out
    assert "topo" in out
    assert "final" in out
    assert "postprocess-memmap" in out
    assert "orbital" in out
    assert "fatband" in out
    assert "chern-post" not in out


def test_unified_tapw_dispatches_to_tool_mains(monkeypatch):
    from tapw import cli
    from tapw import chern_post
    from tapw import config_generator
    from tapw import plot_band_01

    calls = []

    monkeypatch.setattr(config_generator, "main", lambda argv=None, **_kwargs: calls.append(("init", argv)))
    monkeypatch.setattr(plot_band_01, "main", lambda argv=None, **_kwargs: calls.append(("plot", argv)))
    monkeypatch.setattr(chern_post, "main", lambda argv=None, **_kwargs: calls.append(("topo", argv)))

    cli.main(["init", "-o", "cfg"])
    cli.main(["plot", "--config", "bands.yaml"])
    cli.main(["topo", "--config", "config.yaml", "-b", "-1", "-2"])

    assert calls == [
        ("init", ["-o", "cfg"]),
        ("plot", ["--config", "bands.yaml"]),
        ("topo", ["--config", "config.yaml", "-b", "-1", "-2"]),
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


def test_tapw_short_final_alias_dispatches_memmap_postprocess(monkeypatch):
    from tapw import cli
    from tapw import postprocess_memmap

    calls = []
    monkeypatch.setattr(postprocess_memmap, "main", lambda argv=None, **_kwargs: calls.append(argv))

    cli.main(["final", "--root-dir", "Q_shell_6", "--mode", "band"])

    assert calls == [["--root-dir", "Q_shell_6", "--mode", "band"]]


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


def test_tapw_run_help_lists_developer_outputs(capsys):
    from tapw import cli

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["run", "--help"])

    assert excinfo.value.code == 0
    assert "--developer-outputs" in capsys.readouterr().out


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


def test_unified_tapw_dispatches_postprocess_memmap(monkeypatch):
    from tapw import cli
    from tapw import postprocess_memmap

    calls = []
    monkeypatch.setattr(postprocess_memmap, "main", lambda argv=None, **_kwargs: calls.append(argv))

    cli.main(["postprocess-memmap", "--root-dir", "Q_shell_6", "--mode", "band", "--valley", "31", "--efermi", "0"])

    assert calls == [["--root-dir", "Q_shell_6", "--mode", "band", "--valley", "31", "--efermi", "0"]]


def test_tapw_orbital_exits_nonzero_when_required_inputs_are_missing(tmp_path):
    from tapw import orbital_analysis_tool

    config_path = tmp_path / "config.yaml"
    config_path.write_text("paths: {}\ncompute: {}\n", encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        orbital_analysis_tool.main([str(tmp_path), "--config", str(config_path), "--quiet"])

    assert excinfo.value.code == 1


def test_legacy_tapw_subcommands_still_dispatch(monkeypatch):
    from tapw import chern_post
    from tapw import cli
    from tapw import config_generator
    from tapw import orbital_analysis_tool
    from tapw import plot_orbital_tool

    calls = []
    monkeypatch.setattr(cli, "main_calc", lambda argv=None, **_kwargs: calls.append(("calc", argv)))
    monkeypatch.setattr(config_generator, "main", lambda argv=None, **_kwargs: calls.append(("config", argv)))
    monkeypatch.setattr(chern_post, "main", lambda argv=None, **_kwargs: calls.append(("chern-post", argv)))
    monkeypatch.setattr(orbital_analysis_tool, "main", lambda argv=None, **_kwargs: calls.append(("orbital-analyze", argv)))
    monkeypatch.setattr(plot_orbital_tool, "main", lambda argv=None, **_kwargs: calls.append(("orbital-plot", argv)))

    cli.main(["config", "-o", "cfg"])
    cli.main(["calc", "--config", "config.yaml"])
    cli.main(["chern-post", "--config", "config.yaml", "-b", "-1"])
    cli.main(["orbital", "analyze", ".", "--config", "config.yaml"])
    cli.main(["orbital", "plot", ".", "--valley", "Gamma"])

    assert calls == [
        ("config", ["-o", "cfg"]),
        ("calc", ["--config", "config.yaml"]),
        ("chern-post", ["--config", "config.yaml", "-b", "-1"]),
        ("orbital-analyze", [".", "--config", "config.yaml"]),
        ("orbital-plot", [".", "--valley", "Gamma"]),
    ]


def test_legacy_aliases_dispatch_through_unified_main(monkeypatch):
    import sys

    from tapw import chern_post
    from tapw import cli
    from tapw import config_generator
    from tapw import orbital_analysis_tool
    from tapw import plot_band_01
    from tapw import plot_orbital_tool

    calls = []
    monkeypatch.setattr(cli, "main_calc", lambda argv=None, **_kwargs: calls.append(("calc", argv)))
    monkeypatch.setattr(config_generator, "main", lambda argv=None, **_kwargs: calls.append(("config", argv)))
    monkeypatch.setattr(plot_band_01, "main", lambda argv=None, **_kwargs: calls.append(("plot", argv)))
    monkeypatch.setattr(chern_post, "main", lambda argv=None, **_kwargs: calls.append(("chern-post", argv)))
    monkeypatch.setattr(orbital_analysis_tool, "main", lambda argv=None, **_kwargs: calls.append(("orbital", argv)))
    monkeypatch.setattr(plot_orbital_tool, "main", lambda argv=None, **_kwargs: calls.append(("plot-orbital", argv)))

    for alias in [
        "tapw-calc",
        "tapw-config",
        "tapw-plot",
        "tapw-chernpost",
        "tapw-orbital",
        "tapw-plot-orbital",
    ]:
        monkeypatch.setattr(sys, "argv", [alias, "--help"])
        cli.main()

    assert calls == [
        ("calc", ["--help"]),
        ("config", ["--help"]),
        ("plot", ["--help"]),
        ("chern-post", ["--help"]),
        ("orbital", ["--help"]),
        ("plot-orbital", ["--help"]),
    ]


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
