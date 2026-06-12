import pytest


def test_unified_tapw_help_lists_subcommands(capsys):
    from tapw import cli

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--help"])

    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "config" in out
    assert "calc" in out
    assert "plot" in out
    assert "chern-post" in out
    assert "orbital" in out


def test_unified_tapw_dispatches_to_tool_mains(monkeypatch):
    from tapw import cli
    from tapw import chern_post
    from tapw import config_generator
    from tapw import plot_band_01

    calls = []

    monkeypatch.setattr(config_generator, "main", lambda argv=None, **_kwargs: calls.append(("config", argv)))
    monkeypatch.setattr(plot_band_01, "main", lambda argv=None, **_kwargs: calls.append(("plot", argv)))
    monkeypatch.setattr(chern_post, "main", lambda argv=None, **_kwargs: calls.append(("chern-post", argv)))

    cli.main(["config", "-o", "cfg"])
    cli.main(["plot", "--config", "bands.yaml"])
    cli.main(["chern-post", "--config", "config.yaml", "-b", "-1", "-2"])

    assert calls == [
        ("config", ["-o", "cfg"]),
        ("plot", ["--config", "bands.yaml"]),
        ("chern-post", ["--config", "config.yaml", "-b", "-1", "-2"]),
    ]


def test_unified_tapw_dispatches_calc_without_rewriting_algorithm(monkeypatch):
    from tapw import cli

    calls = []
    monkeypatch.setattr(cli, "run_calc", lambda args: calls.append(args.config))

    cli.main(["calc", "--config", "config.yaml"])

    assert calls == ["config.yaml"]


def test_unified_tapw_dispatches_orbital_subcommands(monkeypatch):
    from tapw import cli
    from tapw import orbital_analysis_tool
    from tapw import plot_orbital_tool

    calls = []
    monkeypatch.setattr(orbital_analysis_tool, "main", lambda argv=None, **_kwargs: calls.append(("analyze", argv)))
    monkeypatch.setattr(plot_orbital_tool, "main", lambda argv=None, **_kwargs: calls.append(("plot", argv)))

    cli.main(["orbital", "analyze", ".", "--config", "config.yaml"])
    cli.main(["orbital", "plot", ".", "--valley", "Gamma"])

    assert calls == [
        ("analyze", [".", "--config", "config.yaml"]),
        ("plot", [".", "--valley", "Gamma"]),
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
    monkeypatch.setattr(cli, "main_calc", lambda argv=None: calls.append(("calc", argv)))
    monkeypatch.setattr(config_generator, "main", lambda argv=None: calls.append(("config", argv)))
    monkeypatch.setattr(plot_band_01, "main", lambda argv=None: calls.append(("plot", argv)))
    monkeypatch.setattr(chern_post, "main", lambda argv=None: calls.append(("chern-post", argv)))
    monkeypatch.setattr(orbital_analysis_tool, "main", lambda argv=None: calls.append(("orbital", argv)))
    monkeypatch.setattr(plot_orbital_tool, "main", lambda argv=None: calls.append(("plot-orbital", argv)))

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
