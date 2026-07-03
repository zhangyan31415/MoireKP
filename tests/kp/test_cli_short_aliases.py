from __future__ import annotations

import pytest


def test_kp_help_lists_only_release_facing_commands(capsys):
    from kp import cli

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--help"])

    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "{inspect,project,symm,model}" in out
    for command in ("inspect", "project", "symm", "model"):
        assert command in out
    for removed in ("{show", "plot,", "{proj", " sweep", " fit", " export"):
        assert removed not in out


@pytest.mark.parametrize(
    "argv",
    [
        ["show", "-c", "case.yaml"],
        ["plot", "-c", "case.yaml"],
        ["proj", "-c", "case.yaml"],
        ["fit", "-c", "case.yaml"],
        ["export", "-c", "case.yaml", "-o", "portable"],
        ["model", "export-standalone", "model", "portable"],
        ["model", "-c", "case.yaml", "--export-standalone", "portable"],
    ],
)
def test_removed_kp_commands_are_rejected(argv):
    from kp import cli

    with pytest.raises(SystemExit):
        cli.main(argv)
