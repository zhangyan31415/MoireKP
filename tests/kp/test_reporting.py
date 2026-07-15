from __future__ import annotations

import io


class _TtyBuffer(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_kp_reporter_renders_plain_scientific_summary() -> None:
    from kp.reporting import KpReporter

    stream = io.StringIO()
    reporter = KpReporter("kp project", stream=stream)

    reporter.title("Project effective Hamiltonian")
    reporter.fields(
        [
            ("config", "case.yaml"),
            ("material", "MoTe2"),
        ]
    )
    reporter.stage(2, 4, "Build projectors")
    reporter.check("gauge", passed=True, detail="symmetry-adapted")
    reporter.section("Results")
    reporter.path("Hamiltonian", "outputs/heff.npy")
    reporter.warning("Schur margin needs attention")
    reporter.complete(elapsed=1.25)

    assert stream.getvalue() == (
        "[kp project] Project effective Hamiltonian\n"
        "  config    case.yaml\n"
        "  material  MoTe2\n"
        "\n"
        "[kp project] [2/4] Build projectors\n"
        "  gauge  OK  symmetry-adapted\n"
        "\n"
        "[kp project] Results\n"
        "  Hamiltonian  outputs/heff.npy\n"
        "  WARN  Schur margin needs attention\n"
        "\n"
        "[kp project] OK  Completed in 1.25 s\n"
    )
    assert "\033[" not in stream.getvalue()


def test_kp_reporter_uses_color_only_for_tty(monkeypatch) -> None:
    from kp.reporting import KpReporter

    monkeypatch.delenv("NO_COLOR", raising=False)
    stream = _TtyBuffer()
    reporter = KpReporter("kp model", stream=stream)

    reporter.title("Build continuum model")
    reporter.path("output", "model")
    reporter.warning("check residual")
    reporter.complete()

    out = stream.getvalue()
    assert "\033[" in out
    assert "[kp model]" in out
    assert "Build continuum model" in out
    assert "model" in out
    assert "WARN" in out
    assert "OK" in out


def test_kp_reporter_honors_no_color(monkeypatch) -> None:
    from kp.reporting import KpReporter

    monkeypatch.setenv("NO_COLOR", "1")
    stream = _TtyBuffer()
    reporter = KpReporter("kp symm", stream=stream)

    reporter.title("Project symmetry")
    reporter.complete()

    assert "\033[" not in stream.getvalue()
