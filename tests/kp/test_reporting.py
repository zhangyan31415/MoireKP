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
    assert "\033[2mmodel\033[0m" in out
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


def test_kp_reporter_renders_status_and_aligned_table_in_plain_output() -> None:
    from kp.reporting import KpReporter

    stream = io.StringIO()
    reporter = KpReporter("kp project", stream=stream)

    reporter.status("Selected low-energy basis", "PASS")
    reporter.table(
        ("Layer", "Band", "Spin", "Dominant orbitals"),
        (
            ("L1", "22", "up 100%", "Mo_dxy 37.2% · Mo_dx2-y2 36.1%"),
            ("L2", "22", "up 100%", "Mo_dxy 36.8% · Mo_dx2-y2 36.5%"),
        ),
        right_align=(1,),
    )

    assert stream.getvalue() == (
        "[kp project] Selected low-energy basis  PASS\n"
        "  Layer  Band  Spin     Dominant orbitals\n"
        "  L1       22  up 100%  Mo_dxy 37.2% · Mo_dx2-y2 36.1%\n"
        "  L2       22  up 100%  Mo_dxy 36.8% · Mo_dx2-y2 36.5%\n"
    )


def test_kp_reporter_colors_status_and_selected_table_columns_only_on_tty(monkeypatch) -> None:
    from kp.reporting import KpReporter

    monkeypatch.delenv("NO_COLOR", raising=False)
    stream = _TtyBuffer()
    reporter = KpReporter("kp project", stream=stream)

    reporter.status("Selected low-energy basis", "WARN")
    reporter.table(
        ("Layer", "Band", "Spin", "Dominant orbitals"),
        (("L1", "22", "up 100%", "Mo_dxy 37.2%"),),
        right_align=(1,),
        column_styles={1: "1;36", 3: "1"},
    )

    output = stream.getvalue()
    assert "\033[33mWARN\033[0m" in output
    assert "\033[1;36m22\033[0m" in output
    assert "\033[1mMo_dxy 37.2%\033[0m" in output


def test_kp_reporter_right_aligns_the_last_table_column() -> None:
    from kp.reporting import KpReporter

    stream = io.StringIO()
    reporter = KpReporter("kp model", stream=stream)

    reporter.table(
        ("Family", "Terms", "Components"),
        (("A", 1, 2), ("longer", 300, 600)),
        right_align=(1, 2),
    )

    assert stream.getvalue() == (
        "  Family  Terms  Components\n"
        "  A           1           2\n"
        "  longer    300         600\n"
    )
