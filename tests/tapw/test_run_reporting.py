from types import SimpleNamespace

import logging
import os
import numpy as np
import pandas as pd
import pytest


class _ListLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(str(message))


def test_reporter_default_hides_details_but_verbose_shows_them():
    from tapw.reporting import TapwReporter

    default_logger = _ListLogger()
    default = TapwReporter(default_logger)
    default.detail("atom_type_list", np.array([0, 1, 2]), purpose="internal atom-type packing")
    default.kv("Projected TAPW basis dimension", 1278)
    default.check("Projection normalization check", passed=True, detail="max deviation 0.000e+00")

    default_output = "\n".join(default_logger.messages)
    assert "atom_type_list" not in default_output
    assert "Projected TAPW basis dimension: 1278" in default_output
    assert "Projection normalization check: PASS" in default_output
    assert "max deviation 0.000e+00" in default_output

    verbose_logger = _ListLogger()
    verbose = TapwReporter(verbose_logger, verbose=True)
    verbose.detail("atom_type_list", np.array([0, 1, 2]), purpose="internal atom-type packing")

    verbose_output = "\n".join(verbose_logger.messages)
    assert "atom_type_list:" in verbose_output
    assert "Purpose: internal atom-type packing" in verbose_output


def test_reporter_stage_and_step_make_run_flow_clear():
    from tapw.reporting import TapwReporter

    logger = _ListLogger()
    reporter = TapwReporter(logger)

    reporter.stage("Preprocessing", "Prepare structure and TAPW basis inputs.")
    reporter.step("Layer clustering", "3 physical layers, 2 source orientation groups")

    assert logger.messages == [
        "[TAPW] Preprocessing",
        "  Prepare structure and TAPW basis inputs.",
        "  - Layer clustering: 3 physical layers, 2 source orientation groups",
    ]


def test_reporter_formats_sections_key_values_and_arrays():
    from tapw.reporting import TapwReporter

    logger = _ListLogger()
    reporter = TapwReporter(logger)

    reporter.section("Structure")
    reporter.kv("Twist index", 6)
    reporter.array("Moire lattice vectors (Angstrom)", np.array([[1.0, 0.0], [0.5, 0.866025]]))

    assert logger.messages[0] == "[TAPW] Structure"
    assert logger.messages[1] == "  Twist index: 6"
    assert logger.messages[2] == "  Moire lattice vectors (Angstrom):"
    assert logger.messages[3].startswith("    [[1.")
    assert all("===" not in message for message in logger.messages)


def test_reporter_indents_multiline_key_values():
    from tapw.reporting import TapwReporter

    logger = _ListLogger()
    reporter = TapwReporter(logger)

    reporter.kv("atom_orb_name_list", np.array(["Te7.0-s3p2d2f1", "Mo7.0-s3p2d2", "Te7.0-s3p2d2f1"]))

    assert logger.messages[0] == "  atom_orb_name_list:"
    assert logger.messages[1].startswith("    [")
    assert all(message.startswith("    ") for message in logger.messages[1:])


def test_setup_logging_uses_clean_console_output(capsys, tmp_path):
    from tapw import cli

    logger = cli.setup_logging(str(tmp_path / "run.log"))
    logger.info("[TAPW] Run")

    out = capsys.readouterr().out
    assert out == "[TAPW] Run\n"
    assert "| INFO |" not in out
    assert not out.startswith("20")

    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    logging.shutdown()


def test_tapw_run_help_lists_verbose(capsys):
    from tapw import cli
    import pytest

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["run", "--help"])

    assert excinfo.value.code == 0
    assert "--verbose" in capsys.readouterr().out


def test_openmx_display_properties_uses_english_reporter_output():
    from tapw.io.structure import OpenMXFile
    from tapw.reporting import TapwReporter

    logger = _ListLogger()
    reporter = TapwReporter(logger)
    structure = OpenMXFile.__new__(OpenMXFile)
    structure.bravais = "hex"
    structure.twist_angle = 5.085848
    structure.Tmat = np.array([[39.700745, 0.0, 0.0], [-19.850372, 34.381854, 0.0], [0.0, 0.0, 50.0]])
    structure.reciprocal_Tmat = np.array([[0.158264, 0.091374, 0.0], [0.0, 0.182747, 0.0], [0.0, 0.0, 0.125664]])
    structure.atoms_number = 1143
    structure.species_coordinates_unit = "Ang"
    structure.species_count = {"Te": 762, "Mo": 381}
    structure.orbitals_count = {"Te": 26, "Mo": 19}
    structure.atom_basis = {"Te": "Te7.0-s3p2d2f1", "Mo": "Mo7.0-s3p2d2"}

    structure.display_properties(reporter=reporter)
    output = "\n".join(logger.messages)

    assert "[TAPW] Structure" in output
    assert "Twist angle from twist_index_m (deg): 5.085848" in output
    assert "Moire lattice vectors (Angstrom):" in output
    assert "扭转角度" not in output
    assert "===" not in output


def test_structure_processor_summary_uses_reporter_lines():
    from tapw.io.structure import StructureProcessorSpglib
    from tapw.reporting import TapwReporter

    logger = _ListLogger()
    reporter = TapwReporter(logger)
    processor = StructureProcessorSpglib.__new__(StructureProcessorSpglib)
    processor.reporter = reporter
    processor.df = pd.DataFrame(
        {
            "layer": [0, 0, 0, 1],
            "sublayer": [0, 0, 1, 0],
            "atom_type": [0, 0, 1, 2],
            "species": ["Te", "Te", "Mo", "Te"],
        }
    )

    processor.print_summary()
    output = "\n".join(logger.messages)

    assert "[TAPW] Layer and atom-type summary" in output
    assert "Final atom-type grouping used by the TAPW projector." in output
    assert "  Total layers: 2" in output
    assert "  Layer 0: 2 sublayers" in output
    assert "    Atom type 0 (Te): 2 atoms" in output
    assert "--- Clustering Summary ---" not in output


def test_spglib_stderr_incomplete_cleaning_is_hard_error():
    from tapw.io.structure import StructureProcessorSpglib

    def noisy_spglib_call():
        os.write(2, b"spglib: Primitive lattice cleaning is incomplete\n")
        return object()

    with pytest.raises(RuntimeError, match="rigid/reference OpenMX input"):
        StructureProcessorSpglib._call_spglib_checked(
            noisy_spglib_call,
            "Twist group 0: spglib.standardize_cell",
        )


def test_primitive_basis_count_rejects_relaxed_degenerate_assignment():
    from tapw.io.structure import StructureProcessorSpglib

    processor = StructureProcessorSpglib.__new__(StructureProcessorSpglib)
    processor.twist_index = 8
    processor.Tmat = np.array(
        [
            [60.95086595, 0.0, 0.0],
            [-30.47543298, 52.78499830, 0.0],
            [0.0, 0.0, 50.0],
        ]
    )
    group_df = pd.DataFrame({"species": ["Mg"] * 651})
    mapping = np.arange(651)

    with pytest.raises(RuntimeError, match="one-to-one"):
        processor._validate_primitive_basis_count(0, group_df, mapping)


def test_generate_g_vec_list_reports_summary_by_default_and_arrays_in_verbose():
    import tapw.workflows.band as band_workflow
    from tapw.reporting import TapwReporter

    logger = _ListLogger()
    reporter = TapwReporter(logger)
    config = SimpleNamespace(
        n_g=1,
        valley=5,
        bravais="hex",
        Electric_field_in_eVpA=None,
        Inner_symmetrical_Electric_Field=False,
        symmetrize_hamiltonian=False,
    )
    structure = SimpleNamespace(
        reciprocal_Tmat=np.array([[1.0, 0.0, 0.0], [0.5, 0.8660254, 0.0], [0.0, 0.0, 1.0]]),
        twist_index=1,
        df=pd.DataFrame(),
    )
    params = band_workflow.TAPW_parameters(structure=structure, config=config, reporter=reporter)

    params.generate_g_vec_list()
    output = "\n".join(logger.messages)

    assert "Valley center setup" in output
    assert "G-vector count: K1 set=" in output
    assert "m_g_unitvec_1:" not in output
    assert "K1 G-vectors:" not in output
    assert "======================" not in output

    verbose_logger = _ListLogger()
    verbose = TapwReporter(verbose_logger, verbose=True)
    params = band_workflow.TAPW_parameters(structure=structure, config=config, reporter=verbose)
    params.generate_g_vec_list()
    verbose_output = "\n".join(verbose_logger.messages)

    assert "m_g_unitvec_1:" in verbose_output
    assert "K1 G-vectors:" in verbose_output
    assert "Purpose: reciprocal vectors retained for the first source orientation group" in verbose_output


def test_generate_gr_matrix_reports_user_summary_by_default_and_details_in_verbose():
    import tapw.workflows.band as band_workflow
    from tapw.reporting import TapwReporter

    structure_df = pd.DataFrame(
        {
            "atom_type": [0, 1],
            "orb_num": [1, 1],
            "twist_group": [0, 1],
            "shifted_x": [0.0, 0.5],
            "shifted_y": [0.0, 0.5],
            "orb_name": ["A-s1", "B-s1"],
            "phys_layer": [0, 1],
        }
    )
    structure = SimpleNamespace(df=structure_df, spin=False)

    logger = _ListLogger()
    reporter = TapwReporter(logger)
    params = band_workflow.TAPW_parameters.__new__(band_workflow.TAPW_parameters)
    params.structure = structure
    params.reporter = reporter
    params.g_vec_list_K1 = np.array([[0.0, 0.0]])
    params.g_vec_list_K2 = np.array([[0.0, 0.0]])

    params.generate_gr_matrix()
    output = "\n".join(logger.messages)

    assert "Projected TAPW basis dimension: 2" in output
    assert "Source orbital basis dimension: 2" in output
    assert "Projection normalization check: PASS" in output
    assert "atom_type_list" not in output
    assert "factor_list" not in output
    assert "dim_gr_1" not in output

    verbose_logger = _ListLogger()
    verbose_params = band_workflow.TAPW_parameters.__new__(band_workflow.TAPW_parameters)
    verbose_params.structure = structure
    verbose_params.reporter = TapwReporter(verbose_logger, verbose=True)
    verbose_params.g_vec_list_K1 = np.array([[0.0, 0.0]])
    verbose_params.g_vec_list_K2 = np.array([[0.0, 0.0]])

    verbose_params.generate_gr_matrix()
    verbose_output = "\n".join(verbose_logger.messages)

    assert "atom_type_list:" in verbose_output
    assert "factor_list:" in verbose_output
