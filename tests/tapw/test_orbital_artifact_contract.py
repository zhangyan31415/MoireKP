from types import SimpleNamespace
import numpy as np
import pytest
from tapw.orbital_analysis_tool import OrbitalAnalyzer, OrbitalAnalysisConfig


def _analyzer(tmp_path):
    return OrbitalAnalyzer(OrbitalAnalysisConfig(config_file=str(tmp_path/'config.yaml'),
        result_dir=str(tmp_path), output_dir=str(tmp_path/'orbital'), valley='K1', verbose=False))


def test_orbital_analyzer_uses_current_writer_artifacts(tmp_path):
    band = tmp_path/'band'
    band.mkdir()
    np.savetxt(band/'energies_cbm.txt', np.zeros((2, 1)))
    np.save(band/'wavefunctions_cbm.npy', np.ones((2, 2, 1)))
    analyzer = _analyzer(tmp_path)
    actual = analyzer.detect_valleys_and_files()
    assert actual == {'K1': {'band_file': str(band/'energies_cbm.txt'),
                            'vec_file': str(band/'wavefunctions_cbm.npy'), 'valley_flag': 'K1'}}


def test_orbital_analyzer_requires_saved_g_vectors(tmp_path):
    analyzer = _analyzer(tmp_path)
    with pytest.raises(FileNotFoundError, match='g_vectors_group1'):
        analyzer._get_gvector_count(4, 'K1')


def test_saved_g_vector_count_is_not_inferred_from_wavefunction_dimensions(tmp_path):
    band = tmp_path/'band'
    band.mkdir()
    for group in (1, 2):
        np.save(band/f'g_vectors_group{group}.npy', np.zeros((3, 2)))
    assert _analyzer(tmp_path)._get_gvector_count(100, 'K1') == 3
