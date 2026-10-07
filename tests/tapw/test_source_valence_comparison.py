"""Independent non-Gamma generalized-band reconstruction for DFT H/S."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
from scipy import linalg, sparse


SCRIPT = Path(__file__).resolve().parents[2] / 'examples/dft_backends/compare_dft_tapw_valence.py'


def _tool():
    spec = spec_from_file_location('compare_dft_tapw_valence', SCRIPT)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_non_gamma_bloch_sign_and_generalized_valence_window():
    tool = _tool()
    h0 = sparse.diags([-1.0, 2.0], format='csr', dtype=complex)
    plus = sparse.csr_matrix(([0.2+0.1j], ([0], [1])), shape=(2, 2))
    minus = plus.conj().T.tocsr()
    s0 = sparse.eye(2, format='csr', dtype=complex)
    splus = sparse.diags([0.05, 0.1], format='csr', dtype=complex)
    h = tool.assemble_bloch({(0, 0, 0): h0, (1, 0, 0): plus, (-1, 0, 0): minus},
                            (0.25, 0.0, 0.0))
    s = tool.assemble_bloch({(0, 0, 0): s0, (1, 0, 0): splus, (-1, 0, 0): splus},
                            (0.25, 0.0, 0.0))
    expected_h = np.array([[-1., -0.1+0.2j], [-0.1-0.2j, 2.]])
    np.testing.assert_allclose(h.toarray(), expected_h, atol=1e-14)
    np.testing.assert_allclose(s.toarray(), np.eye(2), atol=1e-14)
    values = tool.solve_gap_window(h, s, occupied_count=1, valence_bands=1)
    np.testing.assert_allclose(values, linalg.eigh(expected_h, eigvals_only=True), atol=1e-14)
