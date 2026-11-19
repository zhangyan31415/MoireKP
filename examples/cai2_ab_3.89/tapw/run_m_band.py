"""Use the existing TAPW M-triplet C3 averaging for the archived convention."""
import sys
from tapw import cli
from tapw.workflows.band import uses_m_valley_threefold_symmetrization

Calculator = cli.BandStructureCalculator


class ArchivedMCalculator(Calculator):
    def __init__(self, *args, **kwargs):
        config = kwargs["config"]
        assert config.valleys == [31] and config.valley == 31
        config.symmetrize_hamiltonian = ["C3z"]
        assert uses_m_valley_threefold_symmetrization(config)
        super().__init__(*args, **kwargs)


cli.BandStructureCalculator = ArchivedMCalculator
print("TAPW M1 uses threefold M-valley averaging, matching the source archive.", flush=True)
raise SystemExit(cli.main_calc(sys.argv[1:]))
