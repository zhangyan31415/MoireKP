"""Shared constants and caches for the continuum model."""

hartree = 27.2113845
CANONICAL_P_TOL = 1.0e-10

# Symmetrization caches shared by ContinuumModelBuilder static methods.
SYMMETRIZE_GLOBAL_CACHE: dict = {}
SYMMETRIZE_MONOMIAL_OP_CACHE: dict = {}
SYMMETRIZE_MONOMIAL_OP_VALIDATED: set = set()
SYMMETRIZE_COMPOSED_OP_CACHE: dict = {}
SYMMETRIZE_COMPOSED_OP_VALIDATED: set = set()
SYMMETRIZE_ORBIT_CACHE: dict = {}
KZ_POW_CACHE: dict = {}


def clear_symmetry_caches() -> None:
    """Clear module-level symmetrization caches."""
    SYMMETRIZE_GLOBAL_CACHE.clear()
    SYMMETRIZE_MONOMIAL_OP_CACHE.clear()
    SYMMETRIZE_MONOMIAL_OP_VALIDATED.clear()
    SYMMETRIZE_COMPOSED_OP_CACHE.clear()
    SYMMETRIZE_COMPOSED_OP_VALIDATED.clear()
    SYMMETRIZE_ORBIT_CACHE.clear()
    KZ_POW_CACHE.clear()
