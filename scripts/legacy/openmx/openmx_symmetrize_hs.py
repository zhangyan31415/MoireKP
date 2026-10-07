#!/usr/bin/env python3
"""OpenMX structure/H/S symmetry CLI; implementation shared by all backends."""
from tapw.io import hs_symmetry as _implementation

globals().update({name: value for name, value in vars(_implementation).items()
                  if not name.startswith('__')})
if __name__ == '__main__':
    raise SystemExit(_implementation.main())
