#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "usage: scripts/build_openmx_symm_hs.sh <openmx-source-dir> [build-dir]" >&2
  exit 2
fi

OPENMX_SOURCE=$(cd "$1" && pwd)
BUILD_DIR=${2:-build/openmx_symm_hs}
mkdir -p "$BUILD_DIR"
BUILD_DIR=$(cd "$BUILD_DIR" && pwd)

if [ -z "${MKLROOT:-}" ]; then
  echo "MKLROOT is not set. Load oneAPI first, e.g. module load oneapi22.3" >&2
  exit 2
fi

if [ ! -f "$OPENMX_SOURCE/read_scfout.o" ]; then
  echo "$OPENMX_SOURCE/read_scfout.o is missing. Build OpenMX analysis_example/read_scfout first." >&2
  exit 2
fi

CXX=${CXX:-mpiicpc}
SRC=$(cd "$(dirname "$0")/.." && pwd)/scripts/openmx_analysis_symm_hs.cpp
OUT="$BUILD_DIR/analysis_symm_hs"

set -x
"$CXX" -std=c++17 -O3 -xHOST -ip -no-prec-div -qopenmp -DOPENMX_ANALYSIS_BUILD \
  -I"$OPENMX_SOURCE" -I"${MKLROOT}/include" -I"${MKLROOT}/include/fftw" \
  "$SRC" "$OPENMX_SOURCE/read_scfout.o" \
  -L"${MKLROOT}/lib/intel64" \
  -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core \
  -lifcore -lmkl_blacs_intelmpi_lp64 -liomp5 -lpthread -lm -ldl \
  -o "$OUT"
set +x

echo "$OUT"
