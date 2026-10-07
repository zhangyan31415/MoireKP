#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/openmx/build_openmx_symm_hs.sh" "$@"
