#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
if [[ "${1:-}" == "--final" ]]; then
    export MOIREKP_RELEASE_FINAL=1
    shift
fi
python -m pytest -p no:cacheprovider -m "not slow and not external_data" \
    tests/test_release_contract.py tests/kp/test_example_dependency_contract.py -q "$@"
