#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export MOIREKP_RELEASE_FINAL=1

python -m pytest \
  -p no:cacheprovider \
  -m "not slow and not external_data" \
  tests/test_release_contract.py \
  tests/kp/test_example_dependency_contract.py \
  examples/minimal_synthetic \
  -q
