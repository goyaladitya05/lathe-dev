#!/usr/bin/env bash
# Stage main (public export) and the fp8 branch into build/modal, then run release/modal_l4.py on Modal, detached.
#   release/modal_l4.sh [bench|fp8|all]   (MODAL_PROFILE picks the account)
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf build/modal && mkdir -p build/modal/fp8
python3 release/export.py HEAD build/modal/main
git archive fp8 | tar -x -C build/modal/fp8
exec build/modal-venv/bin/modal run --detach release/modal_l4.py --only "${1:-all}"
