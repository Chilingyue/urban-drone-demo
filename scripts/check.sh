#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "$0")/env.sh"
cd "$URBAN_PROJECT_ROOT"
python3 -m unittest discover -s tests -v
python3 -m compileall -q urban_drone
bash -n scripts/env.sh scripts/demo.sh scripts/replay.sh
