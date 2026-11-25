#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "$0")/env.sh"
exec python3 -m urban_drone.stop
