#!/usr/bin/env bash
# Source this file, or use scripts/demo.sh. Keep incompatible ~/.local NumPy out.
source /opt/ros/jazzy/setup.bash
export PYTHONNOUSERSITE=1
URBAN_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${URBAN_PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export GZ_IP=127.0.0.1

