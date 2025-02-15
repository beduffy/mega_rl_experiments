#!/bin/bash
# Checks for conda pytest first, falls back to system pytest

CONDA_PYTEST="/home/ben/miniconda3/envs/py38/bin/pytest"
SYSTEM_PYTEST="pytest"

if [ -f "$CONDA_PYTEST" ]; then
    echo "Using conda pytest: $CONDA_PYTEST"
    exec "$CONDA_PYTEST" "$@"
else
    echo "Using system pytest: $(which pytest)"
    exec "$SYSTEM_PYTEST" "$@"
fi
