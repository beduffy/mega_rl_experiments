#!/bin/bash

# Try conda path first, fall back to system pytest
CONDA_PYTEST="/home/ben/miniconda3/envs/py38/bin/pytest"
SYSTEM_PYTEST="pytest"

if [ -f "$CONDA_PYTEST" ]; then
    exec "$CONDA_PYTEST" "$@"
else
    exec "$SYSTEM_PYTEST" "$@"
fi
