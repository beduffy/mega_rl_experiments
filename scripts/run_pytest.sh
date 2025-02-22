#!/bin/bash
# Runs tests based on changed files or specific patterns

# Default test directory
DEFAULT_TESTS="tests/"

echo "Received arguments: $@"

# If arguments are passed from pre-commit (changed files), use those
if [ $# -ne 0 ]; then
    echo "Processing changed files..."
    # Filter for test files related to changed files
    RAW_FILTER=$(echo "$@" | grep -E 'tests/test_.*\.py|src/.*\.py')

    if [ -z "$RAW_FILTER" ]; then
        echo "No relevant tests to run for changed files: $@"
        exit 0
    fi

    SELECTED_TESTS=$(echo "$RAW_FILTER" | xargs -n1 | sort -u | tr '\n' ' ')
    echo "Selected tests: '$SELECTED_TESTS'"
else
    echo "No arguments - running all tests"
    SELECTED_TESTS="$DEFAULT_TESTS"
fi

CONDA_PYTEST="/home/ben/miniconda3/envs/py38/bin/pytest"
SYSTEM_PYTEST="pytest"

if [ -f "$CONDA_PYTEST" ]; then
    echo "Running targeted tests: $SELECTED_TESTS"
    exec "$CONDA_PYTEST" -v -m "not integration" $SELECTED_TESTS
else
    echo "Running all tests"
    exec "$SYSTEM_PYTEST" -v "$DEFAULT_TESTS"
fi
