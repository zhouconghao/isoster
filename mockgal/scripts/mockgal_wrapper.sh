#!/bin/bash
# scripts/mockgal_wrapper.sh

MOCKGAL_PATH="../isophote_test/mockgal.py"
PROFIT_CLI_PATH="../isophote_test/libprofit/mbp"

export PROFIT_CLI_PATH

python "$MOCKGAL_PATH" "$@"
