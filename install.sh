#!/usr/bin/env bash
# Creates a fresh conda environment and installs ptyxplore.
# Usage: bash install.sh
# The environment name is defined in environment.yml (default: ptyxplore)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Creating conda environment from environment.yml"
conda env create -f "$SCRIPT_DIR/environment.yml"

ENV_NAME=$(grep '^name:' "$SCRIPT_DIR/environment.yml" | awk '{print $2}')

echo ""
echo "Done! Activate the environment with:"
echo "  conda activate $ENV_NAME"
