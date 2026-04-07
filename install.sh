#!/usr/bin/env bash
# Creates a fresh conda environment and installs ptyxplore.
# Usage: bash install.sh
# The environment name is defined in environment.yml (default: ptyxplore)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_NAME=$(grep '^name:' "$SCRIPT_DIR/environment.yml" | awk '{print $2}')
ENV_PATH="/gpfs/cfel/user/$USER/envs/$ENV_NAME"

echo "==> Creating conda environment at $ENV_PATH"
conda env create -f "$SCRIPT_DIR/environment.yml" -p "$ENV_PATH"

echo ""
echo "==> Registering Jupyter kernel for $ENV_NAME"
"$ENV_PATH/bin/python" -m ipykernel install --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME)"

echo ""
echo "Done! Activate the environment with:"
echo "  conda activate $ENV_PATH"
