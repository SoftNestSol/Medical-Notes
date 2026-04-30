#!/usr/bin/env bash

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$PROJECT_DIR" || exit 1

if [ ! -d ".venv" ]; then
  echo "No .venv found in $PROJECT_DIR"
  echo "Create it first with: python3 -m venv .venv"
  exit 1
fi

source ".venv/bin/activate"

echo "Medical-Notes project ready"
echo "Project: $PROJECT_DIR"
echo "Python:  $(which python)"
echo ""
echo "Run scripts like:"
echo "  python scripts/train.py"