#!/usr/bin/env bash

if [ -z "$VIRTUAL_ENV" ]; then
  echo "No virtual environment is currently active."
  return 0 2>/dev/null || exit 0
fi

echo "Deactivating virtual environment:"
echo "$VIRTUAL_ENV"

deactivate

echo "Virtual environment deactivated."