#!/bin/bash
# Activate the virtual environment
source "venv-3.12.5/bin/activate"

# Set environment variables from .env if it exists
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

echo "Virtual environment activated with Python 3.12.5. Run 'deactivate' to exit."
