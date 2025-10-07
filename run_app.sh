#!/usr/bin/env bash
set -euo pipefail

# Create virtual environment if it does not exist
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

# Activate the virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
  source .venv/Scripts/activate
else
  source .venv/bin/activate
fi

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# Launch the PySide6 application
python main.py
