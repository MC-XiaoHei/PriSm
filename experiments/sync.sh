#!/bin/sh
set -e
HOST=$(python -c "import tomllib,pathlib;h=tomllib.loads(pathlib.Path('experiments/host.toml').read_text());print(f\"{h['host']}:{h['root']}\")")
rsync -az --delete --exclude __pycache__ --exclude .venv src presets pyproject.toml "$HOST/"
