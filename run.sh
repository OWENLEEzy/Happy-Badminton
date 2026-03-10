#!/usr/bin/env bash
# Happy-Badminton launcher
# Usage:
#   ./run.sh              # auto-train if needed, then start server
#   ./run.sh --train      # force re-train
#   ./run.sh --port 8080  # custom port

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

uv run python main.py "$@"
