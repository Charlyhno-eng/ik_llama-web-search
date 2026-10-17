#!/bin/sh
# Run from any directory. The application has no Python package dependencies.
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
exec python3 start.py "$@"
