#!/usr/bin/env bash
# Add a library to the federation docs (SPEC §10): registry edits + scaffold +
# extension install. Packages the shared extension first so the new library can
# be previewed standalone immediately. Idempotent.
#
# Usage:
#   build/add-library.sh NAME --accent '#RRGGBB' [--blurb "..."] [--status scaffold]
set -euo pipefail
DOC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bash "$DOC/build/package-extension.sh" >/dev/null
python3 "$DOC/build/add_library.py" "$@"
echo "Next: build the hub with  build/build.sh  (or preview standalone: quarto preview <lib>/docs)"
