#!/usr/bin/env bash
# Assemble the distributable `hypercoil/federation` Quarto extension from the
# umbrella's canonical sources. Run before any render and before syncing the
# extension into library repos. Idempotent.
set -euo pipefail
DOC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXT="$DOC/_extensions/hypercoil/federation"
mkdir -p "$EXT"

# _extension.yml is authored in place; sync the generated-from-canonical files.
cp "$DOC/xref/xref.lua"               "$EXT/xref.lua"
cp "$DOC/xref/resolver.lua"           "$EXT/resolver.lua"
cp "$DOC/xref/namespaces.yml"         "$EXT/namespaces.yml"
cp "$DOC/theme/federation.scss"       "$EXT/federation.scss"
cp "$DOC/theme/_metadata.yml"         "$EXT/_metadata.yml"
cp "$DOC/_bibliography/federation.bib" "$EXT/federation.bib"

echo "Packaged federation extension -> $EXT"
