#!/usr/bin/env bash
# Install/refresh the pinned federation Quarto extension into a docs project.
#
# Single source of truth: github.com/hypercoil/quarto-federation, pinned to a tag.
# While that repo is PRIVATE (stabilisation period) `quarto add` cannot fetch it
# (Quarto does not authenticate its archive download), so this script fetches the
# pinned tag over authenticated git and installs it to the SAME path quarto add
# would (`_extensions/hypercoil/federation/`). When the repo goes public, this is
# replaceable 1:1 by:  quarto add hypercoil/quarto-federation@<version>
#
# Usage:  sync-extension.sh <docs-dir> [version]
#   <docs-dir>  the library's Quarto project dir (the one with _quarto.yml)
#   [version]   tag to pin (default: $FED_EXT_VERSION or v0.1.0)
# Auth: reads $GHROPAT (org read PAT) for the private repo; not needed once public.
set -euo pipefail

REPO="hypercoil/quarto-federation"
VERSION="${2:-${FED_EXT_VERSION:-v0.1.0}}"
TARGET_DOCS="${1:?usage: sync-extension.sh <docs-dir> [version]}"
DEST="${TARGET_DOCS%/}/_extensions/hypercoil/federation"

auth=""
[ -n "${GHROPAT:-}" ] && auth="x-access-token:${GHROPAT}@"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT

# --quiet + redirect so a token-bearing URL never lands in logs.
if ! git clone --quiet --depth 1 --branch "$VERSION" \
       "https://${auth}github.com/${REPO}.git" "$tmp/ext" >/dev/null 2>&1; then
  echo "ERROR: clone of ${REPO}@${VERSION} failed (private repo needs GHROPAT)" >&2
  exit 1
fi

src="$tmp/ext/_extensions/federation"
[ -d "$src" ] || { echo "ERROR: _extensions/federation not in ${REPO}@${VERSION}" >&2; exit 1; }

mkdir -p "$DEST"
rm -rf "${DEST:?}"/*
cp -R "$src"/. "$DEST"/
echo "installed federation extension ${VERSION} -> ${DEST#$PWD/}"
