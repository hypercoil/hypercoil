# Source this (do not execute) to put the pinned doc toolchain on PATH.
#   source doc/_tooling/activate.sh
# Honours DOC_TOOLING_PREFIX (same default as bootstrap.sh).
_doc_tooling_here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# shellcheck source=versions.env
source "$_doc_tooling_here/versions.env"
DOC_TOOLING_PREFIX="${DOC_TOOLING_PREFIX:-$_doc_tooling_here/.tooling}"
export DOC_TOOLING_PREFIX
export PATH="$DOC_TOOLING_PREFIX/quarto-${QUARTO_VERSION}/bin:$PATH"
# Activate the venv so `quarto render` finds jupyter and `pagefind` is on PATH.
# shellcheck disable=SC1091
source "$DOC_TOOLING_PREFIX/venv/bin/activate"
