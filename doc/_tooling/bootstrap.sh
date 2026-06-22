#!/usr/bin/env bash
# Reproducible doc-toolchain bootstrap for the hypercoil federation docs.
#
# Installs, under a single prefix, everything needed to build/serve the hub:
#   * a pinned Quarto (bundles its own pandoc + Lua runtime)
#   * a Python venv with Pagefind (search) + a CI-safe figure stack
#
# Identical on a laptop and in CI. Idempotent: re-running is cheap and safe.
#
# Usage:
#   DOC_TOOLING_PREFIX=/path/to/tooling ./bootstrap.sh
#   (defaults to ./.tooling next to this script)
#
# After bootstrapping, `source activate.sh` to put quarto + the venv on PATH.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=versions.env
source "$HERE/versions.env"

PREFIX="${DOC_TOOLING_PREFIX:-$HERE/.tooling}"
mkdir -p "$PREFIX"

# ---- platform detection -----------------------------------------------------
os="$(uname -s)"; arch="$(uname -m)"
case "$os/$arch" in
  Linux/x86_64)   q_asset="quarto-${QUARTO_VERSION}-linux-amd64.tar.gz" ;;
  Linux/aarch64)  q_asset="quarto-${QUARTO_VERSION}-linux-arm64.tar.gz" ;;
  Darwin/*)       q_asset="quarto-${QUARTO_VERSION}-macos.tar.gz" ;;
  *) echo "Unsupported platform $os/$arch. Install Quarto ${QUARTO_VERSION} manually into \$DOC_TOOLING_PREFIX." >&2; exit 1 ;;
esac

# ---- Quarto -----------------------------------------------------------------
if [ ! -x "$PREFIX/quarto-${QUARTO_VERSION}/bin/quarto" ]; then
  url="https://github.com/quarto-dev/quarto-cli/releases/download/v${QUARTO_VERSION}/${q_asset}"
  echo ">> Downloading Quarto ${QUARTO_VERSION}"
  curl -fsSL "$url" -o "$PREFIX/${q_asset}"
  tar -xzf "$PREFIX/${q_asset}" -C "$PREFIX"
  rm -f "$PREFIX/${q_asset}"
  # Normalise: some assets extract to quarto-<ver>, others nest a bin/ deeper.
  if [ ! -x "$PREFIX/quarto-${QUARTO_VERSION}/bin/quarto" ]; then
    found="$(find "$PREFIX" -maxdepth 3 -type f -name quarto -path '*/bin/*' | head -1)"
    [ -n "$found" ] && ln -sfn "$(dirname "$(dirname "$found")")" "$PREFIX/quarto-${QUARTO_VERSION}"
  fi
fi
echo ">> Quarto: $("$PREFIX/quarto-${QUARTO_VERSION}/bin/quarto" --version)"

# ---- Python venv (Pagefind + figure stack) ----------------------------------
VENV="$PREFIX/venv"
[ -x "$VENV/bin/python" ] || "${DOC_PYTHON:-python3}" -m venv "$VENV"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PREFIX/pip-cache}"
"$VENV/bin/python" -m pip install -q --upgrade pip
"$VENV/bin/python" -m pip install -q -r "$HERE/requirements-docs.txt"
# Freeze exact transitive versions for full reproducibility.
"$VENV/bin/python" -m pip freeze > "$HERE/requirements-docs.lock.txt"
echo ">> Python venv ready: $("$VENV/bin/python" -m pagefind --version 2>/dev/null || echo 'pagefind installed')"

echo ">> Done. Run:  source \"$HERE/activate.sh\""
