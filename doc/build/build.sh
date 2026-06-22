#!/usr/bin/env bash
# Canonical federation documentation build (SPEC §8). Identical locally and in
# CI. Stages:
#   1. unit-test the xref resolver (federation-critical)
#   2. package the shared federation extension from canonical sources
#   3. render the umbrella, then each active library (composed mode) into _site/
#   4. unified Pagefind search index over _site/
#   5. negative test: a deliberately broken xref must fail the build
#
# Library docs are taken from sibling checkouts under code/. In CI this is where
# the clone/fetch-at-pinned-ref step lands them (libraries.yml drives it).
set -euo pipefail

DOC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"     # .../hypercoil/doc
CODE="$(cd "$DOC/../.." && pwd)"                            # .../code
SITE="$DOC/_site"

# 0. toolchain
# shellcheck disable=SC1091
source "$DOC/_tooling/activate.sh"

# 1. keystone unit tests
echo "== [1/5] xref resolver unit tests =="
quarto pandoc lua "$DOC/xref/test_resolver.lua"

# 2. package the shared extension
echo "== [2/5] package federation extension =="
bash "$DOC/build/package-extension.sh"

# 3. render umbrella first (it owns _site/ root), then libraries into _site/<lib>
echo "== [3/5] render =="
rm -rf "$SITE"
python "$DOC/build/gen-legend.py"
quarto render "$DOC" -M xref-mode:composed

for name in $(python "$DOC/build/_active_libs.py"); do
  libdocs="$CODE/$name/docs"
  echo "   -- library: $name"
  # Install the umbrella's pinned shared extension (overrides any local copy,
  # guaranteeing hub-wide consistency — SPEC §4.3).
  mkdir -p "$libdocs/_extensions/hypercoil"
  rm -rf "$libdocs/_extensions/hypercoil/federation"
  cp -r "$DOC/_extensions/hypercoil/federation" "$libdocs/_extensions/hypercoil/"
  quarto render "$libdocs" -M xref-mode:composed
  mkdir -p "$SITE/$name"
  cp -r "$libdocs/_site/." "$SITE/$name/"
done

# 4. unified search
echo "== [4/5] pagefind =="
python -m pagefind --site "$SITE" >/dev/null
echo "   pagefind index built under $SITE/pagefind"

# 5. negative test (broken xref must abort a render). Render a throwaway project
#    so the federation extension's shortcode is actually discovered.
echo "== [5/5] negative xref test =="
NEG="$(mktemp -d)"
cp -r "$DOC/_extensions" "$NEG/_extensions"
cp "$DOC/xref/test/negative.qmd" "$NEG/index.qmd"
printf 'project:\n  type: website\n' > "$NEG/_quarto.yml"
set +e
quarto render "$NEG" -M xref-mode:composed >/tmp/xref_negative.log 2>&1
rc=$?
set -e
rm -rf "$NEG"
if [ "$rc" -eq 0 ]; then
  echo "   FAIL: a broken xref did NOT fail the build (see /tmp/xref_negative.log)"; exit 1
fi
echo "   OK: broken xref correctly aborted the build"

echo "== build complete -> $SITE =="
