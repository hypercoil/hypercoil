#!/usr/bin/env bash
# Canonical federation documentation build (SPEC §8). Identical locally and in
# CI. Composes the umbrella (hub root) + each active library into _site/.
#
# Usage:
#   build.sh [--clean] [--serve] [--port N] [LIB ...]
#     (no LIB)   full build: umbrella + every active library
#     LIB ...    incremental: rebuild only these libraries (+ the umbrella),
#                keeping the rest of _site/ — for iterating on one library
#                against the whole-site view
#     --clean    wipe the freeze caches first, forcing full figure re-execution
#                (a fresh deploy clone has no cache, so it already re-runs)
#     --serve    serve _site/ over HTTP after building (Pagefind needs HTTP)
#     --port N   port for --serve (default 8080)
#
# Caching: executable figures are frozen (execute.freeze:auto in the shared
# federation metadata) under each project's `_freeze/` — gitignored, reused
# across builds, and re-executed only when a document's source changes. The
# rendered `_site/` output is always purged and rebuilt (no stale pages); the
# `_freeze/` cache lives outside `_site/` and survives. Library docs come from
# sibling checkouts under code/ (in CI, the clone-at-pinned-ref step; locally,
# your working trees). libraries.yml drives which are active.
set -euo pipefail

DOC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"     # .../hypercoil/doc
CODE="$(cd "$DOC/../.." && pwd)"                            # .../code
SITE="$DOC/_site"

CLEAN=0; SERVE=0; PORT=8080; REQUESTED=()
while [ $# -gt 0 ]; do
  case "$1" in
    --clean)     CLEAN=1 ;;
    --serve)     SERVE=1 ;;
    --port)      PORT="$2"; shift ;;
    --port=*)    PORT="${1#*=}" ;;
    -h|--help)   sed -n '2,20p' "$0"; exit 0 ;;
    --*)         echo "unknown flag: $1" >&2; exit 2 ;;
    *)           REQUESTED+=("$1") ;;
  esac
  shift
done

# 0. toolchain
# shellcheck disable=SC1091
source "$DOC/_tooling/activate.sh"

ACTIVE=($(python "$DOC/build/_active_libs.py"))
if [ "${#REQUESTED[@]}" -eq 0 ]; then
  TARGETS=("${ACTIVE[@]}"); INCREMENTAL=0
else
  for l in "${REQUESTED[@]}"; do
    printf '%s\n' "${ACTIVE[@]}" | grep -qx "$l" \
      || { echo "not an active library: $l (active: ${ACTIVE[*]})" >&2; exit 2; }
  done
  TARGETS=("${REQUESTED[@]}"); INCREMENTAL=1
fi

# --clean: drop freeze caches so every figure re-executes this run.
if [ "$CLEAN" -eq 1 ]; then
  echo "== --clean: wiping freeze caches =="
  rm -rf "$DOC/_freeze"
  for name in "${ACTIVE[@]}"; do rm -rf "$CODE/$name/docs/_freeze"; done
fi

# 1. keystone unit tests
echo "== [1/5] xref resolver unit tests =="
quarto pandoc lua "$DOC/xref/test_resolver.lua"

# 2. package the shared extension
echo "== [2/5] package federation extension =="
bash "$DOC/build/package-extension.sh"

# 3. render
echo "== [3/5] render (${INCREMENTAL:+incremental: }${TARGETS[*]}) =="
python "$DOC/build/gen-legend.py"
if [ "$INCREMENTAL" -eq 0 ]; then
  rm -rf "$SITE"      # full build: purge the whole composed site (freeze survives)
fi
# The umbrella owns _site/ root (index, legend, nav); render it either way. It
# overwrites its own files and leaves library subtrees intact for incremental.
quarto render "$DOC" -M xref-mode:composed

ensure_ignored() {   # keep a rendered library's working tree clean of build cruft
  local gi="$1/.gitignore"
  for pat in '_site/' '_freeze/'; do
    [ -f "$gi" ] && grep -qxF "$pat" "$gi" && continue
    printf '%s\n' "$pat" >> "$gi"
  done
}

for name in "${TARGETS[@]}"; do
  libdocs="$CODE/$name/docs"
  echo "   -- library: $name"
  # Install the umbrella's pinned shared extension (overrides any local copy,
  # guaranteeing hub-wide consistency — SPEC §4.3).
  mkdir -p "$libdocs/_extensions/hypercoil"
  rm -rf "$libdocs/_extensions/hypercoil/federation"
  cp -r "$DOC/_extensions/hypercoil/federation" "$libdocs/_extensions/hypercoil/"
  ensure_ignored "$libdocs"
  rm -rf "$libdocs/_site"          # purge the library's own stale output (keeps _freeze)
  quarto render "$libdocs" -M xref-mode:composed
  rm -rf "$SITE/$name"             # purge this library's stale subtree in the composed site
  mkdir -p "$SITE/$name"
  cp -r "$libdocs/_site/." "$SITE/$name/"
done

# 4. unified search
echo "== [4/5] pagefind =="
python -m pagefind --site "$SITE" >/dev/null
echo "   pagefind index built under $SITE/pagefind"

# 5. negative test (broken xref must abort a render). Full builds only — it is a
#    global correctness gate, not per-library. Render a throwaway project so the
#    federation extension's shortcode is actually discovered.
if [ "$INCREMENTAL" -eq 0 ]; then
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
else
  echo "== [5/5] negative xref test skipped (incremental build) =="
fi

echo "== build complete -> $SITE =="

if [ "$SERVE" -eq 1 ]; then
  echo "== serving $SITE at http://localhost:$PORT/  (Ctrl-C to stop) =="
  ( cd "$SITE" && exec python -m http.server "$PORT" )
fi
