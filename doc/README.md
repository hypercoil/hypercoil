# Federation documentation hub — running it locally

This directory (`hypercoil/doc/`) is the **umbrella** that composes every
federation library's `docs/` into one site with a shared black/`$accent` theme,
cross-library links, and unified search. This README is the practical "get it
running on my laptop" guide. For the *why*, see `SPEC.md` / `DECISIONS.md`; for
*authoring*, see `CONTRIBUTING-docs.md`.

> **TL;DR**
> ```bash
> bash doc/_tooling/bootstrap.sh          # one-time: installs pinned Quarto + Pagefind + a figure venv
> source doc/_tooling/activate.sh         # puts them on PATH (run in each new shell)
> quarto preview nitrix/docs              # fast dev loop: live-reload one library
> # …or build the whole hub:
> bash doc/build/build.sh                 # -> doc/_site
> python -m http.server -d doc/_site 8000 # serve at http://localhost:8000
> ```

---

## Prerequisites

- **git** and **Python ≥ 3.10** (only to create the figure venv; you do *not*
  pre-install Quarto or Pagefind — the bootstrap fetches pinned copies).
- ~1 GB free disk for the toolchain, and internet access on first bootstrap.
- Works on Linux and macOS. (Windows: use WSL.)

All commands below are written to be run from the **`hypercoil` repo root**.

## 1. One-time: install the pinned toolchain

```bash
bash doc/_tooling/bootstrap.sh
```

This installs, under a single prefix, exactly the versions pinned in
`doc/_tooling/versions.env` + `requirements-docs.txt`:

- **Quarto** (standalone — bundles its own pandoc + Lua),
- **Pagefind** (federation-wide search), and
- a **CPU-only Python venv** for executable figures (numpy/matplotlib + the
  Jupyter execution stack).

By default everything lands in `doc/_tooling/.tooling/` (git-ignored). To put it
elsewhere (e.g. a scratch disk), set a prefix — use the **same** value every time:

```bash
export DOC_TOOLING_PREFIX=/path/to/tooling
bash doc/_tooling/bootstrap.sh
```

Then, in every shell where you build or preview:

```bash
source doc/_tooling/activate.sh      # honours the same DOC_TOOLING_PREFIX
quarto --version                     # 1.9.38
```

Re-running `bootstrap.sh` is cheap and idempotent (use it to pick up version
bumps). It writes `requirements-docs.lock.txt` with exact transitive versions.

## 2. Fast dev loop — live-preview one library

The quickest inner loop for writing docs is Quarto's live preview of a single
library (auto-reloads on save):

```bash
source doc/_tooling/activate.sh
quarto preview nitrix/docs           # or tensorbids/docs, niffi/docs, …
```

Notes:
- A library previewed alone is **standalone**: cross-library `{{< xref … >}}`
  links resolve to the *published hub URL* (so they're not dead, but they point
  off-laptop). In the full composed build they become local relative links.
- Preview needs the shared theme/extension present in that library's
  `docs/_extensions/` — see [Composition & branches](#composition--branches)
  if a library's `docs/` isn't in your working tree yet.

To preview just the hub landing page:

```bash
quarto preview doc
```

## 3. Build & serve the whole hub

```bash
source doc/_tooling/activate.sh
bash doc/build/build.sh
```

`build.sh` runs the full pipeline (identical to CI): xref resolver unit tests →
package the shared extension → render the umbrella + every buildable library into
`doc/_site/` → build the unified Pagefind index → a negative-xref test that must
fail. Then serve the static output:

```bash
python -m http.server -d doc/_site 8000   # http://localhost:8000
```

(Plain relative links, so any static server works.) `doc/_site/` is git-ignored.

## 4. Other commands

```bash
# Add a new library to the hub (registry + scaffold + extension install):
bash doc/build/add-library.sh <name> --accent '#RRGGBB' --blurb "one line"

# Seed fillable API-reference stubs from source (static/AST — no import needed):
python3 doc/build/gen_reference_stubs.py <lib> [--modules <lib>.subpkg]

# Flag reference stubs whose signatures have drifted (CI gate); --update to re-bless:
python3 doc/build/check_reference_staleness.py <lib>

# Run the xref resolver unit tests on their own:
quarto pandoc lua doc/xref/test_resolver.lua
```

## Composition & branches

The umbrella build expects each library checked out as a **sibling directory**
next to `hypercoil/` (the layout `setup_repos.sh` produces):

```
code/
  hypercoil/        ← umbrella lives in hypercoil/doc
  nitrix/  tensorbids/  niffi/  ilex/  thrux/  bitsjax/  entense/  gramform/  conveyant/
```

`doc/libraries.yml` is the registry: which libraries exist, their accent, and the
git `ref` the **CI** clones for each. `build.sh` (local) renders whatever is in
each sibling's `docs/` working tree; CI instead clones each repo at its pinned
`ref` and runs the same script.

**Important:** each library's federation `docs/` currently lives on a
`docs/federation-scaffold` branch in that repo (so it stays out of the teams'
active branches). For a library to appear in a **local full build**, its `docs/`
must be present in the checked-out working tree — either:

- preview it from that branch: `git -C ../<lib> switch docs/federation-scaffold`
  then `quarto preview ../<lib>/docs`; or
- merge `docs/federation-scaffold` into the branch you build from; or
- for CI, point that library's `ref` in `libraries.yml` at the branch/tag that
  carries its docs.

Only `active` and `scaffold` libraries are built; `aspirational` ones (e.g.
`hyve`, no code yet) are registry-only and shown as *planned* in the legend.

## Troubleshooting

- **`quarto: command not found`** — run `source doc/_tooling/activate.sh` in this
  shell (and confirm the same `DOC_TOOLING_PREFIX` you bootstrapped with).
- **`build.sh` fails rendering a library** — that sibling's `docs/` isn't in its
  working tree; see [Composition & branches](#composition--branches).
- **An executable figure renders blank** — don't call `matplotlib.use("Agg")` in
  a `{python}` cell; it disables Quarto's inline figure capture.
- **A cross-reference fails the build** (`ERROR [xref] …`) — the namespace/anchor
  isn't registered in `doc/xref/namespaces.yml`. That's by design: broken xrefs
  are build errors, not 404s.
- **Re-bootstrap from clean** — delete the prefix (default `doc/_tooling/.tooling/`)
  and re-run `bootstrap.sh`.

## More documentation

- `SPEC.md` — what the system is and its contracts.
- `DECISIONS.md` — resolved decisions (D-1…D-14) and rationale.
- `RECON.md` — per-library reality and adapt/migrate/greenfield verdicts.
- `CONTRIBUTING-docs.md` — authoring guide (house style, math, figures, xref).
- `PHASE1.md` / `PHASE2.md` / `PHASE3.md` — what each phase built and verified.
