# The federation docs hub has moved

The docs hub that used to live in this directory has moved to diffprog-dev: the theme,
cross-library xref, the `hypercoil/federation` extension sources, the bibliography, the
library template, the toolchain pins, the docstring tools and the design records (SPEC,
DECISIONS, PLAN, PHASE1–3, RECON). It moved from `hypercoil@fd54605`.

| Was here | Now |
|---|---|
| `build/build.sh` | `dpx docs build` (`--local`, `--pinned`, `--lib`, `--serve`) |
| `build/gen_reference_stubs.py`, `check_reference_staleness.py`, `inventory_docstrings.py`, `docstring_migration_check.py` | `diffprog-dev/hub/tools/`, run with `dpx docs stubs\|stale\|inventory\|migration-check` |
| `libraries.yml`, `xref/namespaces.yml` | the `docs` blocks in `diffprog-dev/registry/repos.toml`, plus `hub/hub.toml` |
| `theme/`, `xref/`, `assets/`, `_bibliography/`, `_quarto.yml`, `index.qmd` | `diffprog-dev/hub/` |
| `_extensions/hypercoil/federation/` | packaged from `diffprog-dev/hub/` with `dpx docs package`; `dpx docs extension <lib>` vendors it into a library |
| `_template/` | `diffprog-dev/template/docs/` (`dpx docs scaffold <lib>`) |
| `_tooling/` | `diffprog-dev/hub/tooling/` (`dpx docs setup`) |
| `SPEC.md`, `DECISIONS.md`, `PLAN.md`, `PHASE*.md`, `RECON.md` | `diffprog-dev/dev/proposals/docs-hub.md` and `docs-hub/` |
| `CONTRIBUTING-docs.md` | `diffprog-dev/hub/CONTRIBUTING.md` |
| `.github/workflows/federation-docs.yml` | `diffprog-dev/.github/workflows/hub.yml` |

What stays here is hypercoil's legacy documentation: `_source/`, `_scripts/` and
`_static/`, built by `.github/workflows/doc.yml`.
