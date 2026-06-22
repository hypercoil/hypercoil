# Phase 2 — Templatized and parallel authoring unblocked

**Status: complete.** Every code-present federation library now has a themed,
building docs tree; adding a library is one command; the contributor guide is
written. Teams can author in parallel with no cross-repo coordination.

## What was built

| Artifact | Path | Purpose |
|---|---|---|
| Library docs template | `doc/_template/docs/` | Four Diátaxis sections, `brand.scss` stub, federation-wired `_quarto.yml`, `_resources/{figures,generated}/`, copy-me `explanation/example-chapter.qmd` (full authoring stack) |
| `add-library` procedure | `doc/build/add-library.sh` + `add_library.py` | Idempotent, non-destructive: registry edits + scaffold (backs up collisions to `.legacy`) + extension install |
| Build/legend status model | `doc/build/_active_libs.py`, `gen-legend.py` | `active` / `scaffold` are built; `aspirational` is registry-only; legend tags each |
| Contributor guide | `doc/CONTRIBUTING-docs.md` | Setup, add-a-library, preview/build, house style (fast-path, math/theorem/algorithm, exec-vs-frozen figures, xref + anchor exposure, citations, authored reference) |

## Onboarded (scaffold, distinct accents)

`niffi` `ilex` `thrux` `bitsjax` `entense` `gramform` `conveyant` — each a green
placeholder Diátaxis tree. `nitrix` + `tensorbids` remain the Phase-1 real slice.
`hyve` is registered **aspirational** (no code yet → not built; shown *planned* in
the legend). `thrux` and `conveyant` had pre-existing drifted docs: their frame
files were backed up to `*.legacy.*` and their stale chapters are preserved in-repo
but left out of the render list (not shipped) pending Phase-4 rewrite.

## Verification

- Full hub builds green: **9 libraries render**, unified Pagefind indexes
  **63 pages**, the negative xref test still fails the build.
- Each library is visually distinct by its own accent; the colour-key legend
  (generated from `libraries.yml`) lists all 10 with correct status tags.
- `hyve` has no built site (correct — aspirational).

## Exit criteria (PLAN §2) — met

- [x] All code-present libraries scaffolded and building.
- [x] Adding a library is a single documented procedure (`add-library.sh`).
- [x] Contributor guide complete.
- [x] Legend + nav pick up new libraries automatically from the registries.
- [x] Teams can author in parallel without cross-repo coordination.

## Next

- **Phase 3** — API-reference stub generator (autodoc → fillable stubs, in-repo) +
  a staleness signal.
- **Phase 4** — content build-out (the long pole): real chapters per library;
  rewrite `thrux`/`conveyant` from their `.legacy` source; deepen `nitrix` GP path.
- Optional hardening: full link/figure/citation validation stage (SPEC §7.3–7.4);
  flip on the Pages deploy.
