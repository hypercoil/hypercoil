# Phase 1 — Federation plumbing proven on a two-library slice

**Status: complete.** The federation documentation architecture is proven end to
end on `nitrix` (greenfield textbook authoring) + `tensorbids` (migration of
current content), per PLAN §1. Build everything with:

```bash
DOC_TOOLING_PREFIX=/scratch/hypercoil-doc-tooling bash doc/_tooling/bootstrap.sh   # once
DOC_TOOLING_PREFIX=/scratch/hypercoil-doc-tooling bash doc/build/build.sh          # build -> doc/_site
```

## What was built

| Artifact | Path | Notes |
|---|---|---|
| Reproducible toolchain | `doc/_tooling/{bootstrap,activate}.sh`, `versions.env`, `requirements-docs.txt`, `requirements-docs.lock.txt` | Pinned Quarto 1.9.38 + Pagefind 1.5.2 + figure venv; laptop ≡ CI |
| xref resolver (keystone) | `doc/xref/resolver.lua` + `doc/xref/test_resolver.lua` | Pure, host-agnostic; **8/8 unit tests** via `quarto pandoc lua` |
| xref shortcode | `doc/xref/xref.lua` | `{{< xref ns:anchor "text" >}}`; composed vs standalone; hard-fails on bad ref |
| Namespace registry | `doc/xref/namespaces.yml` | Curated cross-library anchors only |
| Theme contract | `doc/theme/federation.scss` (+ `brand-hub.scss`) | Black base; **all** distinguishing visuals derive from one `$accent` |
| Shared format defaults | `doc/theme/_metadata.yml` | Math, callouts (incl. `.callout-tldr`), crossref, code |
| Bibliography | `doc/_bibliography/federation.bib` | Single source of truth |
| Library registry | `doc/libraries.yml` | Drives the (generated) colour-key legend + build list |
| Distribution | `doc/_extensions/hypercoil/federation/` + `doc/build/package-extension.sh` | Shared surface as a pinned Quarto extension (SPEC §4.3) |
| Canonical build | `doc/build/build.sh` | tests → package → render → pagefind → negative test |
| CI | `.github/workflows/federation-docs.yml` | Clones libs at pinned refs, runs the same `build.sh` |
| Umbrella | `doc/_quarto.yml`, `doc/index.qmd`, `doc/federation/_legend.md` (generated) | Hub landing + colour-key legend |
| Slice — greenfield | `nitrix/docs/` incl. `explanation/fellner-schall.qmd` | Flagship chapter: LaTeX, theorem+proof, algorithm, citation, executable figure, curated anchor |
| Slice — migrate | `tensorbids/docs/` (brand + theme wired) | Current content composed onto the shared theme |

## The six verifications (all pass)

1. **Unified black base, distinct accents** — both libraries on `#0a0a0b`;
   `nitrix` teal `#00e0c6`, `tensorbids` orange `#ff7a45`, no cross-contamination.
2. **Cross-library link in the composed site** — `nitrix` chapter →
   `../tensorbids/explanation/architecture.html#the-three-pillars`; hub →
   both libraries.
3. **Standalone preview resolves the same xref to the hub URL** —
   `https://hypercoil.github.io/tensorbids/...` (not a dead link).
4. **Unified Pagefind search** — one index over the merged `_site/`; 21 pages
   indexed across both `nitrix` and `tensorbids`.
5. **Broken xref fails the build** — the negative test aborts the render
   non-zero with a clear `ERROR [xref] …` diagnostic.
6. **Colour-key legend generated from `libraries.yml`** — cannot drift.

## Frictions found and resolved (informs the patterns going forward)

- **Quarto's `error()` is non-fatal inside shortcodes.** The resolver returns
  `(nil, message)` and the shortcode hard-exits; do not rely on `error()` to
  break a build from a filter.
- **Shortcode files can't assume `PANDOC_SCRIPT_FILE`.** `xref.lua` locates its
  sibling `resolver.lua` via `debug.getinfo`.
- **`-M` vs `metadata-files` precedence.** Mode is not set in `namespaces.yml`;
  `xref.lua` defaults to standalone and the composed build passes
  `-M xref-mode:composed` — no precedence fight.
- **Executable figures need ipykernel's inline backend.** Do **not**
  `matplotlib.use("Agg")` — it disables figure capture.
- **Per-library render scoping.** `nitrix/docs` limits `render:` to the new
  Diátaxis pages so the legacy design/feature-request markdown stays as source
  material without polluting the site.

## Not yet done (deliberately deferred past the Phase-1 gate)

- Full link/figure/citation validation stage (SPEC §7.3–7.4) beyond render-time
  errors + the negative xref test.
- API-reference stub generation (Phase 3) and the `add-library` procedure
  (Phase 2).
- Real deployment (the CI workflow builds + uploads an artifact; the Pages
  deploy step is present but commented).
