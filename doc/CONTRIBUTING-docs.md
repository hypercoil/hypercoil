# Contributing to the federation documentation

This guide is for anyone authoring docs for a federation library. The hub is one
site assembled from per-library `docs/` trees; you author in your own repo and
preview standalone, and the umbrella composes everything at build time. You never
need another library's working tree checked out to work on yours.

See also: `SPEC.md` (contracts), `DECISIONS.md` (what was decided and why),
`RECON.md` (per-library state), `PHASE1.md` / `PHASE2.md` (what's built).

---

## 1. One-time setup

Install the pinned toolchain (Quarto + Pagefind + a CPU-only figure venv) under a
prefix of your choosing — identical on a laptop and in CI:

```bash
DOC_TOOLING_PREFIX=/path/to/tooling bash doc/_tooling/bootstrap.sh
source doc/_tooling/activate.sh        # puts quarto + the venv on PATH
```

## 2. Add a library (SPEC §10)

```bash
doc/build/add-library.sh <name> --accent '#RRGGBB' --blurb "one line"
```

This is idempotent and non-destructive. It:

1. registers the library in `doc/libraries.yml` (drives the colour-key legend and
   the build list) and an xref namespace in `doc/xref/namespaces.yml`;
2. scaffolds `<name>/docs/` from `doc/_template` — the four Diátaxis sections, a
   one-line `brand.scss`, a federation-wired `_quarto.yml`, `_resources/` with the
   `generated/` cache convention, and a copy-me `explanation/example-chapter.qmd`;
3. installs the pinned federation extension so `quarto preview <name>/docs` works
   immediately.

Any pre-existing frame file it would replace is backed up to `<stem>.legacy.<ext>`
(your other content is never touched). Pick an accent not already in
`libraries.yml`; the legend updates itself.

## 3. Preview and build

```bash
quarto preview <name>/docs           # standalone, fully themed (cross-lib links -> hub URLs)
doc/build/build.sh                   # whole hub -> doc/_site (composed; cross-lib links relative)
```

`build.sh` runs the xref unit tests, packages the extension, renders the umbrella
and every buildable library, builds the unified Pagefind index, and runs the
negative xref test. Run it before pushing.

## 4. House style

### 4.1 Fast path / TL;DR (convention, not enforced)
Open each chapter — especially Explanation — with a fast path: the minimal working
understanding before the full treatment. Use the recognisable callout:

```markdown
::: {.callout-tldr title="Fast path"}
The one or two paragraphs a reader needs before the deep dive.
:::
```

Apply it where it helps; omit it where a page shouldn't be forced to conform.

### 4.2 Math, theorems, algorithms
- **Equations**: LaTeX, labelled for cross-reference — `$$ … $$ {#eq-name}`, cited as `@eq-name`.
- **Theorems/proofs**: Quarto built-ins — `::: {#thm-name}` … `:::` and `::: {.proof}` … `:::`. Styling is shared; don't restyle locally.
- **Algorithms**: a framed `::: {.algorithm}` block with a bold "Algorithm N — …" line and a pseudocode fence.

The `explanation/example-chapter.qmd` scaffold shows all three wired up — copy it.

### 4.3 Executable vs frozen figures (SPEC §6.4)
- **Lightweight** figures execute at build time — **numpy/matplotlib only**, never
  JAX, native FFI, or GPU. Do **not** call `matplotlib.use("Agg")` (it disables
  Quarto's figure capture).
- **Heavy / HPC / not-CI-reproducible**: **freeze** it. Compute offline, commit the
  artifact under `docs/_resources/generated/`, and embed it as a static image. The
  doc build must never require HPC or special privileges.

### 4.4 Cross-library references (SPEC §5)
Within your library use ordinary Quarto refs (`@sec-…`, `@eq-…`, `@thm-…`). To link
*into another library*, use the federation shortcode:

```markdown
{{< xref nitrix:fellner-schall "the Fellner–Schall chapter" >}}
```

It resolves to a same-origin relative link in the composed hub and to the published
hub URL in a standalone preview. A reference to an unregistered namespace/anchor
**fails the build** — broken xrefs are errors, not 404s.

**Exposing an anchor**: cross-library anchors are a *curated subset*, not every
heading. To make one referenceable: give the target heading a stable id
(`## Title {#my-anchor}`) and register it in `doc/xref/namespaces.yml` under your
namespace's `anchors:` as `my-anchor: explanation/page.qmd#my-anchor`. Keep the set
small and intentional; renaming a published anchor is a breaking change.

### 4.5 Citations
Cite against the single shared bibliography `doc/_bibliography/federation.bib`
(`[@key]`). One canonical entry per work — don't add a second entry for a source
another library already cites. Additions are additive and must stay deduplicated.

### 4.6 API reference (SPEC §6.5)
Reference pages are **authored** `.qmd`, living in your repo next to the code to
minimise drift. An autodoc generator may seed *stubs* (signatures + skeletons) for
you to fill — but autodoc output is never the final reference, and it does not
replace the hand-authored textbook reference. (The stub generator lands in Phase 3.)

## 5. The rules that keep the hub coherent

- **No format/visual keys in your `_quarto.yml`.** Theme, math, callouts, crossref,
  and code defaults come from the shared layer. Anything visual in a library config
  is a drift bug (SPEC §4.4) — your only visual knob is `brand.scss`'s `$accent`.
- **Add new pages to the `render:` list** in `_quarto.yml` as you write them. The
  scaffold lists only frame pages so half-written or legacy pages don't ship.
- **The build must stay green.** xref/citation errors and the negative test gate
  every build.
