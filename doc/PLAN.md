# Federation Documentation System — Implementation Plan

**Companion to:** `SPEC.md` (defines *what*; this defines *how* and *in what order*).
**Primary executor:** Claude Code, working across the federation repos.
**Operating principle:** Prove the *federation plumbing* on a minimal slice before any large-scale chapter authoring. Mechanism first, content second. Parallel team authoring must be unblocked as early as possible.

---

## Phase 0 — Reconciliation and decisions (do this before building anything)

The system in `SPEC.md` is a target state. First, survey reality.

### 0.1 Inventory existing docs across the federation
For each library (`nitrix`, `niffi`, `ilex`, `thrux`, `bitsjax`, `hyve`, `entense`, `gramform`, `conveyant`, `tensorbids`, `hypercoil`):
- Detect any existing `docs/`, Sphinx (`conf.py`), MkDocs (`mkdocs.yml`), Quarto (`_quarto.yml`), README narrative, notebooks, or docstring conventions.
- Record: tool in use, content volume, content quality/currency, theme assets, any cross-refs or citations already present.
- Produce `RECON.md` in the umbrella with a per-library table and a verdict each: **adapt** / **migrate** / **greenfield**.

### 0.2 Resolve the open decisions (SPEC §11) and record them
Write decisions + one-line rationale into `DECISIONS.md`:
- Umbrella home: `hypercoil-docs` vs `docs/` in `hypercoil`.
- Library-docs ingestion: submodules vs registry-fetch.
- Shared-theme distribution: Quarto extension vs pinned fetched asset.
- Host: confirm GitHub Pages or substitute.

### 0.3 Confirm toolchain
- Pin a Quarto version. Pin Pagefind. Pin the Python/JAX environment used for executable figures (CPU-only, CI-installable — no HPC, no native FFI build needed to render docs).
- Record all pins in `DECISIONS.md` and a lockfile/`environment.yml`.

**Exit criteria for Phase 0:** `RECON.md` and `DECISIONS.md` exist and are internally consistent. No build work has started yet.

---

## Phase 1 — Federation plumbing on a two-library slice

**Goal:** a working, deployed hub composing exactly two libraries — `nitrix` (deep numerics/math, exercises the textbook path) and `thrux` (higher-level glue, exercises cross-library xref into `nitrix`). Minimal content; maximal mechanism. This is the proof of the whole architecture.

### 1.1 Stand up the umbrella skeleton
Create (per DECISIONS.md):
- `_quarto.yml` (umbrella: global nav placeholder, search via Pagefind, shared theme).
- `theme/federation.scss` — black base; define `$accent` and derive **all** library-distinguishing visuals from it (links, active nav, callout edges, eq/section number accents, code-annotation marks). No other per-library visual knobs.
- `theme/_metadata.yml` — shared format defaults: math config, theorem/proof/definition/algorithm environment styling, callout styles (incl. the fast-path/TL;DR callout, SPEC §6.3), citation style, cross-ref label formats.
- `_bibliography/federation.bib` — seed with the handful of works the slice cites (e.g. Fellner–Schall).
- `libraries.yml` — `nitrix` and `thrux` entries (repo, accent, base_path).
- `xref/namespaces.yml` — `nitrix`, `thrux` namespaces.

### 1.2 Build the xref shortcode (highest-leverage single artifact)
Implement `xref/xref.lua` per SPEC §5.3:
- `{{< xref ns:anchor >}}` and `{{< xref ns:anchor "text" >}}`.
- Composed-build resolution → same-origin relative path.
- Standalone-build resolution → published hub URL.
- Unknown namespace/anchor → **loud build failure** with a clear diagnostic.
Write unit tests over the resolver logic (table of inputs → expected URLs, including the failure case). This filter is federation-critical; treat it as real software with an owner and tests.

### 1.3 Shared-theme distribution
Package `theme/` per the chosen mechanism (extension or pinned asset) so a library can resolve it standalone (SPEC §4.3). Verify `quarto preview` works in a library repo with **only** that repo checked out.

### 1.4 Instantiate the two library docs trees
In `nitrix/docs/` and `thrux/docs/`, create the §3.2 layout:
- `brand.scss` with one `$accent` line each (two distinct colours).
- `_quarto.yml` restricted to project type + nav + brand composition (no shared-format keys — SPEC §4.4).
- One real `explanation/` chapter in `nitrix` that genuinely exercises the textbook stack: LaTeX derivation, a theorem+proof, an algorithm listing, a citation, and one lightweight executable figure. The Fellner–Schall chapter is the natural choice and doubles as real content.
- In `thrux`, one page that uses `{{< xref nitrix:... >}}` to link into that chapter, proving cross-library navigation.
- A curated namespaced anchor in `nitrix` (`#nitrix:fellner-schall`) as the xref target.

### 1.5 Build script + validation + search
Implement `build/build.sh` (SPEC §8): resolve pins → render each library into `_site/<lib>/` → render `federation/` → **validation stage** (broken links, unresolved xrefs, unresolved citations all fail) → Pagefind over `_site/` → output.
Wire the same script into CI (GitHub Actions) and confirm it runs identically on a laptop.

### 1.6 Deploy and verify the slice
Deploy `_site/`. Verify end to end:
- Black theme unified; `nitrix` and `thrux` visibly distinct by accent.
- Cross-library link from `thrux` → `nitrix` works in the composed site.
- Standalone `thrux` preview resolves the same xref to the hub URL (not a dead link).
- Unified Pagefind search returns hits across both libraries.
- A deliberately broken xref fails the build (negative test).

**Exit criteria for Phase 1:** the two-library hub is deployed; all six verifications pass; the negative test fails the build as designed. The architecture is proven.

---

## Phase 2 — Templatize and unblock parallel authoring

**Goal:** make every other library team able to start authoring **in parallel** against a proven frame.

### 2.1 Extract the library docs template
From the `nitrix`/`thrux` trees, factor a `docs/` **template** (cookiecutter-style or a documented copy-me directory): the four Diátaxis sections, `brand.scss` stub, minimal `_quarto.yml`, shared-theme dependency declaration, `_resources/` with the `generated/` cache convention, and a starter `explanation/` chapter skeleton including the TL;DR/fast-path block.

### 2.2 Author the contributor guide
Write `CONTRIBUTING-docs.md` in the umbrella covering:
- How to instantiate `docs/` from the template; how to preview standalone.
- The fast-path/TL;DR convention (SPEC §6.3) with an example.
- Math/theorem/algorithm authoring patterns (the environments defined in shared metadata).
- The executable-figure vs. pre-rendered/HPC-frozen split (SPEC §6.4) with the "freeze" workflow.
- The namespaced-anchor convention and when to expose a cross-library anchor (SPEC §5.2).
- Authored API-reference conventions (SPEC §6.5) and the stub-generation workflow (§4 below).
- Citation rules against the shared bib (one canonical entry per work).

### 2.3 Codify "add a library" (SPEC §10) as a script/checklist
A `add-library` procedure (script + checklist) that performs registry edits (`libraries.yml`, `namespaces.yml`), scaffolds `docs/` from the template, and adds the umbrella build entry. The colour-key legend and global nav must pick the new library up automatically from the registries.

### 2.4 Onboard all remaining libraries (scaffold only)
For every remaining library, run `add-library` to create an empty-but-valid, themed, building `docs/` tree with a distinct accent. After this step the **full federation builds green with placeholder content**, and every team has a working surface to author into independently. This is the moment parallel authoring is unblocked.

**Exit criteria for Phase 2:** all libraries scaffolded and building; contributor guide complete; adding a library is a single documented procedure; teams can author in parallel without cross-repo coordination.

---

## Phase 3 — API-reference scaffolding (drift mitigation)

**Goal:** make authored API reference sustainable without trusting autodoc for content.

### 3.1 Stub generator
Build a thin generator that introspects a library and emits **stub** `.qmd` reference pages — signatures and section skeletons only — into `reference/`. These are scaffolding to be filled by hand/LLM, never shipped as-is (SPEC §6.5). Handle JAX pathologies gracefully (transformed functions, pytree-registered classes, decorators): where introspection is unreliable, emit a clearly-marked TODO stub rather than wrong signatures.

### 3.2 Staleness signal (not a guarantee)
Provide a check that flags reference stubs whose underlying signatures have changed since last fill (e.g. a hash of the introspected signature stored in front-matter). This surfaces drift for human attention; it does not auto-edit prose. Records the SPEC §7 "no freshness guarantee" reality while making drift visible.

**Exit criteria for Phase 3:** running the generator on `nitrix` produces fillable stubs; the staleness check flags a deliberately changed signature.

---

## Phase 4 — Content build-out (the long pole; parallelisable)

**Goal:** populate the textbook and Diátaxis content. This phase is where library teams work concurrently and indefinitely; the system is already live and green throughout.

### 4.1 Per-library content priorities
Following the maintainer's stated focus, sequence within each library as: **explanation (textbook) → tutorials/how-to → reference fill → high-level/aspirational scaffold.** Concretely, early high-value chapters include:
- `nitrix`: Fellner–Schall as a core of the statistical suite + the GP-extension path (already seeded in Phase 1; deepen it).
- `niffi`: the "binary drive" FFI model for FreeSurfer — rationale and bundling/compilation/distribution implications for HPC users without a system compiler.
- Per-module chapters wherever design choices diverge.

### 4.2 Cross-library tutorials
Author end-to-end tutorials that legitimately live in higher-level libraries (`thrux`, `bitsjax`, `entense`) while xref-ing concepts down into substrate libraries (SPEC §6.1). These stress-test the xref system under real authoring load.

### 4.3 The aspirational high-level layer
Build the everyday-user landing/explanation scaffold in the hub (`federation/`) and high-level libraries as a deliberately-marked aspirational skeleton, to be fleshed out as those components mature. Keep it clearly delineated from the deep developer/advanced-user material that is today's substance.

### 4.4 Periodic integration
On a cadence (or per sprint), bump library SHA pins in the umbrella, run the full validating build, and redeploy. This is the only cross-team synchronisation point and is a deliberate, reviewable act (SPEC §3.4, §9).

**Exit criteria for Phase 4:** rolling — defined per library/sprint, not a single gate.

---

## Phase 5 — Hardening and ongoing governance

- **PDF/textbook output:** enable Quarto's HTML+PDF rendering for `explanation/` trees so chapters can also ship as a printable textbook artifact (a stated long-term draw of choosing Quarto). Validate math/figures survive the PDF path.
- **Drift control checks in CI:** the config-drift detector (SPEC §4.4 / §7.5), bib-deduplication check, and xref/citation/link validation run on every PR in every library repo, not only in the umbrella.
- **Shared-surface change discipline:** operationalise SPEC §9 — theme/metadata changes versioned and rolled out by pin bumps; bib additive and deduplicated; xref-namespace renames trigger a federation-wide sweep.
- **Owner assignment:** name an owner for each shared surface (theme, bib, xref filter) and for the umbrella build.

---

## Critical path and parallelism summary

```
Phase 0  (recon + decisions)            ── sequential, blocking, fast
   │
Phase 1  (2-library plumbing proof)     ── sequential, the keystone
   │        ⮡ xref.lua + theme contract + build script are the core artifacts
   │
Phase 2  (templatize + scaffold all)    ── unblocks ALL teams
   │
   ├── Phase 3 (API stub tooling)        ┐
   └── Phase 4 (content build-out)       ├─ run concurrently, many teams, ongoing
            ⮡ per-library, parallel      ┘
                  │
Phase 5  (hardening + governance)        ── overlaps Phase 4, continuous
```

The hard gate is Phase 1: nothing scales until the xref mechanism, theme contract, and validating build are proven on two libraries. Once Phase 2 lands, the federation's teams proceed in parallel exactly as required, integrating only at deliberate pin bumps.

---

## First concrete actions for Claude Code

1. Execute Phase 0: produce `RECON.md` and `DECISIONS.md`. **Do not build before these exist.**
2. Stand up the Phase 1 umbrella skeleton and the `xref.lua` filter **with tests** first.
3. Wire `nitrix` (Fellner–Schall chapter) + `thrux` (xref into it) into a deployed two-library hub with unified Pagefind search and a passing negative xref test.
4. Only then extract the template (Phase 2) and scaffold the rest of the federation.

Report back after Phase 1 with the deployed slice and the six verifications (SPEC §1.6 equivalents) before proceeding.
