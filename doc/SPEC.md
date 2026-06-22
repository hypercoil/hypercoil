# Federation Documentation System — Specification

**Status:** Draft for triage
**Audience:** Claude Code (implementation), federation maintainers, per-library doc authors
**Scope:** Defines *what* the documentation system is and the contracts it must satisfy. The companion `PLAN.md` defines *how* and *in what order* to build it.

> **Note for the triaging agent.** This spec describes a target state. Before implementing, reconcile it against what already exists in each library repo (any current `docs/`, Sphinx/MkDocs config, READMEs, docstrings, notebooks). Where prior art exists, prefer adapting it to these contracts over greenfield replacement. Flag any contract here that conflicts with an entrenched existing pattern rather than silently overriding it.

---

## 1. Goals and non-goals

### 1.1 Goals

1. **One federation, many libraries, unified feel.** A single documentation hub spanning all federation libraries, sharing a black theme and global navigation/search, while each library is visually distinguishable by a single bold accent colour that signals location.
2. **Textbook-grade explanation as a first-class deliverable.** LaTeX math, proofs, derivations, algorithm listings, citations, and executable figures are core, not bolted on. The explanatory layer is the primary product; it must read like a textbook with a "fast path" to minimal working understanding.
3. **Full Diátaxis coverage per library.** Tutorials, how-to guides, explanation, and reference, each as a distinct mode, with explanation/tutorial being today's focus.
4. **Authored (not autodoc-dependent) API reference.** API reference is important but is authored with full layout control. Autodoc, if used at all, produces *scaffolding* only, never final content.
5. **Parallel, non-blocking authoring.** Each library team builds and previews its own documentation independently in its own repo, without coordinating working trees with other teams. Integration happens at composition time via version pins.
6. **Cross-library navigation.** Pages in one library can reference concepts, sections, equations, and tutorials in another via a stable, namespaced cross-reference mechanism. Search spans the entire federation.
7. **Low-friction extension.** Adding a new library to the federation is a bounded, documented procedure: pick an accent colour, add a docs tree from a template, register an xref namespace, add a build entry.
8. **Reproducible builds anywhere.** The hub builds identically via CI or on a maintainer's laptop with no CI-only steps. Building does not require HPC or special privileges.

### 1.2 Non-goals

- The hub does **not** require building any federation library's compiled/native components to render docs. Executable figures use lightweight, pre-installable code paths; heavy/HPC-only computations are pre-rendered and cached, never run at doc-build time.
- The hub does **not** enforce the fast-path/deep-path duality mechanically. It is a content-authoring convention (see §6.3).
- The hub is **not** a replacement for in-code docstrings; it consumes/links them where useful but owns the authored narrative separately.
- No mechanical guarantee of API-reference freshness is promised in v1 (see §7 and the drift-mitigation in PLAN.md).

---

## 2. Substrate decision

**Substrate: Quarto.** Rationale (recorded so the triaging agent does not relitigate it without cause):

- Pandoc-native LaTeX math, theorem/proof environments, cross-references, BibTeX/CSL citations, and multi-format output (HTML + PDF) — directly serves Goal 2.
- Native execution of embedded code (Python/JAX) to produce inline figures — serves Goal 2's "executable figures."
- Per-project SCSS theming via variable overrides — serves Goal 1's accent contract cheaply.
- Authored `.qmd` pages give total layout control over API reference — serves Goal 4 and matches the maintainer's prior experience that manual reference beat autodoc for this federation.

Sphinx's decisive advantage (mature autodoc) is explicitly waived (Goal 4). MkDocs is rejected for weaker math, weaker cross-project linking, and weaker executable-content support.

**Cross-project gaps Quarto does not solve natively, and their designated solutions:**

| Gap | Solution | Section |
|---|---|---|
| Cross-project cross-references | Namespaced xref shortcode (federation-owned) | §5 |
| Federation-wide search | Pagefind over final merged `_site/` | §5.4 |
| Per-project config drift | Shared `_metadata.yml` + theme contract | §4.4 |

---

## 3. Repository topology

### 3.1 Split authoring location from build location

**Authoring is per-repo. Composition is centralised.** This is the mechanism that makes parallel team development (Goal 5) work while keeping docs versioned with their code.

### 3.2 Per-library repo layout

Every federation library repo (`nitrix`, `niffi`, `ilex`, `thrux`, `bitsjax`, `hyve`, `entense`, `gramform`, `conveyant`, `tensorbids`, …) contains a `docs/` subtree:

```
<library>/
  docs/
    _quarto.yml          # project config: accent brand, this library's nav tree
    _metadata.yml        # (symlink or fetched copy of) shared format/math defaults
    index.qmd            # library landing page
    explanation/         # textbook chapters (primary focus)
    tutorials/           # Diátaxis: learning-oriented
    howto/               # Diátaxis: task-oriented
    reference/           # Diátaxis: authored API reference
    _resources/
      figures/           # static + generated figure assets
      generated/         # cached outputs of expensive/HPC computations
    brand.scss           # EXACTLY ONE library-specific line: $accent
```

A library team can run `quarto preview docs/` in isolation and see a fully themed, standalone site for their library, with the shared theme pulled as a pinned dependency (see §4.3). No other repo need be present.

### 3.3 The umbrella repo

A federation documentation repo — `hypercoil-docs`, or a `docs/` hub inside the existing `hypercoil` meta-repo (the triaging agent chooses based on what exists; `hypercoil` is already the "keep the federation consistent" package and is the natural owner) — holds the composition layer:

```
hypercoil-docs/
  _quarto.yml            # umbrella site config: global nav, search, shared theme
  theme/
    federation.scss      # black base; ALL shared visual structure; the accent contract
    _metadata.yml        # shared format defaults consumed by every library
  _bibliography/
    federation.bib       # single source of truth for citations
  xref/
    namespaces.yml       # registry of library xref namespaces (see §5.2)
    xref.lua             # the cross-reference shortcode/filter (see §5.3)
  federation/            # cross-cutting material owned by NO single library
    index.qmd            # the hub landing page (+ colour-key legend, §4.2)
    why-this-federation.qmd
    ...
  libraries.yml          # registry: each library, its repo, pinned SHA/tag, accent
  build/
    build.sh             # the canonical build script (§8)
  _site/                 # build output (gitignored)
```

### 3.4 How the umbrella pulls library docs

Each library's `docs/` is brought into the umbrella build at a **pinned SHA or tag**, via either git submodules or a fetch step driven by `libraries.yml`. The triaging agent selects the mechanism (submodules are more standard; a fetch step is lighter to automate) and records the choice. Either way:

- The umbrella always builds against a **coherent, pinned snapshot** of the federation, mirroring how `hypercoil` pins a consistent set of library versions.
- A library advancing its docs does **not** change the hub until its pin is deliberately bumped. This is the answer to dependency drift: integration is an explicit, reviewable act.

### 3.5 Shared mutable surfaces (the only cross-team coupling)

Exactly three artifacts are shared and mutable; everything else is library-local. These are versioned, owned, and changed with discipline (see §9):

1. `theme/federation.scss` + `theme/_metadata.yml` — the visual/format contract.
2. `_bibliography/federation.bib` — citations.
3. `xref/namespaces.yml` — the cross-reference namespace registry.

---

## 4. Theming contract

### 4.1 Black base, one accent variable

`theme/federation.scss` defines the entire black-themed visual structure. Every library-distinguishing visual (link colour, active nav highlight, callout edge, equation/section number accent, code-annotation marks) **must derive from a single SCSS variable `$accent`.** No library overrides anything else visual.

### 4.2 Per-library brand

Each library's `docs/brand.scss` contains effectively one meaningful line:

```scss
$accent: #RRGGBB;   // this library's bold accent colour
```

Its `_quarto.yml` composes the theme:

```yaml
theme:
  - federation.scss     # resolved from the pinned shared theme
  - brand.scss
```

Adding a library to the palette is therefore one line. The hub landing page (`federation/index.qmd`) renders a **colour-key legend** mapping accent → library, generated from `libraries.yml` so it cannot drift from reality.

### 4.3 How standalone library builds get the shared theme

So a team can build in isolation (Goal 5), the shared theme must be resolvable without the umbrella checked out. Designated mechanism: the shared theme is published as a small versioned **Quarto extension** (or a pinned fetched asset) that each library declares as a dependency at a pinned version. Standalone `quarto preview` resolves it; the umbrella build overrides with its own pinned copy to guarantee hub-wide consistency. The triaging agent confirms the chosen distribution mechanism and records it.

### 4.4 Config-drift control

A shared `_metadata.yml` (Quarto's directory-level metadata) carries format defaults: math config, callout styling, code-block/annotation defaults, citation style, cross-ref label formats. Every library consumes it. Library `_quarto.yml` files are restricted to: project type, nav tree, brand composition, and library-local options. **Anything visual or format-wide that appears in a library `_quarto.yml` is a drift bug** and should be lifted into the shared layer.

---

## 5. Cross-library navigation

### 5.1 Requirement

A page in any library must be able to reference a section, equation, theorem, definition, tutorial, or API entry in any other library, with links that survive both standalone and composed builds, and without cross-origin friction.

### 5.2 Namespace registry

`xref/namespaces.yml` is the authoritative registry. Each library owns a short namespace token:

```yaml
namespaces:
  nitrix:     { repo: "...", base_path: "/nitrix/" }
  thrux:      { repo: "...", base_path: "/thrux/" }
  bitsjax:    { repo: "...", base_path: "/bitsjax/" }
  # ...
```

Within a library, authors use ordinary Quarto cross-refs (`@sec-…`, `@eq-…`, `@thm-…`). Anchors that are intended to be referenced *across* libraries must use a **stable, namespaced label convention**: `#<namespace>:<slug>` (e.g. `#nitrix:fellner-schall`). Cross-library-referenceable anchors are a deliberate, curated subset — not every heading is a public anchor.

### 5.3 The xref shortcode

`xref/xref.lua` is a federation-owned Lua filter/shortcode. Authoring form:

```
{{< xref nitrix:fellner-schall >}}
{{< xref thrux:graph-assembly "custom link text" >}}
```

It resolves a namespaced reference to the correct cross-project URL using `namespaces.yml`:

- In **composed (umbrella) builds**, resolves to a same-origin relative path (e.g. `/nitrix/explanation/statistics.html#fellner-schall`).
- In **standalone builds**, resolves to the published hub URL for that target (so a team previewing in isolation gets a working external link rather than a dead one).
- On an **unknown or unregistered namespace/anchor**, fails the build loudly with a clear diagnostic. Broken cross-references are build errors, not silent 404s.

This filter is a real internal tool with an owner and tests (see PLAN.md). It is the documentation analogue of `conveyant`'s semantic-typing role: it makes cross-library composition safe.

### 5.4 Federation-wide search

After all projects render into the merged `_site/`, a **Pagefind** post-build step indexes the final static HTML across the whole tree, producing one search index and one search UI over the entire federation. Per-project Quarto search is disabled in favour of the unified index to avoid fragmented search boxes.

### 5.5 Single-origin deployment

All projects render into one output tree under one origin: `/` (hub), `/nitrix/`, `/thrux/`, …. Cross-library links are therefore plain relative paths with no cross-origin concerns.

---

## 6. Content model

### 6.1 Diátaxis per library

Each library provides the four Diátaxis modes as distinct top-level sections: `tutorials/`, `howto/`, `explanation/`, `reference/`. The hub navigation surfaces these consistently across libraries so users learn one mental model. A tutorial or how-to may legitimately live in a *higher-level* library while referencing concepts in a lower one (e.g. an end-to-end tutorial in `thrux` or `bitsjax` exercising `nitrix` numerics); cross-library xrefs (§5) make this first-class.

### 6.2 Explanation = the textbook layer

The `explanation/` tree is the textbook. Chapters must support and are expected to use:

- LaTeX math, numbered and cross-referenceable equations.
- Theorem/lemma/proof/definition environments with consistent styling (defined once in the shared theme/metadata).
- Algorithm listings (pseudocode) with consistent formatting.
- Citations against the shared `federation.bib`.
- Executable figures: embedded code (JAX/Python) that runs at build time to produce plots and numerical demonstrations, subject to §6.4.
- Per-library and, where design choices differ, **per-module** chapters.

The bar is: a reader who "knows linear algebra" or "can code" can reach deep understanding of design rationale, mathematical theory, algorithms, and extension paths (e.g. *why* Fellner–Schall anchors the `nitrix` statistical suite and *how* one would extend it to Gaussian-process models; *why* a "binary drive" FFI model for FreeSurfer in `niffi` and its bundling/compilation/distribution implications for HPC users without a system compiler).

### 6.3 Fast path / deep path (authoring convention)

Each chapter opens with a brief, self-contained **TL;DR / fast-path** block giving minimal working understanding before the full treatment. This is a **house-style convention, not a mechanism**: applied where it fits, omitted where a page shouldn't be forced to conform. The shared theme provides a recognisable callout style for it; the spec does not enforce its presence.

### 6.4 Executable-content / HPC boundary

Doc builds must not require HPC, special privileges, or heavy native compilation (Goal/Non-goal §1.2). Therefore:

- Lightweight examples execute at build time.
- Expensive, HPC-only, or non-reproducible-in-CI computations are **pre-rendered**; their outputs are cached under `docs/_resources/generated/` and committed (or fetched), and the chapter consumes the cached artifact. Authoring guidelines must make this split explicit and provide a "freeze" workflow.

### 6.5 Authored API reference

`reference/` pages are authored `.qmd` with full layout control. Autodoc is permitted only to emit **stubs** (signatures + section skeletons) that are then filled by hand/LLM (see PLAN.md drift mitigation). No page's final reference content is autodoc output.

---

## 7. Quality and consistency requirements

1. **Build is reproducible and host-agnostic** — identical steps in CI and locally (§8).
2. **Broken cross-references fail the build** (§5.3).
3. **Broken intra-project links and missing figure assets fail the build** (link-check stage).
4. **Citations resolve against the shared bib**; unresolved citation keys fail the build.
5. **Config drift is detectable** — a check flags visual/format keys appearing in library `_quarto.yml` that belong in the shared layer (§4.4).
6. **Accent legend matches reality** — the hub colour-key is generated from `libraries.yml`, not hand-maintained.
7. **No freshness guarantee for API reference in v1** — accepted; mitigated by stub generation, not promised by the system.

---

## 8. Build topology

Canonical build (`build/build.sh`), identical under CI (GitHub Actions preferred) or local laptop:

```
1. Resolve federation snapshot: checkout/fetch each library's docs/ at its pinned SHA
   (per libraries.yml / submodules).
2. For each library: quarto render docs/ → staged into _site/<library>/
   using the shared theme + _metadata.yml + xref filter + shared bib.
3. Render federation/ (umbrella textbook + landing + legend) → _site/.
4. Run link/xref/citation validation across _site/ (fail on error).
5. Pagefind: index _site/ → unified search.
6. Deploy _site/ (Pages, or chosen host).
```

No step is CI-exclusive. Local builds run the same script and are fully debuggable. Per-library standalone builds (`quarto preview docs/`) are a subset of step 2 with the shared theme resolved via §4.3.

---

## 9. Change discipline for shared surfaces

The three shared mutable surfaces (§3.5) are governed:

- **Theme/metadata** (`theme/`): changes are reviewed by the federation-docs owner; because every library renders against them, changes are versioned and libraries pin a theme version (§4.3). A theme change is rolled out by bumping pins, not by surprise.
- **Bibliography** (`federation.bib`): additive by default; one canonical entry per work (no per-library divergent citations of the same source). Deduplication is enforced.
- **xref namespaces** (`namespaces.yml`): adding a namespace is required when onboarding a library; renaming/removing one is a breaking change requiring a federation-wide reference sweep.

---

## 10. Adding a new library (procedure, to be realised by PLAN.md)

1. Pick an unused accent colour; add the library to `libraries.yml` (repo, accent, base_path).
2. Add an xref namespace in `namespaces.yml`.
3. Instantiate the `docs/` tree from the template (§3.2), set `brand.scss`'s `$accent`, declare the shared-theme dependency at the current pinned version.
4. Add the library to the umbrella build (pin its SHA).
5. The colour-key legend and global nav pick it up automatically from the registries.

---

## 11. Open decisions for the triaging agent to resolve and record

- **Umbrella home:** standalone `hypercoil-docs` repo vs. `docs/` inside existing `hypercoil`. Decide from what exists; record rationale.
- **Library-docs ingestion:** git submodules vs. registry-driven fetch. Decide and record.
- **Shared-theme distribution:** Quarto extension vs. pinned fetched asset. Decide and record.
- **Pre-existing docs reconciliation:** for each library, inventory current docs/config/notebooks and produce a per-library migration note (adapt vs. replace vs. greenfield).
- **Host:** GitHub Pages assumed; confirm or substitute.
