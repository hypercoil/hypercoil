# Federation Documentation — Decisions (DECISIONS)

**Phase 0 deliverable** (PLAN §0.2). Resolves SPEC §11 open questions plus the
issues surfaced during triage. Companion to `RECON.md`.

Status legend — **DECIDED**: settled (by SPEC or maintainer); **PROPOSED**:
recommended default, awaiting maintainer confirmation.

---

### D-1 · Substrate — **DECIDED: Quarto**
Per SPEC §2 (LaTeX math, theorem/proof envs, citations, HTML+PDF, executable
figures, authored layout). Not relitigated.

### D-2 · Build posture — **DECIDED: greenfield**
The new hub inherits **no** legacy doc system. Legacy Sphinx (`hypercoil/docs/`)
is retired; the Quarto+quartodoc PoC (`doc/_source` + `_scripts/build.py`) is not
evolved. Legacy *content/structure* is salvaged only where currently accurate and
useful, and is verified against current code before reuse (`RECON.md`).
*Carry-over:* reproduce the **API-reference rendering style** the PoC achieved
(source links, See-Also tables, parameter annotations) in the new reference layer.

### D-3 · API-reference / autodoc policy — **DECIDED (resolves SPEC §6.5)**
Autodoc is **permitted to seed reference _stubs_ only** (signatures + section
skeletons), via a great-docs/quartodoc-style generator. Stubs are then hand-
refined. Autodoc output is **never the final reference**, and the generated
reference **coexists with**, and does not displace, the hand-authored
textbook-grade reference. No mechanical freshness guarantee in v1.

### D-4 · Reference lives in-repo; CI assembles — **DECIDED**
Each library's documentation **including its API reference** lives in that
library's own repo, next to the code, to minimise drift. The CI/build system
**clones the repos at pinned refs and assembles the aggregate** hub.

### D-5 · Library-docs ingestion — **PROPOSED: registry-driven clone/fetch at pinned refs**
A `libraries.yml` registry drives a CI clone/fetch of each repo's `docs/` at a
pinned SHA/tag (lighter to automate than submodules; matches D-4's "CI clones the
repos"). Submodules remain acceptable if preferred. *Confirm.*

### D-6 · Umbrella home — **PROPOSED: `hypercoil/doc/` (this directory)**
`hypercoil` is the federation-consistency package and the natural owner; SPEC/PLAN
and these artifacts already live here. Keep the hub here rather than a separate
`hypercoil-docs` repo, unless a separate repo is wanted for access/release
reasons. *Confirm.*

### D-7 · Theme — **DECIDED: black base + single `$accent` per library**
Per SPEC §4. All library-distinguishing visuals derive from one `$accent` SCSS
variable. The legacy `vapormod.scss` vaporwave theme is **not** carried forward.

### D-8 · Shared-theme distribution — **PROPOSED: pinned Quarto extension**
Publish the shared theme + `_metadata.yml` as a small versioned Quarto extension
each library declares at a pinned version, so standalone `quarto preview` resolves
it without the umbrella; the umbrella overrides with its own pin at compose time.
A pinned fetched asset is the fallback. *Confirm.*

### D-9 · Cross-library xref — **DECIDED: federation-owned `xref.lua` + `namespaces.yml`**
Net-new (no existing mechanism; legacy `interlinks` only targets external docs).
Built with tests as PLAN §1.2's keystone. Unknown namespace/anchor fails the build.

### D-10 · Search — **DECIDED: Pagefind over merged `_site/`**
Single federation index; per-project Quarto search disabled (SPEC §5.4).

### D-11 · Host — **PROPOSED: GitHub Pages**
Per SPEC §11 assumption. *Confirm* (or substitute).

### D-12 · Toolchain pins — **DECIDED (Phase 1)**
Pinned and installed reproducibly under a configurable prefix (default `/scratch`
here) by `doc/_tooling/bootstrap.sh`: **Quarto 1.9.38** (standalone tarball;
bundles pandoc + Lua), **Pagefind 1.5.2** (pip `pagefind[extended]`), and a
CPU-only figure venv (numpy/matplotlib + the jupyter execution stack). Versions
in `doc/_tooling/versions.env` + `requirements-docs.txt`; exact transitive
versions frozen to `requirements-docs.lock.txt`. The same script bootstraps a
laptop and CI. Executable figures use numpy/matplotlib only — no JAX/nitrix/GPU
at doc-build time (SPEC §6.4).

### D-13 · Stale-content handling — **DECIDED**
Drifted existing prose is not migrated as-is. `thrux` docs are treated as harmful
and rewritten (stale pages removed so they cannot mislead). Salvaged structure
(nav, Diátaxis skeleton) is retained; salvaged prose is re-verified against current
code. `tensorbids` is the lone current-content migration.

### D-14 · Roster — **DECIDED (see RECON §3)**
Core: nitrix, niffi, ilex, thrux, bitsjax, entense, gramform, conveyant,
tensorbids, hypercoil. `hyve` = aspirational slot (code in limbo). Drop `paranox`.
`lytemaps`/`ilex-import`/`nitrix-perf-bench` = siblings. `nimox` internal to ilex.

---

## Open items requiring maintainer confirmation
D-5 (ingestion), D-6 (umbrella home), D-8 (theme distribution), D-11 (host),
D-12 (toolchain pins), and the sibling-scope of `lytemaps`/`ilex-import` in the hub.
