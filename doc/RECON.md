# Federation Documentation — Reconciliation (RECON)

**Phase 0 deliverable** (PLAN §0.1). Surveys the *actual* code/doc state of the
federation and assigns each library a verdict. Companion to `DECISIONS.md`.

> **Headline:** Build the new hub as **greenfield**. Inherit *no* legacy doc
> system. Salvage legacy *content/structure* only where it is (a) currently
> accurate and (b) useful — and verify every salvaged line against current code
> before reuse.

---

## 0. Two corrections that override structural appearances

1. **Existing docs have drifted and are mostly stale.** With one exception
   (`tensorbids`), the libraries have been substantially refined since their docs
   were written. The structure may look mature (full Diátaxis trees in `thrux`,
   `conveyant`); the *content* is out of date. **`thrux`'s docs are drifted far
   enough to be actively misleading** and should be treated as harmful, not
   salvaged as prose. `tensorbids` is current only because development focus moved
   to lower substrates after its first-pass scaffold.

2. **No legacy doc system was ever deployed.** The Quarto+quartodoc hub
   (`doc/_source` + `_scripts/build.py`) and its CI (`.github/workflows/doc.yml`,
   present only on `dev`, never merged to `main`) were a **local proof of
   concept** — mostly exploring *what the API reference should look like*. The
   legacy `hypercoil/docs/` Sphinx tree is legacy. Neither is a foundation to
   evolve; both are references at most.

   *Worth keeping from the PoC:* the **rendered API-reference style** of
   `build.py`'s custom renderer (source links, See-Also tables, parameter
   annotations). The maintainer liked that output; it should inform the look of
   the new reference layer — without inheriting the hacked quartodoc plumbing.

---

## 1. Existing documentation systems (none inherited)

| System | Location | Tech | Status | Disposition |
|---|---|---|---|---|
| Quarto + quartodoc hub | `hypercoil/doc/_source`, `_scripts/build.py` | Quarto + quartodoc (custom `HBuilder`/`HRenderer`), `vapormod.scss` | Local PoC; CI never merged to `main` | **Do not inherit.** Mine the reference-rendering ideas only. |
| Legacy Sphinx | `hypercoil/docs/` | Sphinx + pydata + numpydoc (272/287 autodoc) | Legacy; references ghost lib `hyve` | **Retire.** |
| Per-library Diátaxis Quarto | `conveyant`, `thrux`, `tensorbids` (stock `cosmo`) | Quarto; great-docs `auto:true` in conveyant+tensorbids | Standalone, no CI; **content drifted except tensorbids** | **Reuse structure; rewrite prose** (except tensorbids = migrate). |

Genuinely absent everywhere (the new build's net-new core): namespaced
cross-library xref, federation-wide search (Pagefind), a shared bibliography, a
black/`$accent` theme contract, and any shared-theme distribution mechanism.

---

## 2. Per-library reconciliation

Verdict legend — **greenfield**: author fresh; **adapt**: keep structure, rewrite
content against current code; **migrate**: content is current, port it.

| Library | Federation role | Code maturity | Existing docs | Currency | Verdict |
|---|---|---|---|---|---|
| **nitrix** | L0 numerics (stats/geometry/registration) | High (~49k LOC, 19 subpkgs); Fellner–Schall + REML/LME/GLMM/GAMM all ship | design/ + feature-requests/ md; **no Quarto** | Drifted — use as *signal*, verify before lifting | **greenfield** (rich, but stale source) |
| **niffi** | L0 FFI to community C/C++ kernels | ~4k LOC core + 4 suite plugins | root SPEC.md/DESIGN_FORKS/etc. md | Design-current-ish; binary-drive (L3) **retired as build strategy**, kept as oracle | **greenfield** (frame around L0–L3 reachability tiers) |
| **ilex** | L0 ML inference + HL training | ~8k LOC; inference-wrap mature, training partial | `docs/design/*.md`, `notes.md` | Mixed; doc by maturity tier | **greenfield**, staged |
| **thrux** | L1 glue (I/O, in-mem dispatch/graph) | High (~12k LOC) | full Diátaxis Quarto | **Drifted — actively harmful** | **adapt** (keep nav skeleton; **rewrite/delete stale prose**) |
| **bitsjax** | L2 glue (dataset awareness over BITS) | Light/early | 1 feature-request note | n/a | **greenfield** |
| **tensorbids** | util (BITS on-disk reference impl) | High (~9k LOC); + 6 SVGs, spec md | full Diátaxis Quarto + great-docs | **Current** | **migrate** (best-in-federation; verify) |
| **conveyant** | util (semantic-typed composition) | Compact, mature | full Diátaxis Quarto + great-docs | Drifted | **adapt** (keep structure, rewrite) |
| **gramform** | util (Wilkinson/nwx/fslmaths grammars) | Production grammars | `docs/nwx/*.md` design | Design-ish; no Quarto | **greenfield** (rich source) |
| **entense** | HL ETL/workflow into BITS | Phase 0 (design frozen) | root design md; no docs/ | Design-current | **greenfield** |
| **hyve** | HL visualisation | **Real code, in limbo (not in tree)** | legacy stubs only | n/a | **aspirational** until code lands |

### Siblings / out of core scope
- **lytemaps** — neuromaps fork (data access). External sibling; link, don't make a core chapter (confirm).
- **ilex-import** — 28-model porting factory; companion toolkit, document separately from ilex.
- **nitrix-perf-bench** — benchmark dashboard with its own Pages CI; external.
- **nitrix-main** — stale duplicate checkout of the `nitrix` remote; **ignore**.
- **nimox** — vendored subpackage at `ilex/src/ilex/nimox`; not a standalone repo.
- **paranox** — **phased out**; remove from all rosters and the legacy `_pkgidx.yml`.

---

## 3. Roster correction (for SPEC §3.2)

Core federation libraries with code today: `nitrix`, `niffi`, `ilex`, `thrux`,
`bitsjax`, `entense`, `gramform`, `conveyant`, `tensorbids`, `hypercoil`.
Plus `hyve` (real code, in limbo → aspirational docs slot now). Drop `paranox`.
Treat `lytemaps`, `ilex-import`, `nitrix-perf-bench` as siblings. `nimox` is
internal to `ilex`.

---

## 4. Environment note (PLAN §0.3)

Quarto, Pagefind, quartodoc and great-docs are **not installed** in this working
environment. The toolchain must be pinned and installed before any build work
(`DECISIONS.md` D-10).
