# Phase 3 — API-reference scaffolding (drift mitigation)

**Status: complete.** Authored API reference is now sustainable without trusting
autodoc for content: a generator seeds fillable stubs, and a staleness signal
surfaces drift for human attention.

## What was built

| Artifact | Path | Notes |
|---|---|---|
| Stub generator | `doc/build/gen_reference_stubs.py` | **Static / AST-based** — never imports the library, so it is CI-safe and immune to JAX pathologies (jit/vmap-transformed callables, pytree classes, decorators). Reads each module's `__all__`; emits one `.qmd` per public module with rendered signatures + Parameters/Returns skeletons and a `signature-hash`. Non-destructive (skips filled stubs unless `--force`); aliases/re-exports → clearly-marked TODO. |
| Staleness signal | `doc/build/check_reference_staleness.py` | Recomputes each stub's hash from current source (shared logic with the generator) and flags mismatches; exit non-zero for CI. `--update` re-blesses after prose is updated. Does not auto-edit prose (SPEC §7.7). |

## Verification (PLAN §3 exit criteria)

- **Generator produces fillable stubs on nitrix.** Run scoped to `nitrix.stats`:
  17 clean public-surface stubs in `nitrix/docs/reference/` (private `_*` modules
  and re-export `__init__` hubs correctly excluded). Signatures are accurate from
  source — full jaxtyping annotations preserved (`Float[Array, 'V N']`, …) and
  decorators surfaced as hints (`@register_result`, `@dataclass`).
- **Staleness check flags a deliberately changed signature.** Adding a parameter
  to `pca_transform` made the check report `signature DRIFT` on
  `nitrix-stats-pca.qmd` and exit 1; reverting the source returned it to
  `17 current, exit 0`.

## Notes / decisions

- Generated stubs are **not** added to any `render:` list — they are scaffolding to
  be filled, never shipped as-is (SPEC §6.5). Authors wire pages into nav as they
  fill them.
- Static AST introspection is deliberately preferred over runtime import: it
  documents the *source* signature (the authored intent) rather than a transformed
  wrapper, needs no heavy/native deps, and degrades to an explicit TODO rather than
  emitting a wrong signature.
- The generated `nitrix/docs/reference/*.qmd` live in the nitrix repo (committed
  separately from the umbrella).

## Next

- **Phase 4** — content build-out (the long pole): fill the seeded reference stubs;
  author real chapters per library; rewrite `thrux`/`conveyant` from their
  `.legacy` source; deepen the nitrix GP path.
- Optional: add the staleness check as a CI gate per library; full
  link/figure/citation validation stage (SPEC §7.3–7.4).
