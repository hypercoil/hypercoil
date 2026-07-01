#!/usr/bin/env python3
"""Inventory docstring pathology and staleness across a federation library.

Static (``griffe``-based, no import) — walks every module and catalogues, per
function / class / method, the docstring problems that a later migration (or a
human) should fix. It analyses the *docstring text* griffe extracts, so a
``SPEC §x`` in a code comment (legitimate) is never confused with one in a
docstring (a public-doc leak).

Detected categories:

  staleness (docstring vs signature)
    * missing_docstring         — public symbol with no docstring
    * undocumented_params       — signature params absent from the Parameters block
    * documented_absent_params  — Parameters block names not in the signature
    * param_order_mismatch      — documented order ≠ signature order
    * returns_undocumented      — a return annotation but no Returns block
  markup / rendering
    * math_in_code              — RST-literal math the generator now renders ($…$)
    * math_in_code_unrendered   — subscript-only math still rendered as code
    * sphinx_role / unresolvable_role
    * bare_xref_candidate       — a code-span identifier that names a known symbol
    * rst_directive             — ``.. note::`` etc. (won't render in Quarto md)
    * sphinx_field / doctest / rst_link
  content hygiene
    * internal_ref              — SPEC/PLAN/FR/§/v3 § leaking into a docstring
    * citation_without_doi      — a citation with no DOI / URL / arXiv id
    * todo_marker               — TODO / FIXME / XXX / HACK
    * summary_issue             — missing / multi-line / unterminated summary line

Writes ``<out>.json`` (machine-readable, to drive a fan-out migration) and
``<out>.md`` (a human-reviewable report), and prints a summary.

Usage:
  inventory_docstrings.py nitrix [--src P] [--out P]
"""
import argparse
import json
import pathlib
import re
import sys

import gen_reference_stubs as g
from griffe import Parser


CODE = g.CODE

LITERAL_RE = re.compile(r"``([^`]+)``")
INTERNAL_REF_RE = re.compile(
    r"\b(?:SPEC(?:_UPDATE)?|PLAN|RECON|FR)\b\s*§?\s*[\d.]*"
    r"|§\s*\d[\d.]*"
    r"|\bv\d+\s*§\s*[\d.]*")
# Only flag RST directives the reference generator does NOT translate to Quarto;
# math / code / admonitions are handled (rendered) so they are not pathology.
RST_DIRECTIVE_RE = re.compile(
    r"^[ \t]*\.\.[ \t]+(?!(?:math|code-block|code|note|warning|tip|caution|"
    r"important|admonition)::)[\w-]+::", re.M)
SPHINX_FIELD_RE = re.compile(r"^[ \t]*:(?:param|parameter|type|returns?|rtype|raises?|yields?|ivar|cvar|var)\b", re.M)
DOCTEST_RE = re.compile(r"^[ \t]*>>> ", re.M)
RST_LINK_RE = re.compile(r"`[^`]+<[^>]+>`_")
TODO_RE = re.compile(r"\b(TODO|FIXME|XXX|HACK|WIP)\b")
DOI_RE = re.compile(r"\b10\.\d{4,}/|doi\.org|arxiv|https?://", re.I)
CITATION_RE = re.compile(
    r"\bet al\.?"
    r"|\b[A-Z][A-Za-z]+ (?:&|and) [A-Z][A-Za-z]+,? \(?\d{4}\)?"
    r"|\b[A-Z][A-Za-z]+ \(?(?:19|20)\d{2}\)?"
    r"|\[\d+\]")
IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# A Sphinx role, capturing the trailing role word (e.g. func, class, math) so
# :math:`…` (which is math, not a cross-reference) can be excluded.
XREF_ROLE_RE = re.compile(r":(?:[a-zA-Z][\w.+-]*:)*(?P<role>[a-zA-Z][\w+-]*):`(?P<t>[^`]+)`")
# Pytree / protocol plumbing that conventionally carries no docstring.
BOILERPLATE_METHODS = {"__init__", "tree_flatten", "tree_unflatten",
                       "tree_flatten_with_keys"}
GREEK_ALT = ("alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|"
             "nu|xi|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega")
# A math subscript: a single-letter or Greek base, then a single-char / numeric /
# Greek subscript — NOT a snake_case identifier (n_groups, p_value stay code).
SUBSCRIPT_MATH_RE = re.compile(
    r"(?:^|[^A-Za-z0-9_.])(?:[a-zA-Z]|" + GREEK_ALT + r")"
    r"_(?:[0-9]+|[a-zA-Z](?![A-Za-z])|" + GREEK_ALT + r")")


# --------------------------------------------------------------- symbol index

def _kind(obj):
    k = getattr(obj, "kind", None)
    return getattr(k, "value", None)


def build_index(pkg):
    """(name -> [paths], {paths}) over every defined func/class/method/attribute."""
    name_index, paths = {}, set()

    def record(obj):
        paths.add(obj.path)
        name_index.setdefault(obj.name, []).append(obj.path)

    for mod in g.iter_modules(pkg):
        paths.add(mod.path)   # module paths resolve :mod:`nitrix.stats.gp`
        for m in mod.members.values():
            if getattr(m, "is_alias", False):
                continue
            k = _kind(m)
            if k in ("function", "class", "attribute"):
                record(m)
                if k == "class":
                    for meth in m.members.values():
                        if getattr(meth, "is_alias", False):
                            continue
                        mk = _kind(meth)
                        if mk == "function":
                            record(meth)          # :meth:`C.m` (by name + path)
                        elif mk == "attribute":
                            paths.add(meth.path)  # :attr:`C.field` — path only, so
                            # common field names don't inflate bare-xref candidates
    return name_index, paths


def public_paths(pkg):
    """Definition paths reachable from a public module's ``__all__``."""
    pub = set()
    for mod in g.iter_modules(pkg):
        if any(p.startswith("_") for p in mod.path.split(".")[1:]):
            continue
        for name in (mod.exports or []):
            mem = mod.members.get(name)
            if mem is None:
                continue
            try:
                tgt = mem.final_target if getattr(mem, "is_alias", False) else mem
                pub.add(tgt.path)
            except Exception:
                pub.add(f"{mod.path}.{name}")
    return pub


# ------------------------------------------------------------------ detectors

def documented_params(obj):
    if not getattr(obj, "docstring", None):
        return None
    try:
        for sec in obj.docstring.parse(Parser.numpy):
            if sec.kind.value == "parameters":
                return [p.name.lstrip("*") for p in sec.value]
    except Exception:
        return None
    return []


def signature_params(obj):
    out = []
    for p in getattr(obj, "parameters", []) or []:
        if p.name in ("self", "cls"):
            continue
        out.append(p.name)
    return out


def signature_issues(obj):
    issues = []
    doc = documented_params(obj)
    sig = signature_params(obj)
    if doc is not None and (sig or doc):
        undoc = [p for p in sig if p not in doc]
        absent = [p for p in doc if p not in sig]
        if undoc:
            issues.append({"type": "undocumented_params", "detail": ", ".join(undoc)})
        if absent:
            issues.append({"type": "documented_absent_params", "detail": ", ".join(absent)})
        common_doc = [p for p in doc if p in sig]
        common_sig = [p for p in sig if p in doc]
        if not undoc and not absent and common_doc != common_sig:
            issues.append({"type": "param_order_mismatch",
                           "detail": f"doc {common_doc} vs sig {common_sig}"})
    if getattr(obj, "returns", None) is not None and str(obj.returns) != "None":
        has_returns = False
        if getattr(obj, "docstring", None):
            try:
                has_returns = any(s.kind.value == "returns"
                                  for s in obj.docstring.parse(Parser.numpy))
            except Exception:
                has_returns = "return" in obj.docstring.value.lower()
        if not has_returns:
            issues.append({"type": "returns_undocumented",
                           "detail": f"-> {obj.returns}"})
    return issues


def _samples(items, n=4):
    uniq = list(dict.fromkeys(items))
    return uniq[:n]


def text_issues(text, own_name, ctx):
    issues = []
    literals = LITERAL_RE.findall(text)

    math_flagged = [s for s in literals if g._is_math_literal(s)]
    if math_flagged:
        issues.append({"type": "math_in_code", "count": len(math_flagged),
                       "samples": _samples(math_flagged)})
    math_sub = [s for s in literals
                if not g._is_math_literal(s) and SUBSCRIPT_MATH_RE.search(s)
                and not (IDENT_RE.match(s) and s in ctx["names"])]
    if math_sub:
        issues.append({"type": "math_in_code_unrendered", "count": len(math_sub),
                       "samples": _samples(math_sub)})

    # Cross-reference roles, excluding :math:`…` (that is math, not a reference).
    roles = [(m.group("role"), m.group("t")) for m in XREF_ROLE_RE.finditer(text)]
    xrefs = [t for role, t in roles if role not in ("math", "doi")]
    if xrefs:
        issues.append({"type": "sphinx_role", "count": len(xrefs),
                       "samples": _samples(xrefs)})
        unresolved = [t for t in xrefs if not _resolvable(t, ctx)]
        if unresolved:
            issues.append({"type": "unresolvable_role", "count": len(unresolved),
                           "samples": _samples(unresolved)})

    bare = [s for s in literals
            if IDENT_RE.match(s) and len(s) >= 2 and s != own_name and s in ctx["names"]]
    if bare:
        issues.append({"type": "bare_xref_candidate", "count": len(bare),
                       "samples": _samples(bare)})

    if INTERNAL_REF_RE.search(text):
        issues.append({"type": "internal_ref",
                       "samples": _samples(INTERNAL_REF_RE.findall(text) and
                                           [m.group(0) for m in INTERNAL_REF_RE.finditer(text)])})
    if RST_DIRECTIVE_RE.search(text):
        issues.append({"type": "rst_directive",
                       "samples": _samples([m.group(0).strip() for m in RST_DIRECTIVE_RE.finditer(text)])})
    if SPHINX_FIELD_RE.search(text):
        issues.append({"type": "sphinx_field"})
    if DOCTEST_RE.search(text):
        issues.append({"type": "doctest"})
    if RST_LINK_RE.search(text):
        issues.append({"type": "rst_link"})
    if TODO_RE.search(text):
        issues.append({"type": "todo_marker",
                       "samples": _samples(TODO_RE.findall(text))})
    cits = [m.group(0) for m in CITATION_RE.finditer(text)]
    if cits and not DOI_RE.search(text):
        issues.append({"type": "citation_without_doi", "samples": _samples(cits)})

    issues += summary_issues(text)
    return issues


def summary_issues(text):
    stripped = text.strip()
    if not stripped:
        return []
    first_block = stripped.split("\n\n", 1)[0].strip()
    problems = []
    if "\n" in first_block:
        problems.append("multi-line summary")
    elif len(first_block) > 100:
        problems.append(f"long summary ({len(first_block)} chars)")
    if not first_block.rstrip("`*").endswith((".", ":", "?", "!")):
        problems.append("summary not terminated")
    return [{"type": "summary_issue", "detail": "; ".join(problems)}] if problems else []


def _resolvable(raw, ctx):
    raw = raw.strip()
    m = re.match(r"(.+?)\s*<([^>]+)>$", raw)
    if m:
        raw = m.group(2).strip()
    raw = raw.lstrip("~")
    if raw in ctx["paths"]:
        return True
    leaf = raw.rsplit(".", 1)[-1]
    if leaf in ctx["names"]:
        return True
    if raw.startswith("."):
        suffix = raw.lstrip(".")
        return any(p.endswith("." + suffix) for p in ctx["paths"])
    # A class-qualified ref (GPResult.edf) resolves if some path ends with it.
    if "." in raw and any(p.endswith("." + raw) for p in ctx["paths"]):
        return True
    # A dotted target rooted outside the library is an external reference
    # (jax.vmap, numpy.…) — legitimate, not a broken in-repo link.
    if "." in raw and raw.split(".")[0] != ctx["lib"]:
        return True
    return False


# ------------------------------------------------------------------ scanning

def rel(path):
    p = pathlib.Path(str(path))
    try:
        return str(p.relative_to(CODE))
    except ValueError:
        return str(p)


def scan(pkg, ctx):
    records = []

    def add(obj, kind, public):
        text = obj.docstring.value if getattr(obj, "docstring", None) else ""
        issues = []
        if not text.strip():
            if public and not (kind == "method" and obj.name in BOILERPLATE_METHODS):
                issues.append({"type": "missing_docstring"})
        else:
            issues += text_issues(text, obj.name, ctx)
            if kind in ("function", "method"):
                issues += signature_issues(obj)
        if issues:
            records.append({
                "path": obj.path, "name": obj.name, "kind": kind, "public": public,
                "file": rel(getattr(obj, "filepath", "")),
                "line": getattr(obj, "lineno", None),
                "issues": issues,
            })

    def add_module(mod):
        text = mod.docstring.value if getattr(mod, "docstring", None) else ""
        if not text.strip():
            return  # module docstrings are optional; only flag markup pathology
        issues = text_issues(text, mod.name.split(".")[-1], ctx)
        if issues:
            public = not any(p.startswith("_") for p in mod.path.split(".")[1:])
            records.append({
                "path": mod.path, "name": mod.path, "kind": "module", "public": public,
                "file": rel(getattr(mod, "filepath", "")), "line": 1, "issues": issues,
            })

    for mod in g.iter_modules(pkg):
        add_module(mod)
        for m in mod.members.values():
            if getattr(m, "is_alias", False):
                continue
            k = _kind(m)
            if k == "function":
                add(m, "function", m.path in ctx["public"])
            elif k == "class":
                cls_public = m.path in ctx["public"]
                add(m, "class", cls_public)
                for meth in m.members.values():
                    if getattr(meth, "is_alias", False) or _kind(meth) != "function":
                        continue
                    if meth.name != "__init__" and meth.name.startswith("_"):
                        continue
                    add(meth, "method", cls_public)
    return records


# -------------------------------------------------------------------- reports

def totals(records):
    by_type = {}
    for r in records:
        for iss in r["issues"]:
            t = by_type.setdefault(iss["type"], {"symbols": 0, "occurrences": 0})
            t["symbols"] += 1
            t["occurrences"] += iss.get("count", 1)
    return dict(sorted(by_type.items(), key=lambda kv: -kv[1]["symbols"]))


CATEGORY_ORDER = [
    "missing_docstring", "undocumented_params", "documented_absent_params",
    "param_order_mismatch", "returns_undocumented", "internal_ref",
    "citation_without_doi", "rst_directive", "sphinx_field", "doctest",
    "rst_link", "math_in_code", "math_in_code_unrendered", "sphinx_role",
    "unresolvable_role", "bare_xref_candidate", "todo_marker", "summary_issue",
]


def markdown_report(library, records, scanned, tot):
    lines = [f"# Docstring inventory — `{library}`", "",
             f"Scanned **{scanned['symbols']}** symbols "
             f"({scanned['public']} public) across **{scanned['modules']}** modules; "
             f"**{len(records)}** carry at least one flag. "
             "Static griffe scan — analyses docstring text only "
             "(code comments are not docstrings). Generated by "
             "`inventory_docstrings.py`; not documentation.", "",
             "## Summary", "", "| category | symbols | occurrences |",
             "|---|---:|---:|"]
    for t, c in tot.items():
        lines.append(f"| `{t}` | {c['symbols']} | {c['occurrences']} |")
    lines.append("")
    order = [t for t in CATEGORY_ORDER if t in tot] + \
            [t for t in tot if t not in CATEGORY_ORDER]
    for t in order:
        hits = [(r, iss) for r in records for iss in r["issues"] if iss["type"] == t]
        lines += [f"## `{t}`  ({len(hits)})", ""]
        for r, iss in hits[:60]:
            extra = iss.get("detail") or (
                ", ".join(f"`{s}`" for s in iss.get("samples", []) if s) if iss.get("samples") else "")
            cnt = f" ×{iss['count']}" if iss.get("count", 1) > 1 else ""
            vis = "" if r["public"] else " *(internal)*"
            loc = f"{r['file']}:{r['line']}" if r["line"] else r["file"]
            lines.append(f"- `{r['path']}`{vis}{cnt} — {extra}  ({loc})".rstrip())
        if len(hits) > 60:
            lines.append(f"- … and {len(hits) - 60} more")
        lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("library")
    p.add_argument("--src", default=None, help="package source root (default: CODE/<lib>/src/<lib>)")
    p.add_argument("--out", default=None,
                   help="output stem (writes <out>.json + <out>.md; "
                        "default CODE/<lib>/docs/_docstring_inventory)")
    a = p.parse_args()

    src_root = pathlib.Path(a.src) if a.src else CODE / a.library / "src" / a.library
    if not src_root.is_dir():
        sys.exit(f"source root not found: {src_root}")
    out_stem = pathlib.Path(a.out) if a.out else CODE / a.library / "docs" / "_docstring_inventory"

    _, pkg = g.load_package(a.library, src_root)
    name_index, paths = build_index(pkg)
    ctx = {"names": set(name_index), "paths": paths, "public": public_paths(pkg),
           "lib": a.library}

    n_symbols = n_public = n_modules = 0
    for mod in g.iter_modules(pkg):
        n_modules += 1
        for m in mod.members.values():
            if getattr(m, "is_alias", False):
                continue
            k = _kind(m)
            if k in ("function", "class"):
                n_symbols += 1
                if m.path in ctx["public"]:
                    n_public += 1

    records = scan(pkg, ctx)
    tot = totals(records)
    scanned = {"symbols": n_symbols, "public": n_public, "modules": n_modules}

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"library": a.library, "scanned": {**scanned, "with_issues": len(records)},
                "totals": tot, "symbols": records}
    out_stem.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
    out_stem.with_suffix(".md").write_text(markdown_report(a.library, records, scanned, tot))

    print(f"docstring inventory: {len(records)}/{n_symbols} symbols flagged "
          f"({n_public} public) across {n_modules} modules")
    for t, c in tot.items():
        print(f"  {t:26} {c['symbols']:4} symbols  {c['occurrences']:5} occ")
    print(f"wrote {rel(out_stem.with_suffix('.json'))} + .md")


if __name__ == "__main__":
    main()
