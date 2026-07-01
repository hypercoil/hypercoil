#!/usr/bin/env python3
"""Generate fillable API-reference stubs for a federation library (PLAN §3.1).

Static (``griffe``-based) — griffe parses source without importing the library,
so it stays CI-safe and immune to the JAX pathologies (jit/vmap-transformed
callables, pytree-registered classes, decorators) that derail runtime
introspection, while resolving re-export aliases to their defining module and
parsing structured docstring sections (overview + per-parameter descriptions).
It reads each module's curated public surface (``__all__``) and emits one
``.qmd`` page per module.

Beyond a bare signature, each page carries:

  * **named tensor dimensions** — jaxtyping annotations (``Float[Array, 'V N']``)
    are parsed into a per-parameter shape table, and the dimension symbols are
    glossed from a curated ``_dimensions.yml`` beside the reference pages;
  * **docstring prose** — the summary becomes the overview and per-parameter
    descriptions populate the parameter table, straight from source;
  * **benchmarks** — where the sibling ``<lib>-perf-bench`` has a
    ``reports/PERF_<NAME>.md`` for a symbol, a headline + the measurements table
    are transcluded with provenance and a link to the full report.

Generated content lives between markers::

    <!-- nitrix:gen <tag> -->
    …regenerated each run…
    <!-- nitrix:/gen -->

``--refresh`` rewrites only those regions (signatures, parameter tables, the
dimension glossary, the benchmark block) and leaves authored prose — everything
*outside* the markers — untouched. ``--force`` rewrites the whole page.

Stubs are scaffolding to be filled by hand/LLM and NEVER shipped as-is
(SPEC §6.5). ``check_reference_staleness.py`` flags a page whose source
signatures drift from its recorded ``signature-hash``.

Usage:
  gen_reference_stubs.py nitrix [--modules nitrix.geometry.grid] [--src P]
        [--out P] [--perf-bench P] [--force | --refresh]
"""
import argparse
import hashlib
import logging
import pathlib
import re
import sys

import griffe
from griffe import ParameterKind, Parser

logging.getLogger("griffe").setLevel(logging.ERROR)

DOC = pathlib.Path(__file__).resolve().parent.parent
CODE = DOC.parent.parent

GEN_RE = re.compile(
    r"<!-- nitrix:gen (?P<tag>\S+) -->\n(?P<body>.*?)\n<!-- nitrix:/gen -->",
    re.DOTALL,
)
FRONT_RE = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)

# jaxtyping array-dtype constructors we recognise: <Name>[Array, "shape"].
JAXTYPE_NAMES = {
    "Array", "Float", "Float16", "Float32", "Float64", "BFloat16",
    "Int", "Int8", "Int16", "Int32", "Int64", "UInt", "UInt8", "UInt32",
    "Bool", "Complex", "Complex64", "Complex128", "Num", "Inexact",
    "Integer", "Real", "Shaped", "Key", "Scalar",
}

VAR_KINDS = (ParameterKind.var_positional, ParameterKind.var_keyword)


def gen_block(tag, inner):
    return f"<!-- nitrix:gen {tag} -->\n{inner}\n<!-- nitrix:/gen -->"


# ------------------------------------------------------------- text utilities

def rst_to_md(text):
    """Light RST→Markdown: ``x`` -> `x`, trailing `::` -> `:`. Best-effort."""
    if not text:
        return ""
    text = re.sub(r"``([^`]+)``", r"`\1`", text)
    text = re.sub(r"[ \t]*::\s*$", ":", text, flags=re.M)
    return text.strip()


def cell(text):
    """Collapse a docstring description into a single safe Markdown table cell."""
    s = " ".join(rst_to_md(text).split())
    return s.replace("|", r"\|") if s else "—"


def parse_jaxtype(annstr):
    """``Float[Array, '*spatial ndim']`` -> ('Float', ['*spatial', 'ndim'])."""
    if not annstr:
        return None
    m = re.match(r"\s*(\w+)\[", annstr)
    if not m or m.group(1) not in JAXTYPE_NAMES:
        return None
    quoted = re.findall(r"'([^']*)'", annstr)
    if not quoted:
        return None
    return m.group(1), quoted[-1].split()


# ------------------------------------------------------------- griffe helpers

def parsed_doc(obj):
    """(overview_md, {param: description}, returns_desc) from a docstring."""
    if not getattr(obj, "docstring", None):
        return "", {}, ""
    overview, pdescs, ret = "", {}, ""
    try:
        sections = obj.docstring.parse(Parser.numpy)
    except Exception:
        return rst_to_md(obj.docstring.value), {}, ""
    for sec in sections:
        kind = sec.kind.value
        if kind == "text" and not overview:
            overview = rst_to_md(sec.value)
        elif kind == "parameters":
            for p in sec.value:
                pdescs[p.name] = p.description
        elif kind == "returns" and not ret:
            ret = " ".join(r.description for r in sec.value if r.description)
    return overview, pdescs, ret


def render_param_decl(p):
    if p.kind == ParameterKind.var_positional:
        name = "*" + p.name
    elif p.kind == ParameterKind.var_keyword:
        name = "**" + p.name
    else:
        name = p.name
    s = name
    if p.annotation is not None and p.kind not in VAR_KINDS:
        s += f": {p.annotation}"
    if p.default is not None and p.kind not in VAR_KINDS:
        s += (" = " if p.annotation is not None else "=") + str(p.default)
    return s


def signature(obj):
    if obj.kind.value == "class":
        bases = ", ".join(str(b) for b in obj.bases)
        return f"class {obj.name}" + (f"({bases})" if bases else "")
    params = list(obj.parameters)
    has_var_pos = any(p.kind == ParameterKind.var_positional for p in params)
    out, prev, star_done = [], None, False
    for p in params:
        if p.name in ("self", "cls") and prev is None:
            prev = p.kind
            continue
        if prev == ParameterKind.positional_only and p.kind != ParameterKind.positional_only:
            out.append("/")
        if p.kind == ParameterKind.keyword_only and not has_var_pos and not star_done:
            out.append("*")
            star_done = True
        out.append(render_param_decl(p))
        prev = p.kind
    if prev == ParameterKind.positional_only:
        out.append("/")
    sig = f"{obj.name}(" + ", ".join(out) + ")"
    if obj.returns is not None:
        sig += f" -> {obj.returns}"
    return sig


def param_rows(obj, pdescs):
    rows = []
    for p in obj.parameters:
        if p.name in ("self", "cls"):
            continue
        annstr = str(p.annotation) if p.annotation is not None else None
        jt = parse_jaxtype(annstr)
        star = ("*" if p.kind == ParameterKind.var_positional
                else "**" if p.kind == ParameterKind.var_keyword else "")
        rows.append({
            "name": star + p.name,
            "ann": annstr,
            "jdtype": jt[0] if jt else None,
            "dims": jt[1] if jt else [],
            "default": str(p.default) if p.default is not None else None,
            "desc": pdescs.get(p.name, ""),
        })
    return rows


# --------------------------------------------------------------- dim glossary

def load_dimensions(path):
    """Tiny flat-YAML reader: ``dim: meaning`` lines, ``#`` comments. Pure stdlib."""
    glossary = {}
    if not path.is_file():
        return glossary
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if ":" in s:
            k, _, v = s.partition(":")
            glossary[k.strip().strip("\"'")] = v.strip().strip("\"'")
    return glossary


def shape_str(dims):
    return " × ".join(f"`{d}`" for d in dims) if dims else "—"


def dims_glossary_block(dims, glossary):
    if not dims:
        return None
    rows = ["**Tensor dimensions**\n", "| dim | meaning |", "|---|---|"]
    for d in dims:
        key = d.lstrip("*")
        if d == "...":
            meaning = "anonymous leading (batch) axes"
        elif d.startswith("*"):
            meaning = glossary.get(key) or f"a variadic group of axes (`{key}`)"
        else:
            meaning = glossary.get(d, "*(undocumented — add to `_dimensions.yml`)*")
        rows.append(f"| `{d}` | {meaning} |")
    return "\n".join(rows)


def params_table(rows):
    if not rows:
        return "*No parameters.*"
    out = ["| parameter | type | shape | default | description |",
           "|---|---|---|---|---|"]
    for p in rows:
        if p["jdtype"]:
            typ, shp = f"`{p['jdtype']}`", shape_str(p["dims"])
        else:
            typ = f"`{p['ann']}`" if p["ann"] else "—"
            shp = "—"
        default = f"`{p['default']}`" if p["default"] is not None else "—"
        out.append(f"| `{p['name']}` | {typ} | {shp} | {default} | {cell(p['desc'])} |")
    return "\n".join(out)


# ---------------------------------------------------------------- perf-bench

def perf_report_path(symbol, perf_reports):
    if perf_reports is None:
        return None
    p = perf_reports / f"PERF_{symbol.upper()}.md"
    return p if p.is_file() else None


def parse_perf(text, symbol):
    """Return (headline, table_md, provenance) parsed from a PERF_<OP>.md."""
    m = re.search(r"-\s*nitrix:\s*(\w+)\s*\|\s*bench:\s*(\w+)", text)
    prov = f"*source: bench `{m.group(2)[:8]}` · nitrix `{m.group(1)[:8]}`*" if m else ""
    tbl, in_tbl = [], False
    for ln in text.splitlines():
        if ln.startswith("| case ") and "platform" in ln:
            in_tbl = True
        if in_tbl:
            if ln.startswith("|"):
                tbl.append(ln)
            else:
                break
    gpu_med, best_ratio, best_base = None, 0.0, None
    for ln in tbl[2:]:                       # skip header + separator
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(cells) < 10:
            continue
        _sym, plat, _param, base, _status, steady, _comp, _mem, _fid, ratio = cells[:10]
        if plat != "jax-cuda12":
            continue
        base = base.strip("`")
        if base == "nitrix-jax" and "/" in steady and gpu_med is None:
            gpu_med = steady.split("/")[-1].strip()    # smallest-case representative
        rr = re.match(r"([\d.]+)x", ratio)
        if rr and base != "nitrix-jax":
            v = float(rr.group(1))
            if v > best_ratio:
                best_ratio, best_base = v, base
    parts = []
    if gpu_med:
        parts.append(f"GPU steady-state **{gpu_med}** (median, L4)")
    if best_base and best_ratio > 1:
        parts.append(f"up to **{best_ratio:g}× faster** than `{best_base}`")
    headline = "; ".join(parts) if parts else "see the measurements below"
    return headline, "\n".join(tbl), prov


def perf_block(symbol, perf_reports, repo_url):
    path = perf_report_path(symbol, perf_reports)
    if path is None:
        return None
    headline, table_md, prov = parse_perf(path.read_text(), symbol)
    link = f"[full report]({repo_url}/blob/main/reports/{path.name})"
    # Collapsed by default; the wide measurements table side-scrolls within the
    # page (Bootstrap `.table-responsive`) instead of bleeding into the margin.
    body = [
        ':::: {.callout-note title="Benchmarks — nitrix-perf-bench" collapse="true"}',
        f"{headline}.  {link}",
        "",
        "::: {.table-responsive}",
        table_md,
        ":::",
        "",
        prov,
        "::::",
    ]
    return "\n".join(body)


# ---------------------------------------------------------------- rendering

def heading(name):
    return f"## `{name}` {{#{name}}}\n"


def render_function(name, obj, perf_reports, repo_url, dim_acc, note):
    overview, pdescs, ret_desc = parsed_doc(obj)
    rows = param_rows(obj, pdescs)
    for r in rows:
        dim_acc(r["dims"])
    ret_jt = parse_jaxtype(str(obj.returns)) if obj.returns is not None else None
    if ret_jt:
        dim_acc(ret_jt[1])
    parts = [heading(name)]
    if note:
        parts.append(note + "\n")
    if obj.decorators:
        decs = ", ".join(f"`@{d.value}`" for d in obj.decorators)
        parts.append(f"*decorated:* {decs}\n")
    parts.append(gen_block(f"sig:{name}", "```python\n" + signature(obj) + "\n```"))
    parts.append("")
    parts.append(gen_block(f"overview:{name}", overview or "*TODO: write an overview.*"))
    parts.append("")
    ret_line = ""
    if obj.returns is not None:
        if ret_jt:
            ret_line = f"\n\n**Returns** `{ret_jt[0]}` — shape {shape_str(ret_jt[1])}."
        else:
            ret_line = f"\n\n**Returns** `{obj.returns}`."
        if ret_desc:
            ret_line += " " + " ".join(rst_to_md(ret_desc).split())
    parts.append(gen_block(f"params:{name}",
                           "**Parameters**\n\n" + params_table(rows) + ret_line))
    parts.append("")
    pb = perf_block(name, perf_reports, repo_url)
    if pb:
        parts.append(gen_block(f"perf:{name}", pb))
        parts.append("")
    return "\n".join(parts), signature(obj)


def render_class(name, obj, perf_reports, repo_url, dim_acc, note):
    overview, _, _ = parsed_doc(obj)
    bases = ", ".join(str(b) for b in obj.bases)
    sig = f"class {name}" + (f"({bases})" if bases else "")
    parts = [heading(name)]
    if note:
        parts.append(note + "\n")
    parts.append(gen_block(f"sig:{name}", "```python\n" + sig + "\n```"))
    parts.append("")
    parts.append(gen_block(f"overview:{name}", overview or "*TODO: write an overview.*"))
    parts.append("")
    sig_for_hash = sig
    methods = [(n, m) for n, m in obj.members.items()
               if (n == "__init__" or not n.startswith("_"))
               and not getattr(m, "is_alias", False)
               and getattr(m, "kind", None) is not None
               and m.kind.value == "function"]
    if methods:
        parts.append("### Methods\n")
        for mn, m in methods:
            mov, mpd, _ = parsed_doc(m)
            for r in param_rows(m, mpd):
                dim_acc(r["dims"])
            parts.append(gen_block(f"sig:{name}.{mn}",
                                   "```python\n" + signature(m) + "\n```"))
            parts.append("")
            if mov:
                parts.append(mov + "\n")
            sig_for_hash += " | " + signature(m)
    pb = perf_block(name, perf_reports, repo_url)
    if pb:
        parts.append(gen_block(f"perf:{name}", pb))
        parts.append("")
    return "\n".join(parts), sig_for_hash


def render_attribute(name, obj, note, local_names):
    ann = str(obj.annotation) if obj.annotation is not None else None
    val = str(obj.value) if obj.value is not None else None
    parts = [heading(name)]
    if note:
        parts.append(note + "\n")
    # A bare-name assignment that points at another public symbol is an alias.
    if val and re.fullmatch(r"\w+", val) and val in local_names:
        parts.append(gen_block(f"sig:{name}",
                               f"`{name}` is an alias for [`{val}`](#{val})."))
        parts.append("")
        return "\n".join(parts), f"{name}=alias:{val}"
    decl = name + (f": {ann}" if ann else "") + (f" = {val}" if val else "")
    parts.append(gen_block(f"sig:{name}", "```python\n" + decl + "\n```"))
    parts.append("")
    if obj.docstring:
        ov = rst_to_md(obj.docstring.value)
    else:
        ov = "*TODO: describe this type alias / constant.*"
    parts.append(gen_block(f"overview:{name}", ov))
    parts.append("")
    return "\n".join(parts), f"{name}::{decl}"


def warn_stub(name, message):
    md = (heading(name) + "\n"
          "::: {.callout-warning title=\"TODO — could not introspect\"}\n"
          f"{message}\n:::\n")
    return md, f"{name}::TODO"


def render_member(name, member, perf_reports, repo_url, dim_acc, local_names):
    """Render one public member, resolving re-export aliases to their target."""
    note = None
    obj = member
    if getattr(member, "is_alias", False):
        try:
            obj = member.final_target
        except Exception:
            return warn_stub(name, "Unresolved re-export alias. Document by hand.")
        home = getattr(getattr(obj, "module", None), "path", None)
        if home and home != getattr(member.parent, "path", None):
            note = f"*Re-exported here; defined in `{home}`.*"
    kind = obj.kind.value
    if kind == "function":
        return render_function(name, obj, perf_reports, repo_url, dim_acc, note)
    if kind == "class":
        return render_class(name, obj, perf_reports, repo_url, dim_acc, note)
    if kind == "attribute":
        return render_attribute(name, obj, note, local_names)
    if kind == "module":
        return warn_stub(name, "Re-exported submodule — see its own page.")
    return warn_stub(name, "Object whose kind has no renderer. Document by hand.")


def page_for_module(modname, mod, glossary, perf_reports, repo_url):
    names = list(mod.exports or [])
    local_names = set(names)
    dims, seen = [], set()

    def dim_acc(found):
        for d in found:
            if d not in seen:
                seen.add(d)
                dims.append(d)

    body, sig_blob = [], []
    for name in names:
        member = mod.members.get(name)
        if member is None:
            md, sh = warn_stub(name, "Listed in `__all__` but not found in source.")
        else:
            md, sh = render_member(name, member, perf_reports, repo_url,
                                   dim_acc, local_names)
        body.append(md)
        sig_blob.append(sh)
    sig_hash = hashlib.sha256("\n".join(sig_blob).encode()).hexdigest()[:16]
    front = (f"---\ntitle: \"{modname}\"\ngenerated: true\n"
             f"signature-hash: {sig_hash}\n---\n\n"
             "::: {.callout-important title=\"Generated reference stub — fill me\"}\n"
             f"Auto-generated scaffold for `{modname}`. Fill the prose *outside* the\n"
             "generated markers; do **not** ship as-is (SPEC §6.5). "
             "`check_reference_staleness.py` flags signature drift.\n:::\n\n")
    glossary_block = dims_glossary_block(dims, glossary)
    pre = gen_block("dims:__page__", glossary_block) + "\n\n" if glossary_block else ""
    return front + pre + "\n".join(body) + "\n", sig_hash


def refresh(old, new):
    """Rewrite only the generated marker regions (and front matter) of *old*."""
    new_blocks = {m.group("tag"): m.group(0) for m in GEN_RE.finditer(new)}
    merged = GEN_RE.sub(lambda m: new_blocks.get(m.group("tag"), m.group(0)), old)
    nf = FRONT_RE.match(new)
    if nf:
        merged = FRONT_RE.sub(lambda _m: nf.group(0), merged, count=1)
    return merged


# ---------------------------------------------------------------- module walk

def load_package(library, src_root):
    """Statically load *library* from *src_root* with re-export aliases resolved."""
    loader = griffe.GriffeLoader(search_paths=[str(src_root.parent)])
    pkg = loader.load(library)
    loader.resolve_aliases(external=True, implicit=False)
    return loader, pkg


def module_by_name(pkg, modname):
    """Resolve a dotted module path against a loaded package (None if absent)."""
    if modname == pkg.name:
        return pkg
    cur = pkg
    for part in modname.split(".")[1:]:
        cur = cur.members.get(part) if hasattr(cur, "members") else None
        if cur is None or getattr(cur, "is_alias", False):
            return None
    return cur


def iter_modules(mod):
    yield mod
    for m in mod.members.values():
        # `is_alias` is resolution-safe; `is_module` proxies to an alias target
        # (and would raise on an unresolvable one), so gate on `is_alias` first.
        if getattr(m, "is_alias", False):
            continue
        if getattr(m, "is_module", False):
            yield from iter_modules(m)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("library")
    p.add_argument("--modules", default=None,
                   help="comma-separated dotted prefixes to include (default: all public)")
    p.add_argument("--src", default=None, help="package source root (default: CODE/<lib>/src/<lib>)")
    p.add_argument("--out", default=None, help="reference dir (default: CODE/<lib>/docs/reference)")
    p.add_argument("--perf-bench", default=None,
                   help="perf-bench reports dir (default: CODE/<lib>-perf-bench/reports)")
    p.add_argument("--force", action="store_true", help="overwrite existing pages wholesale")
    p.add_argument("--refresh", action="store_true",
                   help="rewrite only generated marker regions, preserving authored prose")
    a = p.parse_args()

    src_root = pathlib.Path(a.src) if a.src else CODE / a.library / "src" / a.library
    out = pathlib.Path(a.out) if a.out else CODE / a.library / "docs" / "reference"
    if a.perf_bench:
        perf_reports = pathlib.Path(a.perf_bench)
    else:
        cand = CODE / f"{a.library}-perf-bench" / "reports"
        perf_reports = cand if cand.is_dir() else None
    repo_url = f"https://github.com/hypercoil/{a.library}-perf-bench"
    if not src_root.is_dir():
        sys.exit(f"source root not found: {src_root}")
    glossary = load_dimensions(out / "_dimensions.yml")
    prefixes = a.modules.split(",") if a.modules else None
    out.mkdir(parents=True, exist_ok=True)

    _, pkg = load_package(a.library, src_root)

    created = skipped = refreshed = 0
    for mod in iter_modules(pkg):
        modname = mod.path
        if any(part.startswith("_") for part in modname.split(".")[1:]):
            continue
        # Page leaf modules, not package __init__ surfaces — the public symbols a
        # package re-exports are documented on their defining submodule's page.
        fp = getattr(mod, "filepath", None)
        if getattr(fp, "name", "") == "__init__.py":
            continue
        if prefixes and not any(modname == pre or modname.startswith(pre + ".")
                                or pre.startswith(modname + ".") for pre in prefixes):
            continue
        if not mod.exports:
            continue
        content, _ = page_for_module(modname, mod, glossary, perf_reports, repo_url)
        dst = out / (modname.replace(".", "-") + ".qmd")
        if dst.exists() and not (a.force or a.refresh):
            skipped += 1
            continue
        if dst.exists() and a.refresh and not a.force:
            dst.write_text(refresh(dst.read_text(), content))
            refreshed += 1
            print(f"  ~ {dst.relative_to(CODE)}  (refreshed generated regions)")
            continue
        dst.write_text(content)
        created += 1
        print(f"  + {dst.relative_to(CODE)}  ({len(mod.exports)} symbols)")

    perf_note = f"perf={perf_reports.relative_to(CODE)}" if perf_reports else "perf=none"
    print(f"stubs: {created} written, {refreshed} refreshed, {skipped} kept. "
          f"out={out.relative_to(CODE)} {perf_note}")


if __name__ == "__main__":
    main()
