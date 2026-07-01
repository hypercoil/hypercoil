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

_MATH_ROLE_RE = re.compile(r":math:`([^`]+)`")
# Any Sphinx cross-reference role, e.g. :func:`x`, :class:`~pkg.mod.C`, :py:meth:`m`.
_XREF_ROLE_RE = re.compile(r":(?:[a-zA-Z][\w.+-]*:)+`([^`]+)`")


def _xref_sub(m):
    """Render a Sphinx cross-reference role as inline code (no live link yet).

    Honours the ``~pkg.mod.Name`` (show only the leaf) and ``Text <target>``
    (show the title) conventions. These don't resolve to pages while reference
    coverage is partial; a live-link pass needs a global symbol→page index.
    """
    target = m.group(1).strip()
    titled = re.match(r"(.+?)\s*<[^>]+>$", target)
    if titled:
        return f"`{titled.group(1).strip()}`"
    if target.startswith("~"):
        target = target[1:].rsplit(".", 1)[-1]
    return f"`{target}`"


# nitrix docstrings write inline math inside RST literals (``X^T W X``, ``A^{-1}``,
# ``Σ_i (·)^2``) with informal ASCII/Unicode notation. We render such a literal as
# LaTeX ($…$) only when it carries an *unambiguous* math signal — a caret or a
# Unicode math glyph — so a plain code identifier (``cov_unscaled``) is never
# misread as math. Subscript-only spans (``mu_i``) have no such signal and stay as
# code; the eventual `:math:`-migration (see the docstring-inventory plan) is the
# durable fix. Authored `:math:` always wins and is left untouched.
_UNICODE_MATH = set("ᵀ⁻⁰¹²³⁴⁵⁶⁷⁸⁹½⅓⅔¼¾·×÷√∑∏ΣΔΘΛΞΠΦΨΩ∇∂∈∉≈≠≤≥→↦⟨⟩⊤∘⊗⊙∥±∓∞")
_UNICODE_MAP = [
    ("⁻¹", "^{-1}"), ("⁻", "^-"), ("⁰", "^0"), ("¹", "^1"), ("²", "^2"),
    ("³", "^3"), ("⁴", "^4"), ("⁵", "^5"), ("⁶", "^6"), ("⁷", "^7"),
    ("⁸", "^8"), ("⁹", "^9"), ("½", r"\tfrac{1}{2}"), ("⅓", r"\tfrac{1}{3}"),
    ("⅔", r"\tfrac{2}{3}"), ("¼", r"\tfrac{1}{4}"), ("¾", r"\tfrac{3}{4}"),
    ("ᵀ", r"^{\mathsf{T}}"), ("·", r"\cdot "), ("×", r"\times "), ("÷", r"\div "),
    ("√", r"\sqrt"), ("∑", r"\sum"), ("∏", r"\prod"), ("Σ", r"\Sigma "),
    ("Δ", r"\Delta "), ("Θ", r"\Theta "), ("Λ", r"\Lambda "), ("Ξ", r"\Xi "),
    ("Π", r"\Pi "), ("Φ", r"\Phi "), ("Ψ", r"\Psi "), ("Ω", r"\Omega "),
    ("∇", r"\nabla "), ("∂", r"\partial "), ("∈", r"\in "), ("∉", r"\notin "),
    ("≈", r"\approx "), ("≠", r"\neq "), ("≤", r"\le "), ("≥", r"\ge "),
    ("→", r"\to "), ("↦", r"\mapsto "), ("⟨", r"\langle "), ("⟩", r"\rangle "),
    ("⊤", r"\top "), ("∘", r"\circ "), ("⊗", r"\otimes "), ("⊙", r"\odot "),
    ("∥", r"\| "), ("±", r"\pm "), ("∓", r"\mp "), ("∞", r"\infty "),
]
_GREEK = ("alpha beta gamma delta epsilon varepsilon zeta eta theta iota kappa "
          "lambda mu nu xi omicron pi rho sigma tau upsilon phi varphi chi psi "
          "omega Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega").split()
_GREEK_RE = re.compile(r"(?<![\\A-Za-z])(" + "|".join(_GREEK) + r")(?![A-Za-z])")
_MATHOP = "tr sym diag vec sgn adj rank cov var corr".split()
_MATHOP_RE = re.compile(r"(?<![\\A-Za-z])(" + "|".join(_MATHOP) + r")(?=\s*\()")


def _is_math_literal(s):
    return "^" in s or any(c in _UNICODE_MATH for c in s)


def normalise_math(s):
    """Best-effort informal-notation → LaTeX for a span already known to be math."""
    for glyph, tex in _UNICODE_MAP:
        s = s.replace(glyph, tex)
    s = _GREEK_RE.sub(r"\\\1", s)                 # spelled-out Greek -> \greek
    s = _MATHOP_RE.sub(r"\\operatorname{\1}", s)  # tr(, sym(, … -> \operatorname{…}(
    # Brace multi-character sub/superscripts so they don't render as one glyph
    # (Σ_axis -> \Sigma_{axis}); already-braced ({…}) and single chars are left.
    s = re.sub(r"\^([A-Za-z0-9]{2,})", r"^{\1}", s)
    s = re.sub(r"_([A-Za-z0-9]{2,})", r"_{\1}", s)
    return s.strip()


def _literal_sub(m):
    inner = m.group(1)
    if _is_math_literal(inner):
        return f"${normalise_math(inner)}$"
    return f"`{inner}`"


_DIRECTIVE_RE = re.compile(
    r"^(?P<ind>[ \t]*)\.\.[ \t]+(?P<d>math|code-block|code|note|warning|"
    r"admonition|tip|caution|important)::(?P<arg>.*)$")


def _convert_rst_blocks(text):
    """RST block directives -> Quarto: .. math:: -> $$…$$, .. code[-block]:: ->
    fenced code, .. note::/.. warning:: etc. -> callouts. (Inline :math: and
    roles are handled separately.)"""
    lines = text.split("\n")
    out, i = [], 0
    while i < len(lines):
        m = _DIRECTIVE_RE.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        ind, directive, arg = m.group("ind"), m.group("d"), m.group("arg").strip()
        i += 1
        while i < len(lines) and re.match(r"^[ \t]*:\w[\w-]*:", lines[i]):
            i += 1  # skip directive options (:label:, :name:, …)
        while i < len(lines) and not lines[i].strip():
            i += 1
        body, base = [], None
        while i < len(lines):
            ln = lines[i]
            if not ln.strip():
                body.append("")
                i += 1
                continue
            cur = len(ln) - len(ln.lstrip())
            if cur <= len(ind):
                break
            base = cur if base is None else min(base, cur)
            body.append(ln)
            i += 1
        while body and not body[-1].strip():
            body.pop()
        inner = "\n".join(ln[base:] if ln.strip() else "" for ln in body)
        if directive == "math":
            if "&" in inner or "\\\\" in inner:
                inner = "\\begin{aligned}\n" + inner + "\n\\end{aligned}"
            out += [ind + "$$", inner, ind + "$$", ""]
        elif directive in ("code-block", "code"):
            out += [ind + "```" + (arg or ""), inner, ind + "```", ""]
        else:
            kind = directive if directive in ("note", "warning", "tip", "caution",
                                              "important") else "note"
            out += [ind + f"::: {{.callout-{kind}}}", inner, ind + ":::", ""]
    return "\n".join(out)


def rst_to_md(text):
    """Light RST→Markdown: block directives, roles, math, ``x`` -> `x`, `::`."""
    if not text:
        return ""
    text = _convert_rst_blocks(text)               # .. math:: / .. code:: blocks
    text = _MATH_ROLE_RE.sub(r"$\1$", text)        # :math:`a` -> $a$ (authored LaTeX)
    text = _XREF_ROLE_RE.sub(_xref_sub, text)      # :func:`f` etc. -> `f`
    text = re.sub(r"``([^`]+)``", _literal_sub, text)   # ``x`` -> `x` or $math$
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


_CALLOUT_KINDS = {"note", "warning", "tip", "caution", "important"}


def extra_sections(obj):
    """Render docstring sections beyond the summary/params/returns.

    NumPy ``Notes`` / ``Warnings`` and custom underlined sections (which griffe
    parses as admonitions) plus any secondary prose are transcluded — as Quarto
    callouts for the standard kinds, and as headings for custom titles — so a
    rich module or function docstring is not truncated to its first paragraph.
    """
    if not getattr(obj, "docstring", None):
        return ""
    try:
        sections = obj.docstring.parse(Parser.numpy)
    except Exception:
        return ""
    out, seen_text = [], False
    for sec in sections:
        kind = sec.kind.value
        if kind == "text":
            if seen_text:
                out.append(rst_to_md(sec.value))
            seen_text = True
        elif kind == "admonition":
            adm = sec.value
            akind = (getattr(adm, "kind", "") or "").lower()
            body = rst_to_md(getattr(adm, "contents", "") or "")
            if akind in _CALLOUT_KINDS:
                out.append(f"::: {{.callout-{akind}}}\n{body}\n:::")
            else:
                title = getattr(adm, "title", None) or akind.replace("-", " ").capitalize()
                out.append(f"### {title}\n\n{body}")
    return "\n\n".join(s for s in out if s.strip())


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
    # Proportional dash counts → relative column widths in the rendered HTML;
    # give the description (which carries the docstring prose) the bulk of it.
    sep = "|" + "|".join("-" * w for w in (12, 8, 8, 10, 62)) + "|"
    out = ["| parameter | type | shape | default | description |", sep]
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
#
# Every public symbol becomes its own page under a per-module subdirectory, and
# the module page is a listing that links them (numpy/scipy style). Body builders
# emit everything *below* the page title (Quarto renders the front-matter title
# as the H1) and return (body, sig_for_hash, dims_used, one-line summary).

def summary_line(text):
    """First paragraph of an overview, flattened for a listing description."""
    if not text:
        return ""
    return " ".join(text.strip().split("\n\n", 1)[0].split())


def function_body(name, obj, perf_reports, repo_url):
    overview, pdescs, ret_desc = parsed_doc(obj)
    rows = param_rows(obj, pdescs)
    dims, seen = [], set()

    def acc(found):
        for d in found:
            if d not in seen:
                seen.add(d)
                dims.append(d)

    for r in rows:
        acc(r["dims"])
    ret_jt = parse_jaxtype(str(obj.returns)) if obj.returns is not None else None
    if ret_jt:
        acc(ret_jt[1])
    parts = []
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
    sec = extra_sections(obj)
    if sec:
        parts.append(gen_block(f"sections:{name}", sec))
        parts.append("")
    return "\n".join(parts), signature(obj), dims, summary_line(overview)


def class_body(name, obj, perf_reports, repo_url):
    overview, _, _ = parsed_doc(obj)
    bases = ", ".join(str(b) for b in obj.bases)
    sig = f"class {name}" + (f"({bases})" if bases else "")
    dims, seen = [], set()
    parts = [gen_block(f"sig:{name}", "```python\n" + sig + "\n```"), ""]
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
                for d in r["dims"]:
                    if d not in seen:
                        seen.add(d)
                        dims.append(d)
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
    sec = extra_sections(obj)
    if sec:
        parts.append(gen_block(f"sections:{name}", sec))
        parts.append("")
    return "\n".join(parts), sig_for_hash, dims, summary_line(overview)


def assign_filenames(names):
    """name -> case-insensitively-unique filename stem (stable across filesystems).

    Two public symbols differing only in case (e.g. a ``TWEEDIE`` family constant
    and a ``tweedie`` constructor) would collide on a case-insensitive filesystem
    (macOS), so later collisions get a numeric suffix — deterministically, and
    identically on macOS and Linux.
    """
    used, out = set(), {}
    for name in names:
        stem, i = name, 2
        while stem.casefold() in used:
            stem, i = f"{name}-{i}", i + 1
        used.add(stem.casefold())
        out[name] = stem
    return out


def attribute_body(name, obj, local_names, filemap):
    ann = str(obj.annotation) if obj.annotation is not None else None
    val = str(obj.value) if obj.value is not None else None
    # A bare-name assignment that points at another public symbol is an alias;
    # link to that symbol's sibling page in the same module subdirectory.
    if val and re.fullmatch(r"\w+", val) and val in local_names:
        target = filemap.get(val, val)
        body = gen_block(f"sig:{name}",
                         f"`{name}` is an alias for [`{val}`]({target}.qmd).") + "\n"
        return body, f"{name}=alias:{val}", [], f"Alias for {val}."
    decl = name + (f": {ann}" if ann else "") + (f" = {val}" if val else "")
    if obj.docstring:
        ov = rst_to_md(obj.docstring.value)
        desc = summary_line(obj.docstring.value)
    else:
        ov, desc = "*TODO: describe this type alias / constant.*", ""
    parts = [gen_block(f"sig:{name}", "```python\n" + decl + "\n```"), "",
             gen_block(f"overview:{name}", ov), ""]
    return "\n".join(parts), f"{name}::{decl}", [], desc


def warn_body(message):
    return ("::: {.callout-warning title=\"TODO — could not introspect\"}\n"
            f"{message}\n:::\n")


def resolve_member(name, member):
    """(obj, note) with re-export aliases resolved; obj is None if unresolvable."""
    if not getattr(member, "is_alias", False):
        return member, None
    try:
        obj = member.final_target
    except Exception:
        return None, None
    home = getattr(getattr(obj, "module", None), "path", None)
    note = None
    if home and home != getattr(member.parent, "path", None):
        note = f"*Re-exported here; defined in `{home}`.*"
    return obj, note


def build_symbol(name, member, perf_reports, repo_url, local_names, filemap):
    """One record {name, note, body, sig_hash, dims, desc}; body has no title."""
    obj, note = resolve_member(name, member)
    if obj is None:
        return {"name": name, "note": None,
                "body": warn_body("Unresolved re-export alias. Document by hand."),
                "sig_hash": f"{name}::TODO", "dims": [], "desc": ""}
    kind = obj.kind.value
    if kind == "function":
        body, sh, dims, desc = function_body(name, obj, perf_reports, repo_url)
    elif kind == "class":
        body, sh, dims, desc = class_body(name, obj, perf_reports, repo_url)
    elif kind == "attribute":
        body, sh, dims, desc = attribute_body(name, obj, local_names, filemap)
    elif kind == "module":
        body, sh, dims, desc = (
            warn_body("Re-exported submodule — see its own page."),
            f"{name}::module", [], "")
    else:
        body, sh, dims, desc = (
            warn_body("Object whose kind has no renderer. Document by hand."),
            f"{name}::TODO", [], "")
    return {"name": name, "note": note, "body": body,
            "sig_hash": sh, "dims": dims, "desc": desc}


def build_records(mod, perf_reports, repo_url):
    names = list(mod.exports or [])
    local = set(names)
    filemap = assign_filenames(names)
    recs = []
    for name in names:
        member = mod.members.get(name)
        if member is None:
            rec = {"name": name, "note": None,
                   "body": warn_body("Listed in `__all__` but not in source."),
                   "sig_hash": f"{name}::MISSING", "dims": [], "desc": ""}
        else:
            rec = build_symbol(name, member, perf_reports, repo_url, local, filemap)
        rec["file"] = filemap[name]
        recs.append(rec)
    return recs


def module_hash(mod, perf_reports=None, repo_url=""):
    """Signature hash over a module's whole public surface (drift signal)."""
    recs = build_records(mod, perf_reports, repo_url)
    blob = "\n".join(r["sig_hash"] for r in recs)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _fm(value):
    """A double-quote-safe YAML scalar for front-matter title/description."""
    return value.replace("\\", " ").replace('"', "'").strip()


STUB_CALLOUT = (
    "::: {.callout-important title=\"Generated reference stub — fill me\"}\n"
    "Auto-generated scaffold for `{modname}`. Fill the prose *outside* the\n"
    "generated markers; do **not** ship as-is (SPEC §6.5). "
    "`check_reference_staleness.py` flags signature drift.\n:::\n"
)


def symbol_page(modname, slug, rec, glossary):
    fm = [f'title: "{_fm(rec["name"])}"', "generated: true"]
    if rec["desc"]:
        fm.append(f'description: "{_fm(rec["desc"])[:200]}"')
    parts = ["---", *fm, "---", "",
             f"[‹ back to `{modname}`](../{slug}.qmd)", ""]
    if rec["note"]:
        parts.append(rec["note"] + "\n")
    parts.append(rec["body"])
    gb = dims_glossary_block(rec["dims"], glossary)
    if gb:
        parts.append("")
        parts.append(gen_block(f"dims:{rec['name']}", gb))
    return "\n".join(parts) + "\n"


def module_index_page(modname, mod, records, glossary):
    slug = modname.replace(".", "-")
    all_dims, seen = [], set()
    for rec in records:
        for d in rec["dims"]:
            if d not in seen:
                seen.add(d)
                all_dims.append(d)
    blob = "\n".join(r["sig_hash"] for r in records)
    sig_hash = hashlib.sha256(blob.encode()).hexdigest()[:16]
    mod_overview, _, _ = parsed_doc(mod)
    desc = summary_line(mod_overview)
    fm = [f'title: "{modname}"', "generated: true", f"signature-hash: {sig_hash}"]
    if desc:
        fm.append(f'description: "{_fm(desc)[:200]}"')
    fm += ["listing:", f"  id: {slug}-symbols", f'  contents: "{slug}/*.qmd"',
           "  type: table", "  fields: [title, description]",
           "  sort: \"title\"", "  filter-ui: false", "  page-size: 500"]
    parts = ["---", *fm, "---", "",
             STUB_CALLOUT.replace("{modname}", modname)]
    parts.append(gen_block("overview:__module__",
                           mod_overview or "*TODO: module overview.*"))
    parts.append("")
    sec = extra_sections(mod)
    if sec:
        parts.append(gen_block("sections:__module__", sec))
        parts.append("")
    gb = dims_glossary_block(all_dims, glossary)
    if gb:
        parts.append(gen_block("dims:__page__", gb))
        parts.append("")
    parts += ["## Symbols", "", f"::: {{#{slug}-symbols}}", ":::"]
    return "\n".join(parts) + "\n", sig_hash


def pages_for_module(modname, mod, glossary, perf_reports, repo_url):
    """[(relative_path, content)] — the module index page and one page per symbol."""
    slug = modname.replace(".", "-")
    records = build_records(mod, perf_reports, repo_url)
    index_content, _ = module_index_page(modname, mod, records, glossary)
    pages = [(f"{slug}.qmd", index_content)]
    for rec in records:
        pages.append((f"{slug}/{rec['file']}.qmd",
                      symbol_page(modname, slug, rec, glossary)))
    return pages


def refresh(old, new):
    """Rewrite only the generated marker regions (and front matter) of *old*."""
    new_blocks = {m.group("tag"): m.group(0) for m in GEN_RE.finditer(new)}
    merged = GEN_RE.sub(lambda m: new_blocks.get(m.group("tag"), m.group(0)), old)
    nf = FRONT_RE.match(new)
    if nf:
        merged = FRONT_RE.sub(lambda _m: nf.group(0), merged, count=1)
    return merged


def read_page_meta(qmd):
    """Front-matter title/description/generated/signature-hash of a stub page."""
    m = FRONT_RE.match(qmd.read_text())
    meta = {}
    if not m:
        return meta
    for line in m.group(0).splitlines():
        if not line.startswith(" ") and ":" in line:   # top-level keys only
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip().strip('"')
    return meta


def build_index_tree(out, library):
    """A hierarchical index of module pages, grouped by import path.

    Leaf-module pages (``generated: true`` with a signature hash) are grouped
    under their parent package as headings, so the top reference index reflects
    the package tree and names each import path — rather than a flat A→Z list.
    """
    mods = []
    for qmd in sorted(out.glob("*.qmd")):
        meta = read_page_meta(qmd)
        if meta.get("generated") == "true" and "signature-hash" in meta:
            mods.append((meta.get("title", qmd.stem), qmd.stem, meta.get("description", "")))
    groups = {}
    for dotted, stem, desc in mods:
        parent = dotted.rsplit(".", 1)[0] if "." in dotted else library
        groups.setdefault(parent, []).append((dotted, stem, desc))
    lines = []
    for parent in sorted(groups):
        level = min(parent.count(".") + 2, 6)       # nitrix.stats -> h3, .inference -> h4
        lines.append(f"{'#' * level} `{parent}`\n")
        for dotted, stem, desc in sorted(groups[parent]):
            leaf = dotted.rsplit(".", 1)[-1]
            tail = f" — {desc}" if desc else ""
            lines.append(f"- [`{leaf}`]({stem}.qmd){tail}")
        lines.append("")
    return "\n".join(lines).strip()


def refresh_index_tree(out, library):
    """Rewrite the ``index-tree`` marker block in reference/index.qmd, if present."""
    index_path = out / "index.qmd"
    if not index_path.is_file():
        return False
    text = index_path.read_text()
    if "nitrix:gen index-tree" not in text:
        return False
    block = gen_block("index-tree", build_index_tree(out, library))
    index_path.write_text(
        GEN_RE.sub(lambda m: block if m.group("tag") == "index-tree" else m.group(0), text))
    return True


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
        slug = modname.replace(".", "-")
        subdir = out / slug
        # On a wholesale rewrite, drop stale per-symbol pages (removed symbols).
        if a.force and subdir.is_dir():
            for old in subdir.glob("*.qmd"):
                old.unlink()
        pages = pages_for_module(modname, mod, glossary, perf_reports, repo_url)
        n_new = n_ref = n_kept = 0
        for relpath, content in pages:
            dst = out / relpath
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() and not (a.force or a.refresh):
                n_kept += 1
            elif dst.exists() and a.refresh and not a.force:
                dst.write_text(refresh(dst.read_text(), content))
                n_ref += 1
            else:
                dst.write_text(content)
                n_new += 1
        created += n_new
        refreshed += n_ref
        skipped += n_kept
        verb = "~" if (n_ref and not n_new) else "+" if n_new else "="
        print(f"  {verb} {slug}  ({len(mod.exports)} symbols → {len(pages)} pages)")

    # Refresh the hierarchical module index from *all* pages present (not just
    # this run's --modules subset), so it stays complete as coverage grows.
    if refresh_index_tree(out, a.library):
        print("  = index.qmd  (refreshed hierarchical module index)")

    perf_note = f"perf={perf_reports.relative_to(CODE)}" if perf_reports else "perf=none"
    print(f"pages: {created} written, {refreshed} refreshed, {skipped} kept. "
          f"out={out.relative_to(CODE)} {perf_note}")


if __name__ == "__main__":
    main()
