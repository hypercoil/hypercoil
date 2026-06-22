#!/usr/bin/env python3
"""Generate fillable API-reference STUBS for a federation library (PLAN §3.1).

Static (AST-based) — never imports the library, so it is CI-safe and immune to
JAX pathologies (jit/vmap-transformed callables, pytree-registered classes,
decorators) that derail runtime introspection. It reads each module's curated
public surface (`__all__`) and emits one `.qmd` stub per module:

  * functions/classes  -> rendered signature + a Parameters/Returns skeleton;
  * aliases / re-exports it cannot resolve from source -> a clearly-marked TODO
    stub (never a wrong signature).

Each page records a `signature-hash` in its front matter for the staleness check
(check_reference_staleness.py). Stubs are scaffolding to be filled by hand/LLM and
NEVER shipped as-is (SPEC §6.5).

Non-destructive: an existing page is left untouched (so fills survive) unless
--force. Pure stdlib.

Usage:
  gen_reference_stubs.py nitrix [--modules nitrix.stats] [--src PATH] [--out PATH] [--force]
"""
import argparse
import ast
import hashlib
import pathlib
import sys

DOC = pathlib.Path(__file__).resolve().parent.parent
CODE = DOC.parent.parent


def module_dotted(src_root, path, pkg):
    rel = path.relative_to(src_root).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join([pkg, *parts]) if parts else pkg


def literal_all(tree):
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "__all__":
                    try:
                        return list(ast.literal_eval(node.value))
                    except Exception:
                        return None
    return None


def func_sig(node):
    sig = f"{node.name}({ast.unparse(node.args)})"
    if node.returns is not None:
        sig += f" -> {ast.unparse(node.returns)}"
    return sig


def first_doc_line(node):
    doc = ast.get_docstring(node)
    if not doc:
        return ""
    for line in doc.splitlines():
        if line.strip():
            return line.strip()
    return ""


def collect(tree):
    """name -> dict(kind, sig, doc, methods, decorators) for top-level defs."""
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[node.name] = {
                "kind": "function",
                "sig": func_sig(node),
                "doc": first_doc_line(node),
                "decorators": [ast.unparse(d) for d in node.decorator_list],
            }
        elif isinstance(node, ast.ClassDef):
            methods = []
            for b in node.body:
                if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if b.name.startswith("_") and b.name != "__init__":
                        continue
                    methods.append({"sig": func_sig(b), "doc": first_doc_line(b)})
            out[node.name] = {
                "kind": "class",
                "sig": f"class {node.name}",
                "doc": first_doc_line(node),
                "decorators": [ast.unparse(d) for d in node.decorator_list],
                "methods": methods,
            }
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out[t.id] = {"kind": "alias", "sig": None, "doc": ""}
    return out


def render_object(name, info):
    """Return (markdown, signature_string_for_hash)."""
    if info is None or info.get("kind") == "alias" or info.get("sig") is None:
        md = (f"## `{name}` {{#{name}}}\n\n"
              "::: {.callout-warning title=\"TODO — could not introspect\"}\n"
              "This name is an alias / re-export / dynamically-built object; its\n"
              "signature can't be read from source. Document it by hand.\n:::\n")
        return md, f"{name}::TODO"

    kind = info["kind"]
    parts = [f"## `{name}` {{#{name}}}\n"]
    if info.get("decorators"):
        parts.append("*decorated:* " + ", ".join(f"`@{d}`" for d in info["decorators"]) + "\n")
    parts.append("```python\n" + info["sig"] + "\n```\n")
    if info.get("doc"):
        parts.append(info["doc"] + "\n")
    sig_for_hash = info["sig"]

    if kind == "function":
        parts.append("**Parameters** — *TODO: document each parameter.*\n\n"
                     "**Returns** — *TODO.*\n")
    elif kind == "class":
        parts.append("*TODO: class overview.*\n")
        if info.get("methods"):
            parts.append("\n### Methods\n")
            for m in info["methods"]:
                parts.append("```python\n" + m["sig"] + "\n```\n")
                if m["doc"]:
                    parts.append(m["doc"] + "\n")
                sig_for_hash += " | " + m["sig"]
    return "\n".join(parts), sig_for_hash


def page_for_module(modname, names, defs):
    body, sig_blob = [], []
    for name in names:
        md, sh = render_object(name, defs.get(name))
        body.append(md)
        sig_blob.append(sh)
    sig_hash = hashlib.sha256("\n".join(sig_blob).encode()).hexdigest()[:16]
    front = (f"---\ntitle: \"{modname}\"\ngenerated: true\n"
             f"signature-hash: {sig_hash}\n---\n\n"
             "::: {.callout-important title=\"Generated reference stub — fill me\"}\n"
             f"Auto-generated scaffold for `{modname}` (signatures only). Replace the\n"
             "skeletons with authored prose; do **not** ship as-is (SPEC §6.5).\n"
             "`check_reference_staleness.py` flags this page if the underlying\n"
             "signatures drift from this stub.\n:::\n\n")
    return front + "\n".join(body) + "\n", sig_hash


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("library")
    p.add_argument("--modules", default=None,
                   help="comma-separated dotted prefixes to include (default: all public)")
    p.add_argument("--src", default=None, help="package source root (default: CODE/<lib>/src/<lib>)")
    p.add_argument("--out", default=None, help="reference dir (default: CODE/<lib>/docs/reference)")
    p.add_argument("--force", action="store_true", help="overwrite existing stubs")
    a = p.parse_args()

    src_root = pathlib.Path(a.src) if a.src else CODE / a.library / "src" / a.library
    out = pathlib.Path(a.out) if a.out else CODE / a.library / "docs" / "reference"
    if not src_root.is_dir():
        sys.exit(f"source root not found: {src_root}")
    prefixes = a.modules.split(",") if a.modules else None
    out.mkdir(parents=True, exist_ok=True)

    created = skipped = empty = 0
    for path in sorted(src_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        # Curated public surface only: skip package __init__ re-export hubs and
        # any private (underscore-prefixed) module.
        if path.name == "__init__.py":
            continue
        modname = module_dotted(src_root, path, a.library)
        if any(part.startswith("_") for part in modname.split(".")[1:]):
            continue
        if prefixes and not any(modname == pre or modname.startswith(pre + ".") or pre.startswith(modname + ".") for pre in prefixes):
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError as e:
            print(f"  ! parse error in {modname}: {e}", file=sys.stderr)
            continue
        names = literal_all(tree)
        if not names:                      # only document curated public surfaces
            continue
        defs = collect(tree)
        content, _ = page_for_module(modname, names, defs)
        dst = out / (modname.replace(".", "-") + ".qmd")
        if dst.exists() and not a.force:
            skipped += 1
            continue
        dst.write_text(content)
        created += 1
        print(f"  + {dst.relative_to(CODE)}  ({len(names)} symbols)")

    print(f"stubs: {created} written, {skipped} kept (use --force to overwrite). out={out.relative_to(CODE)}")


if __name__ == "__main__":
    main()
