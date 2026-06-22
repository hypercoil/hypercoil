#!/usr/bin/env python3
"""Add a library to the federation docs (SPEC §10).

Does three things, all idempotent and non-destructive:
  1. registers the library in libraries.yml (legend + build list) if missing;
  2. registers an xref namespace in xref/namespaces.yml if missing;
  3. scaffolds <code>/<name>/docs/ from doc/_template (backing up any frame file
     it would replace to <stem>.legacy<suffix>; never touches non-template files)
     and installs the pinned federation extension for standalone preview.

Pure stdlib. Run via add-library.sh (which packages the extension first).

Usage:
  add_library.py NAME --accent '#RRGGBB' [--blurb "..."] [--status scaffold]
                 [--repo URL] [--ref main] [--no-scaffold]
"""
import argparse
import pathlib
import re
import shutil
import sys

DOC = pathlib.Path(__file__).resolve().parent.parent          # .../hypercoil/doc
CODE = DOC.parent.parent                                       # .../code
TEMPLATE = DOC / "_template" / "docs"
LIBS_YML = DOC / "libraries.yml"
NS_YML = DOC / "xref" / "namespaces.yml"
EXT = DOC / "_extensions" / "hypercoil" / "federation"
TOKENS = (".qmd", ".yml", ".scss", ".md")


def _append_if_missing(path, key_re, block, label):
    text = path.read_text()
    if re.search(key_re, text, re.M):
        print(f"  = {label}: '{path.name}' already has entry; left as-is")
        return
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text + block)
    print(f"  + {label}: appended entry to {path.name}")


def register(name, blurb, repo, ref, accent, status):
    _append_if_missing(
        LIBS_YML, rf"^  {re.escape(name)}:",
        (f"  {name}:\n"
         f'    blurb: "{blurb}"\n'
         f"    repo: {repo}\n"
         f"    ref: {ref}\n"
         f"    base_path: /{name}/\n"
         f'    accent: "{accent}"\n'
         f"    docs_subdir: docs\n"
         f"    status: {status}\n"),
        "libraries.yml",
    )
    _append_if_missing(
        NS_YML, rf"^  {re.escape(name)}:",
        (f"  {name}:\n"
         f"    base_path: /{name}/\n"
         f"    hub_url: https://hypercoil.github.io/{name}/\n"
         f"    repo: {repo}\n"
         f"    anchors: {{}}\n"),
        "namespaces.yml",
    )


def render(text, name, blurb, accent):
    return (text.replace("@@LIB@@", name)
                .replace("@@BLURB@@", blurb)
                .replace("@@ACCENT@@", accent))


def scaffold(name, blurb, accent):
    target = CODE / name / "docs"
    for src in sorted(TEMPLATE.rglob("*")):
        if src.is_dir():
            continue
        rel = src.relative_to(TEMPLATE)
        dst = target / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        content = src.read_text()
        if src.suffix in TOKENS:
            content = render(content, name, blurb, accent)
        if dst.exists():
            if dst.read_text() == content:
                continue                                       # already current
            backup = dst.with_name(dst.stem + ".legacy" + dst.suffix)
            if not backup.exists():
                shutil.copy2(dst, backup)
                print(f"  ~ backed up {rel} -> {backup.name}")
        dst.write_text(content)
        print(f"  + scaffolded {rel}")

    # Install the pinned federation extension for standalone preview.
    if EXT.is_dir():
        ext_dst = target / "_extensions" / "hypercoil" / "federation"
        if ext_dst.exists():
            shutil.rmtree(ext_dst)
        shutil.copytree(EXT, ext_dst)
        print("  + installed federation extension")
    else:
        print("  ! extension not packaged; run build/package-extension.sh", file=sys.stderr)


def main():
    p = argparse.ArgumentParser(description="Add a library to the federation docs.")
    p.add_argument("name")
    p.add_argument("--accent", required=True, help="#RRGGBB bold accent colour")
    p.add_argument("--blurb", default="(blurb pending)")
    p.add_argument("--status", default="scaffold",
                   choices=["active", "scaffold", "aspirational"])
    p.add_argument("--repo", default=None)
    p.add_argument("--ref", default="main")
    p.add_argument("--no-scaffold", action="store_true",
                   help="registry only (e.g. for code-less aspirational libs)")
    a = p.parse_args()
    repo = a.repo or f"https://github.com/hypercoil/{a.name}"

    print(f"add-library: {a.name} (status={a.status}, accent={a.accent})")
    register(a.name, a.blurb, repo, a.ref, a.accent, a.status)
    if a.no_scaffold:
        print("  (skipping scaffold)")
    elif not (CODE / a.name).is_dir():
        print(f"  ! {CODE / a.name} not present; registry updated, scaffold skipped",
              file=sys.stderr)
    else:
        scaffold(a.name, a.blurb, a.accent)
    print("done.")


if __name__ == "__main__":
    main()
