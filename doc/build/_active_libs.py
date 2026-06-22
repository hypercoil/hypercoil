#!/usr/bin/env python3
"""Print space-separated names of BUILDABLE federation libraries.

Buildable = has a docs tree to render: status `active` (real content) or
`scaffold` (placeholder). `aspirational` libraries (no code yet) are excluded.
"""
import pathlib
import yaml

DOC = pathlib.Path(__file__).resolve().parent.parent
libs = yaml.safe_load((DOC / "libraries.yml").read_text())["libraries"]
BUILDABLE = {"active", "scaffold"}
print(" ".join(n for n, m in libs.items() if m.get("status") in BUILDABLE))
