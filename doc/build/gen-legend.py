#!/usr/bin/env python3
"""Generate the colour-key legend partial from libraries.yml (SPEC §4.2, §7.6).

The legend cannot drift because it is regenerated from the registry on every
build. Output: doc/federation/_legend.md (an underscore-prefixed include).
"""
import pathlib
import yaml

DOC = pathlib.Path(__file__).resolve().parent.parent
libs = yaml.safe_load((DOC / "libraries.yml").read_text())["libraries"]

lines = ["::: {.federation-legend}", ""]
for name, m in libs.items():
    accent = m["accent"]
    blurb = m["blurb"]
    base = m.get("base_path", "#")
    status = m.get("status", "scaffold")
    swatch = (
        f'<span style="display:inline-block;width:0.9em;height:0.9em;'
        f'border-radius:50%;background:{accent};vertical-align:middle;'
        f'margin-right:0.45em;border:1px solid #2a2a31;"></span>'
    )
    if status == "active":
        label = f"[`{name}`]({base})"
    elif status == "scaffold":
        label = f"[`{name}`]({base}) *(scaffold)*"
    else:  # aspirational — no built site yet
        label = f"`{name}` *(planned)*"
    lines.append(f"- {swatch} {label} — {blurb}")
lines += ["", ":::", ""]

out = DOC / "federation" / "_legend.md"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(lines))
print(f"Wrote {out} ({len(libs)} libraries)")
