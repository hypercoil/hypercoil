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
    if status == "active":
        href = base
        status_label = "active"
    elif status == "scaffold":
        href = base
        status_label = "scaffold"
    else:  # aspirational — no built site yet
        href = ""
        status_label = "planned"

    tag = f'<span class="fed-card-status">{status_label}</span>'
    body = (
        f'<span class="fed-card-kicker" style="--lib-accent:{accent}"></span>'
        f'<span class="fed-card-name">{name}</span>'
        f'{tag}'
        f'<span class="fed-card-blurb">{blurb}</span>'
    )
    attrs = f' class="fed-card" style="--lib-accent:{accent}"'
    if href:
        lines.append(f'<a href="{href}"{attrs}>{body}</a>')
    else:
        lines.append(f'<span{attrs}>{body}</span>')
lines += ["", ":::", ""]

out = DOC / "federation" / "_legend.md"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(lines))
print(f"Wrote {out} ({len(libs)} libraries)")
