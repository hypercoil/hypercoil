-- resolver.lua — pure cross-reference resolution logic for the federation.
--
-- This module is deliberately free of any Pandoc/Quarto dependency so it can be
-- unit-tested standalone (`quarto pandoc lua test_resolver.lua`). The Quarto
-- shortcode wrapper (xref.lua) adapts Pandoc metadata into the plain Lua tables
-- this module expects, then delegates here.
--
-- Contract (mirrors SPEC §5.3):
--   * composed mode  -> same-origin relative URL  (/<base_path>/page.html#frag)
--   * standalone mode -> published hub URL         (<hub_url>/page.html#frag)
--   * unknown namespace OR unknown anchor -> returns (nil, message)
--
-- The function RETURNS errors rather than calling error(): Quarto monkeypatches
-- the global `error` to be non-fatal inside shortcodes, so a (nil, msg) contract
-- is the only host-agnostic way to signal a build-breaking failure. The caller
-- (xref.lua) turns a nil result into a hard non-zero exit.

local M = {}

-- namespaces: { [ns] = { base_path=str, hub_url=str, anchors = { [slug]=target } } }
--   target is "relative/page.qmd" or "relative/page.qmd#fragment".
-- mode: "composed" | "standalone"
-- ref:  "ns:anchor"
-- Returns the resolved URL string on success, or (nil, message) on failure.
function M.resolve(namespaces, mode, ref)
  ref = tostring(ref or "")
  local ns, slug = ref:match("^([%w%-_]+):([%w%-_%./]+)$")
  if not ns then
    return nil, "xref: malformed reference '" .. ref ..
                "' — expected '<namespace>:<anchor>'"
  end

  local entry = namespaces and namespaces[ns]
  if not entry then
    return nil, "xref: unknown namespace '" .. ns .. "' in reference '" .. ref ..
                "' — register it in xref/namespaces.yml"
  end

  local target = entry.anchors and entry.anchors[slug]
  if not target then
    return nil, "xref: unknown anchor '" .. slug .. "' in namespace '" .. ns ..
                "' — only curated, registered anchors are cross-referenceable " ..
                "(see xref/namespaces.yml)"
  end

  -- Split "page.qmd#frag"; default the fragment to the slug itself.
  local path, frag = target:match("^(.-)#(.+)$")
  if not path then
    path, frag = target, slug
  end
  local html = path:gsub("%.qmd$", ".html"):gsub("%.md$", ".html")

  local prefix
  if mode == "composed" then
    prefix = entry.base_path or ("/" .. ns .. "/")
  elseif mode == "standalone" then
    prefix = entry.hub_url
    if not prefix or prefix == "" then
      return nil, "xref: namespace '" .. ns .. "' has no hub_url, required to " ..
                  "resolve '" .. ref .. "' in a standalone build"
    end
  else
    return nil, "xref: unknown build mode '" .. tostring(mode) ..
                "' (expected 'composed' or 'standalone')"
  end

  if prefix:sub(-1) ~= "/" then prefix = prefix .. "/" end
  return prefix .. html .. "#" .. frag
end

return M
