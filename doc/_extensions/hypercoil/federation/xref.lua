-- xref.lua — federation cross-reference shortcode (Quarto).
--
-- Authoring forms (SPEC §5.3):
--   {{< xref nitrix:fellner-schall >}}
--   {{< xref nitrix:fellner-schall "custom link text" >}}
--
-- Resolution mode and the namespace registry come from document metadata
-- (`xref-mode`, `xref-namespaces`), supplied by xref/namespaces.yml. The pure
-- resolution logic lives in resolver.lua (unit-tested standalone); this file is
-- only the Pandoc/Quarto adapter. A bad reference raises and fails the build.

-- Locate the sibling resolver.lua regardless of how Quarto invokes us.
-- debug.getinfo gives this chunk's own file path ("@/abs/path/xref.lua"),
-- which is reliable in the shortcode context (PANDOC_SCRIPT_FILE is not).
local src = debug.getinfo(1, "S").source
local here = src:sub(1, 1) == "@" and src:sub(2):gsub("[^/\\]*$", "") or "./"
package.path = here .. "?.lua;" .. package.path
local resolver = require("resolver")

local stringify = pandoc.utils.stringify

-- Convert the `xref-namespaces` MetaMap into the plain table resolver expects.
local function build_namespaces(meta)
  local out = {}
  local nsmeta = meta and meta["xref-namespaces"]
  if not nsmeta then return out end
  for ns, entry in pairs(nsmeta) do
    local e = { anchors = {} }
    if entry.base_path then e.base_path = stringify(entry.base_path) end
    if entry.hub_url   then e.hub_url   = stringify(entry.hub_url)   end
    if entry.anchors then
      for slug, target in pairs(entry.anchors) do
        e.anchors[slug] = stringify(target)
      end
    end
    out[ns] = e
  end
  return out
end

local function xref(args, kwargs, meta)
  local ref  = stringify(args[1])
  local mode = (meta and meta["xref-mode"] and stringify(meta["xref-mode"]))
               or "standalone"
  local namespaces = build_namespaces(meta)

  -- Broken cross-references are build errors, never silent 404s. The resolver
  -- returns (nil, msg) on failure (Quarto's error() is non-fatal in shortcodes),
  -- so we print a clear diagnostic and hard-exit non-zero to fail the build.
  local url, err = resolver.resolve(namespaces, mode, ref)
  if not url then
    io.stderr:write("\nERROR [xref] " .. tostring(err) .. "\n")
    os.exit(1)
  end

  local label
  if args[2] then
    label = args[2]                      -- author-supplied link text (Inlines)
  else
    local slug = ref:match(":([%w%-_%./]+)$") or ref
    label = pandoc.Str(slug)
  end
  return pandoc.Link(label, url)
end

return { ["xref"] = xref }
