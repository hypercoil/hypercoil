-- test_resolver.lua — unit tests for the federation xref resolver.
-- Run:  quarto pandoc lua doc/xref/test_resolver.lua   (exit 0 = all pass)
-- This is federation-critical software; treat failures as build-breaking.

-- Make the sibling resolver.lua importable regardless of CWD.
local here = (arg and arg[0] or ""):gsub("[^/\\]*$", "")
if here == "" then here = "./" end
package.path = here .. "?.lua;" .. package.path
local resolver = require("resolver")

-- A representative registry (a subset of xref/namespaces.yml).
local NS = {
  nitrix = {
    base_path = "/nitrix/",
    hub_url   = "https://hypercoil.github.io/nitrix/",
    anchors   = {
      ["fellner-schall"] = "explanation/fellner-schall.qmd#fellner-schall",
    },
  },
  tensorbids = {
    base_path = "/tensorbids/",
    hub_url   = "https://hypercoil.github.io/tensorbids/",
    anchors   = {
      ["navigator-dag"] = "explanation/navigator-dag.qmd",  -- frag defaults to slug
    },
  },
}

local pass, fail = 0, 0
local function ok(name, cond, detail)
  if cond then pass = pass + 1
  else fail = fail + 1; io.stderr:write("FAIL: " .. name .. "  " .. (detail or "") .. "\n") end
end

-- Resolves to a value; assert equality.
local function expect(name, mode, ref, want)
  local got = resolver.resolve(NS, mode, ref)
  ok(name, got == want, "got=" .. tostring(got) .. " want=" .. want)
end

-- Expect a build-breaking failure: (nil, message) whose message contains `needle`.
local function expect_error(name, mode, ref, needle)
  local got, err = resolver.resolve(NS, mode, ref)
  ok(name, got == nil and tostring(err):find(needle, 1, true) ~= nil,
     "got=" .. tostring(got) .. " err=" .. tostring(err))
end

-- ---- positive cases --------------------------------------------------------
expect("composed/cross-lib", "composed", "nitrix:fellner-schall",
       "/nitrix/explanation/fellner-schall.html#fellner-schall")
expect("standalone/cross-lib", "standalone", "nitrix:fellner-schall",
       "https://hypercoil.github.io/nitrix/explanation/fellner-schall.html#fellner-schall")
expect("composed/frag-defaults-to-slug", "composed", "tensorbids:navigator-dag",
       "/tensorbids/explanation/navigator-dag.html#navigator-dag")
expect("standalone/frag-defaults-to-slug", "standalone", "tensorbids:navigator-dag",
       "https://hypercoil.github.io/tensorbids/explanation/navigator-dag.html#navigator-dag")

-- ---- negative cases (must fail the build) ----------------------------------
expect_error("unknown-namespace", "composed", "nimbus:foo", "unknown namespace 'nimbus'")
expect_error("unknown-anchor", "composed", "nitrix:no-such-anchor", "unknown anchor 'no-such-anchor'")
expect_error("malformed-ref", "composed", "nitrix-fellner-schall", "malformed reference")
expect_error("bad-mode", "frobnicate", "nitrix:fellner-schall", "unknown build mode")

-- ---- report ----------------------------------------------------------------
io.write(string.format("xref resolver: %d passed, %d failed\n", pass, fail))
os.exit(fail == 0 and 0 or 1)
