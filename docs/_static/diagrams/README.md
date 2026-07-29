# Diagrams

The documentation diagrams are **SVG**, and the SVG is what the pages embed. It stays
sharp at any zoom, it is a fraction of the size of the equivalent raster, and it can
restyle itself for the reader's theme, which a PNG cannot.

## Theme awareness

Every diagram carries a `prefers-color-scheme: dark` block inside its own `<style>`
element. CSS overrides SVG presentation attributes, so the authored light palette stays
the default and dark mode remaps the surfaces and text in place.

This matters more than it sounds. Before it, `custom.css` forced a white mat behind
every diagram so the light PNGs would not sit oddly on the page, which meant each one
became a glaring white slab in dark mode. The mat now uses the theme's own surface color,
and only non-SVG images still get a white background.

## Authoring a new diagram

Two ways in, both fine:

- **Write the SVG by hand.** Match the visual language: blue `#2563eb` for the primary
  subject, amber `#f59e0b` for the accent or highlighted path, slate `#1e293b` for text,
  white cards with a soft drop shadow, and a label on every arrow. Copy the `<style>`
  block from an existing file so the diagram is theme-aware.
- **Build it from `tools/diagrams/_authoring.py`.** That module holds the same language as
  functions (`band`, `card`, `arrow`, `curve`, `label`, `note`) with the theme-aware defs
  already wired. Write a small script beside it in `tools/diagrams/`, run it, and commit
  both the script and the SVG it emits here. `transfer_modes.py`, `adaptive_loop.py`, and
  `inference_stages.py` are the examples to copy.

**The scripts live in `tools/diagrams/`, not here, and that separation is load-bearing.**
Sphinx copies `html_static_path` wholesale, so any `.py` in this directory is published as
a website asset. `exclude_patterns` does not filter static files, so moving them out is the
only thing that works.

Design constraints that matter: label every arrow, because an unlabeled one only says
"related"; cap a diagram at roughly seven boxes and split into two zoom levels past that;
never encode meaning in color alone; and keep every label short enough to survive being
scaled to a phone column. A diagram must agree with the prose around it, and it must
never be the only carrier of a fact.

Diagrams that assert something the code decides should name their source module in the
generating script's docstring, so the two can be kept in step. `adaptive_loop.py` and
`transfer_modes.py` both do this.

## Raster output

`tools/diagrams/render.py` rasterizes every `*.svg` to a retina PNG with `rsvg-convert` (librsvg). The
docs no longer need those PNGs, so this is only for contexts that cannot take SVG, such
as a slide deck or a PDF export. It is not part of the docs build, and `rsvg-convert` is
not required to work on the documentation.

## Charts

`gpu_utilization` and `stage_overlap` are charts rather than diagrams, so they answer to
`.claude/rules/documentation.md`'s charts rule as well: every figure traces to a committed
benchmark, named in the generating script's docstring, and the axes carry their units. Both
were color-checked against each surface rather than eyeballed. The light "before" bar in
`stage_overlap` failed the 3:1 contrast floor at its first value and was re-stepped.

Current diagrams: `hub`, `lifecycle`, `mergeable`, `two_planes`, `layer_stack`,
`data_flow`, `pipeline_breakers`, `carbonite_loop`, `adaptive_loop`, `transfer_modes`,
`inference_stages`. Charts: `gpu_utilization`, `stage_overlap`.
