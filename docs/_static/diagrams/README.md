# Diagrams

The documentation diagrams are hand-authored **SVG** files in this directory (the
editable source), rasterized to retina PNGs that the docs embed.

- Edit the `*.svg` you want to change (brand palette: blue `#2563eb`, amber
  `#f59e0b`, slate text `#1e293b`; gradient cards with soft shadows and labeled
  flow arrows).
- Regenerate the PNGs with `python render.py` (or `just diagrams`), which runs
  `rsvg-convert` over every `*.svg` here.

Diagrams: `hub`, `lifecycle`, `mergeable`, `two_planes`, `layer_stack`,
`data_flow`, `pipeline_breakers`, `carbonite_loop`.
