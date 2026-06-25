#!/usr/bin/env python3
"""Render the hand-authored SVG diagrams to retina PNGs.

Each diagram is an SVG in this directory (the editable source); this script
rasterizes every ``*.svg`` to a matching ``*.png`` with ``rsvg-convert`` (librsvg,
``brew install librsvg``). The PNGs are what the docs embed. Run after editing a
diagram::

    python docs/_static/diagrams/render.py     # or: just diagrams
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WIDTH = 2000  # retina-crisp; Furo scales to the column width


def main() -> int:
    rsvg = shutil.which("rsvg-convert")
    if not rsvg:
        print("error: `rsvg-convert` not found — install librsvg (brew install librsvg)", file=sys.stderr)
        return 1
    svgs = sorted(HERE.glob("*.svg"))
    if not svgs:
        print("no SVG diagrams found", file=sys.stderr)
        return 1
    for svg in svgs:
        png = svg.with_suffix(".png")
        subprocess.run([rsvg, "-w", str(WIDTH), str(svg), "-o", str(png)], check=True)
        print(f"rendered {png.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
