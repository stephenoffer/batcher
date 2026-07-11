"""Scan suite: file-layout sensitivity of the parquet scan path.

One logical table, three physical layouts (one big file / ~132 MiB files / many ~1.2 MiB
files), read by every engine in the lineup. Measures scan planning — listing, footer
opens, fixed per-file cost — which the table-shaped standard suites never isolate.
Family modules are auto-discovered, as in the other suites.
"""

from __future__ import annotations

from discover import import_submodules

import_submodules(__name__)
