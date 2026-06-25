"""SQL-first industry-standard suites: TPC-H, TPC-DS, ClickBench.

Each module declares its queries as SQL strings and registers them with
``Suite.sql`` (one query, fanned across every SQL-capable engine via
``registry.sql_case``). Modules are auto-discovered — drop a ``.py`` file here and it
registers itself on import.
"""

from __future__ import annotations

from discover import import_submodules

import_submodules(__name__)
