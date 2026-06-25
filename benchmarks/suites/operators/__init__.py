"""Operator-mix suites: single relational operators over real TPC-H tables.

The dataframe-API counterpart to the SQL-first standard suites, and the place the
non-SQL engines (PyArrow, Ray Data) compete. Family modules are auto-discovered —
drop a ``.py`` file here and its cases register themselves on import.
"""

from __future__ import annotations

from discover import import_submodules

import_submodules(__name__)
