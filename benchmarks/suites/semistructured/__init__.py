"""Semistructured suites: JSON-document parsing + typed path extraction.

The text/JSON counterpart to ``standard`` (SQL tables) and ``multimodal`` (images): every
row is a nested JSON document in a Utf8 column, and each case makes the engines parse it and
pull typed fields out by path — the parse-bound work that dominates real semistructured
analytics. Batcher competes through all three of its surfaces (the ``Dataset`` API, the
``.json`` expression accessor, and SQL ``json_extract_string``); DuckDB, Polars, Daft, and
Spark use their native JSON paths. Family modules are auto-discovered, as in the other suites.
"""

from __future__ import annotations

from discover import import_submodules

import_submodules(__name__)
