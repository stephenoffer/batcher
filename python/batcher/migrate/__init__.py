"""`batcher.migrate`: rewrite code onto, and off, Batcher's API, driven by the migration registry.

Run it as `python -m batcher.migrate`. It needs the `migrate` extra (libcst) and does not need
the compiled engine, so it runs in CI jobs and on machines where Batcher is not built.
`canonicalize` renames Batcher's own removed spellings, `translate` rewrites a PySpark, Polars,
Daft or Ray Data script onto Batcher, and `export` rewrites a Batcher script onto one of them.
"""

from __future__ import annotations

from batcher.migrate.canonical import Edit, RenameReport, canonicalize
from batcher.migrate.finish import Report, Site
from batcher.migrate.outbound import export
from batcher.migrate.translate import translate

__all__ = ["Edit", "RenameReport", "Report", "Site", "canonicalize", "export", "translate"]
