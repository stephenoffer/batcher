"""`batcher.migrate`: rewrite code onto, and off, Batcher's API, driven by the migration registry.

Run it as `python -m batcher.migrate`. It needs the `migrate` extra (libcst) and does not need
the compiled engine, so it runs in CI jobs and on machines where Batcher is not built.
"""

from __future__ import annotations

from batcher.migrate.canonical import Edit, RenameReport, canonicalize

__all__ = ["Edit", "RenameReport", "canonicalize"]
