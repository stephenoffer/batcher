"""Scan benchmark: the same logical table read from three different file layouts.

The Ray benchmark bucket stores one 16-column ``int64`` dataset three ways — as a single
~1 GiB file, as ~132 MiB files, and as ~1.2 MiB files. At scale 1 and 10 the three hold
an *identical* row count (8,388,608 and 83,886,080), so the layout is the only variable
and the ms columns are directly comparable across the three families. (At scale 100 and
above the many-small corpus is not row-count-equivalent — see ``sources.ScanCorpus`` —
so there the per-engine comparison within a family still holds, but the cross-family
one is indicative only.)

This is the benchmark that measures **scan planning**, not compute: how much an engine
pays to list files, open footers, and build a scan before it reads a single value. It is
where Ray Data and Spark are known to struggle and where a query engine's fixed
per-file overhead is exposed — an overhead that a TPC-H run over eight tidy files never
shows. Each engine's reader is therefore constructed *inside* the timed call (see
``Engine.scan_sql_runner``), so listing and metadata are measured, not amortized away.

Cases are named ``scan-<shape>-<layout>`` and grouped into one family per layout, so
``--family scan-many_small`` runs every shape against the pathological layout and
``--only scan-count`` runs one shape against all three.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from registry import EngineQueries, suite
from sources import SCAN_LAYOUTS

from .shapes import SHAPES, Shape

if TYPE_CHECKING:
    from context import Context


def _engine_queries(ctx: Context, layout: str, shape: Shape) -> EngineQueries:
    """Bind ``shape`` to every engine that can express it over the ``layout`` corpus.

    SQL engines get the corpus glob and expand it themselves; PyArrow and Ray Data have
    no SQL surface and no glob support, so they receive the listed file paths. Either
    way the scan is built when the returned callable runs, never before.
    """
    corpus = ctx.corpora[layout]
    queries: EngineQueries = {}
    for engine in ctx.engines:
        if engine.supports_sql:
            runner = engine.scan_sql_runner(corpus.glob)
            if runner is not None:
                queries[engine.name] = lambda run=runner, sql=shape.sql: run(sql)
        elif (native := shape.native.get(engine.name)) is not None:
            queries[engine.name] = lambda eng=engine, fn=native, c=corpus: fn(
                eng.scan_handle(*c.open())
            )
    return queries


def _register() -> None:
    """One family per layout, one case per (shape, layout)."""
    for layout in SCAN_LAYOUTS:
        family = suite(f"scan-{layout}", dataset="scan")
        for shape in SHAPES:

            def build(ctx: Context, _layout: str = layout, _shape: Shape = shape) -> EngineQueries:
                return _engine_queries(ctx, _layout, _shape)

            family.case(f"scan-{shape.name}-{layout}")(build)


_register()
