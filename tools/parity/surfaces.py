"""Snapshot every public name a migrating user can type in PySpark, Polars, Daft and Ray Data.

The migration registry (`batcher._internal.migration`) must classify each of those names
exactly once, and `tests/unit/test_migration_registry.py` holds it to that. The test does not
import the four libraries: they are heavy, two of them are not test dependencies, and a
PySpark install without a JVM still enumerates fine but tells CI nothing more. So the
enumeration runs here, against whatever versions are installed, and is committed as
`tools/parity/surfaces/<engine>.json`. Upgrading a competitor is then a reviewable diff of
names rather than a silent change in what "full parity" means.

A surface is one thing a user holds: a class (`DataFrame`, `pl.Expr.str`), a module whose
`__all__` they call through (`pyspark.sql.functions`), or a dataclass whose fields they set
(`ray.data.DataContext`). Names are recorded per surface because the same word means
different things on different receivers: `count` on a Spark `DataFrame` and on `GroupedData`
are two registry rows.

Run it with ``just parity-snapshot``. Each engine is optional: a library that is not
installed keeps its previous snapshot untouched rather than being written empty.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import sys
import warnings
from pathlib import Path
from typing import Any

OUT_DIR = Path(__file__).resolve().parent / "surfaces"

# (surface label, dotted target, kind). `all` reads a module's `__all__` and `fields` a
# dataclass's fields; any other kind (`class`, and `public` for a module with no `__all__`,
# such as `ray.data.aggregate`) takes every non-underscore name from `dir()`.
SURFACES: dict[str, list[tuple[str, str, str]]] = {
    "pyspark": [
        ("DataFrame", "pyspark.sql.DataFrame", "class"),
        ("GroupedData", "pyspark.sql.GroupedData", "class"),
        ("Column", "pyspark.sql.Column", "class"),
        ("functions", "pyspark.sql.functions", "all"),
        ("Window", "pyspark.sql.Window", "class"),
        ("WindowSpec", "pyspark.sql.WindowSpec", "class"),
        ("DataFrameReader", "pyspark.sql.DataFrameReader", "class"),
        ("DataFrameWriter", "pyspark.sql.DataFrameWriter", "class"),
        ("DataFrameWriterV2", "pyspark.sql.DataFrameWriterV2", "class"),
        ("DataStreamReader", "pyspark.sql.streaming.DataStreamReader", "class"),
        ("DataStreamWriter", "pyspark.sql.streaming.DataStreamWriter", "class"),
        ("StreamingQuery", "pyspark.sql.streaming.StreamingQuery", "class"),
        ("StreamingQueryManager", "pyspark.sql.streaming.StreamingQueryManager", "class"),
        ("SparkSession", "pyspark.sql.SparkSession", "class"),
        ("Catalog", "pyspark.sql.catalog.Catalog", "class"),
        ("RuntimeConfig", "pyspark.sql.conf.RuntimeConfig", "class"),
        ("UDFRegistration", "pyspark.sql.udf.UDFRegistration", "class"),
        ("UDTFRegistration", "pyspark.sql.udtf.UDTFRegistration", "class"),
        ("DataFrameNaFunctions", "pyspark.sql.DataFrameNaFunctions", "class"),
        ("DataFrameStatFunctions", "pyspark.sql.DataFrameStatFunctions", "class"),
        ("datasource", "pyspark.sql.datasource", "all"),
        ("types", "pyspark.sql.types", "all"),
    ],
    "polars": [
        ("Expr", "polars.Expr", "class"),
        ("Expr.str", "polars.expr.string.ExprStringNameSpace", "class"),
        ("Expr.dt", "polars.expr.datetime.ExprDateTimeNameSpace", "class"),
        ("Expr.list", "polars.expr.list.ExprListNameSpace", "class"),
        ("Expr.arr", "polars.expr.array.ExprArrayNameSpace", "class"),
        ("Expr.struct", "polars.expr.struct.ExprStructNameSpace", "class"),
        ("Expr.cat", "polars.expr.categorical.ExprCatNameSpace", "class"),
        ("Expr.bin", "polars.expr.binary.ExprBinaryNameSpace", "class"),
        ("Expr.name", "polars.expr.name.ExprNameNameSpace", "class"),
        ("Expr.meta", "polars.expr.meta.ExprMetaNameSpace", "class"),
        ("LazyFrame", "polars.LazyFrame", "class"),
        ("DataFrame", "polars.DataFrame", "class"),
        ("GroupBy", "polars.dataframe.group_by.GroupBy", "class"),
        ("LazyGroupBy", "polars.lazyframe.group_by.LazyGroupBy", "class"),
        ("DynamicGroupBy", "polars.dataframe.group_by.DynamicGroupBy", "class"),
        ("RollingGroupBy", "polars.dataframe.group_by.RollingGroupBy", "class"),
        ("selectors", "polars.selectors", "all"),
        ("polars", "polars", "all"),
        ("SQLContext", "polars.SQLContext", "class"),
        ("Config", "polars.Config", "class"),
        ("api", "polars.api", "all"),
    ],
    "daft": [
        ("DataFrame", "daft.DataFrame", "class"),
        ("Expression", "daft.Expression", "class"),
        ("functions", "daft.functions", "all"),
        ("GroupedDataFrame", "daft.dataframe.dataframe.GroupedDataFrame", "class"),
        ("Window", "daft.Window", "class"),
        ("daft", "daft", "all"),
        ("Session", "daft.Session", "class"),
        ("Catalog", "daft.Catalog", "class"),
        ("Table", "daft.Table", "class"),
        ("DataType", "daft.DataType", "class"),
    ],
    "ray_data": [
        ("Dataset", "ray.data.Dataset", "class"),
        ("GroupedData", "ray.data.grouped_data.GroupedData", "class"),
        ("DataIterator", "ray.data.DataIterator", "class"),
        ("ray.data", "ray.data", "all"),
        ("aggregate", "ray.data.aggregate", "public"),
        ("expressions", "ray.data.expressions", "all"),
        ("Expr", "ray.data.expressions.Expr", "class"),
        ("Expr.str", "ray.data.namespace_expressions.string_namespace._StringNamespace", "class"),
        ("Expr.list", "ray.data.namespace_expressions.list_namespace._ListNamespace", "class"),
        ("Expr.dt", "ray.data.namespace_expressions.dt_namespace._DatetimeNamespace", "class"),
        (
            "Expr.struct",
            "ray.data.namespace_expressions.struct_namespace._StructNamespace",
            "class",
        ),
        ("Expr.map", "ray.data.namespace_expressions.map_namespace._MapNamespace", "class"),
        ("Expr.arr", "ray.data.namespace_expressions.arr_namespace._ArrayNamespace", "class"),
        ("preprocessors", "ray.data.preprocessors", "all"),
        ("Preprocessor", "ray.data.preprocessor.Preprocessor", "class"),
        ("llm", "ray.data.llm", "all"),
        ("DataContext", "ray.data.DataContext", "fields"),
        ("ExecutionOptions", "ray.data.ExecutionOptions", "class"),
        ("Datasource", "ray.data.Datasource", "class"),
        ("Datasink", "ray.data.Datasink", "class"),
    ],
}

# The distribution whose version a snapshot is pinned to.
_DISTRIBUTION = {"pyspark": "pyspark", "polars": "polars", "daft": "daft", "ray_data": "ray"}

# Names a private namespace module re-exports under `__all__` purely for its own plumbing.
_PRIVATE = ("_",)


def _resolve(target: str) -> Any:
    """Import `a.b.C` as a module when it is one, else as an attribute of its parent."""
    try:
        return importlib.import_module(target)
    except ImportError:
        parent, _, attr = target.rpartition(".")
        return getattr(importlib.import_module(parent), attr)


def _names(obj: Any, kind: str) -> list[str]:
    """The public names one surface exposes."""
    if kind == "all":
        names = list(getattr(obj, "__all__", ()))
    elif kind == "fields":
        names = [f.name for f in dataclasses.fields(obj)]
    else:
        names = list(dir(obj))
    return sorted({n for n in names if not n.startswith(_PRIVATE)})


def snapshot(engine: str) -> dict[str, Any] | None:
    """Enumerate one engine, or `None` when it is not installed."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        pinned = version(_DISTRIBUTION[engine])
    except PackageNotFoundError:
        return None
    surfaces: dict[str, list[str]] = {}
    for label, target, kind in SURFACES[engine]:
        surfaces[label] = _names(_resolve(target), kind)
    return {"engine": engine, "version": pinned, "surfaces": surfaces}


def main(argv: list[str]) -> int:
    """Write one snapshot per installed engine; print a count line for each."""
    warnings.filterwarnings("ignore")
    engines = argv or list(SURFACES)
    OUT_DIR.mkdir(exist_ok=True)
    for engine in engines:
        snap = snapshot(engine)
        if snap is None:
            print(f"{engine}: not installed, snapshot left as is", file=sys.stderr)
            continue
        path = OUT_DIR / f"{engine}.json"
        path.write_text(json.dumps(snap, indent=1, sort_keys=True) + "\n")
        total = sum(len(v) for v in snap["surfaces"].values())
        print(f"{engine} {snap['version']}: {total} names over {len(snap['surfaces'])} surfaces")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
