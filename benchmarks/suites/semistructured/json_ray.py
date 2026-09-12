"""Ray Data pipelines for the JSON path-extraction cases.

Ray Data has no JSON-path expression, so each pipeline parses the document in a
``map_batches`` over the PyArrow blocks Ray already holds and pulls the fields out with
``json.loads``. That is what a Ray Data user would write, and it is the honest comparison:
the other engines' advantage here is a *vectorized* extractor, and the measurement should
show that rather than hide it behind a helper.

The parse runs once per batch and produces the extracted columns; the group-by and the
aggregate are Ray Data's own.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pyarrow as pa

__all__ = ["array_first_tag", "extract", "filter_agg", "group_count", "project5"]


def _paths(doc: dict[str, Any], path: str) -> Any:
    """The value at a dotted ``$.a.b`` path, or `None` where any step is missing."""
    cur: Any = doc
    for step in path.removeprefix("$.").split("."):
        if not isinstance(cur, dict) or step not in cur:
            return None
        cur = cur[step]
    return cur


def extract(ds: Any, fields: dict[str, str]) -> Any:
    """Parse ``payload`` once per row and project `fields` out of it.

    Args:
        ds: The dataset carrying a Utf8 ``payload`` column.
        fields: Output column name -> the ``$.a.b`` path to pull into it.

    Returns:
        A dataset of exactly the extracted columns.
    """

    def pull(batch: pa.Table) -> pa.Table:
        docs = [json.loads(p) if p else {} for p in batch.column("payload").to_pylist()]
        return pa.table(
            {name: pa.array([_paths(d, p) for d in docs]) for name, p in fields.items()}
        )

    return ds.map_batches(pull, batch_format="pyarrow")


def group_count(ds: Any, name: str, path: str) -> pa.Table:
    """``SELECT <path> AS name, COUNT(*) GROUP BY 1`` over the parsed documents."""
    from suites.h2o.join_ray.base import to_arrow

    got = to_arrow(extract(ds, {name: path}).groupby(name).count())
    return pa.table({name: got.column(name), "n": got.column("count()").cast("int64")})


def project5(ds: Any, keys: dict[str, str], measures: dict[str, str]) -> pa.Table:
    """Three group keys, two summed measures and a row count, all from one parse.

    The count comes from the same aggregate as the sums. Asking for it separately and
    joining or zipping the two results is the trap this suite's Ray pipelines hit twice:
    two aggregations need not return their groups in the same order.
    """
    import pyarrow.compute as pc
    from ray.data.aggregate import Count, Sum

    from suites.h2o.join_ray.base import to_arrow

    def pull(batch: pa.Table) -> pa.Table:
        docs = [json.loads(p) if p else {} for p in batch.column("payload").to_pylist()]
        cols = {n: pa.array([_paths(d, p) for d in docs]) for n, p in keys.items()}
        for n, p in measures.items():
            cols[n] = pc.cast(pa.array([_paths(d, p) for d in docs]), "float64")
        return pa.table(cols)

    parsed = ds.map_batches(pull, batch_format="pyarrow")
    aggs = [Sum(m) for m in measures]
    got = to_arrow(parsed.groupby(list(keys)).aggregate(*aggs, Count()))
    out = {k: got.column(k) for k in keys}
    for m in measures:
        out[m] = got.column(f"sum({m})")
    out["n"] = got.column("count()").cast("int64")
    return pa.table(out)


def array_first_tag(ds: Any, name: str) -> pa.Table:
    """``$.tags[0]`` grouped and counted -- the array-index path."""
    from suites.h2o.join_ray.base import to_arrow

    def pull(batch: pa.Table) -> pa.Table:
        docs = [json.loads(p) if p else {} for p in batch.column("payload").to_pylist()]
        tags = [(d.get("tags") or [None])[0] for d in docs]
        return pa.table({name: pa.array(tags)})

    got = to_arrow(ds.map_batches(pull, batch_format="pyarrow").groupby(name).count())
    return pa.table({name: got.column(name), "n": got.column("count()").cast("int64")})


def filter_agg(ds: Any) -> pa.Table:
    """``SUM(event.value), COUNT(*) WHERE event.type = 'purchase'``."""
    import pyarrow.compute as pc

    def pull(batch: pa.Table) -> pa.Table:
        docs = [json.loads(p) if p else {} for p in batch.column("payload").to_pylist()]
        keep = [i for i, d in enumerate(docs) if _paths(d, "$.event.type") == "purchase"]
        vals = [_paths(docs[i], "$.event.value") for i in keep]
        return pa.table({"v": pc.cast(pa.array(vals), "float64")})

    from suites.h2o.join_ray.base import to_arrow

    got = to_arrow(ds.map_batches(pull, batch_format="pyarrow"))
    if got.num_rows == 0:
        return pa.table(
            {
                "s": pa.array([None], type=pa.float64()),
                "n": pa.array([0], type=pa.int64()),
            }
        )
    col = got.column("v")
    return pa.table(
        {
            "s": pa.array([pc.sum(col).as_py()], type=pa.float64()),
            "n": pa.array([got.num_rows], type=pa.int64()),
        }
    )


def ray_fn(ctx: Any, build: Callable[[Any], pa.Table]) -> Callable[[], pa.Table] | None:
    """`build` bound to the ``events`` Ray handle, or `None` when Ray is not in the lineup."""
    if "ray" not in ctx.names():
        return None
    handle = ctx.handle("events", "ray")
    return lambda: build(handle)
