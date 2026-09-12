"""The five h2o ``join`` questions as Ray Data pipelines.

Each is ``x`` joined to one right-hand table on one key, projecting ``x``'s columns plus a
few of the right side's under the names the benchmark's SQL gives them. The renames are the
only fiddly part: the SQL aliases ``medium.id1`` to ``medium_id1``, and Ray Data's join
carries both sides' columns under their original names, so the right side is renamed before
the join rather than after -- which also keeps the join key unambiguous.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from suites.h2o.join_ray.base import impl, join, to_arrow

#: ``x``'s own columns, which every question projects with ``x.*``.
X_COLUMNS = ["id1", "id2", "id3", "id4", "id5", "id6", "v1"]


def _prefixed(ds: Any, key: str, keep: dict[str, str]) -> Any:
    """The right side cut to its key plus `keep`, renamed to the SQL's output names."""

    def rename(batch: pa.Table) -> pa.Table:
        cols = {key: batch.column(key)}
        for src, dst in keep.items():
            cols[dst] = batch.column(src)
        return pa.table(cols)

    return ds.map_batches(rename, batch_format="pyarrow")


def _result(joined: Any, extra: list[str]) -> pa.Table:
    """Materialize and order the columns as ``SELECT x.*, <extra>`` does."""
    got = to_arrow(joined)
    names = [c for c in X_COLUMNS + extra if c in got.schema.names]
    return got.select(names)


@impl("h2o-join-q1")
def q1(t: dict[str, Any]) -> pa.Table:
    """``x JOIN small USING (id1)`` -- the small right side."""
    right = _prefixed(t["small"], "id1", {"id4": "small_id4", "v2": "v2"})
    return _result(join(t["x"], right, "id1"), ["small_id4", "v2"])


@impl("h2o-join-q2")
def q2(t: dict[str, Any]) -> pa.Table:
    """``x JOIN medium USING (id2)`` -- inner, on an integer key."""
    right = _prefixed(
        t["medium"],
        "id2",
        {"id1": "medium_id1", "id4": "medium_id4", "id5": "medium_id5", "v2": "v2"},
    )
    return _result(
        join(t["x"], right, "id2"),
        ["medium_id1", "medium_id4", "medium_id5", "v2"],
    )


@impl("h2o-join-q3")
def q3(t: dict[str, Any]) -> pa.Table:
    """``x LEFT JOIN medium USING (id2)`` -- the same join, outer."""
    right = _prefixed(
        t["medium"],
        "id2",
        {"id1": "medium_id1", "id4": "medium_id4", "id5": "medium_id5", "v2": "v2"},
    )
    return _result(
        join(t["x"], right, "id2", how="left_outer"),
        ["medium_id1", "medium_id4", "medium_id5", "v2"],
    )


@impl("h2o-join-q4")
def q4(t: dict[str, Any]) -> pa.Table:
    """``x JOIN medium USING (id5)`` -- inner, on a factor (string) key."""
    right = _prefixed(
        t["medium"],
        "id5",
        {"id1": "medium_id1", "id2": "medium_id2", "id4": "medium_id4", "v2": "v2"},
    )
    return _result(
        join(t["x"], right, "id5"),
        ["medium_id1", "medium_id2", "medium_id4", "v2"],
    )


@impl("h2o-join-q5")
def q5(t: dict[str, Any]) -> pa.Table:
    """``x JOIN big USING (id3)`` -- the right side as large as the left."""
    right = _prefixed(
        t["big"],
        "id3",
        {
            "id1": "big_id1",
            "id2": "big_id2",
            "id4": "big_id4",
            "id5": "big_id5",
            "id6": "big_id6",
            "v2": "v2",
        },
    )
    return _result(
        join(t["x"], right, "id3"),
        ["big_id1", "big_id2", "big_id4", "big_id5", "big_id6", "v2"],
    )
