"""Every relational operator x every non-default execution flag, on edge-case input.

`test_diff_operator_matrix.py` crosses operator x *scheduling* (`collect`, spill,
`iter_batches`) and lists the four wrong-answer bugs that lived in that gap. This file is the
dimension that matrix does not vary: the `execution.*` switches that pick a different code
path for the same scheduling. `CLAUDE.md` names this shape directly -- "nothing combined an
operator with a non-default flag on a non-default path" -- and one of these flags,
`fast_path`, is **off by default**, so its path is only ever reached by a test that turns it
on.

A flag changes *how* the answer is computed, never *what* it is, so the assertion is equality
with the same query at default settings. Sorts are compared order-sensitively: `assert_same`
is order-independent by design and is how the spilled-descending-sort bug stayed invisible.

The input crosses a morsel boundary on purpose. Without that, `morsel_rows` and
`adaptive_morsel_sizing` are switches over a single batch and the matrix tests nothing.
"""

from __future__ import annotations

import math

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher.config import active_config, reset_option, set_option

pytestmark = pytest.mark.differential

#: Above the 16,384-row default morsel, so every path sees more than one batch.
_ROWS, _GROUPS = 20_000, 300


@pytest.fixture(scope="module")
def table() -> pa.Table:
    """Nulls in every column, NaN and `-0.0` among the float keys, duplicate group keys."""
    rng = np.random.default_rng(7)
    values = rng.normal(size=_ROWS)
    values[::701] = float("nan")
    values[::1103] = -0.0
    return pa.table(
        {
            "k": pa.array(
                [None if i % 257 == 0 else f"k{i % _GROUPS}" for i in range(_ROWS)], pa.string()
            ),
            "v": pa.array(values, pa.float64()),
            "i": pa.array([None if j % 373 == 0 else int(j) for j in range(_ROWS)], pa.int64()),
        }
    )


#: `(builder, order_matters)`. Order matters exactly where the operator defines one.
SHAPES = {
    "sort_desc": (lambda d: d.sort("v", descending=True), True),
    "sort_two_keys": (lambda d: d.sort("k", "i"), True),
    "top_n": (lambda d: d.sort("v", descending=True).limit(25), True),
    "group_sum": (lambda d: d.groupby("k").agg(s=bt.col("v").sum()), False),
    "group_multi": (
        lambda d: d.groupby("k").agg(
            s=bt.col("v").sum(), n=bt.col("i").count(), m=bt.col("v").max()
        ),
        False,
    ),
    "distinct": (lambda d: d.select("k").distinct(), False),
    "window_rank": (
        lambda d: d.with_columns(r=bt.col("i").rank().over(partition_by="k", order_by="i")),
        False,
    ),
    "self_join": (lambda d: d.join(d.select("k", "i"), on="k"), False),
    "filter_conjunction": (
        lambda d: d.filter((bt.col("v") > 0) & bt.col("i").is_not_null()),
        False,
    ),
    "arithmetic": (lambda d: d.select(x=bt.col("v") * 2 + bt.col("i").cast("float64")), False),
}

#: Each is a different code path for the same semantics, not a different answer.
FLAGS = [
    ("execution.parallelism", 1),  # the sequential executor -- invariant #6's seq == par
    ("execution.fast_path", True),  # off by default, so nothing else reaches it
    ("execution.fuse_linear", False),  # operator fusion disabled
    ("execution.adaptive_morsel_sizing", False),
    ("execution.morsel_rows", 512),  # many small morsels instead of few large
    ("execution.streaming", False),
]


def _rows(table: pa.Table, ordered: bool) -> list[tuple]:
    """Rows as comparable tuples, with NaN and `-0.0` given stable spellings.

    NaN != NaN and `-0.0 == 0.0` in Python, so a raw comparison would either report a false
    divergence on every NaN or hide a real `-0.0`/`0.0` split. Both are normalized once here.
    """
    names = table.schema.names
    out = [
        tuple(
            "NaN" if isinstance(x, float) and math.isnan(x) else (0.0 if x == 0 else x) for x in row
        )
        for row in zip(*[table.column(c).to_pylist() for c in names], strict=True)
    ]
    return out if ordered else sorted(out, key=repr)


def _run(build, table: pa.Table) -> pa.Table:
    return build(bt.from_arrow(table)).collect()


@pytest.fixture(params=FLAGS, ids=[f"{n.split('.')[1]}={v}" for n, v in FLAGS])
def flag(request):
    """Set one `execution.*` option for the test, and prove it actually took effect.

    Without the assertion this fixture is the whole matrix's blind spot: a renamed or ignored
    option would leave every case running at default settings and passing, reporting coverage
    of paths it never entered.
    """
    name, value = request.param
    attribute = name.split(".")[1]
    set_option(name, value)
    assert getattr(active_config().execution, attribute) == value, (
        f"{name} did not take effect; this matrix would have tested the default path"
    )
    yield name, value
    reset_option(name)


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_a_non_default_execution_flag_changes_no_answer(shape, flag, table):
    """Same rows, same order where ordered, same column types -- whichever path ran."""
    build, ordered = SHAPES[shape]
    name, value = flag
    with_flag = _run(build, table)  # the fixture has the option set
    reset_option(name)
    at_default = _run(build, table)
    set_option(name, value)  # restore, so the fixture's teardown resets a set option

    assert _rows(with_flag, ordered) == _rows(at_default, ordered), (
        f"{shape} returned different rows under {name}={value}"
    )
    assert [str(t) for t in with_flag.schema.types] == [str(t) for t in at_default.schema.types], (
        f"{shape} returned different column types under {name}={value}"
    )
