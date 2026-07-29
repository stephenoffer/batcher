"""`output_columns` is a promise to the optimizer, so it has to be checked.

`MapBatches.available_columns()` returns the declaration verbatim, which means every
operator above the stage plans against names the engine has never seen produced. A stale or
typo'd declaration therefore failed somewhere else entirely, or resolved a column that
existed only in the plan. These pin that the mismatch is reported at the stage that made it.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError


@pytest.fixture
def ds() -> bt.Dataset:
    return bt.from_pydict({"x": [1, 2, 3, 4]})


def test_declared_but_not_returned(ds: bt.Dataset) -> None:
    with pytest.raises(PlanError, match=r"declared but not returned: \['z'\]"):
        ds.map_batches(lambda b: {"y": [1, 2, 3, 4]}, output_columns=["y", "z"]).to_pydict()


def test_returned_but_not_declared(ds: bt.Dataset) -> None:
    with pytest.raises(PlanError, match=r"returned but not declared: \['w'\]"):
        ds.map_batches(
            lambda b: {"y": [1, 2, 3, 4], "w": [1, 2, 3, 4]}, output_columns=["y"]
        ).to_pydict()


def test_matching_declaration_passes(ds: bt.Dataset) -> None:
    assert ds.map_batches(lambda b: {"y": [1, 2, 3, 4]}, output_columns=["y"]).to_pydict() == {
        "y": [1, 2, 3, 4]
    }


def test_declaration_order_is_not_enforced(ds: bt.Dataset) -> None:
    """The plan relies on the *set* of names; failing on order would be a false alarm."""
    out = ds.map_batches(
        lambda b: {"b": [1, 2, 3, 4], "a": [5, 6, 7, 8]}, output_columns=["a", "b"]
    ).to_pydict()
    assert set(out) == {"a", "b"}


def test_no_declaration_is_unchecked(ds: bt.Dataset) -> None:
    assert ds.map_batches(lambda b: {"y": [1, 2, 3, 4]}).to_pydict() == {"y": [1, 2, 3, 4]}


def test_an_all_empty_result_is_not_evidence() -> None:
    """A filtered-to-zero upstream must not fail here: there is nothing to check against."""
    empty = bt.from_pydict({"x": [1]}).filter(bt.col("x") > 99)
    assert empty.map_batches(lambda b: b, output_columns=["x"]).to_pydict() == {"x": []}


def test_input_columns_excuses_a_pruned_pass_through() -> None:
    """Declaring `input_columns` is what lets pushdown prune the pass-through columns.

    The user guide's own `input_columns` example declares every input column in
    `output_columns` and then has them pruned away, so treating that as a mismatch would
    reject the idiom the guide recommends.
    """
    wide = bt.from_pydict({"a": [1, 2], "b": ["x", "y"], "price": [10.0, 30.0]})
    cheap = wide.map_batches(
        lambda b: b.append_column(
            "cheap", pa.array([v < 25.0 for v in b.column("price").to_pylist()])
        ),
        input_columns=["price"],
        output_columns=["a", "b", "price", "cheap"],
    )
    assert cheap.select("price", "cheap").to_pydict()["cheap"] == [True, False]


def test_input_columns_does_not_excuse_an_extra_column() -> None:
    """Pruning can only remove columns, so an undeclared *extra* is still a real mismatch."""
    wide = bt.from_pydict({"a": [1, 2], "price": [10.0, 30.0]})
    with pytest.raises(PlanError, match="returned but not declared"):
        wide.map_batches(
            lambda b: b.append_column("surprise", pa.array([1, 2])),
            input_columns=["price"],
            output_columns=["price"],
        ).to_pydict()


def _mismatched(dataset: bt.Dataset, **options) -> bt.Dataset:
    return dataset.map_batches(
        lambda b: {"y": [1] * b.num_rows}, output_columns=["y", "missing"], **options
    )


async def _async_fn(batch: pa.RecordBatch) -> dict:
    return {"y": [1] * batch.num_rows}


@pytest.mark.parametrize(
    ("name", "build"),
    [
        ("chained", lambda d: _mismatched(d.map_batches(lambda b: b))),
        ("batch_size", lambda d: _mismatched(d, batch_size=8)),
        ("threads", lambda d: _mismatched(d, num_workers=4)),
        ("error_budget", lambda d: _mismatched(d, max_errored_rows=5)),
        (
            "async",
            lambda d: d.map_batches(_async_fn, output_columns=["y", "missing"]),
        ),
    ],
)
@pytest.mark.parametrize("terminal", ["collect", "iter_batches"])
def test_the_check_fires_on_every_dispatch_path(name: str, build, terminal: str) -> None:
    """A guard that only covers the default path is the failure mode CLAUDE.md names.

    `map_batches` reaches the user's function through five routes — the materializing path,
    the streaming linear chain, the thread pool, the error-budget bisection, and the async
    event loop — and each of those crossed with each terminal is where a check silently
    stops applying.
    """
    plan = build(bt.from_pydict({"x": list(range(100))}))
    with pytest.raises(PlanError, match="output_columns"):
        plan.to_pydict() if terminal == "collect" else list(plan.iter_batches())


def test_the_check_survives_rebatching(ds: bt.Dataset) -> None:
    """The declaration is checked once per stage, so an explicit batch_size changes nothing."""
    with pytest.raises(PlanError, match="output_columns"):
        ds.map_batches(
            lambda b: pa.record_batch({"y": b.column("x")}),
            output_columns=["y", "missing"],
            batch_size=2,
        ).to_pydict()
