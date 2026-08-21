"""An operator *above* a `map_batches` stage must survive the distributed re-plan.

`map_batches(...)` alone distributed fine; adding `.select(...)` or `.filter(...)` above it
raised ``ColumnNotFoundError: ... available: []`` while the identical pipeline worked
single-node. That is the ordinary shape of batch inference — score, then narrow — so the
`ds.ml.predict` / `generate` / `embed` families all hit it, and SQL's `AI_GENERATE` hit it on
every query, since ``SELECT id, response FROM ...`` is a projection above the stage.

The cause was in `dist.executors.plan_analysis`. Splitting a plan into resource stages rebuilt
each later stage onto a boundary `Scan` carrying an empty schema, using `dataclasses.replace`
— which re-runs `__post_init__`, so the node re-validated its column references against that
empty schema and failed. The boundary scan stands in for the upstream stage's published
morsels rather than describing them, and a `MapBatches` cannot describe its output types at
all (`available_schema()` is `None` through an opaque `fn`), so the fix is to re-parent without
re-validating rather than to invent a schema.

CI installs no Ray, so nothing here runs in the PR gate — which is exactly why the defect
survived: every single-node test of the same pipelines passed throughout.
"""

from __future__ import annotations

import pytest

import batcher as bt

pytest.importorskip("ray", reason="ray not installed")
pytest.importorskip("batcher._native", reason="native engine not built")


from _ray_cluster import init_test_ray, shutdown_test_ray


@pytest.fixture(scope="module", autouse=True)
def _ray_session():
    started = init_test_ray(2)
    yield
    shutdown_test_ray(started)


SOURCE = {"id": [1, 2, 3, 4], "body": ["a", "b", "c", "d"]}


def _upper():
    """A closure, not a class or a module-level function, and that is load-bearing.

    Ray pickles a callable defined in an importable module *by reference*, so a worker then
    re-imports it by qualified name — and the tests directory is not on a worker's path, so
    every actor dies with `ModuleNotFoundError` before the plan under test is ever exercised.
    A closure is pickled by value instead. `concurrency=1` is what makes the stage a pool
    stage, which a class `fn` would otherwise have done.
    """
    import pyarrow as pa

    def run(batch):
        values = [s.upper() for s in batch.column("body").to_pylist()]
        return batch.append_column("resp", pa.array(values, pa.string()))

    return run


def _scored(declared: list[str] | None) -> bt.Dataset:
    return bt.from_pydict(SOURCE).ml.map_batches(_upper(), output_columns=declared, concurrency=1)


def _rows(result) -> list[tuple]:
    data = result.to_pydict()
    return sorted(zip(*data.values(), strict=True))


#: `output_columns` omitted means "the input columns pass through", so a *declared* stage is
#: the only one that may select the column the `fn` adds. Both still take a filter on a column
#: that existed all along, and both broke on that.
DECLARED = ["id", "body", "resp"]
DECLARATIONS = [pytest.param(DECLARED, id="declared"), pytest.param(None, id="undeclared")]


def test_a_select_above_map_batches_survives_distribution() -> None:
    out = _scored(DECLARED).select("id", "resp").collect(distributed=True)
    assert _rows(out) == [(1, "A"), (2, "B"), (3, "C"), (4, "D")]


@pytest.mark.parametrize("declared", DECLARATIONS)
def test_a_filter_above_map_batches_survives_distribution(declared) -> None:
    out = _scored(declared).filter(bt.col("id") > 2).collect(distributed=True)
    assert [row[0] for row in _rows(out)] == [3, 4]


def test_distributed_equals_single_node_for_filter_then_select() -> None:
    """The invariant proper, on the shape that used to raise."""

    def shape(ds):
        return ds.filter(bt.col("id") > 1).select("id", "resp")

    single = shape(_scored(DECLARED)).collect()
    distributed = shape(_scored(DECLARED)).collect(distributed=True)
    assert _rows(single) == _rows(distributed)
    assert _rows(distributed) == [(2, "B"), (3, "C"), (4, "D")]


def test_an_undeclared_stage_still_refuses_the_column_it_never_declared() -> None:
    """Not a casualty of the fix: this rejection is the `output_columns` contract working.

    Skipping `__post_init__` at a stage boundary must not become a way to smuggle an
    unvalidated plan past the check that runs when the user builds it.
    """
    from batcher._internal.errors import ColumnNotFoundError

    with pytest.raises(ColumnNotFoundError, match="resp"):
        _scored(None).select("id", "resp")


def test_map_batches_alone_still_distributes() -> None:
    """The case that always worked, kept so a fix cannot regress it."""
    out = _scored(["id", "body", "resp"]).collect(distributed=True)
    assert sorted(out.to_pydict()["resp"]) == ["A", "B", "C", "D"]


def test_sql_ai_generate_distributes() -> None:
    """`SELECT <cols> FROM AI_GENERATE(...)` is a projection above the stage, so it hit this."""
    session = bt.Session()
    session.register("t", bt.from_pydict(SOURCE))
    session.register_engine("shouty", lambda: lambda prompts: [p.upper() for p in prompts])
    query = "SELECT id, response FROM AI_GENERATE(t, shouty, prompt_column => 'body')"
    assert _rows(session.sql(query).collect(distributed=True)) == [
        (1, "A"),
        (2, "B"),
        (3, "C"),
        (4, "D"),
    ]
