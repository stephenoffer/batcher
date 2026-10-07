"""The "What runs when" table in `docs/getting-started/concepts/lazy.md`, executed.

Each label there is a claim about whether a call runs the plan. A `map_batches` callable that
counts the rows it sees is the probe: a plan-building call must leave the count at zero, and
an executing one must move it. The page's two caveats are pinned too: `schema` and an
`INSERT` into a session table probe an opaque callable to learn its output types -- a batch
callback on an empty batch, a per-row one on one row -- and not at all once the output
schema is declared.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

import batcher as bt
from batcher.ml import StandardScaler

pytestmark = pytest.mark.integration


class _Probe:
    """A `map_batches` callable that counts the calls it gets and the rows it is handed."""

    def __init__(self) -> None:
        self.rows = 0
        self.calls = 0

    def __call__(self, batch: Any) -> Any:
        self.calls += 1
        self.rows += batch.num_rows
        return batch


def _rows_seen(call: Callable[[bt.Dataset, bt.Session], object]) -> int:
    probe = _Probe()
    ds = bt.from_pydict({"x": [1.0, 2.0, 3.0]}).map_batches(probe)
    session = bt.Session()
    session.register("t", ds)
    probe.rows = 0  # registration is itself under test below; start every call from zero
    call(ds, session)
    return probe.rows


_PLAN_BUILDING = {
    "filter_select": lambda ds, s: ds.filter(bt.col("x") > 1).select("x"),
    "cache": lambda ds, s: ds.cache(),
    "columns": lambda ds, s: ds.columns,
    "explain": lambda ds, s: ds.explain(),
    "register": lambda ds, s: bt.Session().register("u", ds),
    "sql_select": lambda ds, s: s.sql("SELECT x FROM t WHERE x > 1"),
    "sql_create_view": lambda ds, s: s.sql("CREATE VIEW v AS SELECT x FROM t"),
    "sql_ctas_session": lambda ds, s: s.sql("CREATE TABLE u AS SELECT x FROM t"),
    "iter_batches_unconsumed": lambda ds, s: ds.iter_batches(),
}

_EXECUTING = {
    "collect": lambda ds, s: ds.collect(),
    "to_pandas": lambda ds, s: ds.to_pandas(),
    "count": lambda ds, s: ds.count(),
    "len": lambda ds, s: len(ds),
    "shape": lambda ds, s: ds.shape,
    "explain_analyze": lambda ds, s: ds.explain(analyze=True),
    "fit": lambda ds, s: StandardScaler(["x"]).fit(ds),
    "iter_batches_first_next": lambda ds, s: next(ds.iter_batches()),
}

# The caveats: an opaque callable's output types are only known by asking it, so these probe
# each undeclared callback stage -- a batch callback on an empty batch, a row one on one row.
_PROBING = {
    "schema_over_a_callable": lambda ds, s: ds.schema,
    "sql_insert_over_a_callable": lambda ds, s: s.sql("INSERT INTO t SELECT x FROM t"),
}


@pytest.mark.parametrize("name", sorted(_PLAN_BUILDING))
def test_plan_building_runs_nothing(name: str) -> None:
    assert _rows_seen(_PLAN_BUILDING[name]) == 0


@pytest.mark.parametrize("name", sorted(_EXECUTING))
def test_executing_runs_the_plan(name: str) -> None:
    assert _rows_seen(_EXECUTING[name]) > 0


def _probed(call: Callable[[bt.Dataset, bt.Session], object], **stage: Any) -> _Probe:
    probe = _Probe()
    ds = bt.from_pydict({"x": [1.0, 2.0, 3.0]}).map_batches(probe, **stage)
    session = bt.Session()
    session.register("t", ds)
    probe.rows = probe.calls = 0
    call(ds, session)
    return probe


@pytest.mark.parametrize("name", sorted(_PROBING))
def test_a_type_probe_hands_a_batch_callback_an_empty_batch(name: str) -> None:
    probe = _probed(_PROBING[name])
    assert probe.calls >= 1  # it is asked: its types are only known by asking
    assert probe.rows == 0


@pytest.mark.parametrize("name", sorted(_PROBING))
def test_a_declared_output_schema_needs_no_probe(name: str) -> None:
    probe = _probed(_PROBING[name], output_columns=pa.schema([pa.field("x", pa.float64())]))
    assert (probe.calls, probe.rows) == (0, 0)


@pytest.mark.parametrize("name", sorted(_PROBING))
def test_a_type_probe_hands_a_row_callback_one_row(name: str) -> None:
    seen: list[float] = []

    def row(r: dict[str, Any]) -> dict[str, Any]:
        seen.append(r["x"])
        return r

    ds = bt.from_pydict({"x": [1.0, 2.0, 3.0]}).map(row)
    session = bt.Session()
    session.register("t", ds)
    seen.clear()
    _PROBING[name](ds, session)
    assert len(seen) == 1


def test_a_write_runs_the_plan(tmp_path: Path) -> None:
    assert _rows_seen(lambda ds, s: ds.write.parquet(str(tmp_path / "out.parquet"))) > 0


def test_a_reader_reads_the_schema_at_construction(tmp_path: Path) -> None:
    """`bt.read.*` touches storage when called: a file that is not Parquet fails right away."""
    bad = tmp_path / "bad.parquet"
    bad.write_text("not parquet")
    with pytest.raises(Exception, match="parquet"):
        bt.read.parquet(str(bad))
