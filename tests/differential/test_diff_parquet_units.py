"""A Parquet scan the engine reads itself, row group by row group, returns what DuckDB returns.

`api.orchestration.chunked` hands a plain Parquet driving source to the engine as a list of row
groups (`io.formats.structured.parquet.units`), which each worker decodes and pushes through its
pipeline (`bc_interp::stream::chunked::execute_units`). It engages above a size floor, so these
tests lower it, write files of many small row groups, and hold the result to DuckDB: inner, left,
semi and anti joins on the probe spine, global and grouped aggregates, a sort and limit above,
NULL keys, a row group every row of which is filtered away, a predicate pushed into the read, a
spine with no aggregate returned in order, and pool widths from one worker up. The sources the
path must decline — a file whose schema differs from the declared one — are checked to decline
and still answer correctly.
"""

from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from _harness import assert_same_for_query

pytestmark = pytest.mark.differential

_FILES = 4
_ROWS_PER_FILE = 3_000
_ROW_GROUP = 500

_QUERIES = [
    "SELECT o_prio, sum(l_price * (1 - l_disc)) AS rev, count(*) AS n, avg(l_qty) AS q "
    "FROM lineitem JOIN orders ON l_ok = o_ok WHERE o_prio <> 'LOW' AND l_ship < 40 "
    "GROUP BY o_prio ORDER BY rev DESC LIMIT 10",
    "SELECT count(*) AS n, sum(l_qty) AS q, min(l_price) AS lo, max(l_price) AS hi "
    "FROM lineitem WHERE l_ship BETWEEN 10 AND 20",
    "SELECT o_prio, count(o_ok) AS matched, count(*) AS n "
    "FROM lineitem LEFT JOIN orders ON l_ok = o_ok GROUP BY o_prio",
    "SELECT l_flag, count(*) AS n FROM lineitem "
    "WHERE EXISTS (SELECT 1 FROM orders WHERE o_ok = l_ok AND o_prio = 'HIGH') "
    "GROUP BY l_flag ORDER BY l_flag",
    "SELECT l_flag, count(*) AS n FROM lineitem "
    "WHERE NOT EXISTS (SELECT 1 FROM orders WHERE o_ok = l_ok) GROUP BY l_flag",
    "WITH rev AS (SELECT l_ok AS k, sum(l_price * 0.07 + l_qty * 0.013) AS r FROM lineitem "
    "WHERE l_ship < 50 GROUP BY l_ok) SELECT k, r FROM rev WHERE r = (SELECT max(r) FROM rev)",
    # The last file's rows all fail the predicate, so its row groups contribute nothing.
    "SELECT count(*) AS n, sum(l_price) AS s FROM lineitem WHERE l_ship < 3",
    # Nothing survives at all: the aggregate over no rows still owes its identity row. The
    # predicate is one no footer statistic can decide, so the query really executes.
    "SELECT count(*) AS n, sum(l_price) AS s FROM lineitem WHERE l_price - l_price > 1",
]


def _lineitem(part: int) -> pa.Table:
    base = part * _ROWS_PER_FILE
    rows = range(base, base + _ROWS_PER_FILE)
    return pa.table(
        {
            "l_ok": pa.array([None if i % 211 == 0 else i % 1_700 for i in rows], pa.int64()),
            "l_qty": pa.array([float(i % 50) for i in rows]),
            "l_price": pa.array([float(i % 997) + 0.5 for i in rows]),
            "l_disc": pa.array([(i % 4) / 4 for i in rows]),
            "l_ship": pa.array(
                [60 + i % 30 if part == _FILES - 1 else i % 90 for i in rows], pa.int64()
            ),
            "l_flag": pa.array(["AFRN"[i % 4] for i in rows]),
        }
    )


_ORDERS = pa.table(
    {
        "o_ok": pa.array(list(range(0, 1_500)), pa.int64()),
        "o_prio": pa.array(["HIGH", "MED", "LOW"][i % 3] for i in range(1_500)),
    }
)


def _whole() -> pa.Table:
    return pa.concat_tables(_lineitem(p) for p in range(_FILES))


@pytest.fixture(scope="module")
def lineitem_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("lineitem_units")
    for part in range(_FILES):
        pq.write_table(_lineitem(part), root / f"part-{part}.parquet", row_group_size=_ROW_GROUP)
    return root


@pytest.fixture
def engine_read(monkeypatch):
    """Force the engine's row-group read on these small inputs and count its calls."""
    import batcher.core as core
    from batcher.api.orchestration import chunked

    monkeypatch.setattr(chunked, "units_worthy", lambda _bytes: True)
    calls: list[int] = []
    real = core.execute_local_parquet

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(core, "execute_local_parquet", spy)
    return calls


def _session(lineitem_dir, pattern: str = "*.parquet") -> bt.Session:
    s = bt.Session()
    s.register("lineitem", bt.read.parquet(str(lineitem_dir / pattern)))
    s.register("orders", _ORDERS)
    return s


@pytest.mark.parametrize("query", _QUERIES)
def test_engine_read_matches_duckdb(duck, lineitem_dir, engine_read, query):
    duck.register("lineitem", _whole())
    duck.register("orders", _ORDERS)
    got = _session(lineitem_dir).sql(query).collect()
    assert engine_read, "the query did not take the engine's row-group read"
    assert_same_for_query(got, duck.sql(query), query)


@pytest.mark.parametrize("width", [1, 3, 0])
def test_a_spine_comes_back_in_file_order_at_every_width(lineitem_dir, engine_read, width):
    """With no aggregate or sort, the rows are the files' rows in file and row-group order.

    Compared as ordered lists on purpose: the ranges each worker reads are contiguous and
    taken in order, and an order-independent comparison could not see them come back shuffled.
    `width` 0 is the machine's own width.
    """
    from batcher.config import Config, ExecutionConfig, config_context

    cfg = Config().replace(execution=ExecutionConfig(parallelism=width))
    with config_context(cfg):
        got = (
            _session(lineitem_dir)
            .sql("SELECT l_ok, l_ship FROM lineitem JOIN orders ON l_ok = o_ok WHERE l_ship < 45")
            .collect()
        )
    assert engine_read, "the query did not take the engine's row-group read"
    whole = _whole().to_pydict()
    keys = set(_ORDERS.column("o_ok").to_pylist())
    want = [
        (k, s) for k, s in zip(whole["l_ok"], whole["l_ship"], strict=True) if k in keys and s < 45
    ]
    got_rows = zip(got.column("l_ok").to_pylist(), got.column("l_ship").to_pylist(), strict=True)
    assert list(got_rows) == want


def test_a_file_with_a_different_schema_declines_and_still_answers(duck, tmp_path, engine_read):
    """A file whose column types differ from the declared schema keeps the resident read.

    Per-file conformance (casting to the declared type) is what the row-group read skips, so a
    source that needs it must not take that path; the answer is the same either way.
    """
    for part in range(2):
        t = _lineitem(part)
        if part == 1:
            t = t.set_column(4, "l_ship", t.column("l_ship").cast(pa.int32()))
        pq.write_table(t, tmp_path / f"part-{part}.parquet", row_group_size=_ROW_GROUP)
    query = "SELECT count(*) AS n, sum(l_ship) AS s FROM lineitem WHERE l_ship < 40"
    s = bt.Session()
    s.register("lineitem", bt.read.parquet(str(tmp_path / "*.parquet"), schema_mode="union"))
    got = s.sql(query).collect()
    assert not engine_read, "a file needing conformance took the row-group read"
    duck.register("lineitem", pa.concat_tables([_lineitem(0), _lineitem(1)]))
    assert_same_for_query(got, duck.sql(query), query)


def test_control_the_same_files_with_one_schema_do_take_it(duck, tmp_path, engine_read):
    """The positive control for the decline above: identical files take the row-group read."""
    for part in range(2):
        pq.write_table(
            _lineitem(part), tmp_path / f"part-{part}.parquet", row_group_size=_ROW_GROUP
        )
    query = "SELECT count(*) AS n, sum(l_ship) AS s FROM lineitem WHERE l_ship < 40"
    s = bt.Session()
    s.register("lineitem", bt.read.parquet(str(tmp_path / "*.parquet"), schema_mode="union"))
    got = s.sql(query).collect()
    assert engine_read
    duck.register("lineitem", pa.concat_tables([_lineitem(0), _lineitem(1)]))
    assert_same_for_query(got, duck.sql(query), query)


def test_the_engine_read_still_measures_its_operators(lineitem_dir, engine_read):
    """Every driving row passes through the operators once, so their counts are the query's own.

    The learning loop records them as it records the resident executor's; `explain(analyze=True)`
    reads the same metrics, so an operator with a measured row count here is one the hub was
    handed too. The scan's count is checked against the rows the files hold.
    """
    ds = _session(lineitem_dir).sql("SELECT l_flag, count(*) AS n FROM lineitem GROUP BY l_flag")
    doc = json.loads(ds.explain(analyze=True, format="json"))
    assert engine_read, "the query did not take the engine's row-group read"
    ops = {o["kind"]: o for o in doc["ops"] if o.get("measured")}
    assert ops.get("scan", {}).get("rows_out") == _FILES * _ROWS_PER_FILE
    assert ops.get("aggregate", {}).get("rows_out") == 4


def test_the_q15_equality_holds_when_the_sideways_verdict_is_set(duck, tmp_path, engine_read):
    """TPC-H q15's main query and its scalar subquery must sum on one executor.

    The subquery's `max` is evaluated first and folded in as a literal, and the main query keeps
    the group whose sum *equals* it; two executors summing in different orders disagree in the
    last bit and keep nothing. Here the main query joins a small table to the aggregate, which
    is exactly what sets Kyber's sideways verdict, and the aggregate is over the streamed table
    itself — so the verdict must not move the main query off the row-group read. Inexact float
    factors on purpose, over enough rows to clear the verdict's size floor; the measured failure
    (TPC-H sf10 q15 returning no row) did not reproduce at this size, so the test pins that both
    evaluations take the same read rather than relying on the sums to disagree.
    """
    rows = 100_000
    for part in range(4):
        base = part * rows
        pq.write_table(
            pa.table(
                {
                    "l_k": pa.array([(base + i) % 3_000 for i in range(rows)], pa.int64()),
                    "l_p": pa.array([float((base + i) % 997) + 0.37 for i in range(rows)]),
                    "l_q": pa.array([float((base + i) % 50) for i in range(rows)]),
                }
            ),
            tmp_path / f"part-{part}.parquet",
            row_group_size=20_000,
        )
    supp = pa.table({"s_k": pa.array(range(3_000), pa.int64())})
    query = (
        "WITH rev AS (SELECT l_k AS k, sum(l_p * 0.07 + l_q * 0.013) AS r FROM lineitem "
        "GROUP BY l_k) SELECT s_k, r FROM supp JOIN rev ON s_k = k "
        "WHERE r = (SELECT max(r) FROM rev)"
    )
    s = bt.Session()
    s.register("lineitem", bt.read.parquet(str(tmp_path / "*.parquet")))
    s.register("supp", supp)
    got = s.sql(query).collect()
    # The property itself, since the last-bit disagreement is data-dependent and did not show at
    # this size: the subquery and the main query both took the row-group read.
    assert len(engine_read) == 2, f"{len(engine_read)} of 2 evaluations took the row-group read"
    assert got.num_rows >= 1, "the equality kept nothing: the two sums came from two executors"
    duck.register(
        "lineitem", pa.concat_tables(pq.read_table(p) for p in sorted(tmp_path.glob("*.parquet")))
    )
    duck.register("supp", supp)
    assert_same_for_query(got, duck.sql(query), query)
