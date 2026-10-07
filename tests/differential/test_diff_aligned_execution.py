"""Joins over tables stored in key order, run one key range at a time, return what DuckDB does.

`dist.executors.aligned` gives each unit one key range of every table laid out in join-key
order and runs the join, and any aggregate above it, inside that unit. These tests run the
executor in process (its one Ray seam, `_run_units`, replaced by a loop) over a TPC-H-shaped
fact/order pair written in `ok` order, and hold every result to DuckDB over the same files.

The fixture is built to break the executor if its range rules are wrong:

* two adjacent files share a boundary key, so a range filter that is not half-open reads
  that key's rows twice;
* a second copy of the fact table carries NULL join keys, which fall in no range; a key
  that may be NULL is not aligned on, and the query must still come back right;
* the order table is split differently from the fact table, so units cannot be matched file
  to file;
* a customer dimension stored in *random* key order is joined on a key neither fact table is
  ordered by, so it must be read whole (broadcast) and its filter applied once.

What cannot be aligned is covered too: a join whose broadcast side is the preserved one, and
a table whose files do not follow the key.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from _harness import assert_same_for_query
from batcher import col, kyber, lit
from batcher.dist.executors.aligned import choose_plan, find_plan, key_classes
from batcher.dist.executors.aligned import run as aligned_run
from batcher.plan.logical import Join
from batcher.plan.visitor import scanned_source_ids

pytestmark = pytest.mark.differential

_LI_FILES = 6
_LI_ROWS = 2_000  # per file; keys advance by ~1 per 4 rows
_ORDER_FILES = 4
_TABLES = (
    "lineitem",
    "nullkeys",
    "orders",
    "customer",
    "shuffled",
    "segment",
    "flag",
    "custflag",
)


def _inline_units(calls, held, empties, unit_cpus, slots, depth=2):
    return aligned_run._aligned_units_task(calls, held, empties)


@pytest.fixture(autouse=True)
def _no_ray(monkeypatch):
    from batcher.dist.executors.aligned import units

    monkeypatch.setattr(aligned_run, "_run_units", _inline_units)
    # Cut this small fixture as finely as a large input, so key ranges, the boundary key and
    # the NULL keys genuinely split across units rather than sharing one of two.
    monkeypatch.setattr(units, "SMALL_UNIT_BYTES", 1)
    # And hold its broadcasts as a large input's are, rather than reading each on every node,
    # which is what a fixture this size would otherwise get (`local.LOCAL_FILTERED_BYTES`).
    from batcher.dist.executors.aligned import local

    monkeypatch.setattr(local, "LOCAL_FILTERED_BYTES", 0)


def _lineitem(part: int, nulls: bool = False) -> pa.Table:
    base = part * _LI_ROWS
    rows = range(base, base + _LI_ROWS)
    # Key = i // 4, so each file covers 500 keys; the last row of each file repeats the
    # first key of the next, which makes adjacent files share a boundary key.
    keys = [((i + 1) // 4 if (i + 1) % _LI_ROWS == 0 else i // 4) for i in rows]
    return pa.table(
        {
            "l_ok": pa.array(
                [None if nulls and i % 97 == 0 else k for i, k in zip(rows, keys, strict=True)],
                pa.int64(),
            ),
            "l_qty": pa.array([float(i % 50) for i in rows]),
            "l_price": pa.array([float(i % 997) + 0.5 for i in rows]),
            "l_flag": pa.array(["AFRN"[i % 4] for i in rows]),
        }
    )


def _orders(part: int, total: int) -> pa.Table:
    per = total // _ORDER_FILES
    ks = range(part * per, (part + 1) * per if part < _ORDER_FILES - 1 else total)
    return pa.table(
        {
            "o_ok": pa.array(list(ks), pa.int64()),
            "o_ck": pa.array([(k * 7919) % 400 for k in ks], pa.int64()),
            "o_prio": pa.array([["HIGH", "MED", "LOW"][k % 3] for k in ks]),
        }
    )


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    root = tmp_path_factory.mktemp("aligned")
    for name in _TABLES:
        (root / name).mkdir()
    for part in range(_LI_FILES):
        pq.write_table(_lineitem(part), root / "lineitem" / f"part-{part}.parquet")
        pq.write_table(_lineitem(part, nulls=True), root / "nullkeys" / f"part-{part}.parquet")
    # Keys 0..2899: a little past the fact table's last key, and missing none of its keys.
    for part in range(_ORDER_FILES):
        pq.write_table(_orders(part, 2_900), root / "orders" / f"part-{part}.parquet")
    customers = pa.table(
        {
            "c_ck": pa.array([(i * 131) % 400 for i in range(400)], pa.int64()),
            "c_seg": pa.array(["BUILDING" if i % 5 == 0 else "OTHER" for i in range(400)]),
        }
    )
    for part in range(4):
        pq.write_table(customers.slice(part * 100, 100), root / "customer" / f"p{part}.parquet")
    segments = pa.table({"g_seg": ["BUILDING", "OTHER"], "g_region": ["EAST", "WEST"]})
    pq.write_table(segments, root / "segment" / "p0.parquet")
    flags = pa.table(
        {"f_flag": ["A", "F", "N", "R"], "f_name": ["alpha", "foxtrot", "nov", "romeo"]}
    )
    pq.write_table(flags, root / "flag" / "p0.parquet")
    # Every (customer, flag) pair: joined on a fact column and a customer column at once.
    pairs = [(ck, f) for ck in range(400) for f in "AFNR"]
    custflag = pa.table({"x_ck": [p[0] for p in pairs], "x_flag": [p[1] for p in pairs]})
    pq.write_table(custflag, root / "custflag" / "p0.parquet")
    # The order table again, rows dealt round-robin: every file spans the whole key range.
    whole = pa.concat_tables([_orders(p, 2_900) for p in range(_ORDER_FILES)])
    for part in range(4):
        idx = pa.array(range(part, whole.num_rows, 4))
        pq.write_table(whole.take(idx), root / "shuffled" / f"p{part}.parquet")
    return root


def _read(root, name: str) -> bt.Dataset:
    return bt.read.parquet(str(root / name / "*.parquet"))


def _run_aligned(ds: bt.Dataset) -> pa.Table:
    opt = kyber.optimize_logical(ds._plan, sources=ds._sources)
    found = choose_plan(opt, ds._sources, strict=False)
    assert found is not None, "the plan was expected to align"
    out = aligned_run.run_plan(found, ds._sources, workers=2)
    assert out is not None, "the aligned executor declined a plan it chose"
    return out


def _duck(root, query: str):
    import duckdb

    con = duckdb.connect()
    for name in _TABLES:
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{root / name}/*.parquet')")
    return con.sql(query)


def test_the_fixture_cuts_several_units(tables):
    li, orders = _read(tables, "lineitem"), _read(tables, "orders")
    ds = li.join(orders, left_on="l_ok", right_on="o_ok").agg(n=bt.count())
    opt = kyber.optimize_logical(ds._plan, sources=ds._sources)
    cut = choose_plan(opt, ds._sources, strict=False)
    assert cut is not None
    from batcher.dist.executors.aligned.units import plan_units, source_key_bounds

    bounds = {s: source_key_bounds(ds._sources[s], cut.key.column_of(s)) for s in cut.aligned}
    units = plan_units(bounds, {}, 1 << 40, min_units=12)
    # Every fact file its own unit: the boundary key and the NULL keys are then genuinely
    # split across units, which is what the tests below rely on.
    assert units is not None and len(units) == _LI_FILES


def test_partial_aggregate_over_a_key_join(tables):
    li, orders = _read(tables, "lineitem"), _read(tables, "orders")
    ds = (
        li.join(orders, left_on="l_ok", right_on="o_ok")
        .filter(col("o_prio") != "LOW")
        .group_by("o_prio")
        .agg(rev=col("l_price").sum(), n=bt.count(), q=col("l_qty").mean())
    )
    query = (
        "SELECT o_prio, sum(l_price) AS rev, count(*) AS n, avg(l_qty) AS q FROM lineitem "
        "JOIN orders ON l_ok = o_ok WHERE o_prio <> 'LOW' GROUP BY o_prio"
    )
    assert_same_for_query(_run_aligned(ds), _duck(tables, query), query)


def test_key_grouped_aggregate_sorted_and_limited_above(tables):
    li, orders = _read(tables, "lineitem"), _read(tables, "orders")
    ds = (
        li.join(orders, left_on="l_ok", right_on="o_ok")
        .group_by("l_ok", "o_prio")
        .agg(s=col("l_qty").sum())
        .sort("s", "l_ok", descending=[True, False])
        .limit(25)
    )
    query = (
        "SELECT l_ok, o_prio, sum(l_qty) AS s FROM lineitem JOIN orders ON l_ok = o_ok "
        "GROUP BY l_ok, o_prio ORDER BY s DESC, l_ok LIMIT 25"
    )
    assert_same_for_query(_run_aligned(ds), _duck(tables, query), query)


def test_left_join_keeps_unmatched_rows(tables):
    """Fact keys 2900-2999 have no order, so the left join emits them null-padded."""
    li, orders = _read(tables, "lineitem"), _read(tables, "orders")
    ds = (
        li.join(orders, left_on="l_ok", right_on="o_ok", how="left")
        .group_by("o_prio")
        .agg(n=bt.count(), s=col("l_price").sum())
    )
    query = (
        "SELECT o_prio, count(*) AS n, sum(l_price) AS s FROM lineitem "
        "LEFT JOIN orders ON l_ok = o_ok GROUP BY o_prio"
    )
    assert_same_for_query(_run_aligned(ds), _duck(tables, query), query)


def test_semi_and_anti_join(tables):
    li, orders = _read(tables, "lineitem"), _read(tables, "orders")
    high = orders.filter(col("o_prio") == "HIGH")
    for how, word in (("semi", "EXISTS"), ("anti", "NOT EXISTS")):
        ds = (
            li.join(high, left_on="l_ok", right_on="o_ok", how=how)
            .group_by("l_flag")
            .agg(n=bt.count())
        )
        query = (
            f"SELECT l_flag, count(*) AS n FROM lineitem WHERE {word} "
            "(SELECT 1 FROM orders WHERE o_ok = l_ok AND o_prio = 'HIGH') GROUP BY l_flag"
        )
        assert_same_for_query(_run_aligned(ds), _duck(tables, query), query)


def test_filtered_dimension_is_broadcast_once(tables):
    li, orders, cust = _read(tables, "lineitem"), _read(tables, "orders"), _read(tables, "customer")
    building = cust.filter(col("c_seg") == "BUILDING")
    ds = (
        li.join(orders, left_on="l_ok", right_on="o_ok")
        .join(building, left_on="o_ck", right_on="c_ck")
        .group_by("l_ok")
        .agg(rev=col("l_price").sum())
    )
    query = (
        "SELECT l_ok, sum(l_price) AS rev FROM lineitem JOIN orders ON l_ok = o_ok "
        "JOIN customer ON o_ck = c_ck WHERE c_seg = 'BUILDING' GROUP BY l_ok"
    )
    assert_same_for_query(_run_aligned(ds), _duck(tables, query), query)


def test_a_dimension_filter_joined_above_the_spine_moves_to_its_dimension(tables):
    """`(facts JOIN customer) JOIN segment` keyed on a customer column: the segment filter
    must meet `customer` in the broadcast, not after every unit's join."""
    from batcher.dist.executors.aligned.rewrite import group_broadcast_joins

    li, orders = _read(tables, "lineitem"), _read(tables, "orders")
    cust, seg = _read(tables, "customer"), _read(tables, "segment")
    ds = (
        li.join(orders, left_on="l_ok", right_on="o_ok")
        .join(cust, left_on="o_ck", right_on="c_ck")
        .join(seg.filter(col("g_region") == "EAST"), left_on="c_seg", right_on="g_seg")
        .group_by("g_region", "l_flag")
        .agg(rev=col("l_price").sum(), n=bt.count())
    )
    # The plan as written, fact-first: at this size Kyber would join the two small tables
    # first itself, which is the shape the rewrite produces and leaves it nothing to do.
    cut = choose_plan(ds._plan, ds._sources, strict=False)
    assert cut is not None
    # Positive control: the rewrite has something to move in this plan.
    assert group_broadcast_joins(cut.cuts[0].body, cut.aligned) != cut.cuts[0].body
    query = (
        "SELECT g_region, l_flag, sum(l_price) AS rev, count(*) AS n FROM lineitem "
        "JOIN orders ON l_ok = o_ok JOIN customer ON o_ck = c_ck "
        "JOIN segment ON c_seg = g_seg WHERE g_region = 'EAST' GROUP BY g_region, l_flag"
    )
    got = aligned_run.run_plan(cut, ds._sources, workers=2)
    assert got is not None
    assert_same_for_query(got, _duck(tables, query), query)


def test_a_filter_joined_on_the_facts_key_moves_through_the_key_pair(tables):
    """TPC-H q9's shape: `(facts JOIN customer ON o_ck = c_ck) JOIN vip ON o_ck = v_ck`.

    `vip` is keyed on a *fact* column, so it has no broadcast leaf to move to by name; the
    inner join above makes `o_ck = c_ck`, so it moves onto `customer` through the pair.
    """
    from batcher.dist.executors.aligned.rewrite import group_broadcast_joins

    li, orders, cust = _read(tables, "lineitem"), _read(tables, "orders"), _read(tables, "customer")
    vip = cust.filter(col("c_seg") == "BUILDING").select(v_ck=col("c_ck"))
    ds = (
        li.join(orders, left_on="l_ok", right_on="o_ok")
        .join(cust, left_on="o_ck", right_on="c_ck")
        .join(vip, left_on="o_ck", right_on="v_ck")
        .group_by("c_seg", "l_flag")
        .agg(rev=col("l_price").sum(), n=bt.count())
    )
    cut = choose_plan(ds._plan, ds._sources, strict=False)
    assert cut is not None
    body = cut.cuts[0].body
    grouped = group_broadcast_joins(body, cut.aligned)
    # Positive control: the join moved, and `customer JOIN vip` is now one broadcast input.
    assert grouped != body
    fact_free = [
        n
        for n in _walk(grouped)
        if isinstance(n, Join)
        and n.join_type == "inner"
        and not (scanned_source_ids(n) & cut.aligned)
    ]
    assert fact_free, "customer and vip were expected to meet below the fact joins"
    query = (
        "SELECT c_seg, l_flag, sum(l_price) AS rev, count(*) AS n FROM lineitem "
        "JOIN orders ON l_ok = o_ok JOIN customer ON o_ck = c_ck "
        "JOIN (SELECT c_ck AS v_ck FROM customer WHERE c_seg = 'BUILDING') ON o_ck = v_ck "
        "GROUP BY c_seg, l_flag"
    )
    got = aligned_run.run_plan(cut, ds._sources, workers=2)
    assert got is not None
    assert_same_for_query(got, _duck(tables, query), query)


def test_a_large_broadcast_is_read_by_the_fleet_file_by_file(tables, monkeypatch):
    """A broadcast input past the driver threshold is split by file across units.

    `customer` joined to `segment` aligns on no key, so `route.scan_plan` reads it one file
    per unit, joining each against the whole of `segment`; the union must be the whole join.
    """
    from batcher.dist.executors.aligned import hoist, route

    # The threshold is read where the broadcast is evaluated (`hoist.evaluate_broadcast`);
    # `run` imports the name for its residual path, so patching it there changes nothing here.
    monkeypatch.setattr(hoist, "_NESTED_BROADCAST_BYTES", 1)
    taken = []
    real = route.scan_plan

    def spy(node, sources):
        plan = real(node, sources)
        taken.append(plan)
        return plan

    monkeypatch.setattr(route, "scan_plan", spy)
    li, orders = _read(tables, "lineitem"), _read(tables, "orders")
    cust, seg = _read(tables, "customer"), _read(tables, "segment")
    east = cust.join(seg.filter(col("g_region") == "EAST"), left_on="c_seg", right_on="g_seg")
    ds = (
        li.join(orders, left_on="l_ok", right_on="o_ok")
        .join(east, left_on="o_ck", right_on="c_ck")
        .group_by("l_flag")
        .agg(rev=col("l_price").sum(), n=bt.count())
    )
    # Grouped by a fact column only, so `reassociate_star_measures` (which would aggregate the
    # facts first and join the customers after, leaving no broadcast) has no dimension key to
    # carry and leaves the broadcast in place.
    query = (
        "SELECT l_flag, sum(l_price) AS rev, count(*) AS n FROM lineitem "
        "JOIN orders ON l_ok = o_ok JOIN customer ON o_ck = c_ck "
        "JOIN segment ON c_seg = g_seg WHERE g_region = 'EAST' GROUP BY l_flag"
    )
    assert_same_for_query(_run_aligned(ds), _duck(tables, query), query)
    # Positive control: the broadcast really went through the file-split plan.
    assert any(p is not None and p.cuts[0].key.keyless for p in taken)


def test_a_broadcast_is_cut_to_the_keys_another_broadcast_can_match(tables, monkeypatch):
    """TPC-H q5's shape: `custflag` joins on `(l_flag, o_ck)`, and `o_ck` equals the key of
    the filtered `customer` broadcast, so only its customers' rows of `custflag` can match.
    """
    from batcher.dist.executors.aligned import reduce

    sizes = []
    real = reduce.reduce_broadcasts

    def spy(body, held, local=None):
        before = {k: t.num_rows for k, t in held.items()}
        real(body, held, local)
        sizes.append((before, {k: t.num_rows for k, t in held.items()}))

    monkeypatch.setattr(aligned_run, "reduce_broadcasts", spy)
    li, orders = _read(tables, "lineitem"), _read(tables, "orders")
    cust, cf = _read(tables, "customer"), _read(tables, "custflag")
    building = cust.filter(col("c_seg") == "BUILDING")
    ds = (
        li.join(orders, left_on="l_ok", right_on="o_ok")
        .join(building, left_on="o_ck", right_on="c_ck")
        .join(cf, left_on=["l_flag", "o_ck"], right_on=["x_flag", "x_ck"])
        .group_by("l_flag")
        .agg(rev=col("l_price").sum(), n=bt.count())
    )
    query = (
        "SELECT l_flag, sum(l_price) AS rev, count(*) AS n FROM lineitem "
        "JOIN orders ON l_ok = o_ok JOIN customer ON o_ck = c_ck "
        "JOIN custflag ON l_flag = x_flag AND o_ck = x_ck "
        "WHERE c_seg = 'BUILDING' GROUP BY l_flag"
    )
    # The plan as written, fact-first: at this size Kyber joins the two small tables itself.
    cut = choose_plan(ds._plan, ds._sources, strict=False)
    assert cut is not None
    got = aligned_run.run_plan(cut, ds._sources, workers=2)
    assert got is not None
    assert_same_for_query(got, _duck(tables, query), query)
    # Positive control: a held broadcast was actually cut (1,600 custflag rows to 320).
    assert any(after[k] < before[k] for before, after in sizes for k in before)


def test_a_table_read_in_several_cuts_is_weighed_once(tables):
    """TPC-H q21's shape: `EXISTS` and `NOT EXISTS` over `lineitem` itself.

    Its membership sides cut to distinct keys, split by file, make one cut per scan of
    `lineitem`: three passes over the table where aligning the three scans makes one. The
    spread counts the table once, so that plan cannot outweigh the aligned one.
    """
    from batcher.dist.executors.aligned import route
    from batcher.dist.executors.aligned.rewrite import distinct_membership_sides
    from batcher.dist.executors.aligned.units import projected_bytes

    li = _read(tables, "lineitem")
    ds = (
        li.join(_read(tables, "lineitem").filter(col("l_flag") == "F"), on="l_ok", how="semi")
        .join(_read(tables, "lineitem").filter(col("l_qty") > 40), on="l_ok", how="anti")
        .group_by("l_flag")
        .agg(n=bt.count())
    )
    opt = kyber.optimize_logical(ds._plan, sources=ds._sources)
    membership = distinct_membership_sides(opt)
    by_file = route.find_plan(
        membership, route.KeyClass.by_file(0), frozenset({0, 1, 2}), len(ds._sources)
    )
    # Positive control: the by-file plan really is one cut per scan.
    assert by_file is not None and len(by_file.cuts) == 3
    whole = projected_bytes(ds._sources[0], None)
    assert route._weigh(by_file, ds._sources, strict=False) <= whole
    found = choose_plan(opt, ds._sources, strict=False)
    assert found is not None and len(found.cuts) == 1 and not found.cuts[0].key.keyless


def _recipes(monkeypatch) -> list:
    """Every node-read broadcast recipe the next aligned run builds, forced for any size."""
    from batcher.dist.executors.aligned import local

    monkeypatch.setattr(local, "LOCAL_BROADCAST_BYTES", 0)
    built = []
    real = aligned_run.local_broadcast

    def spy(*args, **kwargs):
        recipe = real(*args, **kwargs)
        built.append(recipe)
        return recipe

    monkeypatch.setattr(aligned_run, "local_broadcast", spy)
    return built


def test_an_unfiltered_broadcast_is_read_on_each_node(tables, monkeypatch):
    """TPC-H q10's shape at SF100: all of `customer`, which nothing filters, joined per unit.

    Each node reads and evaluates it for itself rather than receiving it from the driver.
    """
    built = _recipes(monkeypatch)
    li, orders, cust = _read(tables, "lineitem"), _read(tables, "orders"), _read(tables, "customer")
    ds = (
        li.join(orders, left_on="l_ok", right_on="o_ok")
        .join(cust, left_on="o_ck", right_on="c_ck")
        .group_by("l_flag")
        .agg(rev=col("l_price").sum(), n=bt.count(), segs=col("c_seg").count_distinct())
    )
    query = (
        "SELECT l_flag, sum(l_price) AS rev, count(*) AS n, count(DISTINCT c_seg) AS segs "
        "FROM lineitem JOIN orders ON l_ok = o_ok JOIN customer ON o_ck = c_ck GROUP BY l_flag"
    )
    found = choose_plan(ds._plan, ds._sources, strict=False)
    assert found is not None
    got = aligned_run.run_plan(found, ds._sources, workers=2)
    assert got is not None
    assert_same_for_query(got, _duck(tables, query), query)
    # Positive control: `customer` travelled as a recipe, not as rows.
    assert built, "no broadcast was read on the nodes"


def test_a_small_filtered_broadcast_is_read_on_each_node(tables, monkeypatch):
    """Under `LOCAL_FILTERED_BYTES` a filtered dimension is a recipe too: each node reads
    it alongside its first unit, rather than the driver reading it before any unit runs."""
    from batcher.dist.executors.aligned import local

    built = _recipes(monkeypatch)
    monkeypatch.setattr(local, "LOCAL_BROADCAST_BYTES", 1 << 40)
    monkeypatch.setattr(local, "LOCAL_FILTERED_BYTES", 1 << 40)
    li, orders, cust = _read(tables, "lineitem"), _read(tables, "orders"), _read(tables, "customer")
    ds = (
        li.join(orders, left_on="l_ok", right_on="o_ok")
        .join(cust.filter(col("c_seg") == "BUILDING"), left_on="o_ck", right_on="c_ck")
        .group_by("l_flag")
        .agg(rev=col("l_price").sum(), n=bt.count())
    )
    query = (
        "SELECT l_flag, sum(l_price) AS rev, count(*) AS n FROM lineitem "
        "JOIN orders ON l_ok = o_ok JOIN customer ON o_ck = c_ck "
        "WHERE c_seg = 'BUILDING' GROUP BY l_flag"
    )
    found = choose_plan(ds._plan, ds._sources, strict=False)
    assert found is not None
    got = aligned_run.run_plan(found, ds._sources, workers=2)
    assert got is not None
    assert_same_for_query(got, _duck(tables, query), query)
    # Positive control: the filtered `customer` travelled as a recipe carrying its filter.
    assert any("BUILDING" in recipe.ir for recipe in built), [r.ir[:120] for r in built]


def test_a_key_range_every_file_satisfies_does_not_count_as_a_filter(tables):
    """The range Kyber derives from a join's other side keeps every row; a real one does not."""
    from batcher.dist.executors.aligned.local import reduces

    orders = _read(tables, "orders")

    def judged(ds: bt.Dataset) -> tuple[bool, bool]:
        return reduces(ds._plan), reduces(ds._plan, ds._sources)

    whole = orders.filter((col("o_ok") >= 0) & (col("o_ok") <= 2_899)).select("o_ok", "o_ck")
    assert judged(whole) == (True, False)
    # The first file only: a range that genuinely prunes, and a predicate no footer answers.
    assert judged(orders.filter(col("o_ok") <= 700).select("o_ok")) == (True, True)
    assert judged(orders.filter(col("o_prio") == "HIGH").select("o_ok")) == (True, True)


def test_a_node_read_broadcast_is_cut_by_a_held_one(tables, monkeypatch):
    """TPC-H q9's shape at SF100: unfiltered `custflag` joined on `(l_flag, o_ck)`, where
    `o_ck` equals the key of a filtered `customer` broadcast. Read on each node, it is first
    semi-joined to that broadcast, so the units build only its matching rows.
    """
    built = _recipes(monkeypatch)
    li, orders = _read(tables, "lineitem"), _read(tables, "orders")
    cust, cf = _read(tables, "customer"), _read(tables, "custflag")
    building = cust.filter(col("c_seg") == "BUILDING")
    ds = (
        li.join(orders, left_on="l_ok", right_on="o_ok")
        .join(building, left_on="o_ck", right_on="c_ck")
        .join(cf, left_on=["l_flag", "o_ck"], right_on=["x_flag", "x_ck"])
        .group_by("l_flag")
        .agg(rev=col("l_price").sum(), n=bt.count())
    )
    query = (
        "SELECT l_flag, sum(l_price) AS rev, count(*) AS n FROM lineitem "
        "JOIN orders ON l_ok = o_ok JOIN customer ON o_ck = c_ck "
        "JOIN custflag ON l_flag = x_flag AND o_ck = x_ck "
        "WHERE c_seg = 'BUILDING' GROUP BY l_flag"
    )
    found = choose_plan(ds._plan, ds._sources, strict=False)
    assert found is not None
    got = aligned_run.run_plan(found, ds._sources, workers=2)
    assert got is not None
    assert_same_for_query(got, _duck(tables, query), query)
    # Positive control: `custflag` was read on the nodes, restricted by a semi join.
    assert any('"semi"' in recipe.ir for recipe in built), [r.ir[:200] for r in built]


def test_a_broadcast_no_node_can_hold_is_left_to_the_residual(tables, monkeypatch):
    """TPC-H q18's shape: large orders, found per key range, joined to all of `customer`.

    With `customer` too large to broadcast and unfiltered, the cut must stop below its join:
    units return the qualifying orders, and the residual joins them to `customer`.
    """
    monkeypatch.setattr(aligned_run, "BROADCAST_BYTES", 1)
    li, orders, cust = _read(tables, "lineitem"), _read(tables, "orders"), _read(tables, "customer")
    big = li.group_by("l_ok").agg(q=col("l_qty").sum()).filter(col("q") > 100)
    ds = (
        orders.join(big, left_on="o_ok", right_on="l_ok")
        .join(cust, left_on="o_ck", right_on="c_ck")
        .select("o_ok", "c_seg", "q")
    )
    query = (
        "SELECT o_ok, c_seg, q FROM orders JOIN (SELECT l_ok, sum(l_qty) AS q FROM lineitem "
        "GROUP BY l_ok HAVING sum(l_qty) > 100) ON o_ok = l_ok JOIN customer ON o_ck = c_ck"
    )
    # The plan as written, which is q18's order: at this size Kyber would join `customer` to
    # `orders` below the aggregate, and then there is no aggregate for a cut to end at.
    found = choose_plan(ds._plan, ds._sources, strict=False)
    assert found is not None
    customer_id = next(i for i, src in enumerate(ds._sources) if "c_ck" in src.schema().names)
    # Positive control: no cut broadcasts `customer`; it is read by the residual.
    assert all(customer_id not in cut.broadcast for cut in found.cuts)
    got = aligned_run.run_plan(found, ds._sources, workers=2)
    assert got is not None
    assert_same_for_query(got, _duck(tables, query), query)


@pytest.mark.parametrize("grouped", [False, True])
def test_a_dimension_between_two_aligned_tables_is_moved_out_of_their_way(
    tables, monkeypatch, grouped
):
    """TPC-H q10's shape at SF1000: `lineitem JOIN (customer JOIN orders)`, `customer` too
    large to hold on every node.

    The two key-ordered tables meet only above the dimension, so no cut can join them per
    range without broadcasting it. Moved down to `orders`, `lineitem` joins it per range and
    `customer` is left to the residual. Grouped, the facts are also pre-aggregated by the
    customer key beneath it, which is what keeps the units' results small.
    """
    from batcher.dist.executors.aligned import route

    monkeypatch.setattr(aligned_run, "BROADCAST_BYTES", 1)
    li, orders, cust = _read(tables, "lineitem"), _read(tables, "orders"), _read(tables, "customer")
    ds = li.join(
        cust.join(orders, left_on="c_ck", right_on="o_ck"), left_on="l_ok", right_on="o_ok"
    )
    if grouped:
        # A second dimension joined on top, as q10 joins `nation` to `customer`: that is what
        # keeps the plain star re-association from pre-aggregating the facts on its own.
        seg = _read(tables, "segment")
        ds = (
            ds.join(seg, left_on="c_seg", right_on="g_seg")
            .group_by("c_ck", "c_seg", "g_region")
            .agg(rev=col("l_price").sum(), n=bt.count())
        )
        query = (
            "SELECT c_ck, c_seg, g_region, sum(l_price) AS rev, count(*) AS n FROM lineitem "
            "JOIN (SELECT * FROM customer JOIN orders ON c_ck = o_ck) ON l_ok = o_ok "
            "JOIN segment ON c_seg = g_seg GROUP BY c_ck, c_seg, g_region"
        )
    else:
        ds = ds.select("l_ok", "l_price", "c_seg", "o_prio")
        query = (
            "SELECT l_ok, l_price, c_seg, o_prio FROM lineitem "
            "JOIN (SELECT * FROM customer JOIN orders ON c_ck = o_ck) ON l_ok = o_ok"
        )
    li_id, o_id = (
        next(i for i, s in enumerate(ds._sources) if c in s.schema().names)
        for c in ("l_ok", "o_ok")
    )

    def pair_aligned(found) -> bool:
        return found is not None and any({li_id, o_id} <= cut.aligned for cut in found.cuts)

    # The plan as written: at this size Kyber would join the two facts first itself.
    found = choose_plan(ds._plan, ds._sources, strict=False)
    assert pair_aligned(found)
    if grouped:
        assert any(cut.aggregate is not None for cut in found.cuts if {li_id, o_id} <= cut.aligned)
    # Positive control: without the move, the pair never shares a cut.
    monkeypatch.setattr(route, "colocate_aligned_joins", lambda plan, aligned: plan)
    assert not pair_aligned(choose_plan(ds._plan, ds._sources, strict=False))
    got = aligned_run.run_plan(found, ds._sources, workers=2)
    assert got is not None
    assert_same_for_query(got, _duck(tables, query), query)


def test_a_broadcast_too_large_to_hold_is_cut_to_the_keys_another_broadcast_admits(
    tables, monkeypatch
):
    """TPC-H q9's shape: `partsupp` joined on a fact key that green `part` already filters.

    Here `custflag` is joined on `(o_ck, l_flag)` after a filtered `customer` was joined on
    `o_ck = c_ck`, so every row reaching it carries a BUILDING customer's key. Whole, it is
    past the broadcast budget and the cut must stop below it; semi-joined to the filtered
    customers, it is a held input like any filtered one, and the aggregate stays in the cut.
    """
    from batcher.dist.executors.aligned import rewrite, units

    li, orders, cust = _read(tables, "lineitem"), _read(tables, "orders"), _read(tables, "customer")
    custflag = _read(tables, "custflag")
    building = cust.filter(col("c_seg") == "BUILDING")
    ds = (
        li.join(orders, left_on="l_ok", right_on="o_ok")
        .join(building, left_on="o_ck", right_on="c_ck")
        .join(custflag, left_on=["o_ck", "l_flag"], right_on=["x_ck", "x_flag"])
        .group_by("l_flag")
        .agg(rev=col("l_price").sum(), n=bt.count())
    )
    query = (
        "SELECT l_flag, sum(l_price) AS rev, count(*) AS n FROM lineitem "
        "JOIN orders ON l_ok = o_ok JOIN customer ON o_ck = c_ck "
        "JOIN custflag ON o_ck = x_ck AND l_flag = x_flag "
        "WHERE c_seg = 'BUILDING' GROUP BY l_flag"
    )
    cf_id = next(i for i, s in enumerate(ds._sources) if "x_ck" in s.schema().names)
    # The planner's budget just under `custflag`'s size; the run's own bound is untouched.
    monkeypatch.setattr(
        aligned_run, "BROADCAST_BYTES", units.projected_bytes(ds._sources[cf_id], None) - 1
    )

    def held_in_a_cut(found) -> bool:
        return found is not None and any(cf_id in cut.broadcast for cut in found.cuts)

    found = choose_plan(ds._plan, ds._sources, strict=False)
    assert held_in_a_cut(found)
    assert any(cut.aggregate is not None for cut in found.cuts)
    got = aligned_run.run_plan(found, ds._sources, workers=2)
    assert got is not None
    assert_same_for_query(got, _duck(tables, query), query)
    # Positive control: without the reduction, `custflag` is left to the residual.
    monkeypatch.setattr(rewrite, "_semi_reduced", lambda node, aligned, worth: None)
    assert not held_in_a_cut(choose_plan(ds._plan, ds._sources, strict=False))


@pytest.mark.parametrize("how", ["anti", "semi"])
def test_a_membership_join_to_an_unclustered_table_slices_its_keys(tables, monkeypatch, how):
    """TPC-H q22's shape: rows of a key-ordered table with (or without) a match in a table
    that is not. The other side is held, and each unit reads only the keys of its own range.
    """
    from batcher.dist.executors.aligned import reduce

    seen = []
    real = reduce.sliceable_broadcasts

    def spy(body, cut, held):
        found = real(body, cut, held)
        seen.append(found)
        return found

    monkeypatch.setattr(aligned_run, "sliceable_broadcasts", spy)
    li = _read(tables, "lineitem")
    high = _read(tables, "shuffled").filter(col("o_prio") == "HIGH").select(s_ok=col("o_ok"))
    ds = (
        li.join(high, left_on="l_ok", right_on="s_ok", how=how)
        .group_by("l_flag")
        .agg(n=bt.count(), s=col("l_price").sum())
    )
    op = "NOT EXISTS" if how == "anti" else "EXISTS"
    query = (
        f"SELECT l_flag, count(*) AS n, sum(l_price) AS s FROM lineitem WHERE {op} "
        "(SELECT 1 FROM shuffled WHERE o_prio = 'HIGH' AND o_ok = l_ok) GROUP BY l_flag"
    )
    assert_same_for_query(_run_aligned(ds), _duck(tables, query), query)
    # Positive control: the broadcast keys really were sliced to each unit's range.
    assert any(seen), "the membership side was expected to be sliced per unit"


def test_only_a_membership_side_too_large_to_hold_is_cut_to_its_keys(tables, monkeypatch):
    """Cut to distinct keys by file, every unit returns its own slice's keys and the driver
    merges them all, so a side the nodes can hold as it is is left whole."""
    from batcher.dist.executors.aligned import route
    from batcher.dist.executors.aligned.rewrite import distinct_membership_sides

    li, shuffled = _read(tables, "lineitem"), _read(tables, "shuffled")
    ds = li.join(shuffled.select(s_ok=col("o_ok")), left_on="l_ok", right_on="s_ok", how="semi")
    opt = kyber.optimize_logical(ds._plan, sources=ds._sources)

    def gated() -> object:
        return distinct_membership_sides(opt, lambda side: route.unbroadcastable(side, ds._sources))

    assert gated() is opt
    # Positive control: the same side, past the broadcast budget, is cut to its keys.
    monkeypatch.setattr(aligned_run, "BROADCAST_BYTES", 1)
    assert gated() is not opt


def test_a_file_split_cuts_units_inside_files_by_row_group(tmp_path, monkeypatch):
    """A keyless cut over files of several row groups each: units own row groups, not files,
    and a pushed predicate that prunes some of them still reads every surviving row once."""
    import pyarrow.parquet as pq

    from batcher.dist.executors.aligned import route

    for part in range(3):
        keys = list(range(part * 3_000, (part + 1) * 3_000))
        table = pa.table({"k": keys, "v": [float(k % 17) for k in keys]})
        pq.write_table(table, tmp_path / f"p{part}.parquet", row_group_size=500)
    seen = []
    real = aligned_run.plan_units
    monkeypatch.setattr(
        aligned_run, "plan_units", lambda *a, **k: seen.append(real(*a, **k)) or seen[-1]
    )
    ds = bt.read.parquet(str(tmp_path / "*.parquet")).filter(col("k") >= 1_700).select("k", "v")
    opt = kyber.optimize_logical(ds._plan, sources=ds._sources)
    found = route.scan_plan(opt, ds._sources)
    assert found is not None and found.cuts[0].key.keyless
    got = aligned_run.run_plan(found, ds._sources, workers=2)
    assert got is not None
    query = "SELECT k, v FROM rows WHERE k >= 1700"
    import duckdb

    con = duckdb.connect()
    con.execute(f"CREATE VIEW rows AS SELECT * FROM read_parquet('{tmp_path}/*.parquet')")
    assert_same_for_query(got, con.sql(query), query)
    # Positive control: three files became more than three units.
    assert seen and seen[0] is not None and len(seen[0]) > 3


def test_a_join_rejoined_with_its_own_per_key_average(tables):
    """TPC-H q17's shape: the aggregate is not key-local, so it is its own cut, and the
    driver joins it back to the per-unit join result."""
    li, flags = _read(tables, "lineitem"), _read(tables, "flag")
    joined = li.join(flags.filter(col("f_flag") != "N"), left_on="l_flag", right_on="f_flag")
    avg = joined.group_by("l_flag").agg(avg_q=col("l_qty").mean())
    ds = (
        joined.join(avg, on="l_flag")
        .filter(col("l_qty") < lit(0.8) * col("avg_q"))
        .agg(s=col("l_price").sum(), n=bt.count())
    )
    query = (
        "WITH j AS (SELECT * FROM lineitem JOIN flag ON l_flag = f_flag WHERE f_flag <> 'N'), "
        "a AS (SELECT l_flag AS k, avg(l_qty) AS avg_q FROM j GROUP BY l_flag) "
        "SELECT sum(l_price) AS s, count(*) AS n FROM j JOIN a ON l_flag = k "
        "WHERE l_qty < 0.8 * avg_q"
    )
    assert_same_for_query(_run_aligned(ds), _duck(tables, query), query)


def test_groups_compared_with_a_global_total(tables):
    """TPC-H q11's shape: a per-group sum over a key join, kept where it beats a fraction of
    the global sum of the same rows."""
    li, orders = _read(tables, "lineitem"), _read(tables, "orders")
    g = li.join(orders, left_on="l_ok", right_on="o_ok").select(
        "o_prio", v=col("l_price") * col("l_qty")
    )
    total = g.agg(t=col("v").sum() * lit(0.3))
    ds = (
        g.group_by("o_prio")
        .agg(v=col("v").sum())
        .join(total, how="cross")
        .filter(col("v") > col("t"))
        .select("o_prio", "v")
        .sort("v", descending=True)
    )
    query = (
        "WITH g AS (SELECT o_prio, l_price * l_qty AS v FROM lineitem JOIN orders ON l_ok = o_ok) "
        "SELECT o_prio, sum(v) AS v FROM g GROUP BY o_prio "
        "HAVING sum(v) > (SELECT sum(v) * 0.3 FROM g) ORDER BY v DESC"
    )
    assert_same_for_query(_run_aligned(ds), _duck(tables, query), query)


def test_the_top_group_joined_to_a_dimension(tables):
    """TPC-H q15's shape: a sum per non-key group, its maximum, and a dimension join."""
    li, orders, flags = _read(tables, "lineitem"), _read(tables, "orders"), _read(tables, "flag")
    rev = (
        li.join(orders, left_on="l_ok", right_on="o_ok")
        .filter(col("o_prio") == "HIGH")
        .group_by("l_flag")
        .agg(r=col("l_price").sum())
    )
    top = rev.join(rev.agg(m=col("r").max()), how="cross").filter(col("r") == col("m"))
    ds = flags.join(top, left_on="f_flag", right_on="l_flag").select("f_flag", "f_name", "r")
    query = (
        "WITH rev AS (SELECT l_flag, sum(l_price) AS r FROM lineitem JOIN orders "
        "ON l_ok = o_ok WHERE o_prio = 'HIGH' GROUP BY l_flag) "
        "SELECT f_flag, f_name, r FROM flag JOIN rev ON f_flag = l_flag "
        "WHERE r = (SELECT max(r) FROM rev)"
    )
    assert_same_for_query(_run_aligned(ds), _duck(tables, query), query)


def test_splits_spanning_several_files_are_still_read(tables, monkeypatch):
    """A split over several files must still be read by every unit that needs its rows.

    Split planning packs small whole files into one split. Regression: splits were matched
    to units by their `path`, which such a split does not have, so at TPC-H SF1000 every unit
    read nothing from `partsupp` and q2 returned 0 rows where the answer has 100. The pairing
    below produces that exact shape on local files, which never pack on their own.
    """
    import batcher.io.source as io_source
    from batcher.io.splits import MultiFileSplit

    real = io_source.plan_splits
    seen = []

    def paired(source, *args, **kwargs):
        planned = real(source, *args, **kwargs)
        paths = list(
            dict.fromkeys(
                p for s in planned for p in ([s.path] if getattr(s, "path", None) else [])
            )
        )
        if len(paths) < 2:
            return planned
        seen.append(len(paths))
        return [
            MultiFileSplit("parquet", tuple(paths[k : k + 2]), {}) for k in range(0, len(paths), 2)
        ]

    monkeypatch.setattr(io_source, "plan_splits", paired)
    li, orders = _read(tables, "lineitem"), _read(tables, "orders")
    ds = li.join(orders, left_on="l_ok", right_on="o_ok").group_by("o_prio").agg(n=bt.count())
    query = "SELECT o_prio, count(*) AS n FROM lineitem JOIN orders ON l_ok = o_ok GROUP BY o_prio"
    got = _run_aligned(ds)
    assert seen, "the executor never planned the multi-file splits this test is about"
    assert_same_for_query(got, _duck(tables, query), query)


def test_no_matching_rows(tables):
    li, orders = _read(tables, "lineitem"), _read(tables, "orders")
    ds = (
        li.join(orders, left_on="l_ok", right_on="o_ok")
        .filter(col("l_qty") > 1_000.0)
        .group_by("o_prio")
        .agg(n=bt.count())
    )
    query = (
        "SELECT o_prio, count(*) AS n FROM lineitem JOIN orders ON l_ok = o_ok "
        "WHERE l_qty > 1000 GROUP BY o_prio"
    )
    assert_same_for_query(_run_aligned(ds), _duck(tables, query), query)


def test_a_key_that_may_be_null_is_not_aligned_on(tables):
    """NULL keys sit in every file, so no key range owns them.

    The join is never aligned on such a key. Split by file instead (every file in exactly one
    unit, the other side read whole) it is still correct, and when that is what is chosen
    its result is held to DuckDB like any other.
    """
    li, orders = _read(tables, "nullkeys"), _read(tables, "orders")
    ds = li.join(orders, left_on="l_ok", right_on="o_ok", how="anti").agg(n=bt.count())
    plan = kyber.optimize_logical(ds._plan, sources=ds._sources)
    found = choose_plan(plan, ds._sources, strict=False)
    assert found is None or found.key.keyless
    query = (
        "SELECT count(*) AS n FROM nullkeys "
        "WHERE NOT EXISTS (SELECT 1 FROM orders WHERE o_ok = l_ok)"
    )
    got = (
        ds.collect(distributed=False)
        if found is None
        else aligned_run.run_plan(found, ds._sources, 2)
    )
    assert_same_for_query(got, _duck(tables, query), query)


def test_a_preserved_broadcast_side_is_not_aligned(tables):
    """A right join that keeps every customer row would emit its unmatched rows per unit."""
    li, orders, cust = _read(tables, "lineitem"), _read(tables, "orders"), _read(tables, "customer")
    ds = li.join(orders, left_on="l_ok", right_on="o_ok").join(
        cust, left_on="o_ck", right_on="c_ck", how="right"
    )
    plan = kyber.optimize_logical(ds._plan, sources=ds._sources)
    for key in key_classes(plan):
        aligned = frozenset(s for s, _ in key.columns if s in (0, 1))
        found = find_plan(plan, key, aligned, len(ds._sources))
        assert found is None or not any(
            getattr(n, "join_type", None) == "right" for cut in found.cuts for n in _walk(cut.body)
        )


def test_a_table_not_in_key_order_is_not_aligned(tables):
    li, shuffled = _read(tables, "lineitem"), _read(tables, "shuffled")
    ds = li.join(shuffled, left_on="l_ok", right_on="o_ok").agg(n=bt.count())
    plan = kyber.optimize_logical(ds._plan, sources=ds._sources)
    cut = choose_plan(plan, ds._sources, strict=False)
    shuffled_id = 1
    assert cut is None or shuffled_id not in cut.aligned


def _walk(node):
    from batcher.plan.visitor import walk

    return walk(node)
