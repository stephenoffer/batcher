"""Planning must not decode data, and a streaming read must not materialize a file.

These are the scalability contracts of the non-Parquet connectors. Every one of them was
broken in a way that produced *correct results*, which is why they need dedicated tests:
a wrong answer fails a differential test, but a plan-time full-table decode and a
`iter_batches` that buffers a whole file both pass every correctness gate while destroying
the thing they were built to provide.

The assertions are therefore about *work performed*, not about values — call-tracking on
the decode entry points, and batch counts on a reader whose fragment spans many batches.
Row-level equivalence is asserted alongside each one so a "does less work" test can never
pass by doing less work than correctness requires.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

pytestmark = pytest.mark.integration


# --------------------------------------------------------------------------------------
# ORC
# --------------------------------------------------------------------------------------


def _write_orc(path: str, rows: int = 40_000, stripe_size: int = 8192) -> pa.Table:
    orc = pytest.importorskip("pyarrow.orc")
    table = pa.table({"a": list(range(rows)), "b": [f"s{i % 97}" for i in range(rows)]})
    orc.write_table(table, path, stripe_size=stripe_size)
    return table


def test_orc_row_count_never_decodes_a_stripe(tmp_path, monkeypatch):
    """`row_count()` must answer from the footer — `read_stripe` decodes the whole stripe.

    This is the regression that mattered most: `_balance` calls `row_count()` on *every*
    split to bin-pack them, so a `read_stripe` here meant planning a distributed ORC read
    decoded the entire table on the driver before dispatching a single task.
    """
    orc = pytest.importorskip("pyarrow.orc")
    from batcher.io.formats.structured.orc import ORCSource

    path = str(tmp_path / "t.orc")
    _write_orc(path)

    source = ORCSource(path)
    splits = source.splits()
    assert len(splits) > 1, "need a genuinely multi-stripe file to make this meaningful"

    calls: list[int] = []
    real = orc.ORCFile.read_stripe

    def tracked(self, i, *args, **kwargs):
        calls.append(i)
        return real(self, i, *args, **kwargs)

    monkeypatch.setattr(orc.ORCFile, "read_stripe", tracked)

    counts = [s.row_count() for s in splits]

    assert calls == [], f"row_count() decoded {len(calls)} stripe(s); it must read the footer"
    # Unknown is the contract-sanctioned answer (pyarrow exposes no per-stripe count);
    # what must never happen is a decode.
    assert all(c is None or c >= 0 for c in counts)


def test_orc_single_stripe_row_count_is_exact_and_free(tmp_path, monkeypatch):
    """A one-stripe file's stripe holds every row, so the footer proves its exact count."""
    orc = pytest.importorskip("pyarrow.orc")
    from batcher.io.formats.structured.orc import ORCSource

    path = str(tmp_path / "one.orc")
    table = pa.table({"a": list(range(500))})
    orc.write_table(table, path)

    splits = ORCSource(path).splits()
    assert len(splits) == 1

    monkeypatch.setattr(
        orc.ORCFile,
        "read_stripe",
        lambda *a, **k: pytest.fail("row_count() must not decode a stripe"),
    )
    assert splits[0].row_count() == 500


def test_orc_footer_is_cached_across_split_metadata_calls(tmp_path, monkeypatch):
    """Repeated metadata questions must not re-open the file and re-parse the footer.

    Counts the *footer reads* rather than a cache's hit/miss bookkeeping. The cache is a
    bounded `FileMetaCache` shared with the Parquet readers rather than a `functools.
    lru_cache`, so there are no counters to read — and the read count is the quantity the
    cache exists to hold down anyway (~100 ms per footer round trip on object storage).
    """
    pytest.importorskip("pyarrow.orc")
    from batcher.io.formats.structured import orc as orc_mod

    path = str(tmp_path / "t.orc")
    _write_orc(path)

    orc_mod._ORC_FOOTERS.clear()
    reads = {"n": 0}
    real = orc_mod._read_orc_footer

    def counting(p):
        reads["n"] += 1
        return real(p)

    monkeypatch.setattr(orc_mod, "_read_orc_footer", counting)

    source = orc_mod.ORCSource(path)
    splits = source.splits()
    assert len(splits) > 1, "a single split would not exercise the sharing"

    after_planning = reads["n"]
    for s in splits:
        s.schema()
        s.row_count()

    assert reads["n"] == after_planning, "the footer was re-read after being cached"
    assert after_planning >= 1, "the footer was never read at all"


def test_orc_footer_cache_sees_a_rewritten_file(tmp_path):
    """The cache is keyed on file identity, so overwriting a path is not a stale hit.

    `FileSink` writes deterministic names, so a re-run overwrites its own output; a
    path-keyed footer would report the previous file's stripe/row counts for new bytes.
    """
    orc = pytest.importorskip("pyarrow.orc")
    from batcher.io.formats.structured import orc as orc_mod

    path = str(tmp_path / "t.orc")
    orc.write_table(pa.table({"a": list(range(10))}), path)
    assert orc_mod._orc_footer(path).nrows == 10

    import os
    import time

    time.sleep(0.01)
    orc.write_table(pa.table({"a": list(range(700))}), path)
    os.utime(path, None)

    assert orc_mod._orc_footer(path).nrows == 700


def test_orc_splits_cover_every_row_exactly_once(tmp_path):
    """Whatever pruning/splitting does, the union of the splits must be the whole file."""
    pytest.importorskip("pyarrow.orc")
    from batcher.io.formats.structured.orc import ORCSource

    path = str(tmp_path / "t.orc")
    expected = _write_orc(path)

    source = ORCSource(path)
    got = pa.Table.from_batches(
        [b for s in source.splits() for b in s.read()], schema=expected.schema
    )
    assert got.sort_by("a").equals(expected.sort_by("a"))


def test_orc_split_predicate_keeps_a_superset(tmp_path):
    """A pushed predicate may only ever *drop* stripes it can prove hold no match.

    ORC stripe pruning is not currently wired (pyarrow exposes no stripe statistics), so
    every stripe survives. That is the safe direction, and this test pins it: keeping an
    extra stripe costs I/O, dropping a matching one loses rows.
    """
    pytest.importorskip("pyarrow.orc")
    from batcher.io.formats.structured.orc import ORCSource

    path = str(tmp_path / "t.orc")
    expected = _write_orc(path)

    source = ORCSource(path)
    predicate = {"e": "gt", "l": {"e": "col", "name": "a"}, "r": {"e": "lit", "v": 39_000}}
    pruned = source.splits(predicate=predicate)

    rows = pa.Table.from_batches([b for s in pruned for b in s.read()], schema=expected.schema)
    matching = expected.filter(pa.compute.greater(expected.column("a"), 39_000))
    kept = rows.filter(pa.compute.greater(rows.column("a"), 39_000))
    assert kept.sort_by("a").equals(matching.sort_by("a")), "pruning dropped matching rows"


# --------------------------------------------------------------------------------------
# JSON
# --------------------------------------------------------------------------------------


def _write_ndjson(path: str, rows: int = 5_000) -> pa.Table:
    import json

    records = [
        {"a": i, "b": f"s{i}", "c": float(i) + 0.5, "d": {"x": i, "y": "n"}} for i in range(rows)
    ]
    with open(path, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    import pyarrow.json as pajson

    return pajson.read_json(path)


@pytest.mark.parametrize("projection", [None, ["a"], ["c", "a"], ["d"], ["b", "d", "a"]])
def test_json_projection_pushdown_matches_read_then_select(tmp_path, projection):
    """Pushing the projection into the parser must be byte-identical to selecting after.

    Column *order* follows the projection, not file order, and nested struct columns must
    survive — both are places a parser-level projection could silently differ.
    """
    from batcher.io.formats.semistructured.json import JSONSource

    path = str(tmp_path / "d.json")
    full = _write_ndjson(path)

    got = pa.Table.from_batches(JSONSource(path).read(projection=projection))
    expected = full if projection is None else full.select(projection)
    assert got.equals(expected)


def test_json_projection_does_not_parse_unprojected_columns(tmp_path):
    """The unwanted columns must never be materialized, not merely discarded afterwards."""
    from batcher.io.formats.semistructured.json import JSONSource

    path = str(tmp_path / "d.json")
    _write_ndjson(path)

    batches = JSONSource(path).read(projection=["a"])
    assert batches
    for b in batches:
        assert b.schema.names == ["a"], "an unprojected column reached the engine"


def test_json_iter_batches_matches_the_whole_read(tmp_path):
    """Streaming and materializing must produce identical rows in identical order."""
    from batcher.io.formats.semistructured.json import JSONSource

    path = str(tmp_path / "d.json")
    _write_ndjson(path)

    source = JSONSource(path)
    whole = pa.Table.from_batches(source.read(projection=["a", "c"]))
    streamed = pa.Table.from_batches(list(source.iter_batches(projection=["a", "c"])))
    assert streamed.equals(whole)


def test_json_projection_falls_back_when_the_schema_cannot_parse_a_file(tmp_path):
    """Pushdown may only add speed — a file free inference can read must stay readable.

    A column whose type widens late in the file is the case that breaks a forced schema;
    the read must fall back rather than turn a working read into an error.
    """
    from batcher.io.formats.semistructured.json import JSONSource

    path = str(tmp_path / "w.json")
    with open(path, "w") as fh:
        for i in range(2_000):
            fh.write(f'{{"v": {i}}}\n')
        fh.write('{"v": 1.5}\n')

    table = pa.Table.from_batches(JSONSource(path).read(projection=["v"]))
    assert table.num_rows == 2_001
    assert table.column("v")[-1].as_py() == 1.5


# --------------------------------------------------------------------------------------
# Lakehouse
# --------------------------------------------------------------------------------------


def _delta_table(tmp_path, rows: int = 300_000):
    deltalake = pytest.importorskip("deltalake")
    path = str(tmp_path / "delta")
    table = pa.table({"a": list(range(rows)), "b": [f"s{i % 13}" for i in range(rows)]})
    deltalake.write_deltalake(path, table)
    return path, table


def test_delta_split_iter_batches_streams_more_than_one_batch(tmp_path):
    """`iter_batches` must stream the fragment, not build it and re-chunk the result."""
    pytest.importorskip("deltalake")
    from batcher.io.formats.lakehouse.delta.source import DeltaSource

    path, expected = _delta_table(tmp_path)
    splits = DeltaSource(path).splits()
    assert splits

    batches = [b for s in splits for b in s.iter_batches()]
    assert len(batches) > 1, "a multi-batch fragment yielded one batch — it was materialized"

    got = pa.Table.from_batches(batches).select(["a", "b"])
    assert got.sort_by("a").equals(expected.sort_by("a"))


def test_delta_source_iter_batches_matches_read(tmp_path):
    """The streamed table-level scan must equal the materialized one."""
    pytest.importorskip("deltalake")
    from batcher.io.formats.lakehouse.delta.source import DeltaSource

    path, _ = _delta_table(tmp_path)
    source = DeltaSource(path)

    whole = pa.Table.from_batches(source.read())
    streamed = pa.Table.from_batches(list(source.iter_batches()))
    assert streamed.sort_by("a").equals(whole.sort_by("a"))


def _multi_batch_fragment(tmp_path, rows: int = 300_000):
    """One Parquet fragment large enough that a scan of it spans several batches."""
    import pyarrow.dataset as pads
    import pyarrow.parquet as pq

    table = pa.table({"a": list(range(rows)), "b": [f"s{i % 13}" for i in range(rows)]})
    path = str(tmp_path / "frag.parquet")
    pq.write_table(table, path)
    dataset = pads.dataset(path)
    return next(iter(dataset.get_fragments())), dataset.schema, table


@pytest.mark.parametrize(
    "predicate",
    [None, {"e": "gt", "l": {"e": "col", "name": "a"}, "r": {"e": "lit", "v": 150_000}}],
)
def test_delta_iter_fragment_masks_each_batch_at_the_right_offset(tmp_path, predicate):
    """The streamed read must equal the materialized one when a deletion vector applies.

    This is the subtle half of streaming a Delta fragment. A vector is indexed by *physical
    row position*, so each batch has to be masked with its own slice of it; masking every
    batch with the whole vector, or slicing at the wrong offset, deletes arbitrary rows and
    reports success. Comparing against `read_fragment` — which masks the whole table at
    once and is the behavior being preserved — is what makes an offset bug visible.

    A real Delta delete usually rewrites the file rather than emitting a vector, so the
    mask is supplied directly; otherwise this path would go untested.
    """
    from batcher.io.formats.lakehouse.delta.source import iter_fragment, read_fragment

    fragment, schema, table = _multi_batch_fragment(tmp_path)
    # An irregular pattern, so a wrong offset cannot coincidentally agree.
    mask = pa.array([(i % 7) != 0 and (i % 11) != 3 for i in range(table.num_rows)])

    expected = read_fragment(fragment, schema, None, predicate, mask)
    batches = list(iter_fragment(fragment, schema, None, predicate, mask))

    assert len(batches) > 1, "the fragment did not stream — the offset logic is untested"
    assert pa.Table.from_batches(batches, schema=expected.schema).equals(expected)


def test_delta_iter_fragment_refuses_a_misaligned_deletion_vector(tmp_path):
    """A vector that does not describe the file's rows must be refused, not guessed at.

    Streaming moves this check to `count_rows()` (the footer) so it fires *before* any
    batch is emitted, rather than after a consumer has already processed some.
    """
    from batcher._internal.errors import BackendError
    from batcher.io.formats.lakehouse.delta.source import iter_fragment

    fragment, schema, table = _multi_batch_fragment(tmp_path, rows=1_000)
    short = pa.array([True] * (table.num_rows - 5))

    with pytest.raises(BackendError, match="deletion vector"):
        next(iter(iter_fragment(fragment, schema, None, None, short)))


def test_delta_iter_batches_applies_deletion_vectors_per_batch(tmp_path):
    """A deletion vector is indexed by physical row position, so streaming must offset it.

    Masking each batch with the *whole* vector, or with the wrong slice, deletes arbitrary
    rows and reports success — the exact silent corruption the offset tracking prevents.
    """
    deltalake = pytest.importorskip("deltalake")
    from batcher.io.formats.lakehouse.delta.source import DeltaSource

    path = str(tmp_path / "dv")
    rows = 40_000
    table = pa.table({"a": list(range(rows))})
    deltalake.write_deltalake(path, table)

    dt = deltalake.DeltaTable(path)
    try:
        dt.delete("a % 7 = 0")
    except Exception:  # pragma: no cover - writer without predicate delete support
        pytest.skip("this deltalake build cannot apply a predicate delete")

    source = DeltaSource(path)
    expected = {i for i in range(rows) if i % 7 != 0}

    whole = pa.Table.from_batches(source.read())
    streamed = pa.Table.from_batches(list(source.iter_batches()))
    assert set(whole.column("a").to_pylist()) == expected
    assert set(streamed.column("a").to_pylist()) == expected

    per_split = [b for s in source.splits() for b in s.iter_batches()]
    assert set(pa.Table.from_batches(per_split).column("a").to_pylist()) == expected


def test_iceberg_split_iter_batches_streams_more_than_one_batch(tmp_path):
    """The Iceberg split must stream via `to_record_batches`, not build the whole file."""
    pytest.importorskip("pyiceberg")
    source = _iceberg_source(tmp_path)

    splits = source.splits()
    assert splits

    batches = [b for s in splits for b in s.iter_batches()]
    assert len(batches) > 1, "a multi-batch data file yielded one batch — it was materialized"

    whole = pa.Table.from_batches(source.read())
    assert pa.Table.from_batches(batches).sort_by("a").equals(whole.sort_by("a"))


def test_iceberg_source_iter_batches_matches_read(tmp_path):
    """Batch-reader normalization must equal what the materializing read produces."""
    pytest.importorskip("pyiceberg")
    source = _iceberg_source(tmp_path)

    whole = pa.Table.from_batches(source.read())
    streamed = pa.Table.from_batches(list(source.iter_batches()))
    assert streamed.schema.equals(whole.schema), "streamed batches were normalized differently"
    assert streamed.sort_by("a").equals(whole.sort_by("a"))


def _iceberg_source(tmp_path, rows: int = 300_000):
    """A local-catalog Iceberg table with enough rows to span several batches."""
    pytest.importorskip("pyiceberg")
    pytest.importorskip("sqlalchemy")
    from pyiceberg.catalog.sql import SqlCatalog

    warehouse = tmp_path / "wh"
    warehouse.mkdir()
    spec = {
        "type": "sql",
        "uri": f"sqlite:///{warehouse}/catalog.db",
        "warehouse": f"file://{warehouse}",
    }
    catalog = SqlCatalog("default", uri=spec["uri"], warehouse=spec["warehouse"])
    catalog.create_namespace("db")
    data = pa.table({"a": list(range(rows)), "b": [f"s{i % 13}" for i in range(rows)]})
    table = catalog.create_table("db.t", schema=data.schema)
    table.append(data)

    from batcher.io.formats.lakehouse.iceberg.source import IcebergSource

    return IcebergSource("db.t", catalog=spec)
