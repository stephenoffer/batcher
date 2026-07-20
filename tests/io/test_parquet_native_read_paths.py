"""The Parquet reader's fast paths must be indistinguishable from its slow ones.

`ParquetSource` now routes three cases to the native Rust reader that previously went to
pyarrow: a *filtered* read (`_native_read_filtered`), a *streaming* read (`_iter_file` over
row-group windows), and a remote read's concurrency sizing. Each has a pyarrow fallback that
triggers silently, so nothing in a query's result reveals which one ran — which is precisely
why the equivalence has to be asserted rather than assumed.

Three properties are load-bearing here and are what these tests pin:

* **Superset-safety is the contract.** A pushed-down read may return more rows than match,
  never fewer — the engine's `Filter` re-checks them. A test that only compared row *counts*
  would pass on a reader that silently dropped a matching row, so these compare the rows
  themselves against the unfiltered-then-filtered answer.
* **...but the superset must stay small.** Pruning skips row-groups the statistics rule out,
  which does nothing when the matches are scattered across all of them — returning the whole
  file, legally, and OOMing a large scan. So the native path also *filters*, and
  `test_a_scattered_predicate_does_not_hand_the_driver_the_whole_file` is what keeps it
  honest on the layout where pruning cannot help.
* **Read-ahead changes throughput, not results.** Depth is a scheduling knob; every depth
  must yield the same batches in the same file order. It also decides *which reader* streams
  each file, so the tests cover both sides of that rule.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from batcher.io.formats.structured import _parquet_native
from batcher.io.formats.structured.parquet import _native_stream
from batcher.io.formats.structured.parquet._native_stream import row_group_windows
from batcher.io.formats.structured.parquet.source import ParquetSource

pytestmark = pytest.mark.integration


def _write(tmp_path, n_files: int = 3, rows: int = 5_000, row_group_size: int = 500):
    """`n_files` parquet files with many small row-groups, so pruning has something to do."""
    rng = np.random.default_rng(7)
    paths = []
    for f in range(n_files):
        table = pa.table(
            {
                "k": np.arange(f * rows, (f + 1) * rows, dtype="int64"),
                "bucket": rng.integers(0, 10, rows).astype("int64"),
                "val": rng.random(rows),
                "name": pa.array([f"n{i % 97}" for i in range(rows)]),
            }
        )
        p = tmp_path / f"part-{f:03d}.parquet"
        pq.write_table(table, p, row_group_size=row_group_size)
        paths.append(str(p))
    return str(tmp_path), paths


def _pred(col: str, op: str, lit: dict) -> dict:
    return {
        "e": "binary",
        "op": op,
        "left": {"e": "col", "name": col},
        "right": {"e": "lit", "value": lit},
    }


def _pathonly_source(path: str) -> ParquetSource:
    """A `ParquetSource` carrying only its path — no filesystem resolution, no credentials.

    `_is_remote` / `_read_concurrency` are decided from the path alone, and constructing a
    real `s3://` source would try to resolve a backend this test has no business needing.
    """
    src = ParquetSource.__new__(ParquetSource)
    src._path = path
    return src


def _rows(batches) -> list[dict]:
    """Batches as a sorted list of row dicts — order-independent, value-exact."""
    if not batches:
        return []
    table = pa.Table.from_batches(batches)
    return sorted(table.to_pylist(), key=lambda r: r["k"])


def _apply(rows: list[dict], keep) -> list[dict]:
    return [r for r in rows if keep(r)]


# --- 1. a filtered read == the unfiltered read, then filtered ----------------------


@pytest.mark.parametrize(
    ("predicate", "keep"),
    [
        (_pred("k", "ge", {"int": 7000}), lambda r: r["k"] >= 7000),
        (_pred("bucket", "eq", {"int": 3}), lambda r: r["bucket"] == 3),
        (_pred("name", "eq", {"str": "n5"}), lambda r: r["name"] == "n5"),
        (_pred("val", "lt", {"float": 0.25}), lambda r: r["val"] < 0.25),
        # A predicate that matches nothing: pruning should be able to drop every
        # row-group, and an empty result must still be a *correct* empty result.
        (_pred("k", "gt", {"int": 10**9}), lambda r: r["k"] > 10**9),
        # ...and one that matches everything, where pruning must drop nothing.
        (_pred("k", "ge", {"int": -1}), lambda r: r["k"] >= -1),
        (
            {
                "e": "binary",
                "op": "and",
                "left": _pred("bucket", "eq", {"int": 3}),
                "right": _pred("k", "lt", {"int": 6000}),
            },
            lambda r: r["bucket"] == 3 and r["k"] < 6000,
        ),
        ({"e": "is_not_null", "input": {"e": "col", "name": "bucket"}}, lambda r: True),
    ],
)
def test_filtered_read_is_a_superset_containing_exactly_the_matching_rows(
    tmp_path, predicate, keep
):
    """The engine keeps its `Filter`, so the contract is: pushed-read ⊇ answer, and
    filtering the pushed read afterwards reproduces the answer exactly."""
    path, _ = _write(tmp_path)
    src = ParquetSource(path)

    baseline = _rows(src.read())
    expected = _apply(baseline, keep)

    pushed = _rows(src.read(predicate=predicate))

    # Superset: nothing that matches was pruned away.
    assert _apply(pushed, keep) == expected
    # ...and no row was invented, duplicated, or corrupted on the way through.
    assert len(pushed) <= len(baseline)
    assert all(row in baseline for row in pushed)


def test_a_scattered_predicate_does_not_hand_the_driver_the_whole_file(tmp_path):
    """Pruning alone is contract-legal but can return everything, and that is an OOM.

    Row-group pruning only helps when the matching rows are *clustered*. With a selective
    predicate whose matches land in every row-group, the statistics rule nothing out and a
    prune-only reader returns the entire file — measured at 20,000,000 rows / 320 MB where
    pyarrow's `filters=` returned 199,575 / 3.2 MB on the same input. The engine's `Filter`
    still gives the right answer, which is exactly why nothing else catches this. So the
    native path filters the pruned batches too, and this pins that it does.
    """
    rng = np.random.default_rng(3)
    rows = 60_000
    # Uniform random key: every row-group's [min, max] spans the predicate, so no row-group
    # and no page can be excluded. This is the adversarial layout, not a contrived one — it
    # is what any unsorted natural key looks like.
    pq.write_table(
        pa.table({"k": rng.integers(0, 1_000_000, rows).astype("int64"), "v": rng.random(rows)}),
        tmp_path / "scattered.parquet",
        row_group_size=1_000,
    )
    src = ParquetSource(str(tmp_path))
    predicate = _pred("k", "lt", {"int": 10_000})
    expected = [r for r in pa.Table.from_batches(src.read()).to_pylist() if r["k"] < 10_000]

    native = src._native_read_filtered(None, predicate)
    assert native is not None
    got = pa.Table.from_batches(native).to_pylist() if native else []

    assert sorted(r["k"] for r in got) == sorted(r["k"] for r in expected)
    # The point of the test: it returned the matching rows, not the file.
    assert len(got) < rows / 10, "a prune-only result would return every row"


def test_native_and_pyarrow_filtered_paths_agree(tmp_path):
    """The two filtered readers must produce the same *answer* (post-`Filter`), even though
    the native one returns a superset and pyarrow returns the exact rows."""
    path, _ = _write(tmp_path)
    src = ParquetSource(path)
    predicate = _pred("bucket", "eq", {"int": 3})

    def keep(r):
        return r["bucket"] == 3

    native = src._native_read_filtered(None, predicate)
    assert native is not None, "the native filtered read should be available for local files"

    pa_filter = src._pa_filter(predicate)
    pyarrow_rows = _rows(
        [b for f in src._files() for b in src._read_table(f, None, pa_filter).to_batches()]
    )

    assert _apply(_rows(native), keep) == pyarrow_rows
    # pyarrow filters exactly; native prunes. Same answer, different row counts allowed.
    assert len(_rows(native)) >= len(pyarrow_rows)


def test_filtered_read_falls_back_when_the_native_reader_is_unavailable(tmp_path, monkeypatch):
    """A native read returning `None` must be invisible: the pyarrow path takes over and
    the rows are unchanged. This is the silent-fallback contract."""
    path, _ = _write(tmp_path)
    src = ParquetSource(path)
    predicate = _pred("bucket", "eq", {"int": 3})

    with_native = _rows(src.read(predicate=predicate))
    monkeypatch.setattr(
        _parquet_native, "read_row_groups_filtered", lambda *a, **k: None, raising=True
    )
    assert src._native_read_filtered(None, predicate) is None
    without_native = _rows(src.read(predicate=predicate))

    def keep(r):
        return r["bucket"] == 3

    assert _apply(with_native, keep) == _apply(without_native, keep)


def test_a_temporal_predicate_declines_native_and_still_prunes_via_pyarrow(tmp_path):
    """`to_native_predicate` refuses date literals (it cannot verify the parquet physical
    unit without risking an unsound prune), so the native filtered read must *decline* and
    leave the capable pyarrow reader to prune — never silently swallow the predicate."""
    import datetime as dt

    epoch = dt.date(1970, 1, 1)
    days = [epoch + dt.timedelta(days=18_000 + i) for i in range(1000)]
    pq.write_table(
        pa.table({"k": np.arange(1000, dtype="int64"), "d": pa.array(days, pa.date32())}),
        tmp_path / "dates.parquet",
        row_group_size=100,
    )
    src = ParquetSource(str(tmp_path))
    cutoff_days = 18_800
    predicate = _pred("d", "ge", {"date": cutoff_days})

    assert src._native_read_filtered(None, predicate) is None  # declined, as it must

    def keep(r):
        return (r["d"] - epoch).days >= cutoff_days

    pushed = _rows(src.read(predicate=predicate))
    assert _apply(pushed, keep) == _apply(_rows(src.read()), keep)
    # pyarrow really did prune: this is not just an unfiltered read wearing a predicate.
    assert len(pushed) < 1000


def test_a_pushed_predicate_through_the_engine_returns_exactly_the_matching_rows(tmp_path):
    """The property that actually matters, and that the source-level tests cannot see.

    Every other test here asserts what the *reader* returns, where a superset is legal. This
    asserts what the *query* returns, end to end through Kyber and the engine — which must be
    exactly the matching rows no matter which reader path ran or how much it over-read. It is
    the assertion that stays true if the pushdown implementation is rewritten again.
    """
    import batcher as bt

    pq.write_table(pa.table({"a": pa.array([1, 2, 3, 4], type=pa.int64())}), tmp_path / "t.parquet")
    ds = bt.read(str(tmp_path), format="parquet").filter(bt.col("a") > 2)
    assert sorted(ds.to_pydict()["a"]) == [3, 4]


# --- 2. streaming through native row-group windows == pyarrow streaming -------------


@pytest.mark.parametrize(
    ("depth", "remote", "expected"),
    [
        (1, False, True),  # nothing above is fanning out -> native wins (measured 2-7x)
        (2, False, True),  # still the native-winning region
        (4, False, False),  # the measured crossover: the two fan-outs start to multiply
        (16, False, False),  # the default local depth; native measured 3x SLOWER here
        (32, True, True),  # remote: in-flight requests, not cores, are the scarce thing
        (16, True, True),
    ],
)
def test_native_stream_rule_matches_the_measured_crossover(depth, remote, expected):
    """The native reader's row-group concurrency and the read-ahead's file concurrency are
    two routes to the same parallelism; using both at once oversubscribes the decode. This
    pins the rule that keeps exactly one of them active on local disk."""
    assert _native_stream.use_native_stream(depth, remote) is expected


def test_many_local_files_do_not_stream_natively(tmp_path, monkeypatch):
    """The regression this rule exists to prevent: at the default read-ahead depth a
    many-file local scan must NOT also fan out inside each file."""
    from batcher.io.base import source as base_source

    monkeypatch.setattr(base_source, "available_cpu_count", lambda: 16, raising=True)
    path, _ = _write(tmp_path, n_files=8, rows=2_000, row_group_size=200)
    src = ParquetSource(path)
    assert src._iter_readahead_depth(len(src._files())) == 8  # capped by file count

    calls = []
    real = _parquet_native.read_row_groups_filtered
    monkeypatch.setattr(
        _parquet_native,
        "read_row_groups_filtered",
        lambda *a, **k: (calls.append(a[0]), real(*a, **k))[1],
        raising=True,
    )
    streamed = _rows(list(src.iter_batches()))
    assert calls == [], "a deep read-ahead must not also fan out natively per file"
    # ...and the rows are exactly what the native path would have produced anyway.
    assert streamed == _rows(src.read())


def test_a_single_local_file_does_stream_natively(tmp_path, monkeypatch):
    """The other side of the rule: with no outer fan-out the native path must still be taken
    (it measured 4-7.8x there), otherwise the win is silently given up."""
    from batcher.io.base import source as base_source

    monkeypatch.setattr(base_source, "available_cpu_count", lambda: 16, raising=True)
    path, _ = _write(tmp_path, n_files=1, rows=4_000, row_group_size=200)
    src = ParquetSource(path)

    calls = []
    real = _parquet_native.read_row_groups_filtered
    monkeypatch.setattr(
        _parquet_native,
        "read_row_groups_filtered",
        lambda *a, **k: (calls.append(a[0]), real(*a, **k))[1],
        raising=True,
    )
    streamed = _rows(list(src.iter_batches()))
    assert calls, "a single file has no outer fan-out, so it must stream natively"
    assert streamed == _rows(src.read())


def test_iter_batches_matches_read(tmp_path):
    path, _ = _write(tmp_path)
    src = ParquetSource(path)
    assert _rows(list(src.iter_batches())) == _rows(src.read())


def test_iter_batches_preserves_file_order_and_projection(tmp_path):
    path, _ = _write(tmp_path)
    src = ParquetSource(path)
    streamed = list(src.iter_batches(["k", "bucket"]))
    assert all(b.schema.names == ["k", "bucket"] for b in streamed)
    # File order: `k` is globally increasing across the parts, so a correct stream is sorted.
    keys = [k for b in streamed for k in b.column("k").to_pylist()]
    assert keys == sorted(keys)


def test_native_window_stream_matches_pyarrow_stream(tmp_path, monkeypatch):
    """`_iter_file` with the native windows vs. the same file read by pyarrow: same rows."""
    path, files = _write(tmp_path, n_files=1)
    src = ParquetSource(path)

    native_rows = _rows(list(src._iter_file(files[0], None)))

    monkeypatch.setattr(
        _parquet_native, "read_row_groups_filtered", lambda *a, **k: None, raising=True
    )
    fallback_rows = _rows(list(src._iter_file(files[0], None)))

    assert native_rows == fallback_rows
    assert native_rows == _rows(src.read())


def test_a_mid_stream_native_failure_falls_back_per_window(tmp_path, monkeypatch):
    """A window whose native read fails is re-read from pyarrow rather than aborting or
    duplicating — the stream must be seamless even if only some windows go native."""
    path, files = _write(tmp_path, n_files=1)
    src = ParquetSource(path)
    expected = _rows(src.read())

    real = _parquet_native.read_row_groups_filtered
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        return None if calls["n"] % 2 == 0 else real(*args, **kwargs)

    monkeypatch.setattr(_parquet_native, "read_row_groups_filtered", flaky, raising=True)
    assert _rows(list(src._iter_file(files[0], None))) == expected
    assert calls["n"] > 1, "the file should span several windows for this to mean anything"


def test_row_group_windows_cover_every_group_once_and_in_order(tmp_path):
    _, files = _write(tmp_path, n_files=1, rows=5_000, row_group_size=250)
    meta = pq.ParquetFile(files[0]).metadata
    windows = list(row_group_windows(meta))

    assert [i for w in windows for i in w] == list(range(meta.num_row_groups))
    assert all(w for w in windows), "a window is never empty"
    assert all(len(w) <= 8 for w in windows), "the row-group cap holds"


def test_row_group_windows_of_an_empty_file(tmp_path):
    p = tmp_path / "empty.parquet"
    pq.write_table(pa.table({"k": pa.array([], type=pa.int64())}), p)
    src = ParquetSource(str(tmp_path))
    assert src.read() == [] or sum(b.num_rows for b in src.read()) == 0
    assert sum(b.num_rows for b in src.iter_batches()) == 0


def test_a_single_oversized_row_group_still_forms_a_window(tmp_path, monkeypatch):
    """A row-group larger than the whole byte budget must still be readable, as one window."""
    monkeypatch.setattr(_native_stream, "_NATIVE_WINDOW_BYTES", 1, raising=True)
    path, files = _write(tmp_path, n_files=1, rows=2_000, row_group_size=1_000)
    windows = list(row_group_windows(pq.ParquetFile(files[0]).metadata))
    assert all(len(w) == 1 for w in windows)
    assert _rows(list(ParquetSource(path)._iter_file(files[0], None))) == _rows(
        ParquetSource(path).read()
    )


def test_bring_your_own_credentials_never_reach_the_native_reader(tmp_path):
    """The native FFI takes a bare URI and resolves the backend itself, so it cannot honor
    a caller-supplied `filesystem=`/`storage_options=`. Routing such a read to it would not
    be slower — it would address a *different* store (the `endpoint_override` case). Both
    new native paths must decline, and the results must be unchanged."""
    import pyarrow.fs as pafs

    path, files = _write(tmp_path, n_files=2, rows=1_000, row_group_size=100)
    plain = ParquetSource(path)
    expected = _rows(plain.read())
    predicate = _pred("bucket", "eq", {"int": 3})

    for byo in (
        ParquetSource(path, filesystem=pafs.LocalFileSystem()),
        ParquetSource(path, storage_options={"endpoint_override": "http://minio:9000"}),
    ):
        assert byo._native_uri_is_addressable(files[0]) is False
        assert byo._native_read_filtered(None, predicate) is None
        # ...and it still reads correctly, through pyarrow.
        assert _rows(byo.read()) == expected
        assert _rows(list(byo._iter_file(files[0], None))) == _rows(
            list(plain._iter_file(files[0], None))
        )


# --- 3. read-ahead depth is a throughput knob, not a semantic one -------------------


@pytest.mark.parametrize("depth", [1, 2, 4, 32, 128])
def test_readahead_depth_does_not_change_results(tmp_path, monkeypatch, depth):
    path, _ = _write(tmp_path, n_files=6, rows=2_000, row_group_size=200)
    baseline = _rows(list(ParquetSource(path).iter_batches()))

    monkeypatch.setattr("batcher.io.base.source._REMOTE_READ_CONCURRENCY", depth, raising=True)
    monkeypatch.setattr("batcher.io.base.source._ITER_READAHEAD_FILES", depth, raising=True)
    monkeypatch.setattr("batcher.io.base.source.available_cpu_count", lambda: depth, raising=True)
    assert _rows(list(ParquetSource(path).iter_batches())) == baseline
    assert _rows(ParquetSource(path).read()) == baseline


@pytest.mark.parametrize("byte_budget", [1 << 20, 1 << 30])
def test_readahead_byte_budget_does_not_change_results(tmp_path, monkeypatch, byte_budget):
    path, _ = _write(tmp_path, n_files=4, rows=3_000, row_group_size=300)
    baseline = _rows(ParquetSource(path).read())
    monkeypatch.setattr("batcher.io.base.source._ITER_READAHEAD_BYTES", byte_budget, raising=True)
    assert _rows(list(ParquetSource(path).iter_batches())) == baseline


def test_remote_sources_are_sized_by_requests_and_local_ones_by_cores(tmp_path, monkeypatch):
    """The whole point of the change: a low core count no longer caps a remote read's
    concurrency, while a local read is left exactly as it was."""
    from batcher.io.base import source as base_source

    monkeypatch.setattr(base_source, "available_cpu_count", lambda: 4, raising=True)

    path, _ = _write(tmp_path, n_files=1)
    local = ParquetSource(path)
    assert local._is_remote() is False
    assert local._read_concurrency(100) == 8  # 4 cores x 2 — unchanged

    # An `s3://` source without credentials, built without touching the network: only the
    # path is needed to answer the sizing question.
    remote = _pathonly_source("s3://bucket/events/")
    assert remote._is_remote() is True
    assert remote._read_concurrency(100) == base_source._REMOTE_READ_CONCURRENCY
    # Never more files than exist, whichever ruler is used.
    assert remote._read_concurrency(3) == 3


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/data/events", False),
        ("data/events/*.parquet", False),
        ("file:///data/events", False),
        ("s3://bucket/events/", True),
        ("gs://bucket/events/", True),
        ("abfs://c@acct.dfs.core.windows.net/events", True),
        ("https://host/events.parquet", True),
    ],
)
def test_remoteness_is_decided_by_scheme(path, expected):
    assert _pathonly_source(path)._is_remote() is expected
