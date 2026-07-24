"""The read contract under messy data: a declared schema that must hold, and `on_error`.

Two defects are pinned here, both of the "passes every gate while being wrong" kind.

**A source declared a schema it did not enforce.** ``schema_mode="strict"`` (the default)
advertises file 0's schema as the schema of the whole source, and the plan's output
schema, the optimizer's column pruning and the final concat all trust that. Nothing
checked it. A directory whose files disagreed therefore produced, with no error:

- an *extra* column in a later file silently dropped from the result; and
- a *differing type* surfacing at the very end as a bare `pyarrow.lib.ArrowInvalid`
  from `Table.from_batches`, naming neither the file, nor the column, nor the fix,
  after the entire read had been paid for.

**`on_error="skip"` did not survive its own headline case.** A directory with one
corrupt file — the thing the flag exists for — failed anyway, in *both* schema modes and
wherever the bad file sorted, because the per-file *metadata* probes ran outside the
policy. Schema inference died before the read began; and worse, `row_count()` is called
from the post-run learned-stats loop, so a read that had already succeeded, with the
corrupt file correctly skipped and the answer computed, was killed on the way out by a
code path whose only job is to record statistics for the next run.

The oracle for the drift semantics is DuckDB's non-``union_by_name`` reader, which has
the same "file 0 defines the contract" model: extra column dropped, missing column an
error, types cast to the contract. The one deliberate departure is that Batcher's cast is
*safe* — DuckDB reads a ``float64`` file against an ``int64`` contract by truncating, so
``2.5`` silently becomes ``2``, and silently corrupting a value is the failure mode this
whole area exists to remove. ``schema_mode="union"`` returns ``2.5`` intact.
"""

from __future__ import annotations

import os
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher._internal.errors import FormatError, SchemaError

pytestmark = pytest.mark.differential


def _dir(**tables: pa.Table) -> str:
    d = tempfile.mkdtemp()
    for name, t in tables.items():
        pq.write_table(t, os.path.join(d, f"{name}.parquet"))
    return d


def _corrupt(d: str, name: str, kind: str) -> None:
    p = os.path.join(d, f"{name}.parquet")
    if kind == "zero":
        open(p, "wb").close()
    elif kind == "garbage":
        with open(p, "wb") as f:
            f.write(b"not a parquet file at all")
    else:  # truncated: a real file cut in half, so the footer is gone
        pq.write_table(pa.table({"a": [1, 2, 3]}), p)
        with open(p, "rb") as f:
            raw = f.read()
        with open(p, "wb") as f:
            f.write(raw[: len(raw) // 2])


# --------------------------------------------------------------- the declared contract
def test_strict_drops_a_column_the_contract_does_not_have_like_duckdb() -> None:
    """An extra column in a later file is not in the contract, so it is dropped.

    This is the one drift shape where dropping is right rather than data loss: file 0's
    schema *is* the declared schema, and DuckDB's plain `read_parquet` does the same.
    Pinned so the conformance check added for the other shapes cannot turn it into a raise.
    """
    duckdb = pytest.importorskip("duckdb")
    d = _dir(p1=pa.table({"a": [1]}), p2=pa.table({"a": [2], "b": [9]}))

    got = bt.read.parquet(d).to_pydict()
    assert got == {"a": [1, 2]}

    files = [os.path.join(d, "p1.parquet"), os.path.join(d, "p2.parquet")]
    expected = duckdb.connect().execute(f"select * from read_parquet({files!r})").fetchall()
    assert sorted(got["a"]) == sorted(r[0] for r in expected)


def test_strict_raises_naming_the_file_when_a_declared_column_is_missing() -> None:
    """A file lacking a declared column must raise, naming the file and the column.

    Before the fix this reached `pa.Table.from_batches` and died as ``ArrowInvalid:
    Schema at index 1 was different``, which names neither.
    """
    d = _dir(p1=pa.table({"a": [1], "b": [8]}), p2=pa.table({"a": [2]}))
    with pytest.raises(SchemaError) as exc:
        bt.read.parquet(d).to_pydict()
    msg = str(exc.value)
    assert "p2.parquet" in msg and "'b'" in msg
    assert "schema_mode='union'" in msg  # the message carries the fix


def test_strict_raises_on_a_lossy_type_mismatch_and_union_reads_it() -> None:
    """float64 against an int32 contract cannot conform without loss, so it raises.

    The deliberate departure from DuckDB, which truncates 2.5 to 2. `union` promotes the
    column and returns the value intact, which is what the error tells the user to do.
    """
    d = _dir(
        p1=pa.table({"a": pa.array([1], pa.int32())}),
        p2=pa.table({"a": pa.array([2.5], pa.float64())}),
    )
    with pytest.raises(SchemaError) as exc:
        bt.read.parquet(d).to_pydict()
    assert "p2.parquet" in str(exc.value) and "'a'" in str(exc.value)

    assert sorted(bt.read.parquet(d, schema_mode="union").to_pydict()["a"]) == [1.0, 2.5]


def test_strict_still_conforms_a_lossless_width_difference() -> None:
    """int32 and int64 across files is a lossless widening, so it must still just read."""
    d = _dir(
        p1=pa.table({"a": pa.array([1], pa.int32())}),
        p2=pa.table({"a": pa.array([2], pa.int64())}),
    )
    assert sorted(bt.read.parquet(d).to_pydict()["a"]) == [1, 2]


def test_strict_conformance_holds_on_the_streaming_path_too() -> None:
    """`iter_batches` shares the contract: the drift must not slip through streaming.

    The materializing and streaming paths reach the reader differently, and the streaming
    one interleaves files, so this is where a per-file check is easiest to lose.
    """
    d = _dir(p1=pa.table({"a": [1], "b": [8]}), p2=pa.table({"a": [2]}))
    with pytest.raises(SchemaError):
        list(bt.read.parquet(d).iter_batches())


def test_strict_conformance_holds_under_a_pushed_predicate() -> None:
    """A filter takes the native pushdown reader, which bypassed `_normalize` entirely."""
    d = _dir(p1=pa.table({"a": [1], "b": [8]}), p2=pa.table({"a": [5]}))
    with pytest.raises(SchemaError):
        bt.read.parquet(d).filter(bt.col("a") > 0).to_pydict()


# ------------------------------------------------------------------------- on_error
@pytest.mark.parametrize("kind", ["zero", "truncated", "garbage"])
@pytest.mark.parametrize("position", ["first", "last"])
@pytest.mark.parametrize("mode", ["strict", "union"])
def test_skip_tolerates_a_corrupt_file_wherever_it_sorts(kind, position, mode) -> None:
    """The headline case: one corrupt member of a directory must not fail the read.

    Every one of these twelve combinations raised before the fix. Position matters
    because strict-mode inference reads only the *first* file's schema, so a corrupt file
    sorting first killed inference; and mode matters because evolution-mode inference
    reads *every* file's schema, so there any position killed it.
    """
    good, bad = ("a_good", "z_bad") if position == "last" else ("z_good", "a_bad")
    d = _dir(**{good: pa.table({"a": [1, 2, 3]})})
    _corrupt(d, bad, kind)

    got = bt.read.parquet(d, on_error="skip", schema_mode=mode).to_pydict()
    assert got == {"a": [1, 2, 3]}


def test_skip_survives_the_post_run_stats_loop_that_used_to_kill_it() -> None:
    """A tolerated file must not kill the query *after* the answer was computed.

    The sharpest form of the bug: the read succeeded and the corrupt file was correctly
    skipped, then `_close_learning_loops` called `row_count()`, which read every footer
    outside the error policy and raised — discarding a finished, correct result from a
    code path that only records statistics for the next run. `count()` is asserted
    alongside because it is answered *from* that same metadata rather than by scanning,
    so it is the other way the unguarded probe becomes user-visible.
    """
    d = _dir(a_good=pa.table({"a": [1, 2, 3]}))
    _corrupt(d, "z_bad", "garbage")
    ds = bt.read.parquet(d, on_error="skip")

    assert ds.to_pydict() == {"a": [1, 2, 3]}
    # The skipped file contributes no rows, so the metadata count must agree with the data.
    assert ds.agg(n=bt.count()).to_pydict()["n"] == [3]


def test_skip_never_swallows_a_schema_disagreement() -> None:
    """`on_error` is about unreadable *bytes*, never about a schema that disagrees.

    Tolerating a `SchemaError` would silently drop every row of a perfectly readable file
    because one column's type differed — reintroducing, through the tolerance path, the
    exact silent data loss the conformance check above exists to remove.
    """
    d = _dir(p1=pa.table({"a": [1], "b": [8]}), p2=pa.table({"a": [2]}))
    with pytest.raises(SchemaError):
        bt.read.parquet(d, on_error="skip").to_pydict()


def test_raise_mode_gives_a_typed_error_naming_the_file_and_the_flag() -> None:
    """The default must not leak `pyarrow.lib.ArrowInvalid` for a corrupt file."""
    d = _dir(good=pa.table({"a": [1]}))
    _corrupt(d, "bad", "garbage")
    with pytest.raises(FormatError) as exc:
        bt.read.parquet(d).to_pydict()
    msg = str(exc.value)
    assert "bad.parquet" in msg and "on_error='skip'" in msg


def test_corrupt_files_lists_each_skipped_path_once() -> None:
    """One unreadable file is met by inference, the footer count, planning and the read.

    `corrupt_files()` answers *which* files were dropped, not how many times a drop was
    attempted, so the four encounters must not become four entries (nor four warnings).
    """
    from batcher.io.formats.structured.parquet.source import ParquetSource

    d = _dir(a_good=pa.table({"a": [1, 2, 3]}))
    _corrupt(d, "z_bad", "garbage")
    src = ParquetSource(d, on_error="skip")
    src.schema()
    src.row_count()
    src.read()
    assert src.corrupt_files() == [os.path.join(d, "z_bad.parquet")]


def test_skip_raises_when_every_file_is_unreadable() -> None:
    """Skipping the whole source has no schema to infer, so it must say so, not crash."""
    d = tempfile.mkdtemp()
    _corrupt(d, "a", "garbage")
    _corrupt(d, "b", "zero")
    with pytest.raises(SchemaError, match="unreadable"):
        bt.read.parquet(d, on_error="skip").to_pydict()


def test_read_and_iter_agree_on_a_tolerated_single_file() -> None:
    """A source's materializing and streaming reads must give the same rows — including
    the empty case. `read()` returned [] for a tolerated corrupt file while `iter_batches`
    raised, because the single-file stream used the non-tolerant reader and the conform
    step resolved the (absent) schema eagerly. Both now yield nothing."""
    from batcher.io.formats.structured.parquet.source import ParquetSource

    d = tempfile.mkdtemp()
    _corrupt(d, "only", "garbage")
    src = ParquetSource(f"{d}/only.parquet", on_error="skip")
    assert src.read() == []
    assert list(src.iter_batches()) == []


def test_iter_tolerates_a_corrupt_file_among_good_ones() -> None:
    """The streaming path drops the bad file and streams the rest, like the materializing one."""
    d = _dir(a_good=pa.table({"a": [1, 2, 3]}))
    _corrupt(d, "z_bad", "garbage")
    ds = bt.read.parquet(d, on_error="skip")
    streamed = [b.to_pydict()["a"] for b in ds.iter_batches()]
    assert [v for batch in streamed for v in batch] == [1, 2, 3]
    assert ds.to_pydict() == {"a": [1, 2, 3]}


# ----------------------------------------------------------------------------- CSV text
def test_csv_invalid_utf8_raises_instead_of_silently_becoming_binary() -> None:
    """pyarrow types an undecodable column `binary` and reports success; both oracles reject.

    That made a column's *type* depend on whether a stray byte landed in the inference
    block — a clean file gives `string`, the same file with one bad byte gives `binary`,
    and nothing downstream can tell that from a column of genuine bytes.
    """
    p = tempfile.mktemp(suffix=".csv")
    with open(p, "wb") as f:
        f.write(b"a\nvalid\n\xff\xfe\n")
    with pytest.raises(FormatError, match="not valid UTF-8"):
        bt.read.csv(p).to_pydict()


def test_csv_invalid_utf8_after_the_inference_block_says_the_same_thing() -> None:
    """Bad bytes past the first block used to be reported as an inference mismatch.

    The advice attached to that error is "declare the type", which does nothing for
    undecodable bytes — the two failures have opposite fixes, so they need distinct errors.
    """
    p = tempfile.mktemp(suffix=".csv")
    with open(p, "wb") as f:
        f.write(b"a,b\n")
        f.write(b"hello,1\n" * 200_000)
        f.write(b"\xff\xfe,2\n")
    with pytest.raises(FormatError, match="not valid UTF-8"):
        bt.read.csv(p).to_pydict()


def test_csv_binary_column_is_readable_when_declared_deliberately() -> None:
    """The refusal is about *inferring* binary; asking for it explicitly still works."""
    p = tempfile.mktemp(suffix=".csv")
    with open(p, "wb") as f:
        f.write(b"a\nvalid\n\xff\xfe\n")
    got = bt.read.csv(p, schema=pa.schema([("a", pa.binary())])).to_pydict()
    assert got == {"a": [b"valid", b"\xff\xfe"]}


def test_csv_invalid_utf8_file_is_skippable_like_any_other_bad_file() -> None:
    """An undecodable file is a corrupt file, so `on_error='skip'` should drop it."""
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "good.csv"), "w") as f:
        f.write("a\nhello\n")
    with open(os.path.join(d, "bad.csv"), "wb") as f:
        f.write(b"a\n\xff\xfe\n")
    assert bt.read.csv(d, on_error="skip").to_pydict() == {"a": ["hello"]}
