"""A file format writable by name must be readable by name.

`ds.write.<fmt>()` and `bt.read.<fmt>()` are the discoverable surface: someone who wrote a
file with one reaches for the other, and `dir()`/tab-completion is how they find it. A format
on one namespace and absent from the other is a dead end no per-format test catches, because
each one exercises the accessor it already knows exists.

MessagePack was exactly that. `ds.write.msgpack(p)` wrote the file, the `SOURCES` registry
held a working `MsgpackSource`, and `bt.read.msgpack(p)` raised
``FormatError: Unknown format 'msgpack'`` — the reader was reachable only through the untyped
`bt.read(p, format="msgpack")`, which is not what anyone reaches for.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

bt = pytest.importorskip("batcher")

#: Writers that are not file formats, so there is nothing to read back by the same name.
#: Streaming sinks address a destination rather than a file (`console` prints, `memory`
#: accumulates, `for_each*` calls back, `noop` discards — Spark's `format("noop")`, which
#: exists precisely to write nothing), and the MERGE builders mutate a table in place.
_NOT_A_READABLE_FORMAT = {
    "console",
    "for_each",
    "for_each_batch",
    "memory",
    "merge",
    "merge_into",
    "noop",
}


def _named(namespace) -> set[str]:
    return {name for name in dir(namespace) if not name.startswith("_")}


def test_every_named_file_format_writer_has_a_named_reader():
    ds = bt.from_pydict({"x": [1]})
    writers = _named(ds.write) - _NOT_A_READABLE_FORMAT
    missing = sorted(writers - _named(bt.read))
    assert not missing, (
        f"writable by name but not readable by name: {missing}. Add the reader method, or — "
        "if the format genuinely cannot be read back — list it in `_NOT_A_READABLE_FORMAT` "
        "with the reason."
    )


def test_the_not_a_format_list_has_not_gone_stale():
    """An entry that gains a reader must leave the list, or the exemption hides a real gap."""
    gained = sorted(_NOT_A_READABLE_FORMAT & _named(bt.read))
    assert not gained, f"listed as not readable but now on bt.read: {gained}"
    ds = bt.from_pydict({"x": [1]})
    gone = sorted(_NOT_A_READABLE_FORMAT - _named(ds.write))
    assert not gone, f"listed as a writer but no longer one: {gone}"


def test_msgpack_round_trips_through_both_namespaces(tmp_path):
    """The specific gap: write by name, read by name, same rows."""
    pytest.importorskip("msgpack")
    import pyarrow as pa

    table = pa.table({"x": pa.array([1, 2, 3], pa.int64()), "s": pa.array(["a", "b", None])})
    path = str(tmp_path / "events.msgpack")
    bt.from_arrow(table).write.msgpack(path)
    assert bt.read.msgpack(path).collect().to_pydict() == table.to_pydict()


def test_every_top_level_reader_is_reachable_on_the_namespaces_too():
    """`bt.read_<fmt>` is public API, so its name has to lead somewhere on both namespaces.

    The symmetry above is writer-to-reader. This is the third spelling: `bt.read_ipc` and
    `bt.read_ndjson` are public, and neither `ipc` nor `ndjson` is a namespace name — the
    formats are `arrow` and `json`. Someone who read a file with one reaches for the same
    word on the writer and lands on ``Unknown format 'ipc'`` beside a list of thirty names
    that does not visibly contain it, because no number of character edits gets from `ipc`
    to `arrow`.

    A name may be reachable directly or through `_SPELLINGS`; what it may not be is a dead
    end.
    """
    from batcher.api.io_namespace._discovery import _SPELLINGS

    ds = bt.from_pydict({"x": [1]})
    readable = _named(bt.read) | _named(ds.write)
    dead_ends = []
    for attr in dir(bt):
        if not attr.startswith("read_"):
            continue
        fmt = attr[len("read_") :]
        if fmt in readable or _SPELLINGS.get(fmt) in readable:
            continue
        dead_ends.append(fmt)
    assert not dead_ends, (
        f"bt.read_<fmt> names with no namespace spelling and no synonym: {dead_ends}. Add "
        "the namespace method, or map the name in `_SPELLINGS` to the one it means."
    )


@pytest.mark.parametrize(
    ("asked", "meant"), [("ipc", "arrow"), ("ndjson", "json"), ("jsonl", "json")]
)
def test_a_synonym_names_the_format_it_means(asked, meant):
    """The message has to name the *right* thing, not the nearest-looking one.

    Before the synonym table, `bt.read.database` drew "did you mean 'webdataset'" from the
    edit-distance guess — a confident wrong answer, which is worse than none.
    """
    ds = bt.from_pydict({"x": [1]})
    with pytest.raises(bt.FormatError) as excinfo:
        getattr(ds.write, asked)
    message = str(excinfo.value)
    assert f"Did you mean {meant!r}?" in message
    assert f"ds.write.{meant}(...)" in message


def test_an_unrelated_name_still_gets_the_ordinary_error():
    """The synonym table must not swallow the generic case."""
    ds = bt.from_pydict({"x": [1]})
    with pytest.raises(bt.FormatError, match="Unknown format 'nonsense'"):
        _ = ds.write.nonsense


def test_a_synonym_and_the_name_it_means_read_the_same_file(tmp_path):
    """The table claims "same bytes", so the claim is exercised rather than asserted.

    A pair that is merely *similar* would answer a different question, which is the failure
    a synonym table invites.
    """
    path = str(tmp_path / "t.arrow")
    table = {"a": [1, 2], "s": ["x", "y"]}
    bt.from_pydict(table).write.arrow(path)
    assert bt.read.arrow(path).collect().to_pydict() == table

    json_path = str(tmp_path / "t.json")
    bt.from_pydict(table).write.json(json_path)
    assert bt.read.json(json_path).collect().to_pydict() == table
