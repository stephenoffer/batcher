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
