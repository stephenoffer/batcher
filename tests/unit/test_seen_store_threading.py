"""The seen-store may be opened on one thread and used from another.

A streaming query's loop runs on a background thread, but recovery runs on the main one
before that thread starts — and recovery calls `seek` → `confirm`, which is what opens
this connection. Without `check_same_thread=False` the first discovery pass after such a
restore raises ``SQLite objects created in a thread can only be used in that same
thread``, which is the restart path the checkpoint exists for.

The checkpoint logs already disable that check, with a comment saying why; the seen-store
was written first and did not.
"""

from __future__ import annotations

import threading

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from batcher.io.formats.streaming.autoloader import IncrementalFileSource
from batcher.io.formats.streaming.seen_store import SeenStore

pytestmark = pytest.mark.unit


def _in_thread(fn):
    """Run `fn` on a fresh thread, re-raising whatever it raised."""
    box: dict[str, object] = {}

    def run() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:
            box["error"] = exc

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]  # type: ignore[misc]
    return box.get("value")


def test_a_store_opened_on_one_thread_is_usable_from_another(tmp_path):
    store = SeenStore(str(tmp_path / "seen.sqlite"))
    store.mark("a.parquet", size=1, mtime=1.0)

    assert _in_thread(lambda: store.unseen(["a.parquet", "b.parquet"])) == ["b.parquet"]


def test_recovery_on_the_main_thread_does_not_poison_the_loop_thread(tmp_path):
    """`seek` with pending files is what opens the connection during recovery."""
    landing = tmp_path / "landing"
    landing.mkdir()
    for name in ("f1.parquet", "f2.parquet"):
        pq.write_table(pa.table({"a": pa.array([1], type=pa.int64())}), str(landing / name))

    source = IncrementalFileSource(str(landing), "parquet", state_dir=str(tmp_path / "state"))
    # Recovery, on this thread: the restored position names a file already published.
    source.seek({"pending": [str(landing / "f1.parquet")]})
    assert source._store_obj is not None, "seek did not open the store, so this proves nothing"

    # The loop, on its own thread.
    discovered = _in_thread(source.discover)
    assert [str(p).endswith("f2.parquet") for p in discovered] == [True]
