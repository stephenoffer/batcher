"""The thread-safety statements in `docs/api/relational/sessions-and-catalogs.md`, executed.

That page tells a user what may be shared between threads and what may not. Each statement
is one test here, so the page cannot drift from the code:

* a `Dataset` is immutable, so one instance can be collected from many threads at once;
* `config_context` is scoped to the calling context, so a block in one thread is invisible
  to another;
* `set_config` sets the *calling* context's config, so a thread started afterwards does not
  see it (it starts from the defaults);
* the default session is process-global, so `set_session` in one thread changes what
  `bt.sql` resolves against in every thread;
* `iter_batches` and `iter_rows` return plain generators, which belong to one consumer.
"""

from __future__ import annotations

import inspect
import threading
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pytest

import batcher as bt
from batcher.config import Config, ExecutionConfig, active_config, config_context, set_config

pytestmark = pytest.mark.integration


def _run_in_thread(fn):
    out: list = []
    t = threading.Thread(target=lambda: out.append(fn()))
    t.start()
    t.join(timeout=60)
    return out[0]


def test_one_dataset_collected_from_many_threads_at_once() -> None:
    table = pa.table({"k": [i % 7 for i in range(20_000)], "v": list(range(20_000))})
    ds = bt.from_arrow(table).group_by("k").agg(s=bt.col("v").sum()).sort("k")
    expected = ds.to_pydict()
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: ds.to_pydict(), range(16)))
    assert all(r == expected for r in results)


def test_config_context_does_not_leak_across_threads() -> None:
    default = active_config().execution.morsel_rows
    entered, release = threading.Event(), threading.Event()

    def hold() -> None:
        with config_context(Config().replace(execution=ExecutionConfig(morsel_rows=2048))):
            entered.set()
            release.wait(timeout=60)

    t = threading.Thread(target=hold)
    t.start()
    try:
        assert entered.wait(timeout=60)
        assert active_config().execution.morsel_rows == default
    finally:
        release.set()
        t.join(timeout=60)


def test_set_config_is_not_inherited_by_a_new_thread() -> None:
    default = _run_in_thread(lambda: active_config().execution.morsel_rows)
    set_config(Config().replace(execution=ExecutionConfig(morsel_rows=4096)))
    try:
        assert active_config().execution.morsel_rows == 4096
        assert _run_in_thread(lambda: active_config().execution.morsel_rows) == default
    finally:
        set_config(Config())


def test_set_session_is_process_global() -> None:
    previous = bt.current_session()
    mine = bt.Session()
    try:
        _run_in_thread(lambda: bt.set_session(mine))
        assert bt.current_session() is mine
    finally:
        bt.set_session(previous)


def test_iterators_are_single_consumer_generators() -> None:
    ds = bt.from_pydict({"a": [1, 2]})
    assert inspect.isgenerator(ds.iter_batches())
    assert inspect.isgenerator(ds.iter_rows())
