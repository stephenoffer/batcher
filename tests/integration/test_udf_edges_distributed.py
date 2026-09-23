"""The UDF verbs' edges under `distributed=True`, held to their single-node answers.

Each comparison runs `collect(distributed=False)` against `collect(distributed=True,
num_workers=4)` over four Parquet files. That pairing is the one `.claude/rules/testing.md`
asks for: a bare `distributed=True` runs one worker, which computes what single-node
computes, so it would agree for the wrong reason. `test_the_control_diverges` is the proof
the lever moves: a `LIMIT` over an unordered `group_by` is one of the stated exceptions and
keeps different groups once four workers split the input.

What is pinned here:

* a device request no node can meet (`num_gpus=1` on this GPU-less Ray) fails fast with a
  `PlanError`, where it used to wait forever;
* an unpicklable closure fails with a short message naming the captured variable;
* `ds.map` whose rows carry different keys keeps every key on both paths;
* `batch_format` polars / torch / jax, `retry_on`, `max_concurrency` and `fn_args` agree
  with single-node.

Every `fn` is a closure or is defined inside the test: Ray pickles a module-level callable by
reference, and the tests directory is not importable on a worker.
"""

from __future__ import annotations

import concurrent.futures
import threading
import time

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.integration

pytest.importorskip("ray", reason="ray not installed")

from _ray_cluster import init_test_ray, shutdown_test_ray  # noqa: E402  (after importorskip)

_WORKERS = 4
_ROWS_PER_FILE = 500
_FAIL_FAST_S = 60


@pytest.fixture(scope="module", autouse=True)
def _ray_session():
    started = init_test_ray(4)
    yield
    shutdown_test_ray(started)


@pytest.fixture(scope="module")
def source(cluster_scratch) -> str:
    """Four Parquet files, so the read genuinely fans out across the workers."""
    directory = cluster_scratch("udf_edges")
    for part in range(4):
        start = part * _ROWS_PER_FILE
        xs = list(range(start, start + _ROWS_PER_FILE))
        table = pa.table({"x": pa.array(xs, pa.int64()), "g": pa.array([x % 17 for x in xs])})
        pq.write_table(table, directory / f"p{part}.parquet")
    return str(directory)


def _both(query: bt.Dataset) -> tuple[pa.Table, pa.Table]:
    single = query.collect(distributed=False)
    spread = query.collect(distributed=True, num_workers=_WORKERS)
    return single, spread


def _assert_same(query: bt.Dataset, key: str = "x") -> pa.Table:
    """Same rows, names and types on both paths; rows compared in `key` order."""
    single, spread = _both(query)
    assert single.schema == spread.schema
    assert single.sort_by(key).equals(spread.sort_by(key))
    return single


def test_the_control_diverges(source):
    """A LIMIT over an unordered group_by keeps different groups once the workers split."""
    query = bt.read.parquet(source).group_by("g").agg(n=bt.col("x").sum()).limit(3)
    single, spread = _both(query)
    assert single.num_rows == spread.num_rows == 3
    assert set(single["g"].to_pylist()) != set(spread["g"].to_pylist())


# --- fail fast rather than hang ---------------------------------------------------------


def _within(seconds: float, fn) -> BaseException:
    """Run `fn` on a thread and return what it raised, failing if it takes over `seconds`."""
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = pool.submit(fn)
    try:
        future.result(timeout=seconds)
    except concurrent.futures.TimeoutError:
        pytest.fail(f"still running after {seconds}s: the stage is waiting to be scheduled")
    except BaseException as exc:  # the raise is the outcome under test
        return exc
    finally:
        pool.shutdown(wait=False)
    pytest.fail("expected the stage to be refused")


@pytest.mark.parametrize("request_kwargs", [{"num_gpus": 1}, {"resources": {"TPU": 1}}])
def test_an_unplaceable_device_request_fails_fast(source, request_kwargs):
    ray = pytest.importorskip("ray")
    wanted = "GPU" if "num_gpus" in request_kwargs else "TPU"
    if ray.cluster_resources().get(wanted, 0) > 0:
        pytest.skip(f"this Ray cluster has {wanted}")

    class Model:
        def __call__(self, batch):
            return batch

    query = bt.read.parquet(source).map_batches(Model, **request_kwargs)
    started = time.monotonic()
    err = _within(_FAIL_FAST_S, lambda: query.collect(distributed=True, num_workers=_WORKERS))
    assert isinstance(err, PlanError), err
    assert wanted in str(err) and "distributed=False" in str(err)
    assert time.monotonic() - started < _FAIL_FAST_S


def test_an_unpicklable_closure_fails_briefly(source):
    lock = threading.Lock()
    query = bt.read.parquet(source).map_batches(lambda b: (lock, b)[1])
    with pytest.raises(PlanError, match="'lock'") as err:
        query.collect(distributed=True, num_workers=_WORKERS)
    assert len(str(err.value)) < 600
    assert query.collect(distributed=False).num_rows == 4 * _ROWS_PER_FILE


# --- the same answer on both paths ------------------------------------------------------


def test_map_key_drift_keeps_every_key_on_both_paths(source):
    query = bt.read.parquet(source).map(
        lambda r: {"x": r["x"], "a": 1} if r["x"] % 2 == 0 else {"x": r["x"], "b": 2}
    )
    out = _assert_same(query)
    assert out.column_names == ["x", "a", "b"]
    assert out["a"].null_count == out["b"].null_count == 2 * _ROWS_PER_FILE


@pytest.mark.parametrize("fmt", ["polars", "torch", "jax"])
def test_batch_formats_agree(source, fmt):
    pytest.importorskip({"polars": "polars", "torch": "torch", "jax": "jax"}[fmt])
    _assert_same(bt.read.parquet(source).map_batches(lambda b: b, batch_format=fmt))


def test_fn_args_and_kwargs_agree(source):
    query = bt.read.parquet(source).map_batches(
        lambda b, a, k=0: {"x": pc.add(b["x"], a + k)}, fn_args=(1,), fn_kwargs={"k": 2}
    )
    out = _assert_same(query)
    assert out["x"].to_pylist()[:2] == [3, 4]


def test_max_concurrency_agrees(source):
    async def double(batch):
        return {"x": pc.multiply(batch["x"], 2)}

    _assert_same(bt.read.parquet(source).map_batches(double, max_concurrency=2))


def test_retry_on_recovers_on_both_paths(source):
    """A transient failure that `retry_on` names is retried away, so both paths finish.

    The failure is keyed on the batch's first value and a per-process attempt count, so it
    fails exactly once per batch in whichever process runs it.
    """

    def build():
        attempts: dict[int, int] = {}

        def flaky(batch):
            first = batch["x"][0].as_py()
            attempts[first] = attempts.get(first, 0) + 1
            if attempts[first] == 1:
                raise ConnectionError("transient")
            return batch

        return bt.read.parquet(source).map_batches(
            flaky, max_retries=2, retry_on=ConnectionError, retry_backoff=0
        )

    single = build().collect(distributed=False)
    spread = build().collect(distributed=True, num_workers=_WORKERS)
    assert single.schema == spread.schema
    assert single.sort_by("x").equals(spread.sort_by("x"))
    assert single.num_rows == 4 * _ROWS_PER_FILE
