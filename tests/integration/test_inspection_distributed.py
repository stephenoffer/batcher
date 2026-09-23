"""Inspection, metadata and data-quality answers are the same on one node and on many.

The inspection surface is mostly built from ordinary queries: `null_count` returns a lazy
Dataset, `describe`/`profile`/`corr_matrix`/`cov_matrix` run one aggregate on call, and `ds.meta`'s
fallbacks, `ds.dq.validate()` and the scalar terminals execute through `distributed="auto"`
and take no `distributed=` argument of their own. Two things therefore need showing:

- the lazy one (`null_count`) agrees collected single-node and across several Ray workers,
  while the eager ones (`describe`, `profile`, `corr_matrix`, `cov_matrix`, which compute on
  call and return a small in-memory table) agree when computed under each mode, and
- the implicit ones really reach Ray under the session pin `distributed.mode="always"`,
  and agree with `"never"` when they do.

The fan-out is proved rather than assumed. `.claude/rules/testing.md` records that
`collect(distributed=True)` with no `num_workers` runs one worker, which computes what
single-node computes, so every equivalence here uses `num_workers=_WORKERS` and the module
opens with a control that is known to diverge across workers: a `LIMIT` over an unordered
`group_by`, measured on this very fixture.

CI installs no Ray, so this suite never runs in the PR gate — see `just lint-skips`.
"""

from __future__ import annotations

import math

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from _ray_cluster import init_test_ray, shutdown_test_ray
from batcher.config import option_context

pytest.importorskip("ray", reason="ray not installed")
pytest.importorskip("batcher._native", reason="native engine not built")

_WORKERS = 4
_FILES = 4
_ROWS_PER_FILE = 50_000


@pytest.fixture(scope="module", autouse=True)
def _ray_session():
    started = init_test_ray(_WORKERS)
    yield
    shutdown_test_ray(started)


@pytest.fixture(scope="module")
def parquet_dir(tmp_path_factory) -> str:
    """Four files, 200,000 rows: past `MIN_ROWS_TO_SHARD`, with nulls, NaN and a key column."""
    root = tmp_path_factory.mktemp("inspect")
    rng = np.random.default_rng(7)
    for i in range(_FILES):
        n = _ROWS_PER_FILE
        amount = rng.normal(50.0, 20.0, n)
        amount[rng.random(n) < 0.01] = np.nan
        qty = pa.array(rng.integers(0, 40, n), mask=rng.random(n) < 0.05)
        pq.write_table(
            pa.table(
                {
                    "g": rng.integers(0, 17, n).astype("int64"),
                    "order_id": np.arange(i * n, (i + 1) * n, dtype="int64"),
                    "amount": amount,
                    "qty": qty,
                    "region": rng.choice(["n", "s", "e", "w"], n),
                }
            ),
            root / f"part-{i}.parquet",
        )
    return str(root)


@pytest.fixture
def ds(parquet_dir) -> bt.Dataset:
    return bt.read.parquet(parquet_dir)


def _local(d: bt.Dataset) -> pa.Table:
    return d.collect(distributed=False)


def _remote(d: bt.Dataset) -> pa.Table:
    return d.collect(distributed=True, num_workers=_WORKERS)


def _close(a, b) -> bool:
    """Equal, or equal up to float reassociation — the one tolerance distribution may take."""
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
    return a == b


def _assert_rows_close(left: pa.Table, right: pa.Table, key: str) -> None:
    """Row-for-row equality after ordering both sides by `key` (never order-blind)."""
    assert left.column_names == right.column_names
    assert left.schema.types == right.schema.types
    lrows = sorted(left.to_pylist(), key=lambda r: str(r[key]))
    rrows = sorted(right.to_pylist(), key=lambda r: str(r[key]))
    assert len(lrows) == len(rrows)
    for lr, rr in zip(lrows, rrows, strict=True):
        for col in left.column_names:
            assert _close(lr[col], rr[col]), (col, lr[col], rr[col])


def test_the_fixture_really_fans_out(ds):
    """Positive control: across workers an unordered LIMIT keeps different groups.

    If this ever agrees, the other tests in this file are comparing one worker with one
    worker and prove nothing about distribution.
    """
    q = ds.group_by("g").agg(s=bt.col("amount").sum()).limit(3)
    local = sorted(_local(q).column("g").to_pylist())
    remote = sorted(_remote(q).column("g").to_pylist())
    assert len(local) == len(remote) == 3
    assert local != remote


def test_null_count_frame_agrees_across_workers(ds):
    """`null_count()` is the one lazy inspection frame, so it takes `collect(distributed=)`."""
    frame = ds.null_count()
    _assert_rows_close(_local(frame), _remote(frame), "g")


@pytest.mark.parametrize(
    ("build", "key"),
    [
        (lambda d: d.describe(), "statistic"),
        (lambda d: d.profile(), "column"),
        (lambda d: d.corr_matrix(["amount", "qty", "g"]), "column"),
        (lambda d: d.cov_matrix(["amount", "qty", "g"]), "column"),
    ],
    ids=["describe", "profile", "corr_matrix", "cov_matrix"],
)
def test_eager_inspection_frames_agree_under_both_modes(ds, monkeypatch, build, key):
    """These compute their aggregate when called and hand back a small in-memory table, so
    collecting the result distributed would only ship finished numbers. The session pin is
    what moves the aggregate itself; the spy proves it did."""
    routes = _spy_routes(monkeypatch)
    never = _under("never", lambda: build(ds).collect(distributed=False))
    assert routes and not any(routes)
    routes.clear()
    always = _under("always", lambda: build(ds).collect(distributed=False))
    assert any(routes), routes
    _assert_rows_close(never, always, key)


def test_dq_row_splits_agree_across_workers(ds):
    gate = ds.dq.not_null("qty").in_range("amount", 0.0, 100.0).unique("order_id")
    local_clean, local_bad = (_local(d) for d in gate.quarantine())
    remote_clean, remote_bad = (_remote(d) for d in gate.quarantine())
    _assert_rows_close(local_clean, remote_clean, "order_id")
    _assert_rows_close(local_bad, remote_bad, "order_id")
    # The split stays a total partition of the input on the distributed path.
    assert remote_clean.num_rows + remote_bad.num_rows == _FILES * _ROWS_PER_FILE
    _assert_rows_close(_local(gate.annotate()), _remote(gate.annotate()), "order_id")


def _spy_routes(monkeypatch) -> list[bool]:
    """Record every `distributed="auto"` routing decision a terminal makes.

    Spied at the resolver rather than the executor so the streaming terminals count too:
    `approx_*` streams through `iter_batches`, which never reaches `executors.select`.
    """
    from batcher.api.terminal import routing

    routes: list[bool] = []
    original = routing._resolve_distributed

    def spy(distributed, plan=None, sources=None):
        decision = original(distributed, plan, sources)
        routes.append(decision)
        return decision

    monkeypatch.setattr(routing, "_resolve_distributed", spy)
    return routes


def _under(mode: str, fn):
    with option_context("distributed.mode", mode):
        return fn()


def test_implicit_terminals_reach_ray_under_mode_always(ds, monkeypatch):
    """Without the pin these all stay local; with it every one of them runs on Ray."""
    routes = _spy_routes(monkeypatch)
    filtered = ds.filter(bt.col("amount") > 10.0)  # defeats the footer shortcuts
    calls = {
        "count": lambda: filtered.count(),
        "min": lambda: filtered.min("amount"),
        "mean": lambda: filtered.mean("qty"),
        "meta.n_unique": lambda: filtered.meta.col("g").n_unique(),
        "meta.null_counts": lambda: filtered.meta.nulls.counts(),
        "dq.validate": lambda: filtered.dq.not_null("qty").unique("order_id").validate().violations,
        "approx_median": lambda: filtered.approx_median("amount"),
    }
    for name, call in calls.items():
        routes.clear()
        never = _under("never", call)
        assert routes and not any(routes), (name, routes)
        routes.clear()
        always = _under("always", call)
        assert routes and all(routes), (name, routes)
        if name == "approx_median":
            # A TDigest merges partials in arrival order, so it lands within its own error.
            assert always == pytest.approx(never, rel=0.01), name
        elif isinstance(never, dict):
            assert never.keys() == always.keys(), name
            assert all(_close(never[k], always[k]) for k in never), (name, never, always)
        else:
            assert _close(never, always), (name, never, always)


def test_meta_fast_path_needs_no_cluster(ds, monkeypatch):
    """A footer-answered question stays a footer read under `always`: nothing executes."""
    routes = _spy_routes(monkeypatch)
    with option_context("distributed.mode", "always"):
        assert ds.meta.shape() == (_FILES * _ROWS_PER_FILE, 5)
        assert ds.meta.col("order_id").bounds() == (0, _FILES * _ROWS_PER_FILE - 1)
    assert routes == []


def _scd_run(root, mode: str) -> dict:
    """Two SCD loads and one CDC batch under `mode`, returning the final tables."""
    history, current = str(root / f"type2-{mode}"), str(root / f"cdc-{mode}")
    with option_context("distributed.mode", mode):
        bt.from_pydict({"id": [1, 2, 3], "tier": ["a", "b", "c"]}).scd.type2(
            history, keys="id", track=["tier"], as_of="2026-01-01", format="parquet"
        )
        bt.from_pydict({"id": [1, 2, 4], "tier": ["a", "B", "d"]}).scd.type2(
            history, keys="id", track=["tier"], as_of="2026-02-01", format="parquet"
        )
        changes = bt.from_pydict(
            {
                "id": [1, 1, 2, 3],
                "v": ["x", "y", "z", "gone"],
                "seq": [1, 2, 1, 1],
                "op": ["u", "u", "u", "d"],
            }
        )
        changes.scd.apply_changes(
            current,
            keys="id",
            sequence_by="seq",
            deletes=bt.col("op") == "d",
            columns=["id", "v", "seq"],
            format="parquet",
        )
        # Read back inside the block too, so every query this helper runs is under `mode`.
        return {
            "type2": bt.read.parquet(history).sort(["id", "valid_from"]).to_pydict(),
            "cdc": bt.read.parquet(current).sort("id").to_pydict(),
        }


def test_scd_maintenance_is_identical_under_both_modes(tmp_path, monkeypatch):
    """Dimension maintenance is merges and joins, so it must route like any query and land
    the same history. These inputs are small, so this checks routing and results, not
    fan-out; the fan-out itself is `test_the_fixture_really_fans_out`'s job."""
    routes = _spy_routes(monkeypatch)
    never = _scd_run(tmp_path, "never")
    assert routes and not any(routes)
    routes.clear()
    always = _scd_run(tmp_path, "always")
    assert routes and all(routes)
    assert never == always
    assert never["type2"]["is_current"] == [True, False, True, True, True]
    assert never["cdc"] == {"id": [1, 2], "v": ["y", "z"], "seq": [2, 1]}
