"""A preprocessor learns the same state, and transforms to the same rows, on one node or four.

`fit` takes no `distributed=` argument: its aggregate is a bare `collect()`, which follows
``distributed="auto"`` and, on a single-node test cluster, stays single-node. So three ways
of fitting are compared here, over 80,000 rows in eight Parquet files:

- ``distributed.mode="never"``, the single-node baseline;
- ``distributed.mode="always"``, the documented session pin, with a routing spy proving every
  terminal a fit ran really resolved to the Ray path;
- every fit's `collect` forced onto ``num_workers=4``, because `.claude/rules/testing.md`
  records that a distributed run without `num_workers` may use one worker, which computes
  what single-node computes.

Each fitted state must match the baseline up to float reassociation and dict order, and each
fitted object's `transform` must collect to the same rows with ``distributed=True,
num_workers=4`` as with ``distributed=False``. The module opens with a control known to
diverge across workers, a `LIMIT` over an unordered `group_by`, so a vacuous one-worker
comparison cannot pass as a distributed one.

CI installs no Ray, so this suite never runs in the PR gate; see `just lint-skips`.
Run it with ``RAY_ADDRESS=local``.
"""

from __future__ import annotations

import math

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from _ray_cluster import init_test_ray, shutdown_test_ray
from batcher.api.dataset.frame import Dataset
from batcher.config import option_context

pytest.importorskip("ray", reason="ray not installed")
pytest.importorskip("batcher._native", reason="native engine not built")

pytestmark = pytest.mark.integration

_WORKERS = 4
_FILES = 8
_ROWS_PER_FILE = 10_000


@pytest.fixture(scope="module", autouse=True)
def _ray_session():
    started = init_test_ray(_WORKERS)
    yield
    shutdown_test_ray(started)


@pytest.fixture(scope="module")
def parquet_dir(tmp_path_factory) -> str:
    """Eight files, 80,000 rows, with NaN in `a`, a skewed category and a text column."""
    root = tmp_path_factory.mktemp("preprocessors")
    rng = np.random.default_rng(7)
    for i in range(_FILES):
        n = _ROWS_PER_FILE
        a = rng.normal(3.0, 2.0, n)
        a[rng.random(n) < 0.05] = np.nan
        pq.write_table(
            pa.table(
                {
                    "id": np.arange(i * n, (i + 1) * n, dtype="int64"),
                    "a": a,
                    "b": rng.exponential(5.0, n),
                    "g": rng.choice(["x", "y", "z", "w", "v"], n, p=[0.4, 0.3, 0.2, 0.07, 0.03]),
                    "k": rng.integers(0, 300, n).astype("int64"),
                    "y": (rng.random(n) < 0.3).astype("int64"),
                    "s": rng.random(n),
                    "t": rng.choice(["red cat", "blue dog sat", "the cat and the dog", "green"], n),
                }
            ),
            root / f"part-{i}.parquet",
        )
    return str(root)


@pytest.fixture
def ds(parquet_dir) -> bt.Dataset:
    return bt.read.parquet(parquet_dir)


def _preprocessors() -> dict[str, object]:
    """Fresh, unfitted instances of a representative preprocessor from each family."""
    from batcher.ml import preprocessors as pp

    return {
        "StandardScaler": pp.StandardScaler(["a", "b"]),
        "RobustScaler": pp.RobustScaler(["a", "b"]),
        "SimpleImputer-mean": pp.SimpleImputer("a"),
        "SimpleImputer-mode": pp.SimpleImputer("g", strategy="most_frequent"),
        "OneHotEncoder": pp.OneHotEncoder("g"),
        "OrdinalEncoder": pp.OrdinalEncoder(["g", "k"]),
        "TargetEncoder": pp.TargetEncoder("g", "y"),
        "FrequencyEncoder": pp.FrequencyEncoder("g"),
        "CountVectorizer": pp.CountVectorizer("t", dense=True),
        "TfidfVectorizer": pp.TfidfVectorizer("t", dense=True),
        "PCA": pp.PCA(["b", "s", "k"], n_components=2),
        "QuantileTransformer": pp.QuantileTransformer("b", n_quantiles=20),
        "KBinsDiscretizer": pp.KBinsDiscretizer("b", n_bins=8),
        "PowerTransformer": pp.PowerTransformer("b"),
        "GroupImputer": pp.GroupImputer("a", by="g"),
        "IsotonicCalibrator": pp.IsotonicCalibrator("s", "y"),
        "PlattCalibrator": pp.PlattCalibrator("s", "y"),
        "Chain-uncached": pp.Chain(
            pp.SimpleImputer("a"), pp.StandardScaler(["a", "b"]), cache=False
        ),
    }


def _spy_routes(monkeypatch) -> list[bool]:
    """Record every ``distributed="auto"`` routing decision a terminal makes."""
    from batcher.api.terminal import routing

    routes: list[bool] = []
    original = routing._resolve_distributed

    def spy(distributed, plan=None, sources=None):
        decision = original(distributed, plan, sources)
        routes.append(decision)
        return decision

    monkeypatch.setattr(routing, "_resolve_distributed", spy)
    return routes


def _fit_all(ds: bt.Dataset) -> dict[str, object]:
    return {name: pre.fit(ds) for name, pre in _preprocessors().items()}


def _normalized(value: object) -> object:
    """A `to_dict` document with its ordered pair lists turned back into (unordered) dicts."""
    if isinstance(value, dict):
        if set(value) == {"__items__"}:
            return {repr(_normalized(k)): _normalized(v) for k, v in value["__items__"]}
        return {k: _normalized(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalized(v) for v in value]
    return value


def _close(left: object, right: object, path: str = "") -> None:
    """Assert equality up to float reassociation, recursing through dicts and lists."""
    if isinstance(left, float) or isinstance(right, float):
        assert isinstance(left, (int, float)) and isinstance(right, (int, float)), path
        if math.isnan(left) or math.isnan(right):
            assert math.isnan(left) and math.isnan(right), (path, left, right)
            return
        assert math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9), (path, left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys(), (path, left, right)
        for key in left:
            _close(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, list):
        assert isinstance(right, list) and len(left) == len(right), (path, left, right)
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            _close(a, b, f"{path}[{index}]")
    else:
        assert left == right, (path, left, right)


def _assert_same_state(baseline: dict, other: dict) -> None:
    from batcher.ml.preprocessors import to_dict

    assert baseline.keys() == other.keys()
    for name in baseline:
        _close(_normalized(to_dict(baseline[name])), _normalized(to_dict(other[name])), name)


def test_the_fixture_really_fans_out(ds):
    """Positive control: across four workers an unordered LIMIT keeps different groups.

    If this ever agrees, the comparisons below are one worker against one worker and prove
    nothing about distribution.
    """
    query = ds.group_by("k").agg(total=bt.col("b").sum()).limit(3)
    local = sorted(query.collect(distributed=False).column("k").to_pylist())
    remote = sorted(query.collect(distributed=True, num_workers=_WORKERS).column("k").to_pylist())
    assert len(local) == len(remote) == 3
    assert local != remote


def test_fits_under_mode_always_route_to_ray_and_match_never(ds, monkeypatch):
    routes = _spy_routes(monkeypatch)
    with option_context("distributed.mode", "never"):
        never = _fit_all(ds)
    assert routes and not any(routes), "a fit under 'never' reached Ray"
    routes.clear()
    with option_context("distributed.mode", "always"):
        always = _fit_all(ds)
    assert routes and all(routes), "a fit under 'always' stayed single-node"
    _assert_same_state(never, always)


def test_fits_forced_onto_four_workers_match_single_node(ds, monkeypatch):
    with option_context("distributed.mode", "never"):
        baseline = _fit_all(ds)
    original = Dataset.collect
    calls: list[dict] = []

    def four_workers(self, *args, **kwargs):
        kwargs.setdefault("distributed", True)
        kwargs.setdefault("num_workers", _WORKERS)
        calls.append(kwargs)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Dataset, "collect", four_workers)
    forced = _fit_all(ds)
    assert calls, "no fit went through Dataset.collect, so nothing was forced"
    _assert_same_state(baseline, forced)


def _rows(table: pa.Table) -> list[dict]:
    return sorted(table.to_pylist(), key=lambda row: row["id"])


def test_transforms_collect_the_same_rows_on_four_workers(ds):
    with option_context("distributed.mode", "never"):
        fitted = _fit_all(ds)
    for name, pre in fitted.items():
        out = pre.transform(ds)
        local = out.collect(distributed=False)
        remote = out.collect(distributed=True, num_workers=_WORKERS)
        assert local.num_rows == remote.num_rows == _FILES * _ROWS_PER_FILE, name
        assert local.column_names == remote.column_names, name
        assert local.schema.types == remote.schema.types, name
        _close(_rows(local), _rows(remote), name)
