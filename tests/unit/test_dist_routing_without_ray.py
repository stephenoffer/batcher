"""Every operation has a distributed path -- asserted without a cluster.

`tests/integration/test_distributed_engagement.py` asks which executor each shape reaches,
and it is the stronger test because it really runs them. But it needs a Ray cluster, and
**CI installs no Ray** (`CLAUDE.md`), so on the PR gate it does not run at all. The routing
*decision*, though, is pure plan analysis: `_dispatch` picks an executor by looking at the
plan, before a single task is submitted. That part can be checked with no cluster, and this
file is the check -- the first coverage the PR gate has of distributed routing.

Ray is made unreachable rather than merely unused: `_ensure_ray` is neutered and every
executor is replaced by a sentinel raise, so a shape that would contact a cluster fails here
instead of quietly connecting to whatever cluster happens to be up on the machine. That is
not hypothetical tidiness -- the probe this file grew from *did* attach to a long-running
shared cluster on the box, which is precisely what a test must never do.

Sources are a four-file Parquet directory, never in-memory tables. `_unsupported` runs an
in-memory plan on one node *by design* (there is no distributed data to speak of), so on
in-memory sources a missing distributed path is indistinguishable from the correct answer.
The first version of this audit made exactly that mistake and reported ten false gaps.

Three outcomes are legitimate, and the test names which one each shape takes:

* it routes to a distributed executor;
* `requires_staging` is true, so it distributes stage by stage through the adaptive layer
  (`sort` beneath a partitioned window is this -- calling `_dispatch` directly bypasses
  staging, which is a property of the harness, not a gap in the engine);
* it refuses with a `PlanError`, for a reason recorded in `_REFUSES` below.

Anything else is a shape that would silently run the whole job on one node.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.unit


class _Routed(Exception):
    """Raised in place of a distributed executor, naming which one was chosen."""

    def __init__(self, name: str):
        super().__init__(name)
        self.name = name


#: Prefixes identifying a distributed executor entry point.
_EXECUTOR_PREFIXES = ("_distributed_", "execute_global_window_")


def _dist_modules():
    """Every importable module under `batcher.dist`.

    Discovered rather than listed. The first version of this file hand-maintained a table
    of nine entry points against the twenty-one that exist, and patched `_ensure_ray` in
    two of the twenty-two modules that bind it -- so `join_then_agg` reached the fused
    `_distributed_join_aggregate`, missed every sentinel, and **connected to a real Ray
    cluster** from a unit test that advertises itself as Ray-free. A list that has to be
    updated by hand when an executor is added is a list that will be wrong.

    Import failures are skipped: the GPU modules under `dist/gpu/` need a device.
    """
    import importlib
    import pkgutil

    import batcher.dist

    modules = [batcher.dist]
    for info in pkgutil.walk_packages(batcher.dist.__path__, "batcher.dist."):
        try:
            modules.append(importlib.import_module(info.name))
        except Exception:  # a module needing a GPU is absent, not broken
            continue
    return modules


#: Shapes that legitimately have no distributed path, with the reason each is refused.
#: A shape may only be added here with a reason that is a property of the *algebra*, not a
#: gap someone has not got to yet -- the whole point of `_unsupported` raising is that a
#: missing path stays loud.
_REFUSES = {
    "window_lead_global": (
        "an unpartitioned `lead` reads the ordered bucket the offset walk has not reached, so "
        "no rolling tail carries it -- unlike `lag`, whose source rows are the bounded set "
        "immediately behind it and which routes through `global_window/boundary.py`"
    ),
}


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    """A splittable four-file Parquet directory, plus a small right-hand side."""
    left = pa.table(
        {"k": ["a", "b", "a", "c"], "v": [1, 2, 3, 4], "t": [1, 2, 3, 4], "g": ["x", "x", "y", "y"]}
    )
    right = pa.table({"k": ["a", "b", "c"], "w": [10, 20, 30]})
    ldir = tmp_path_factory.mktemp("left")
    rdir = tmp_path_factory.mktemp("right")
    for part in range(4):
        pq.write_table(left, ldir / f"p{part}.parquet")
    pq.write_table(right, rdir / "r.parquet")
    return str(ldir), str(rdir)


#: Shape names, declared statically so `parametrize` does not have to build a `Dataset` at
#: collection time (doing so read a Parquet path the fixture had not created yet).
#: `test_the_name_list_matches_the_builders` keeps the two in step.
_SHAPE_NAMES = (
    "agg_keyless",
    "agg_then_sort",
    "aggregate",
    "distinct",
    "distinct_subset",
    "filter",
    "filter_then_agg",
    "join_anti",
    "join_inner",
    "join_left",
    "join_semi",
    "join_then_agg",
    "limit",
    "project",
    "repartition",
    "row_index",
    "sample_n",
    "sort",
    "sort_then_window",
    "tail",
    "topn",
    "union",
    "window_agg",
    "window_global",
    "window_lag_global",
    "window_lead_global",
    "window_partitioned",
    "with_columns",
)


#: Built lazily from the fixture, so every shape reads a genuinely splittable source.
def _shapes(ldir: str, rdir: str):
    def ds():
        return bt.read.parquet(ldir)

    def rs():
        return bt.read.parquet(rdir)

    return {
        "project": ds().select("k", "v"),
        "filter": ds().filter(bt.col("v") > 1),
        "with_columns": ds().with_columns(z=bt.col("v") * 2),
        "limit": ds().limit(2),
        "tail": ds().tail(2),
        "sort": ds().sort("v"),
        "topn": ds().sort("v").limit(2),
        "aggregate": ds().group_by("k").agg(s=bt.col("v").sum()),
        "agg_keyless": ds().agg(s=bt.col("v").sum()),
        "distinct": ds().distinct(),
        "distinct_subset": ds().distinct(subset=["k"]),
        "join_inner": ds().join(rs(), on="k"),
        "join_left": ds().join(rs(), on="k", how="left"),
        "join_semi": ds().join(rs(), on="k", how="semi"),
        "join_anti": ds().join(rs(), on="k", how="anti"),
        "window_partitioned": ds().with_columns(
            r=bt.row_number().over(partition_by="k", order_by="v")
        ),
        "window_global": ds().with_columns(r=bt.row_number().over(order_by="v")),
        "window_agg": ds().with_columns(s=bt.col("v").sum().over(partition_by="k")),
        "window_lag_global": ds().with_columns(p=bt.lag(bt.col("v"), 1).over(order_by="v")),
        "window_lead_global": ds().with_columns(p=bt.lead(bt.col("v"), 1).over(order_by="v")),
        "row_index": ds().with_row_index("i"),
        "union": ds().union(ds()),
        "sample_n": ds().sample(n=2, seed=1),
        "repartition": ds().repartition(2),
        "agg_then_sort": ds().group_by("k").agg(s=bt.col("v").sum()).sort("s"),
        "filter_then_agg": ds().filter(bt.col("v") > 1).group_by("k").agg(s=bt.col("v").sum()),
        "join_then_agg": ds().join(rs(), on="k").group_by("k").agg(s=bt.col("v").sum()),
        "sort_then_window": ds()
        .sort("v")
        .with_columns(r=bt.row_number().over(partition_by="k", order_by="v")),
    }


@pytest.fixture
def no_ray(monkeypatch):
    """Make a cluster unreachable and every distributed executor a sentinel.

    `_ensure_ray` is patched in **every** module that binds the name, not only where it is
    defined: a monkeypatch follows the name, and `from .ray_runtime.lifecycle import
    _ensure_ray` binds a separate reference that patching the defining module never reaches.
    """
    patched_executors = 0
    for module in _dist_modules():
        for attr in ("_ensure_ray", "ensure_ray"):
            if hasattr(module, attr):
                monkeypatch.setattr(module, attr, lambda *_a, **_k: None, raising=False)
        for name in dir(module):
            if not name.startswith(_EXECUTOR_PREFIXES):
                continue
            if not callable(getattr(module, name, None)):
                continue

            def sentinel(*_a, _name=name, **_k):
                raise _Routed(_name)

            monkeypatch.setattr(module, name, sentinel, raising=False)
            patched_executors += 1
    assert patched_executors >= 20, (
        f"only {patched_executors} executor bindings were patched; discovery is not seeing "
        "the dist package, so a shape could reach a real cluster"
    )


def _route(ds) -> tuple[str, str]:
    """`(outcome, detail)` for one shape: routed / staged / refused."""
    from batcher.dist import executor as dist_executor
    from batcher.dist.executors.plan_analysis import requires_staging

    try:
        dist_executor._dispatch(ds._plan, ds._sources, 2, "disk")
    except _Routed as routed:
        return "routed", routed.name
    except PlanError as refused:
        if requires_staging(ds._plan):
            return "staged", "requires_staging"
        return "refused", str(refused)
    if requires_staging(ds._plan):
        return "staged", "requires_staging"
    return "inline", "returned without reaching an executor"


@pytest.mark.usefixtures("no_ray")
@pytest.mark.parametrize("shape", _SHAPE_NAMES)
def test_every_operation_has_a_distributed_path(shape, tables):
    ds = _shapes(*tables)[shape]
    outcome, detail = _route(ds)

    if shape in _REFUSES:
        assert outcome == "refused", (
            f"{shape} is listed in _REFUSES as having no distributed path, but it "
            f"{outcome} ({detail}) -- if it gained one, delete the entry"
        )
        return

    assert outcome in {"routed", "staged", "inline"}, (
        f"{shape} has no distributed path: {detail}. On splittable data that is a shape "
        "that would run the whole job on one node -- add the executor, or record it in "
        "_REFUSES with a reason from the algebra"
    )


@pytest.mark.usefixtures("no_ray")
def test_the_sentinels_are_actually_installed(tables):
    """Guard against a vacuous suite.

    Every assertion above is "this shape reached an executor". If `monkeypatch` silently
    stopped applying -- an entry point renamed, a caller switched to a spelling nothing
    wraps -- the shapes would run for real (or fail on a missing cluster) and nothing would
    distinguish that from the engine having changed. A plain sort has the longest-standing
    distributed path, so it must come back as `routed`.
    """
    outcome, detail = _route(_shapes(*tables)["sort"])
    assert (outcome, detail) == ("routed", "_distributed_sort")


@pytest.mark.usefixtures("no_ray")
def test_a_refusal_is_still_reachable(tables):
    """The other half of the control: the harness can still observe a refusal.

    Without this, `test_every_operation_has_a_distributed_path` could pass for every shape
    because `_route` never returns "refused" at all -- which would make the whole file
    assert nothing.
    """
    assert _REFUSES, "the refusal control needs at least one entry to be meaningful"
    shape = next(iter(_REFUSES))
    outcome, _detail = _route(_shapes(*tables)[shape])
    assert outcome == "refused"


def test_the_name_list_matches_the_builders(tables):
    """`_SHAPE_NAMES` drives the parametrization; the builders do the work.

    They are two lists, so they can drift -- and a shape dropped from `_SHAPE_NAMES` would
    silently stop being checked while every remaining test still passed.
    """
    assert sorted(_SHAPE_NAMES) == sorted(_shapes(*tables))
