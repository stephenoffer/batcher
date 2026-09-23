"""The graph algorithms give the same answer on one node and across Ray workers.

Seven of them used to raise `PlanError` ("did not stage") under
``distributed.mode="always"``: `triangle_count`, `clustering_coefficient`,
`degree_distribution`, `jaccard_similarity`, `k_core`, `structural_features` and `summarize`.
Each built an aggregate over a `union` (a degree table, a symmetrized edge list) and fed it to
a second aggregate or a join, a shape the staged executor has no decomposition for. They now
count endpoints with `explode` over a two-element array instead, or materialize a per-node
table before the join. This file holds the main algorithms, including all of those but
`structural_features` (which only adds a join over the others and takes the longest), to
single-node's answer on a graph large enough to shard: about 80,000 edges over four Parquet
files.

The graph functions call `collect` internally and take no `distributed=` argument, so the
session pin ``distributed.mode`` is the only lever, and two things are proved rather than
assumed about it. A routing spy records that ``"always"`` really sends terminals to Ray and
``"never"`` keeps them local, and a positive control shows the pinned run fans out over more
than one worker: `.claude/rules/testing.md` records that one worker computes what single-node
computes, which would make every comparison below vacuous.

CI installs no Ray, so this suite never runs in the PR gate; see `just lint-skips`.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
import batcher.graph as bg
from _ray_cluster import init_test_ray, shutdown_test_ray
from batcher.config import option_context

pytest.importorskip("ray", reason="ray not installed")
pytest.importorskip("batcher._native", reason="native engine not built")

pytestmark = pytest.mark.integration

_WORKERS = 4
_FILES = 4
_NODES = 20_000
_EDGES = 80_000
#: Pairs to score, several inside the dense pocket where they share neighbours.
_PAIRS = bt.from_pydict({"a": [1, 2, 3, 150, 7], "b": [4, 5, 6, 151, 19_000]})


@pytest.fixture(scope="module", autouse=True)
def _ray_session():
    started = init_test_ray(_WORKERS)
    yield
    shutdown_test_ray(started)


@pytest.fixture(scope="module")
def edge_dir(tmp_path_factory) -> str:
    """A random directed graph: 80,000 edges among 20,000 nodes, in four files."""
    root = tmp_path_factory.mktemp("graph_edges")
    rng = np.random.default_rng(11)
    src = rng.integers(0, _NODES, _EDGES)
    dst = rng.integers(0, _NODES, _EDGES)
    # A few dense pockets, so the triangle and core counts are not all zero.
    pocket = rng.integers(0, 200, (4_000, 2))
    src[: len(pocket)], dst[: len(pocket)] = pocket[:, 0], pocket[:, 1]
    per = _EDGES // _FILES
    for i in range(_FILES):
        part = slice(i * per, (i + 1) * per)
        pq.write_table(pa.table({"src": src[part], "dst": dst[part]}), root / f"part-{i}.parquet")
    return str(root)


@pytest.fixture
def graph(edge_dir) -> bg.Graph:
    return bg.Graph.from_edges(bt.read.parquet(edge_dir))


def _spy_routes(monkeypatch) -> list[bool]:
    """Record every routing decision a terminal makes, as `test_inspection_distributed` does."""
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
    with option_context("distributed.mode", mode), warnings.catch_warnings():
        # A capped PageRank below warns; both modes hit the same cap by design.
        warnings.simplefilter("ignore", bg.ConvergenceWarning)
        return fn()


def _same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def _assert_same_by_key(local: dict, remote: dict, key: str) -> None:
    """Row-for-row equality keyed by `key`, so neither row order nor a lost row can hide."""
    assert list(local) == list(remote), "column names differ"
    lrows = dict(zip(local[key], zip(*local.values(), strict=True), strict=True))
    rrows = dict(zip(remote[key], zip(*remote.values(), strict=True), strict=True))
    assert len(lrows) == len(local[key]), "duplicate keys in the single-node result"
    assert set(lrows) == set(rrows)
    bad = [k for k in lrows if not all(map(_same, lrows[k], rrows[k]))]
    assert not bad, f"{len(bad)} rows differ, e.g. {bad[0]}: {lrows[bad[0]]} vs {rrows[bad[0]]}"


def test_mode_always_fans_out_over_several_workers(graph):
    """Positive control: across workers an unordered LIMIT keeps different groups.

    If this ever agrees, the pinned runs below compare one worker with one worker.
    """
    q = graph.edges.group_by("src").agg(n=bt.count()).limit(3)
    local = sorted(_under("never", lambda: q.to_pydict())["src"])
    remote = sorted(_under("always", lambda: q.to_pydict())["src"])
    assert len(local) == len(remote) == 3
    assert local != remote


@pytest.mark.parametrize(
    ("run", "key"),
    [
        (lambda g: bg.degree_distribution(g).to_pydict(), "degree"),
        (lambda g: bg.jaccard_similarity(g, _PAIRS).to_pydict(), "a"),
        (lambda g: bg.triangle_count(g).to_pydict(), "node"),
        (lambda g: bg.clustering_coefficient(g).to_pydict(), "node"),
        (lambda g: bg.k_core(g, 3).nodes().to_pydict(), "node"),
        (lambda g: bg.summarize(g).to_pydict(), "nodes"),
        (lambda g: {"t": [bg.transitivity(g)], "k": [0]}, "k"),
        (lambda g: bg.connected_components(g).to_pydict(), "node"),
        (lambda g: bg.pagerank(g, max_iterations=5).to_pydict(), "node"),
        (lambda g: bg.bfs(g, bt.from_pydict({"node": [0]})).to_pydict(), "node"),
    ],
    ids=[
        "degree_distribution",
        "jaccard_similarity",
        "triangle_count",
        "clustering_coefficient",
        "k_core",
        "summarize",
        "transitivity",
        "connected_components",
        "pagerank",
        "bfs",
    ],
)
def test_graph_algorithm_agrees_across_workers(graph, monkeypatch, run, key):
    routes = _spy_routes(monkeypatch)
    local = _under("never", lambda: run(graph))
    assert routes and not any(routes), "the single-node baseline reached Ray"
    routes.clear()
    remote = _under("always", lambda: run(graph))
    assert routes and all(routes), "a terminal under mode='always' stayed local"
    _assert_same_by_key(local, remote, key)


def test_the_fixture_has_the_structure_the_comparisons_need(graph):
    """Guards against a vacuous comparison: triangles and a 3-core must actually exist."""
    assert sum(bg.triangle_count(graph).to_pydict()["triangles"]) > 0
    assert bg.k_core(graph, 3).num_nodes() > 0
