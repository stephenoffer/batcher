"""Graph ML: sampling a graph a model can read, and building features from it.

Two paths, and they compose. Sampling bounds what a batch has to see, so a GNN layer's
cost is set by the sample rather than by the graph's worst node. Feature building turns
the graph into columns an ordinary tabular model can train on, which is a strong baseline
that needs no GNN at all.

    python examples/graph/graph_ml.py
"""

from __future__ import annotations

import batcher as bt
import batcher.graph as bg


def skewed_graph() -> bg.Graph:
    """A hub with many neighbours plus a sparse fringe: the shape sampling exists for."""
    src = ["hub"] * 8 + ["p1", "p2", "p3", "p4", "p5"]
    dst = [f"n{i}" for i in range(8)] + ["p2", "p3", "p4", "p5", "p1"]
    return bg.Graph.from_edges(bt.from_pydict({"src": src, "dst": dst}))


def bound_the_batch(g: bg.Graph) -> None:
    """`neighbor_sample` is GraphSAGE's fan-out cap, applied once per layer."""
    print("--- unsampled: the hub dominates the batch ---")
    before = g.edges.group_by("src").agg(n=bt.count()).sort("n", descending=True).to_pydict()
    print(before)
    assert max(before["n"]) == 8, "the hub is the worst node, and it is what sets the cost"

    print("--- sampled to at most 3 neighbours per node ---")
    sampled = bg.neighbor_sample(g, 3, seed=7)
    after = sampled.group_by("src").agg(n=bt.count()).sort("n", descending=True).to_pydict()
    print(after)
    # The whole purpose of a fan-out cap: no node may exceed it, and nothing under the cap
    # may be thinned. A sampler that simply took 3 edges overall would satisfy the first
    # half and quietly destroy the batch.
    assert max(after["n"]) == 3
    assert set(after["src"]) == set(before["src"]), "every source must survive sampling"
    for src, n in zip(after["src"], after["n"], strict=True):
        assert n == min(3, dict(zip(before["src"], before["n"], strict=True))[src])

    print("--- and it is stable: the same seed gives the same sample ---")
    again = bg.neighbor_sample(g, 3, seed=7)
    stable = sampled.sort("src", "dst").to_pydict() == again.sort("src", "dst").to_pydict()
    print(stable)
    assert stable, "a seeded sampler that is not reproducible cannot be debugged"

    print("--- a different seed gives a different one ---")
    other = bg.neighbor_sample(g, 3, seed=8)
    differs = sampled.sort("src", "dst").to_pydict() != other.sort("src", "dst").to_pydict()
    print(differs)
    # Without this the seed could be ignored entirely and the line above would still pass.
    assert differs, "the seed must actually reach the sampler"


def walk_the_graph(g: bg.Graph) -> None:
    """Random walks are how DeepWalk and node2vec turn a graph into sequences."""
    starts = bt.from_pydict({"node": ["p1", "p1", "p2", "hub"]})
    walks = bg.random_walks(g, starts, 4, seed=3)
    print("--- four walks of up to four steps ---")
    by_walk = walks.sort("walk", "step").to_pydict()
    current, path = None, []
    for w, node in zip(by_walk["walk"], by_walk["node"], strict=True):
        if w != current:
            if path:
                print(f"  walk {current}: {' -> '.join(path)}")
            current, path = w, []
        path.append(str(node))
    if path:
        print(f"  walk {current}: {' -> '.join(path)}")

    # Each walk starts at the node it was seeded with and steps are consecutive from 0.
    seeds = starts.to_pydict()["node"]
    for w in set(by_walk["walk"]):
        steps = [s for s, ww in zip(by_walk["step"], by_walk["walk"], strict=True) if ww == w]
        nodes = [n for n, ww in zip(by_walk["node"], by_walk["walk"], strict=True) if ww == w]
        assert steps == list(range(len(steps))), f"walk {w} has a gap in its steps"
        assert nodes[0] == seeds[w], f"walk {w} must start at its seed"

    print("--- a walk that reaches a dead end stops there rather than being padded ---")
    dead_end = bg.Graph.from_edges(bt.from_pydict({"src": ["a"], "dst": ["b"]}))
    short = bg.random_walks(dead_end, bt.from_pydict({"node": ["a"]}), 5)
    stopped = short.sort("step").to_pydict()
    print(stopped)
    # Asked for 5 steps, but `b` has no out-edge. Padding with repeats or nulls would be the
    # easy implementation and would silently teach a model that `b` links to itself.
    assert stopped["node"] == ["a", "b"], "the walk stops at the dead end, unpadded"


def split_for_link_prediction(g: bg.Graph) -> None:
    """Hold out edges, not nodes: removing nodes changes the task."""
    train = bg.edge_sample(g, 0.7, seed=11)
    print("--- train/test edge split ---")
    counts = {"all": g.num_edges(), "train": train.num_edges()}
    print(counts)
    print("nodes survive the split:", train.num_nodes(), "of", g.num_nodes())
    # A held-out split must remove edges and only edges. It must not be everything (no test
    # set) or nothing (no training set), and it must not silently drop nodes as a side
    # effect — which is the mistake this function's docstring warns about, because removing
    # a node changes the task rather than holding data back from it.
    assert 0 < counts["train"] < counts["all"]
    assert train.num_nodes() <= g.num_nodes()

    def edge_set(graph: bg.Graph) -> set[tuple[str, str]]:
        columns = graph.edges.select("src", "dst").to_pydict()
        return set(zip(columns["src"], columns["dst"], strict=True))

    train_edges, all_edges = edge_set(train), edge_set(g)
    assert train_edges <= all_edges, "the split may only remove edges, never invent them"


def build_features(g: bg.Graph) -> None:
    """Two ways to get columns out of a graph, both of which feed an ordinary model."""
    print("--- structural features need no node attributes at all ---")
    feats = bg.structural_features(g).sort("pagerank", descending=True).limit(4)
    got = feats.to_pydict()
    # PageRank is a distribution, and these are the top 4 of it, so they must be ordered.
    assert got["pagerank"] == sorted(got["pagerank"], reverse=True)
    assert all(0.0 <= v <= 1.0 for v in got["pagerank"])
    assert all(0.0 <= c <= 1.0 for c in got["clustering"])
    for i, node in enumerate(got["node"]):
        print(
            f"  {node:>4}  deg={got['degree'][i]}  tri={got['triangles'][i]}"
            f"  clust={got['clustering'][i]:.3f}  pr={got['pagerank'][i]:.4f}"
        )

    print("--- message passing: each node summarizes its neighbours ---")
    attrs = bt.from_pydict(
        {
            "node": ["hub", "p1", "p2", "p3", "p4", "p5"] + [f"n{i}" for i in range(8)],
            # Distinct scores on the fringe, so mean, sum and max are three different
            # numbers. With every neighbour scoring 1.0 the three agree, and the section
            # would print the same value three times while appearing to contrast them.
            "score": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0] + [float(i) for i in range(8)],
        }
    )
    # Direction first, because it is the trap. `aggregate_neighbors` reads a node's *in*
    # edges, and `hub` has eight out-edges and no in-edges — so on the directed graph its
    # summary is null, not zero. This example used to print exactly that for all three
    # aggregations and read as a demonstration.
    directed = bg.aggregate_neighbors(g, attrs, ["score"], how="sum")
    hub_directed = directed.filter(bt.col("node") == "hub").to_pydict()["score_sum"][0]
    print(f"  hub has no in-edges, so the directed summary is: {hub_directed}")
    assert hub_directed is None, "no in-neighbours means no summary, and null is not 0.0"

    neighbours = g.to_undirected()
    expected = {"mean": 3.5, "sum": 28.0, "max": 7.0}
    for how in ("mean", "sum", "max"):
        one = bg.aggregate_neighbors(neighbours, attrs, ["score"], how=how)
        row = one.filter(bt.col("node") == "hub").to_pydict()
        value = row[f"score_{how}"][0]
        print(f"  hub's neighbours by {how}: {value}")
        # The fringe scores 0..7: sum 28, mean 3.5, max 7.
        assert value == expected[how], f"{how}: expected {expected[how]}, got {value}"

    print("--- stacking hops gives a model a multi-scale view ---")
    stacked = bg.propagate_features(g, attrs, ["score"], 3)
    hops = stacked.filter(bt.col("node").is_in(["p1", "p2", "p3"])).sort("node").to_pydict()
    print(hops)
    # p1 -> p2 -> p3 is a chain, and only p1 carries a score, so the 1.0 must appear one hop
    # further along at each level. A propagation that leaked across levels, or that failed to
    # advance, would print a plausible table of zeros.
    assert hops["node"] == ["p1", "p2", "p3"]
    assert hops["score"][0] == 1.0, "p1 holds the signal at hop 0"
    assert hops["score_hop1"][1] == 1.0, "and p2 sees it one hop out"
    assert hops["score_hop2"][2] == 1.0, "and p3 two hops out"

    print("--- weighting by edge weight, where that is meaningful ---")
    weighted_graph = bg.Graph.from_edges(
        bt.from_pydict({"src": ["a", "a"], "dst": ["c", "c"], "w": [10.0, 1.0]}),
        weight="w",
    )
    vals = bt.from_pydict({"node": ["a", "c"], "v": [2.0, 0.0]})
    weighted = (
        bg.aggregate_neighbors(weighted_graph, vals, ["v"], how="sum", weighted=True)
        .sort("node")
        .to_pydict()
    )
    print(weighted)
    # Two a->c edges of weight 10 and 1 over a's value of 2.0: 2*10 + 2*1 = 22. Unweighted
    # the same call would answer 4.0, so this number is what proves `weighted=True` is read.
    assert dict(zip(weighted["node"], weighted["v_sum"], strict=True))["c"] == 22.0


def focus_on_one_node(g: bg.Graph) -> None:
    """An ego network is the unit a graph-ML batch is usually built from."""
    print("--- everything within one and two hops of p1 ---")
    egos = {}
    for radius in (1, 2):
        ego = bg.ego_network(g, "p1", radius)
        egos[radius] = sorted(ego.nodes().to_pydict()["node"])
        print(f"  radius {radius}: {egos[radius]}")
    # A wider radius can only ever add nodes, and both must contain the ego itself.
    assert "p1" in egos[1] and set(egos[1]) <= set(egos[2])
    assert len(egos[2]) > len(egos[1]), "radius 2 must reach further than radius 1"

    print("--- a deterministic node sample, stable as the graph grows ---")
    sample = sorted(bg.node_sample(g, 0.4, seed=5).to_pydict()["node"])
    print(sample)
    assert sample == sorted(bg.node_sample(g, 0.4, seed=5).to_pydict()["node"]), "deterministic"
    assert 0 < len(sample) < g.num_nodes(), "a 40% sample is neither empty nor everything"


def main() -> None:
    g = skewed_graph().cache()
    bound_the_batch(g)
    walk_the_graph(g)
    split_for_link_prediction(g)
    build_features(g)
    focus_on_one_node(g)


if __name__ == "__main__":
    main()
