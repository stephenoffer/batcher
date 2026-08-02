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
    print(g.edges.group_by("src").agg(n=bt.count()).sort("n", descending=True).to_pydict())

    print("--- sampled to at most 3 neighbours per node ---")
    sampled = bg.neighbor_sample(g, 3, seed=7)
    print(sampled.group_by("src").agg(n=bt.count()).sort("n", descending=True).to_pydict())

    print("--- and it is stable: the same seed gives the same sample ---")
    again = bg.neighbor_sample(g, 3, seed=7)
    print(sampled.sort("src", "dst").to_pydict() == again.sort("src", "dst").to_pydict())

    print("--- a different seed gives a different one ---")
    other = bg.neighbor_sample(g, 3, seed=8)
    print(sampled.sort("src", "dst").to_pydict() != other.sort("src", "dst").to_pydict())


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

    print("--- a walk that reaches a dead end stops there rather than being padded ---")
    dead_end = bg.Graph.from_edges(bt.from_pydict({"src": ["a"], "dst": ["b"]}))
    short = bg.random_walks(dead_end, bt.from_pydict({"node": ["a"]}), 5)
    print(short.sort("step").to_pydict())


def split_for_link_prediction(g: bg.Graph) -> None:
    """Hold out edges, not nodes: removing nodes changes the task."""
    train = bg.edge_sample(g, 0.7, seed=11)
    print("--- train/test edge split ---")
    print({"all": g.num_edges(), "train": train.num_edges()})
    print("nodes survive the split:", train.num_nodes(), "of", g.num_nodes())


def build_features(g: bg.Graph) -> None:
    """Two ways to get columns out of a graph, both of which feed an ordinary model."""
    print("--- structural features need no node attributes at all ---")
    feats = bg.structural_features(g).sort("pagerank", descending=True).limit(4)
    got = feats.to_pydict()
    for i, node in enumerate(got["node"]):
        print(
            f"  {node:>4}  deg={got['degree'][i]}  tri={got['triangles'][i]}"
            f"  clust={got['clustering'][i]:.3f}  pr={got['pagerank'][i]:.4f}"
        )

    print("--- message passing: each node summarizes its neighbours ---")
    attrs = bt.from_pydict(
        {
            "node": ["hub", "p1", "p2", "p3", "p4", "p5"] + [f"n{i}" for i in range(8)],
            "score": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0] + [1.0] * 8,
        }
    )
    for how in ("mean", "sum", "max"):
        one = bg.aggregate_neighbors(g, attrs, ["score"], how=how)
        row = one.filter(bt.col("node") == "hub").to_pydict()
        print(f"  hub's neighbours by {how}: {row[f'score_{how}'][0]}")

    print("--- stacking hops gives a model a multi-scale view ---")
    stacked = bg.propagate_features(g, attrs, ["score"], 3)
    print(stacked.filter(bt.col("node").is_in(["p1", "p2", "p3"])).sort("node").to_pydict())

    print("--- weighting by edge weight, where that is meaningful ---")
    weighted_graph = bg.Graph.from_edges(
        bt.from_pydict({"src": ["a", "a"], "dst": ["c", "c"], "w": [10.0, 1.0]}),
        weight="w",
    )
    vals = bt.from_pydict({"node": ["a", "c"], "v": [2.0, 0.0]})
    print(
        bg.aggregate_neighbors(weighted_graph, vals, ["v"], how="sum", weighted=True)
        .sort("node")
        .to_pydict()
    )


def focus_on_one_node(g: bg.Graph) -> None:
    """An ego network is the unit a graph-ML batch is usually built from."""
    print("--- everything within one and two hops of p1 ---")
    for radius in (1, 2):
        ego = bg.ego_network(g, "p1", radius)
        print(f"  radius {radius}: {sorted(ego.nodes().to_pydict()['node'])}")

    print("--- a deterministic node sample, stable as the graph grows ---")
    print(sorted(bg.node_sample(g, 0.4, seed=5).to_pydict()["node"]))


def main() -> None:
    g = skewed_graph().cache()
    bound_the_batch(g)
    walk_the_graph(g)
    split_for_link_prediction(g)
    build_features(g)
    focus_on_one_node(g)


if __name__ == "__main__":
    main()
