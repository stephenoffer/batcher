"""Graph algorithms against hand-checkable answers and their defining invariants.

Every algorithm here runs over a graph small enough to verify by hand, and each test
pins either a value that can be worked out on paper or a property the algorithm is
*defined* by. Two of the tests exist because the bug they describe was real and passed
every other check.
"""

from __future__ import annotations

import pytest

import batcher as bt
import batcher.graph as bg

pytestmark = pytest.mark.unit


def star() -> bg.Graph:
    """Four leaves pointing at one centre, which has no outgoing edges."""
    return bg.Graph.from_edges(bt.from_pydict({"src": [1, 2, 3, 4], "dst": [0, 0, 0, 0]}))


def two_triangles() -> bg.Graph:
    """Two disjoint triangles: the textbook two-community graph."""
    return bg.Graph.from_edges(
        bt.from_pydict({"src": [1, 2, 1, 4, 5, 4], "dst": [2, 3, 3, 5, 6, 6]})
    )


# --- degree ------------------------------------------------------------------------


def test_degree_obeys_the_handshake_identity():
    """Sum of degrees == 2 * edges, which is what makes a self-loop count 2."""
    g = bg.Graph.from_edges(bt.from_pydict({"src": [1, 2, 3], "dst": [2, 3, 3]}))
    total = sum(bg.degree(g).to_pydict()["degree"])
    assert total == 2 * g.num_edges()


def test_a_sink_appears_with_zero_out_degree_rather_than_being_absent():
    """A left join from the node set is what makes the denominator right downstream."""
    got = bg.out_degree(star()).sort("node").to_pydict()
    assert got == {"node": [0, 1, 2, 3, 4], "out_degree": [0, 1, 1, 1, 1]}


def test_isolated_nodes_are_invisible_without_a_node_table():
    g = bg.Graph.from_edges(bt.from_pydict({"src": [1], "dst": [2]}))
    assert bg.isolated_nodes(g).count() == 0
    with_all = g.with_nodes(bt.from_pydict({"node": [1, 2, 3]}))
    assert with_all.num_nodes() == 3
    assert bg.isolated_nodes(with_all).to_pydict()["node"] == [3]
    # And the average moves, which is the whole reason it matters.
    assert bg.average_degree(g) == 1.0
    assert bg.average_degree(with_all) == pytest.approx(2 / 3)


def test_degree_double_counts_on_a_symmetrized_graph():
    """The trap `k_core` and `modularity` both had to be fixed for.

    After `to_undirected` every edge is present in both directions, so counting both
    endpoints reports twice the neighbour count. `out_degree` is the neighbour count.
    """
    g = bg.Graph.from_edges(bt.from_pydict({"src": [1], "dst": [2]})).to_undirected()
    assert bg.degree(g).sort("node").to_pydict()["degree"] == [2, 2]
    assert bg.out_degree(g).sort("node").to_pydict()["out_degree"] == [1, 1]


# --- centrality --------------------------------------------------------------------


def test_pagerank_sums_to_one_even_with_a_dangling_node():
    """The centre of a star has no outgoing edge, so its mass has nowhere to go.

    Without redistributing it the ranks silently stop summing to 1 while still looking
    plausible, which is the classic way a hand-rolled PageRank comes out wrong.
    """
    got = bg.pagerank(star()).to_pydict()
    assert sum(got["pagerank"]) == pytest.approx(1.0, abs=1e-9)


def test_pagerank_ranks_the_hub_first_and_the_leaves_equally():
    got = bg.pagerank(star()).sort("pagerank", descending=True).to_pydict()
    assert got["node"][0] == 0
    leaves = got["pagerank"][1:]
    assert all(v == pytest.approx(leaves[0]) for v in leaves)


def test_personalized_pagerank_decays_away_from_its_seed():
    chain = bg.Graph.from_edges(bt.from_pydict({"src": [1, 2, 3], "dst": [2, 3, 4]}))
    got = bg.personalized_pagerank(chain, bt.from_pydict({"node": [1]})).sort("node")
    ranks = got.to_pydict()["pagerank"]
    assert sum(ranks) == pytest.approx(1.0, abs=1e-9)
    assert ranks == sorted(ranks, reverse=True), "closer to the seed must score higher"


def test_personalized_pagerank_refuses_a_seed_set_outside_the_graph():
    g = bg.Graph.from_edges(bt.from_pydict({"src": [1], "dst": [2]}))
    with pytest.raises(Exception, match="teleport"):
        bg.personalized_pagerank(g, bt.from_pydict({"node": [99]}))


def test_eigenvector_centrality_is_uniform_on_a_cycle():
    """Every node of a cycle is identical, so the leading eigenvector is flat."""
    cycle = bg.Graph.from_edges(bt.from_pydict({"src": [0, 1, 2], "dst": [1, 2, 0]}))
    got = bg.eigenvector_centrality(cycle).to_pydict()["eigenvector_centrality"]
    assert all(v == pytest.approx(3**-0.5, abs=1e-6) for v in got)


def test_hits_separates_hubs_from_authorities():
    got = bg.hits(star()).sort("node").to_pydict()
    assert got["authority"][0] == pytest.approx(1.0)
    assert got["hub"][0] == pytest.approx(0.0)
    assert all(h == pytest.approx(4**-0.5) for h in got["hub"][1:])


def test_damping_outside_its_range_is_refused():
    with pytest.raises(Exception, match="damping"):
        bg.pagerank(star(), damping=1.0)


# --- components --------------------------------------------------------------------


def test_components_are_labelled_by_their_smallest_node():
    """Stable labels, so a component's id is comparable across runs."""
    g = bg.Graph.from_edges(bt.from_pydict({"src": [1, 2, 8], "dst": [2, 3, 9]}))
    assert bg.connected_components(g).sort("node").to_pydict() == {
        "node": [1, 2, 3, 8, 9],
        "component": [1, 1, 1, 8, 8],
    }
    assert not bg.is_connected(g)


def test_largest_component_restricts_the_declared_nodes_too():
    """The bug this pins: the edges were right and every per-node denominator was wrong.

    `subgraph` used to keep the original declared node table, so `largest_component`
    returned a graph whose `nodes()` still held every node in the input.
    """
    g = bg.Graph.from_edges(bt.from_pydict({"src": [1, 2, 8], "dst": [2, 3, 9]})).with_nodes(
        bt.from_pydict({"node": [1, 2, 3, 8, 9, 77]})
    )
    biggest = bg.largest_component(g)
    assert sorted(biggest.nodes().to_pydict()["node"]) == [1, 2, 3]


def test_k_core_peels_cascading_and_terminates():
    """A triangle with a pendant: the pendant goes at k=2, the triangle at k=3.

    This also pins that peeling terminates. Each round's subgraph is a join over the
    previous round's plan, and without cutting the plan the nesting overflowed the
    engine's plan walk and crashed the process rather than raising.
    """
    g = bg.Graph.from_edges(bt.from_pydict({"src": [1, 2, 3, 1], "dst": [2, 3, 1, 9]}))
    assert sorted(bg.k_core(g, 1).nodes().to_pydict()["node"]) == [1, 2, 3, 9]
    assert sorted(bg.k_core(g, 2).nodes().to_pydict()["node"]) == [1, 2, 3]
    assert bg.k_core(g, 3).num_edges() == 0


# --- triangles and communities -----------------------------------------------------


def test_each_triangle_is_counted_exactly_once():
    """Orienting edges by node id is what stops six orderings becoming six triangles."""
    g = bg.Graph.from_edges(bt.from_pydict({"src": [1, 2, 1, 3], "dst": [2, 3, 3, 4]}))
    assert bg.triangles(g).to_pydict() == {"a": [1], "b": [2], "c": [3]}
    assert bg.triangle_count(g).sort("node").to_pydict()["triangles"] == [1, 1, 1, 0]


def test_clustering_and_transitivity_answer_different_questions():
    g = bg.Graph.from_edges(bt.from_pydict({"src": [1, 2, 1, 3], "dst": [2, 3, 3, 4]}))
    # Node 3 has three neighbours and one connected pair among them: 2/(3*2) = 1/3.
    assert bg.clustering_coefficient(g).sort("node").to_pydict()["clustering"] == pytest.approx(
        [1.0, 1.0, 1 / 3, 0.0]
    )
    assert bg.average_clustering(g) == pytest.approx(7 / 12)
    # Globally: 3 * 1 triangle / 5 connected triples.
    assert bg.transitivity(g) == pytest.approx(0.6)


def test_modularity_scores_a_perfect_split_and_a_trivial_one_correctly():
    """The values are exact for this graph, and the bug they caught was a factor of two.

    `weighted_degree` on a symmetrized edge table is twice the real strength, and
    modularity squares that term, so the error drove a perfect split to -1.0.
    """
    g = two_triangles()
    found = bg.label_propagation(g)
    assert bg.modularity(g, found) == pytest.approx(0.5)
    one_community = found.select("node", community=bt.lit(0))
    assert bg.modularity(g, one_community) == pytest.approx(0.0)
    singletons = found.select("node", community=bt.col("node"))
    assert bg.modularity(g, singletons) < 0.0


def test_label_propagation_finds_the_two_triangles():
    got = bg.label_propagation(two_triangles()).sort("node").to_pydict()["community"]
    assert got[:3] == [got[0]] * 3
    assert got[3:] == [got[3]] * 3
    assert got[0] != got[3]


# --- distances ---------------------------------------------------------------------


def test_bfs_records_the_shortest_hop_distance():
    path = bg.Graph.from_edges(bt.from_pydict({"src": [1, 2, 3], "dst": [2, 3, 4]}))
    assert bg.bfs(path, bt.from_pydict({"node": [1]})).sort("node").to_pydict() == {
        "node": [1, 2, 3, 4],
        "depth": [0, 1, 2, 3],
    }
    assert bg.diameter_estimate(path, bt.from_pydict({"node": [1]})) == 3


def test_weighted_distance_finds_the_cheap_detour_not_the_short_hop():
    e = bt.from_pydict({"src": [1, 1, 2], "dst": [2, 3, 3], "w": [1.0, 9.0, 1.0]})
    g = bg.Graph.from_edges(e, weight="w")
    got = bg.shortest_path_lengths(g, bt.from_pydict({"node": [1]})).sort("node")
    assert got.to_pydict()["distance"] == [0.0, 1.0, 2.0]


def test_a_negative_weight_is_refused_rather_than_diverging():
    e = bt.from_pydict({"src": [1], "dst": [2], "w": [-1.0]})
    g = bg.Graph.from_edges(e, weight="w")
    with pytest.raises(Exception, match="negative"):
        bg.shortest_path_lengths(g, bt.from_pydict({"node": [1]}))


# --- similarity --------------------------------------------------------------------


def test_the_link_prediction_scores_agree_on_a_hand_computable_pair():
    """1 and 3 share exactly node 2, which has degree 2."""
    g = bg.Graph.from_edges(bt.from_pydict({"src": [1, 3], "dst": [2, 2]}))
    pairs = bt.from_pydict({"a": [1], "b": [3]})
    assert bg.common_neighbors(g, pairs).to_pydict()["common_neighbors"] == [1]
    assert bg.adamic_adar(g, pairs).to_pydict()["adamic_adar"] == pytest.approx(
        [1 / 0.693147], rel=1e-4
    )
    assert bg.resource_allocation(g, pairs).to_pydict()["resource_allocation"] == [0.5]
    # Jaccard: one shared neighbour, one distinct neighbour between them.
    assert bg.jaccard_similarity(g, pairs).to_pydict()["jaccard"] == [1.0]
    assert bg.preferential_attachment(g, pairs).to_pydict()["preferential_attachment"] == [1]


def test_candidate_pairs_finds_only_pairs_that_share_a_neighbour():
    g = bg.Graph.from_edges(bt.from_pydict({"src": [1, 3, 5], "dst": [2, 2, 6]}))
    assert bg.candidate_pairs(g).to_pydict() == {"a": [1], "b": [3], "common": [1]}


# --- sampling ----------------------------------------------------------------------


def test_neighbor_sample_bounds_the_fan_out_and_is_reproducible():
    e = bt.from_pydict({"src": [1] * 5 + [2], "dst": [2, 3, 4, 5, 6, 3]})
    g = bg.Graph.from_edges(e)
    once = bg.neighbor_sample(g, 2, seed=4).sort("src", "dst").to_pydict()
    twice = bg.neighbor_sample(g, 2, seed=4).sort("src", "dst").to_pydict()
    assert once == twice, "the same seed must give the same sample"
    per_node = bg.neighbor_sample(g, 2, seed=4).group_by("src").agg(n=bt.count())
    assert max(per_node.to_pydict()["n"]) <= 2


def test_a_walk_follows_real_edges_and_stops_at_a_dead_end():
    cycle = bg.Graph.from_edges(bt.from_pydict({"src": [1, 2, 3], "dst": [2, 3, 1]}))
    assert bg.random_walks(cycle, bt.from_pydict({"node": [1]}), 3).sort("step").to_pydict()[
        "node"
    ] == [1, 2, 3, 1]
    dead = bg.Graph.from_edges(bt.from_pydict({"src": [1], "dst": [2]}))
    short = bg.random_walks(dead, bt.from_pydict({"node": [1]}), 5)
    assert short.sort("step").to_pydict()["node"] == [1, 2], "not padded past the dead end"


def test_ego_network_grows_with_the_radius():
    g = bg.Graph.from_edges(bt.from_pydict({"src": [1, 2, 3], "dst": [2, 3, 4]}))
    assert sorted(bg.ego_network(g, 1, 1).nodes().to_pydict()["node"]) == [1, 2]
    assert sorted(bg.ego_network(g, 1, 2).nodes().to_pydict()["node"]) == [1, 2, 3]


# --- features and summary ----------------------------------------------------------


def test_message_passing_averages_the_neighbours_and_nulls_the_unreached():
    g = bg.Graph.from_edges(bt.from_pydict({"src": [1, 2], "dst": [3, 3]}))
    feats = bt.from_pydict({"node": [1, 2, 3], "x": [10.0, 20.0, 0.0]})
    got = bg.aggregate_neighbors(g, feats, ["x"]).sort("node").to_pydict()
    assert got["x_mean"] == [None, None, 15.0], "no neighbours is null, not zero"


def test_a_weighted_maximum_is_refused_because_it_is_not_a_maximum():
    g = bg.Graph.from_edges(bt.from_pydict({"src": [1], "dst": [2]}))
    feats = bt.from_pydict({"node": [1, 2], "x": [1.0, 0.0]})
    with pytest.raises(Exception, match="weighted"):
        bg.aggregate_neighbors(g, feats, ["x"], how="max", weighted=True)


def test_propagated_features_carry_signal_one_hop_further_each_round():
    g = bg.Graph.from_edges(bt.from_pydict({"src": [1, 2], "dst": [2, 3]}))
    feats = bt.from_pydict({"node": [1, 2, 3], "x": [1.0, 0.0, 0.0]})
    got = bg.propagate_features(g, feats, ["x"], 2).sort("node").to_pydict()
    assert got["x_hop1"] == [None, 1.0, 0.0]
    assert got["x_hop2"] == [None, 0.0, 1.0]


def test_summary_statistics_match_hand_computation():
    g = bg.Graph.from_edges(bt.from_pydict({"src": [1, 2, 3], "dst": [2, 3, 1]}))
    row = bg.summarize(g).to_pydict()
    assert (row["nodes"][0], row["edges"][0]) == (3, 3)
    # 3 of the 6 possible ordered pairs exist.
    assert row["density"][0] == pytest.approx(0.5)
    assert bg.reciprocity(g) == 0.0
    # A star is maximally disassortative: the hub attaches only to leaves.
    assert bg.assortativity(star()) == pytest.approx(-1.0)


# --- constructors: building a graph from data that is not an edge list --------------


def test_knn_graph_links_each_vector_to_its_nearest():
    vecs = bt.from_pydict({"node": ["a", "b", "c"], "vector": [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]})
    got = bg.knn_graph(vecs, 1).edges.sort("src").to_pydict()
    assert got["dst"] == ["b", "a", "b"]
    # A distance metric must rank the same way a similarity metric does, which is the
    # thing a single ascending window ordering has to get right for both.
    assert bg.knn_graph(vecs, 1, metric="euclidean").edges.sort("src").to_pydict()["dst"] == [
        "b",
        "a",
        "b",
    ]


def test_threshold_graph_keeps_a_row_that_matched_nothing():
    """Entity resolution needs the unmatched rows back, as singleton clusters."""
    vecs = bt.from_pydict(
        {"node": ["a", "b", "c"], "vector": [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]]}
    )
    g = bg.threshold_graph(vecs, 0.9)
    assert bg.connected_components(g).sort("node").to_pydict() == {
        "node": ["a", "b", "c"],
        "component": ["a", "a", "c"],
    }


def test_spatial_graph_uses_a_geodesic_radius_in_metres():
    sites = bt.from_pydict(
        {
            "node": ["ferry", "pier", "opera"],
            "geometry": [
                "POINT(-122.3937 37.7955)",
                "POINT(-122.3930 37.7960)",
                "POINT(151.2153 -33.8568)",
            ],
        }
    )
    near = bg.spatial_graph(sites, 200.0)
    assert sorted(near.nodes().to_pydict()["node"]) == ["ferry", "opera", "pier"]
    assert near.num_edges() == 2, "one undirected edge, materialized both ways"
    # The two San Francisco sites are about 83 m apart, so a tighter radius drops them.
    assert bg.spatial_graph(sites, 50.0).num_edges() == 0


def test_co_occurrence_graph_counts_shared_groups_and_prunes():
    log = bt.from_pydict(
        {
            "user": ["u1", "u1", "u2", "u2", "u3"],
            "item": ["bread", "jam", "bread", "jam", "shovel"],
        }
    )
    g = bg.co_occurrence_graph(log, min_count=2)
    assert g.edges.sort("src").to_pydict() == {
        "src": ["bread", "jam"],
        "dst": ["jam", "bread"],
        "weight": [2.0, 2.0],
    }
    assert sorted(g.nodes().to_pydict()["node"]) == ["bread", "jam", "shovel"]
    # Raising the threshold past the observed count leaves the nodes and no edges.
    assert bg.co_occurrence_graph(log, min_count=3).num_edges() == 0


def test_a_blocking_key_restricts_comparison_to_its_own_block():
    """Blocking is what makes these constructors affordable, and it is exact per block."""
    vecs = bt.from_pydict(
        {
            "node": ["a", "b", "c"],
            "vector": [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            "day": ["mon", "mon", "tue"],
        }
    )
    unblocked = bg.threshold_graph(vecs, 0.99)
    blocked = bg.threshold_graph(vecs, 0.99, block="day")
    assert unblocked.num_edges() > blocked.num_edges()
    assert blocked.edges.to_pydict()["src"] == ["a", "b"], "no cross-day pair"


def test_an_unknown_metric_is_refused():
    vecs = bt.from_pydict({"node": ["a"], "vector": [[1.0]]})
    with pytest.raises(Exception, match="metric"):
        bg.knn_graph(vecs, 1, metric="jaccard")


# --- direction-respecting components and dependency order --------------------------


def test_strong_components_respect_direction_where_weak_ones_do_not():
    """A cycle with a tail: one weak component, two strong ones."""
    g = bg.Graph.from_edges(bt.from_pydict({"src": [1, 2, 3, 3], "dst": [2, 3, 1, 4]}))
    assert bg.strongly_connected_components(g).sort("node").to_pydict() == {
        "node": [1, 2, 3, 4],
        "component": [3, 3, 3, 4],
    }
    assert bg.connected_components(g).sort("node").to_pydict()["component"] == [1, 1, 1, 1]


def test_a_chain_has_one_weak_component_and_a_strong_one_per_node():
    g = bg.Graph.from_edges(bt.from_pydict({"src": [1, 2], "dst": [2, 3]}))
    strong = bg.strongly_connected_components(g).sort("node").to_pydict()["component"]
    assert strong == [1, 2, 3], "nothing gets back, so nothing is mutually reachable"


def test_two_cycles_joined_one_way_stay_two_strong_components():
    g = bg.Graph.from_edges(bt.from_pydict({"src": [1, 2, 3, 4, 4], "dst": [2, 1, 4, 3, 1]}))
    got = bg.strongly_connected_components(g).sort("node").to_pydict()["component"]
    assert got[0] == got[1] and got[2] == got[3] and got[0] != got[2]


def test_topological_order_groups_independent_nodes_into_one_level():
    """Levels rather than a flat sequence, so a scheduler can run a level at once."""
    e = bt.from_pydict({"src": ["a", "a", "b", "c"], "dst": ["b", "c", "d", "d"]})
    assert bg.topological_order(bg.Graph.from_edges(e)).sort("node").to_pydict() == {
        "node": ["a", "b", "c", "d"],
        "level": [0, 1, 1, 2],
    }


def test_a_cycle_is_absent_from_the_order_which_is_what_makes_is_dag_work():
    acyclic = bg.Graph.from_edges(bt.from_pydict({"src": ["a", "b"], "dst": ["b", "c"]}))
    cyclic = bg.Graph.from_edges(bt.from_pydict({"src": ["a", "b", "c"], "dst": ["b", "c", "a"]}))
    assert bg.is_dag(acyclic)
    assert not bg.is_dag(cyclic)
    assert bg.topological_order(cyclic).count() == 0


def test_nodes_in_cycles_names_the_nodes_blocking_the_order():
    """A boolean says the graph is broken; this says which nodes to look at."""
    e = bt.from_pydict({"src": ["a", "b", "c", "c"], "dst": ["b", "c", "b", "d"]})
    g = bg.Graph.from_edges(e)
    assert sorted(bg.nodes_in_cycles(g).to_pydict()["node"]) == ["b", "c", "d"]
    assert (
        bg.nodes_in_cycles(
            bg.Graph.from_edges(bt.from_pydict({"src": ["a"], "dst": ["b"]}))
        ).count()
        == 0
    )


def test_betweenness_finds_the_bridge_not_the_hub():
    """Everything from the left must pass through 'c', then 'd'; endpoints score zero."""
    e = bt.from_pydict({"src": ["a", "b", "c", "d"], "dst": ["c", "c", "d", "e"]})
    g = bg.Graph.from_edges(e)
    got = bg.betweenness_centrality(g, bt.from_pydict({"node": ["a", "b"]}))
    assert got.sort("betweenness", descending=True).to_pydict() == {
        "node": ["c", "d", "a", "b", "e"],
        "betweenness": [4.0, 2.0, 0.0, 0.0, 0.0],
    }


def test_betweenness_refuses_sources_outside_the_graph():
    g = bg.Graph.from_edges(bt.from_pydict({"src": ["a"], "dst": ["b"]}))
    with pytest.raises(Exception, match="shortest paths"):
        bg.betweenness_centrality(g, bt.from_pydict({"node": ["zzz"]}))


def test_betweenness_splits_credit_between_tied_shortest_paths():
    """Two equally short routes each carry half the traffic, which is what sigma is for.

    A version that tracked only distance and not the *number* of shortest paths would
    give both intermediates full credit and rank them above a node that really does carry
    everything.
    """
    # a reaches d via b or via c, both in two hops; e sits on the only route to f.
    e = bt.from_pydict(
        {"src": ["a", "a", "b", "c", "a", "e"], "dst": ["b", "c", "d", "d", "e", "f"]}
    )
    got = bg.betweenness_centrality(
        bg.Graph.from_edges(e), bt.from_pydict({"node": ["a"]})
    ).to_pydict()
    scores = dict(zip(got["node"], got["betweenness"], strict=True))
    assert scores["b"] == pytest.approx(0.5)
    assert scores["c"] == pytest.approx(0.5)
    assert scores["e"] == pytest.approx(1.0), "the sole route carries the whole path"


def _ops(ds: bt.Dataset) -> set[str]:
    """Every `op` tag in a dataset's lowered IR, at any depth."""
    seen: set[str] = set()

    def walk(node: object) -> None:
        if isinstance(node, dict):
            if isinstance(node.get("op"), str):
                seen.add(node["op"])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(ds._plan.to_ir())
    return seen


@pytest.mark.parametrize("fn", [bg.degree, bg.out_degree, bg.in_degree, bg.weighted_degree])
def test_degree_stays_join_free_because_a_join_here_cannot_distribute(fn):
    """The degree plans must stay `union -> group_by`, with no join anywhere.

    These once restored their zero rows with a left join from `nodes()`, making the plan
    `union -> distinct  LEFT JOIN  union -> group_by`. That shape has no distributed path:
    `collect(distributed=True)` raised `PlanError` rather than running, so degree was a
    single-node-only algorithm on exactly the graphs big enough to need a cluster.

    A join reappearing here is that bug returning, and it is invisible to every other test
    in this file because all of them run single-node, where the join is merely slower.
    """
    e = bt.from_pydict({"src": [1, 1, 2], "dst": [2, 3, 3], "w": [1.0, 2.0, 3.0]})
    g = bg.Graph.from_edges(e, weight="w").with_nodes(bt.from_pydict({"node": [1, 2, 3, 9]}))
    ops = _ops(fn(g))
    assert not {o for o in ops if "join" in o}, f"{fn.__name__} plan has a join: {sorted(ops)}"


def test_degree_keeps_the_nodes_that_have_no_edge_on_the_counted_side():
    """The zero rows the union arm exists to produce, on a sink and an isolated node."""
    e = bt.from_pydict({"src": [1, 1], "dst": [2, 3]})
    g = bg.Graph.from_edges(e).with_nodes(bt.from_pydict({"node": [1, 2, 3, 9]}))
    out = dict(zip(*(bg.out_degree(g).to_pydict()[k] for k in ("node", "out_degree")), strict=True))
    assert out == {1: 2, 2: 0, 3: 0, 9: 0}, "sinks and the isolated node must survive with 0"
    assert bg.isolated_nodes(g).to_pydict()["node"] == [9]
    assert sum(bg.degree(g).to_pydict()["degree"]) == 2 * 2, "handshake identity"
