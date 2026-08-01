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
