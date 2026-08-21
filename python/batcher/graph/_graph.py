"""The `Graph` handle: an edge table, plus what the algorithms need to know about it.

A graph here is not a new data structure. It is a `Dataset` of edges with two named
columns, and every algorithm in this package is a sequence of joins and aggregations over
that table. That choice is the whole design:

* **Distribution and spill come for free.** A join is a join, so PageRank over a billion
  edges runs on a Ray cluster and spills under memory pressure using exactly the same
  machinery as any other query. There is no second execution path to keep in agreement.
* **The graph can live anywhere a table can.** Parquet on object storage, a lakehouse
  table, a database extract. Nothing has to be loaded into a graph-shaped index first.
* **Node identity is whatever the column holds.** Integers, strings, UUIDs. No
  compaction step, no vertex-id mapping to keep on the side.

What it costs is that per-iteration state is materialized, which is bounded by the node
count rather than the edge count. `iterate` documents why that is required rather than
merely convenient.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.api.dataset import Dataset

__all__ = ["Graph"]

#: The column names the algorithms read. An input table using different names is renamed
#: once, at construction, so nothing below has to thread three column names through.
SRC = "src"
DST = "dst"
WEIGHT = "weight"
NODE = "node"


@dataclass(frozen=True)
class Graph:
    """A graph as an edge table, with the column and direction conventions fixed.

    Build one with `Graph.from_edges` rather than by calling this constructor: it does the
    renaming and validation that lets every algorithm assume `src`, `dst` and `weight`.

    Args:
        edges: The edge table, already normalized to `src`/`dst`/`weight` columns.
        directed: Whether edge direction is meaningful.
        weighted: Whether the `weight` column carries real weights rather than the
            constant 1.0 the constructor supplies when none was given.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph
            >>> g = Graph.from_edges(bt.from_pydict({"src": [1, 2], "dst": [2, 3]}))
            >>> g.num_edges()
            2
    """

    edges: Dataset
    directed: bool = True
    weighted: bool = False
    #: Whether the edge table already holds *both* directions of every edge. Distinct from
    #: `directed`, which says only whether direction is meaningful: `from_edges` documents
    #: that it does not symmetrize, so a graph can be undirected and one-sided at once. That
    #: is the state `to_undirected` used to refuse to act on, because it keyed off `directed`.
    symmetrized: bool = False

    @staticmethod
    def from_edges(
        edges: Dataset,
        *,
        src: str = "src",
        dst: str = "dst",
        weight: str | None = None,
        directed: bool = True,
    ) -> Graph:
        """Build a graph from an edge table.

        Args:
            edges: A dataset with one row per edge.
            src: The column holding the source node id.
            dst: The column holding the destination node id.
            weight: The column holding an edge weight, or `None` for an unweighted graph
                (every edge weighs 1.0).
            directed: Whether direction is meaningful. An undirected graph is *not*
                symmetrized here; call `to_undirected` when an algorithm needs both
                directions materialized, and see its documentation for why that is a
                separate step.

        Returns:
            The graph handle.

        Raises:
            PlanError: If a named column is not in `edges`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.graph import Graph
                >>> follows = bt.from_pydict(
                ...     {"a": ["ann", "bob"], "b": ["bob", "cy"], "w": [2.0, 1.0]}
                ... )
                >>> g = Graph.from_edges(follows, src="a", dst="b", weight="w")
                >>> g.edges.columns
                ['src', 'dst', 'weight']
        """
        have = set(edges.columns)
        for role, name in (("src", src), ("dst", dst)):
            if name not in have:
                raise PlanError(
                    f"Graph.from_edges(): {role} column {name!r} is not in the edge table; "
                    f"available: {sorted(have)}"
                )
        if weight is not None and weight not in have:
            raise PlanError(
                f"Graph.from_edges(): weight column {weight!r} is not in the edge table; "
                f"available: {sorted(have)}"
            )
        if weight is None:
            normalized = edges.select(**{SRC: bt.col(src), DST: bt.col(dst), WEIGHT: bt.lit(1.0)})
        else:
            normalized = edges.select(
                **{SRC: bt.col(src), DST: bt.col(dst), WEIGHT: bt.col(weight).cast("float64")}
            )
        return Graph(normalized, directed=directed, weighted=weight is not None)

    def nodes(self) -> Dataset:
        """Every node that appears as a source or a destination, once each.

        Isolated nodes are invisible to an edge table by construction: a node with no
        edges appears in no row. Attach a node table with `with_nodes` when they matter,
        which they do for a degree distribution and for any per-node average.

        Returns:
            A one-column dataset named `node`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.graph import Graph
                >>> g = Graph.from_edges(bt.from_pydict({"src": [1, 2], "dst": [2, 3]}))
                >>> sorted(g.nodes().to_pydict()["node"])
                [1, 2, 3]
        """
        left = self.edges.select(**{NODE: bt.col(SRC)})
        right = self.edges.select(**{NODE: bt.col(DST)})
        return left.union(right).distinct()

    def extra_nodes(self) -> Dataset | None:
        """The nodes an edge table cannot name, or `None` when the endpoints are all of them.

        This exists so a per-node aggregate can restore its zero rows with a `union` arm
        instead of an outer join from `nodes()`. The two are equivalent, but the join form
        puts a `distinct` under a `group_by`, and that shape has no distributed path -- it
        raises rather than running. Returning `None` here lets the caller build the cheaper
        plan for the common graph and pay for the extra arm only when there is one.

        Returns:
            A one-column dataset of `node`, or `None` if no node table was attached.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.graph import Graph
                >>> g = Graph.from_edges(bt.from_pydict({"src": [1], "dst": [2]}))
                >>> g.extra_nodes() is None
                True

                >>> # With a node table attached, that table is the extra union arm. It is
                >>> # the whole declared table, not only the isolated nodes: the arm is
                >>> # unioned with the endpoints and deduplicated, so overlap is harmless
                >>> # and finding the isolated ones would cost the anti-join this avoids.
                >>> g = g.with_nodes(bt.from_pydict({"node": [1, 2, 3]}))
                >>> sorted(g.extra_nodes().to_pydict()["node"])
                [1, 2, 3]
        """
        return None

    def with_nodes(self, nodes: Dataset, *, node: str = "node") -> Graph:
        """Attach an explicit node table, so isolated nodes are counted.

        Args:
            nodes: A dataset with one row per node.
            node: The column holding the node id.

        Returns:
            A graph whose `nodes()` is the union of this table and the edge endpoints.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.graph import Graph
                >>> g = Graph.from_edges(bt.from_pydict({"src": [1], "dst": [2]}))
                >>> g = g.with_nodes(bt.from_pydict({"node": [1, 2, 3]}))
                >>> g.num_nodes()
                3
        """
        if node not in nodes.columns:
            raise PlanError(
                f"with_nodes(): column {node!r} is not in the node table; "
                f"available: {sorted(nodes.columns)}"
            )
        declared = nodes.select(**{NODE: bt.col(node)})
        # Kept beside the edge table rather than folded into it. The tempting encoding is
        # a zero-weight self-loop per declared node, which needs no second field — but a
        # self-loop is a real edge to a degree count and to a triangle count, so inventing
        # one would corrupt exactly the results this feature exists to make correct.
        # Keyword-passed, and carrying `symmetrized` explicitly: positionally, `declared`
        # used to be the fourth argument, so adding a field to the base class silently routed
        # the node table into it. Naming them makes a future field an error rather than a
        # misdelivery.
        return _GraphWithNodes(
            edges=self.edges,
            directed=self.directed,
            weighted=self.weighted,
            symmetrized=self.symmetrized,
            declared=declared,
        )

    def num_nodes(self) -> int:
        """The number of distinct nodes.

        Returns:
            The node count.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.graph import Graph
                >>> Graph.from_edges(bt.from_pydict({"src": [1], "dst": [2]})).num_nodes()
                2
        """
        return self.nodes().count()

    def num_edges(self) -> int:
        """The number of edges, counting parallel edges separately.

        Returns:
            The edge count.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.graph import Graph
                >>> Graph.from_edges(bt.from_pydict({"src": [1, 1], "dst": [2, 2]})).num_edges()
                2
        """
        return self.edges.count()

    def reverse(self) -> Graph:
        """The graph with every edge's direction flipped.

        Returns:
            The reversed graph. An undirected graph is returned unchanged, because
            reversing one is the identity.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.graph import Graph
                >>> g = Graph.from_edges(bt.from_pydict({"src": [1], "dst": [2]}))
                >>> g.reverse().edges.to_pydict()["src"]
                [2]
        """
        if not self.directed:
            return self
        flipped = self.edges.select(**{SRC: bt.col(DST), DST: bt.col(SRC), WEIGHT: bt.col(WEIGHT)})
        return replace(self, edges=flipped)

    def to_undirected(self) -> Graph:
        """Materialize both directions of every edge.

        This is a real doubling of the edge table, not a flag, and that is deliberate: an
        algorithm that walks outgoing edges gives a different answer on a symmetrized
        graph than on a directed one, and the difference should be a step you took rather
        than a flag you set. Self-loops are emitted once, since a self-loop reversed is
        itself.

        Idempotent, and keyed on `symmetrized` rather than on `directed`. Those are not the
        same question, and conflating them is what made this a no-op exactly where it was
        needed: `from_edges` documents that an undirected graph is *not* symmetrized and that
        you should call this when an algorithm needs both directions -- but the guard read
        ``if not self.directed: return self``, so the call the documentation sends you to did
        nothing for the graph it sends you there with. Fifteen call sites across `community`,
        `components`, `similarity`, `summary` and `sampling` assume this method symmetrizes;
        every one of them silently read a one-sided edge list for such a graph, and `pagerank`
        on it disagreed with a reference implementation while `triangle_count` returned zero.

        **`degree` double-counts on the result.** Every edge is now present in both
        directions, so a node with one neighbour has two edge endpoints. Use `out_degree`
        for a neighbour count on a symmetrized graph; that is what `k_core` does, and
        getting it wrong is why a k-core would keep the pendant nodes it exists to peel.

        Returns:
            The symmetrized graph, marked undirected.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.graph import Graph
                >>> g = Graph.from_edges(bt.from_pydict({"src": [1], "dst": [2]}))
                >>> g.to_undirected().num_edges()
                2
        """
        if self.symmetrized:
            return self
        loops = self.edges.filter(bt.col(SRC) == bt.col(DST))
        non_loops = self.edges.filter(bt.col(SRC) != bt.col(DST))
        both = non_loops.union(
            non_loops.select(**{SRC: bt.col(DST), DST: bt.col(SRC), WEIGHT: bt.col(WEIGHT)})
        )
        return replace(self, edges=both.union(loops), directed=False, symmetrized=True)

    def without_self_loops(self) -> Graph:
        """The graph with every edge from a node to itself removed.

        Returns:
            The filtered graph.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.graph import Graph
                >>> g = Graph.from_edges(bt.from_pydict({"src": [1, 2], "dst": [1, 3]}))
                >>> g.without_self_loops().num_edges()
                1
        """
        return replace(self, edges=self.edges.filter(bt.col(SRC) != bt.col(DST)))

    def simple(self) -> Graph:
        """Collapse parallel edges into one, summing their weights.

        Several algorithms are only defined on a simple graph, and a duplicated edge
        otherwise silently doubles a node's influence: PageRank sends twice the mass along
        it, and a triangle count counts the same triangle twice.

        Returns:
            The graph with at most one edge per ordered pair.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.graph import Graph
                >>> g = Graph.from_edges(bt.from_pydict({"src": [1, 1], "dst": [2, 2]}))
                >>> g.simple().num_edges()
                1
        """
        collapsed = (
            self.edges.group_by(SRC, DST).agg(**{WEIGHT: bt.sum(WEIGHT)}).select(SRC, DST, WEIGHT)
        )
        return replace(self, edges=collapsed)

    def subgraph(self, nodes: Dataset, *, node: str = "node") -> Graph:
        """The induced subgraph on a set of nodes: edges with both endpoints inside.

        Args:
            nodes: A dataset of node ids to keep.
            node: The column holding the node id.

        Returns:
            The induced subgraph.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.graph import Graph
                >>> g = Graph.from_edges(bt.from_pydict({"src": [1, 2], "dst": [2, 3]}))
                >>> keep = bt.from_pydict({"node": [1, 2]})
                >>> g.subgraph(keep).num_edges()
                1
        """
        keep = nodes.select(**{NODE: bt.col(node)}).distinct()
        kept_src = self.edges.join(
            keep.select(_src_key=bt.col(NODE)), left_on=SRC, right_on="_src_key", how="semi"
        )
        both = kept_src.join(
            keep.select(_dst_key=bt.col(NODE)), left_on=DST, right_on="_dst_key", how="semi"
        )
        return self._rebuild(both, keep)

    def _rebuild(self, edges: Dataset, keep: Dataset | None) -> Graph:
        """A graph over `edges`, restricted to `keep` when this graph declares nodes.

        A hook rather than a plain `replace` because a declared node table has to shrink
        with the edges. Without it `largest_component` returned a graph whose `nodes()`
        still held every node in the original — the edges were right and every per-node
        denominator was wrong.

        The base graph derives its nodes from the edges, so `keep` is already implied and
        this ignores it; `_GraphWithNodes` overrides to apply it.
        """
        del keep
        return replace(self, edges=edges)

    def cache(self) -> Graph:
        """Materialize the edge table once, so repeated algorithms do not rebuild it.

        Worth doing before running several algorithms over one graph, and before any
        iterative algorithm, because each iteration reads the edge table again.

        Returns:
            The graph with a cached edge table.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.graph import Graph
                >>> g = Graph.from_edges(bt.from_pydict({"src": [1], "dst": [2]})).cache()
                >>> g.num_edges()
                1
        """
        return replace(self, edges=self.edges.cache())


@dataclass(frozen=True)
class _GraphWithNodes(Graph):
    """A graph carrying an explicit node table, so isolated nodes are visible.

    A separate class rather than an optional field on `Graph` because the field would be
    `None` for every graph that does not have one, and every algorithm would then have to
    decide what `None` means. Overriding `nodes()` puts the decision in one place.
    """

    declared: Dataset | None = None

    def nodes(self) -> Dataset:
        """The declared nodes, plus any endpoint the edge table mentions."""
        endpoints = Graph.nodes(self)
        if self.declared is None:
            return endpoints
        return self.declared.union(endpoints).distinct()

    def extra_nodes(self) -> Dataset | None:
        """The declared table, which is the only place an isolated node can come from."""
        return self.declared

    def _rebuild(self, edges: Dataset, keep: Dataset | None) -> Graph:
        """Restrict the declared nodes alongside the edges."""
        if self.declared is None or keep is None:
            return replace(self, edges=edges)
        return replace(self, edges=edges, declared=self.declared.join(keep, on=NODE, how="semi"))
