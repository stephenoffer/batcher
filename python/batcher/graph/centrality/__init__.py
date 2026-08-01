"""Centrality: which nodes matter, by five different definitions of "matter".

Every function here is a power iteration expressed as joins and aggregations, so it
distributes and spills like any other query. They differ in what they consider important:

| Function | A node is important when |
| --- | --- |
| `degree_centrality` | many edges touch it |
| `pagerank` | important nodes point at it, and they do not point at much else |
| `personalized_pagerank` | it is close to a set of nodes you care about |
| `eigenvector_centrality` | important nodes point at it, without the damping |
| `katz_centrality` | many short paths reach it, with longer paths counting less |
| `hits` | it points at good sources (hub) or is pointed at by good hubs (authority) |

PageRank is the one to reach for by default. Eigenvector centrality is what PageRank
becomes without damping, which makes it exact on a strongly connected graph and unstable
on anything else: on a graph with a sink component it concentrates all the score there.
Katz is the fix for that, and takes a decay instead.
"""

from __future__ import annotations

from batcher.graph.centrality.rank import (
    degree_centrality,
    pagerank,
    personalized_pagerank,
)
from batcher.graph.centrality.spectral import (
    eigenvector_centrality,
    hits,
    katz_centrality,
)

__all__ = [
    "degree_centrality",
    "eigenvector_centrality",
    "hits",
    "katz_centrality",
    "pagerank",
    "personalized_pagerank",
]
