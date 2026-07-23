"""Benchmarks of Batcher's own subsystems, each with its own reporting.

These are not engine comparisons: they measure a Batcher subsystem against a
*second Batcher path* — the same query planned with more rules, executed across
partitions, answered from metadata, or shuffled over a different transport — and
report the pair themselves rather than through the cross-engine ``compare()``
table in ``harness.py``.

- ``optimizer_bench`` — Kyber planning latency as the rule set grows.
- ``metadata_bench`` — metadata-answered queries vs the O(rows) computation.
- ``distributed`` — single-node == many-partition mergeable equivalence, timed.
- ``shuffle_vs_object_store`` — Carbonite's Flight transport vs the Ray object store.

Submodules are imported on demand by ``run.py --benchmark distributed|optimizer|
shuffle``, so that selecting one never pays for another's optional dependency
(Ray in particular). Each also runs directly as a script.
"""

from __future__ import annotations
