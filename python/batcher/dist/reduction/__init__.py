"""Reduction shapes shared by the distributed operators.

A shuffle's reduce side is an associative fold over mergeable partial state, and the shape
of that fold — a line or a tree — is what decides whether the reduce phase shrinks or grows
as nodes are added. `tree` holds the arithmetic; the operators supply the combine.
"""

from __future__ import annotations

from batcher.dist.reduction.tree import chunks, reduce_levels, tree_reduce

__all__ = ["chunks", "reduce_levels", "tree_reduce"]
