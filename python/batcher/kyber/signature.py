"""Structural plan signatures.

A signature identifies "the same operator in the same query shape" across runs,
ignoring literal values — so learned statistics (a filter's selectivity, a join's
output size) recorded on one execution apply to the next execution of the same
shape. This is the key the MetadataHub feedback loop is indexed by.
"""

from __future__ import annotations

import hashlib
import json

from batcher.plan.logical import (
    Aggregate,
    AsofJoin,
    Distinct,
    Filter,
    Join,
    Limit,
    LogicalPlan,
    MapBatches,
    Project,
    Sample,
    Scan,
    Sort,
    Union,
    Unnest,
    Unpivot,
)

__all__ = ["plan_signature"]


# Instance-`__dict__` cache slot, matching the `_memoize_noarg` convention plan nodes
# already use for `to_ir` / `available_schema` (see `plan/logical/base.py`).
_SIG_SLOT = "_c_plan_signature"


def plan_signature(node: LogicalPlan) -> str:
    """A stable short hash of a node's structure (literal values normalized).

    Memoized on the node. Plan nodes are immutable, so the signature is a pure function
    of the node — but it JSON-encodes and hashes the *whole subtree*, and the conductor
    asks for it several times per query (result-cache key, learned-stats lookup, feedback
    recording). Measured on a small query that was ~4 encodes per query, making
    `json.iterencode` as expensive as the entire native execution. The write goes through
    `__dict__` to bypass the frozen `__setattr__`, exactly as `_memoize_noarg` does.

    **Composed from the children's signatures, not from their structure.** The memo above
    only helps a node that is asked twice; it does nothing for a node built fresh over
    children that were already signed, which is the case that dominates. The join-order DP
    constructs a candidate `Join` for every subset it costs, and under the old whole-subtree
    encode each candidate re-encoded everything beneath it — so signing cost O(subtree) per
    candidate and O(2^n x n) for the search. Measured on `join_star(8)`
    (`benchmarks/internals/optimizer_bench.py`): 491 signature computations per `optimize`,
    **22% of a 66 ms plan**, on a rule set whose pattern index had already made the *rule
    count* free. Hashing the node's own token plus its children's memoized digests makes a
    fresh parent O(1), and an incrementally built plan O(depth) in total rather than
    O(depth^2).

    Identity is unchanged by construction: two nodes hash equal exactly when their local
    tokens and their children's signatures agree, which recursively is the same structural
    equality the flat encoding expressed. The digests themselves *do* change, so the
    persisted learned store re-learns once — entries are keyed by signature and simply miss
    until re-measured.
    """
    cached = node.__dict__.get(_SIG_SLOT)
    if cached is not None:
        return cached
    payload = json.dumps(_struct(node), sort_keys=True, default=str)
    # `usedforsecurity=False`: this is a plan-identity cache key, and a FIPS-enforcing host
    # rejects a bare `sha1()` outright — which would fail *every* query, since the signature
    # is taken several times per plan. Declaring the non-security use is what makes OpenSSL
    # allow it, and the digest is unchanged.
    sig = hashlib.sha1(payload.encode(), usedforsecurity=False).hexdigest()[:16]
    node.__dict__[_SIG_SLOT] = sig
    return sig


def _struct(node: LogicalPlan):
    """The node's own shape, with each child standing in as its signature, not its structure.

    Recursion happens through `plan_signature`, so every child is memoized on first use and
    a node built over already-signed children costs one small JSON encode. See
    `plan_signature` for why that matters.
    """
    if isinstance(node, Scan):
        # The source's identity, not its position. `source_id` is an index into *this plan's*
        # own source list, so the first source of every query is `0` — which made this token
        # `["scan"]` for every relation in the process and every signature built over it a
        # shared entry. Two filters of the same shape over different tables then averaged
        # their measured selectivities into a figure wrong for both: `k < 40` keeping 40 of
        # 20,000 rows in one table and every row in another estimated the second at **40 rows
        # against an actual 20,000**, a 500x error and worse than the structural guess it
        # replaced.
        #
        # This is the defect the `MapBatches` arm below already names and fixes for itself by
        # carrying the UDF's identity, and that `estimator._CORRECTABLE` works around by
        # excluding `Scan` from learned row counts. Fixing the token fixes it for every
        # consumer at once instead of one exclusion at a time.
        #
        # `""` — a synthetic scan over an intermediate, or a source that cannot name itself —
        # keeps the old shared token, which is the honest answer for a relation with no
        # cross-run identity and is what those scans always had.
        return ["scan", node.source_key]
    if isinstance(node, Filter):
        return ["filter", _norm(node.predicate.to_ir()), plan_signature(node.input)]
    if isinstance(node, Project):
        return ["project", [i.alias for i in node.items], plan_signature(node.input)]
    if isinstance(node, Aggregate):
        return [
            "agg",
            [k.alias for k in node.group_keys],
            [(s.alias, s.agg.func) for s in node.aggregates],
            plan_signature(node.input),
        ]
    if isinstance(node, Join):
        return [
            "join",
            node.join_type,
            list(node.left_keys),
            list(node.right_keys),
            plan_signature(node.left),
            plan_signature(node.right),
        ]
    if isinstance(node, AsofJoin):
        return [
            "asof_join",
            node.left_on,
            node.right_on,
            list(node.left_by),
            list(node.right_by),
            node.direction,
            plan_signature(node.left),
            plan_signature(node.right),
        ]
    if isinstance(node, Sort):
        return ["sort", plan_signature(node.input)]
    if isinstance(node, Limit):
        return ["limit", plan_signature(node.input)]
    if isinstance(node, Distinct):
        # The dedup key is part of the shape. `distinct(["user_id"])` and a whole-row
        # `distinct()` over the same input produce wildly different row counts, so sharing one
        # learned entry between them would teach the cost model an average of the two — the
        # same collision this module's docstring describes for scans.
        return ["distinct", list(node.keys), plan_signature(node.input)]
    if isinstance(node, Union):
        return ["union", [plan_signature(i) for i in node.inputs]]
    if isinstance(node, MapBatches):
        # The UDF's identity is part of the shape: without it every `map_batches` over the
        # same input signature collapses to one learned entry, so a filtering UDF (0.1x) and
        # an exploding one (20x) — the two ends of an AI pipeline — would share, and
        # whichever ran last would answer for the other. The same defect the `Scan` arm above
        # now fixes by carrying the source's identity, and the reason `MapBatches` can only be
        # a learnable (`_CORRECTABLE`) fan-out once its signature tells the UDFs apart.
        # Best-effort by qualified name (stable across runs for a named function/class);
        # anonymous lambdas still collide, the floor.
        return ["map_batches", _udf_identity(node.fn), plan_signature(node.input)]
    if isinstance(node, Unnest):
        return ["unnest", node.column, node.alias, plan_signature(node.input)]
    if isinstance(node, Unpivot):
        return [
            "unpivot",
            list(node.index),
            list(node.on),
            node.variable_name,
            node.value_name,
            plan_signature(node.input),
        ]
    if isinstance(node, Sample):
        return ["sample", node.fraction, node.seed, plan_signature(node.input)]
    return [type(node).__name__]


def _udf_identity(fn: object) -> str:
    """A best-effort, run-stable identity for a `map_batches` UDF.

    A named function or factory class is identified by ``module.qualname`` — stable across
    processes, so a correction learned on one run applies to the next. A callable *instance*
    (the "build the model once" factory pattern) is identified by its class, since that is
    what determines its behaviour. An anonymous lambda degrades to ``<lambda>`` and may
    collide with another lambda over the same input — the unavoidable floor for a UDF with
    no name — which is still strictly better than every `map_batches` sharing one key.
    """
    qual = getattr(fn, "__qualname__", None)
    if qual is not None:
        module = getattr(fn, "__module__", "") or ""
    else:  # a callable instance: identify it by its type
        cls = type(fn)
        qual = getattr(cls, "__qualname__", cls.__name__)
        module = getattr(cls, "__module__", "") or ""
    return f"{module}.{qual}"


def _norm(ir):
    """Normalize an expression IR, replacing literal values with a placeholder."""
    if isinstance(ir, dict):
        if ir.get("e") == "lit":
            return {"e": "lit"}
        return {k: _norm(v) for k, v in ir.items()}
    if isinstance(ir, list):
        return [_norm(x) for x in ir]
    return ir
