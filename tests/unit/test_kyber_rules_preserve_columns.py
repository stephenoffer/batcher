"""No Kyber rule may change a plan's output columns — in set *or* in order.

A rule is required to be semantics-preserving: it changes which plan runs, never which
relation comes back. Column order is part of that relation. `SELECT *, a, b` has one defined
order in SQL, a caller reading an Arrow schema positionally depends on it, and a headerless
write puts it on disk.

`transpose_adjacent_windows` broke it and stood for as long as it did because its own docstring
recorded the wrong property: the aliases must be disjoint "so the column **set** above the pair
is unchanged either way". A set is not an order. The rule swapped two independent `Window`
nodes into canonical spec order so their specs could collapse, and
`Window.available_columns()` is `input.available_columns() + [aliases]` — so whichever node
ended up outer contributed its aliases last, and `with_columns(a=rank(...), b=sum(...))` came
back as `g, v, b, a`.

Nothing caught it: every value was correct, and the differential harness compares rows as a
sorted multiset with column names as a *set*, so an output permutation was precisely the
property it could not see.

This runs **every registered rule** against a corpus of plan shapes and asserts the columns
are untouched. It is a negative check — it passes by finding nothing — so
`test_the_audit_detects_the_defect_it_was_built_for` reinstates the original bug in memory and
proves the audit reports it. Without that control this file would certify an absence it cannot
see, which is worse than not having it.
"""

from __future__ import annotations

import contextlib
import dataclasses

import pyarrow as pa
import pytest

import batcher as bt

pytestmark = pytest.mark.unit

_T = pa.table(
    {
        "a": pa.array([3, 1, 2, 1], pa.int64()),
        "b": pa.array([1, 2, 3, 4], pa.int64()),
        "g": pa.array(["x", "y", "x", "y"]),
        "f": pa.array([1.5, 2.5, 3.5, 4.5], pa.float64()),
    }
)
_R = pa.table({"a": pa.array([1, 2], pa.int64()), "w": pa.array(["p", "q"])})


def _plans() -> dict[str, object]:
    """Shapes chosen so the rules that *rewrite structure* have something to match.

    Two and three stacked windows are the shape the known defect needed; the rest give the
    join, aggregate, sort, dedup and projection families a plan each.
    """
    ds, rs = bt.from_arrow(_T), bt.from_arrow(_R)
    return {
        "two_windows": ds.with_columns(
            r=bt.col("b").rank().over(partition_by="g", order_by="b"),
            s=bt.col("b").sum().over(partition_by="g"),
        ),
        "three_windows": ds.with_columns(
            r=bt.col("b").rank().over(partition_by="g", order_by="b"),
            s=bt.col("b").sum().over(partition_by="g"),
            m=bt.col("b").max().over(partition_by="g", order_by="b"),
        ),
        "aggregate": ds.group_by("g").agg(n=bt.col("a").count(), s=bt.col("b").sum()),
        "join_then_project": ds.join(rs, on="a").select("a", "w", "b"),
        "sort_limit": ds.sort("a").limit(2),
        "distinct": ds.select("g", "a").distinct(),
        "filter_project": ds.filter(bt.col("a") > 1).select("b", "a"),
        "window_over_aggregate": ds.group_by("g")
        .agg(s=bt.col("b").sum())
        .with_columns(r=bt.col("s").rank().over(order_by="s")),
        "window_over_sort": ds.sort("b").with_columns(r=bt.col("b").rank().over(order_by="b")),
    }


def _nodes(plan):
    out, stack = [], [plan]
    while stack:
        node = stack.pop()
        out.append(node)
        for attr in ("input", "left", "right"):
            child = getattr(node, attr, None)
            if child is not None:
                stack.append(child)
        stack.extend(getattr(node, "inputs", ()) or ())
    return out


def _context(ds):
    from batcher.config import Config
    from batcher.kyber.cardinality import CardinalityEstimator
    from batcher.kyber.pass_base import OptimizerContext

    return OptimizerContext(
        config=Config(),
        sources=ds._sources,
        hub=None,
        estimator=CardinalityEstimator(ds._sources, None),
    )


def _column_changes(rules) -> list[tuple[str, str, list[str], list[str]]]:
    """Every `(rule, plan, before, after)` where a rule altered the output columns."""
    found = []
    for label, ds in _plans().items():
        ctx = _context(ds)
        for node in _nodes(ds._plan):
            try:
                before = node.available_columns()
            # A node that cannot type itself is not what this audit is about.
            except Exception:
                continue
            for rule_obj in rules:
                fn = getattr(rule_obj, "fn", None) or getattr(rule_obj, "func", None) or rule_obj
                matches = getattr(rule_obj, "matches", None)
                if matches:
                    try:
                        if not isinstance(node, tuple(matches)):
                            continue
                    except TypeError:
                        pass
                try:
                    out = fn(node, ctx)
                # A rule that declines by raising is not this defect.
                except Exception:
                    continue
                if out is None or out is node:
                    continue
                try:
                    after = out.available_columns()
                except Exception:
                    continue
                if after != before:
                    found.append((getattr(rule_obj, "name", str(rule_obj)), label, before, after))
    return found


def _registered_rules():
    from batcher.kyber.registry import DEFAULT_REGISTRY, register_builtin_rules

    # Registration is idempotent in effect but raises if already done.
    with contextlib.suppress(Exception):
        register_builtin_rules(DEFAULT_REGISTRY)
    return list(DEFAULT_REGISTRY._rules)


def test_no_rule_changes_a_plans_output_columns():
    rules = _registered_rules()
    assert len(rules) > 100, f"only {len(rules)} rules registered — the audit sees almost nothing"
    changes = _column_changes(rules)
    assert not changes, "these rules altered the output columns:\n" + "\n".join(
        f"  {name} on {label}\n    before={before}\n    after ={after}"
        for name, label, before, after in changes
    )


def test_the_audit_detects_the_defect_it_was_built_for():
    """The positive control. A negative check that cannot fail certifies nothing.

    Reinstates `transpose_adjacent_windows` as it was — swap the pair, do not restore the
    order — and asserts the audit reports it. If this ever stops failing, the audit above has
    stopped looking rather than the tree having stayed clean.
    """
    import batcher.kyber.rules.relational.windows as windows

    def buggy(node, _ctx):
        inner = node.input
        if not isinstance(inner, windows.Window):
            return None
        if windows._window_inputs(node) & windows._outputs(inner):
            return None
        if windows._outputs(node) & windows._outputs(inner):
            return None
        if windows._spec_key(inner) <= windows._spec_key(node):
            return None
        return dataclasses.replace(inner, input=dataclasses.replace(node, input=inner.input))

    buggy.name = "transpose_adjacent_windows__as_it_was"
    changes = _column_changes([buggy])
    assert changes, "the audit no longer detects the transposition it was written for"
    # And specifically an ORDER change, not a SET one — the distinction the original
    # docstring got wrong is the distinction this file exists to hold.
    _name, label, before, after = changes[0]
    assert set(after) == set(before), f"expected an order change, got a set change on {label}"
    assert after != before
