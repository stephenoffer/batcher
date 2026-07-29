"""What the adaptive loop decides, and on what evidence: routing, then staging.

Two questions, both answered in counts rather than wall time so a shared or loaded
machine cannot move the numbers.

Counts routing *decisions*, never wall time, so a shared or loaded machine cannot move
the number. That matters here: the effect this measures is worth about 20% of the sf10
suite, and the harness swings +/-25% under load, so a timing could neither prove nor
disprove it on a busy box.

The question it answers: `resolve_adaptive("auto", ...)` decides whether a query pays for
stage-by-stage re-optimization. Its confidence gate used to read `Provenance.DEFAULT`, a
label describing where a size estimate came from, and treat that as evidence the estimate
was *wrong*. The two are different, and on the one-shot path the label never clears,
because nothing records an intermediate operator's measured cardinality against it.

Run:  python benchmarks/internals/adaptive_gate_routing.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import duckdb

import batcher as bt
from batcher.api.adaptive import gating as gate
from batcher.api.adaptive import resolve_adaptive
from batcher.api.adaptive.plan_surgery import joins
from batcher.kyber.signature import plan_signature
from batcher.metadata import MetadataHub
from batcher.metadata.backends import InProcessBackend
from batcher.plan.feedback import OperatorFeedback
from batcher.plan.logical import is_streamable
from suites.standard.tpch import QUERIES

# The size floor is a separate gate and would short-circuit every shape at this scale.
# Lowering it isolates the confidence gate, exactly as tests/unit/test_adaptive_resolution
# does, so the counts below are about evidence and not about table size.
gate._ADAPTIVE_MIN_INPUT_ROWS = 1

_SCALE = 0.05  # enough rows that operands are estimated rather than known exactly


def _fresh_hub() -> MetadataHub:
    return MetadataHub(InProcessBackend())


def _teach(hub: MetadataHub, plan, ratio: float, runs: int = 4) -> None:
    """Record `runs` executions of every breaker-produced join operand in `plan`.

    `ratio` is actual/estimated: 1.0 is an estimator that held up, 8.0 one that missed by
    far more than `optimizer.reoptimize_error`.
    """
    for join in joins(plan):
        for operand in (join.left, join.right):
            if is_streamable(operand):
                continue
            signature = plan_signature(operand)
            for _ in range(runs):
                hub.record(
                    OperatorFeedback(
                        op_id=1,
                        kind="aggregate",
                        n_actual=int(1000 * ratio),
                        t_op_ms=1.0,
                        m_peak_bytes=0,
                        selectivity=1.0,
                        batch_size=1024,
                        signature=signature,
                        n_estimated=1000.0,
                    )
                )


def _stage_waste(session) -> tuple[int, int, int]:
    """Breaker stages a staged run executes, and how many were already sized exactly.

    A breaker whose output size the optimizer already knows exactly costs a
    materialization and returns a number the planner had. Those are the stages worth
    not taking.
    """
    from batcher.api.adaptive import staging
    from batcher.plan.stats import Provenance

    counts = {"stages": 0, "exact": 0, "queries": 0}
    original = staging._run_stage

    def counting(target, srcs, hub, *args, **kwargs):
        counts["stages"] += 1
        try:
            estimate = gate._build_estimator(srcs, hub).estimate(target)
            if estimate.provenance < Provenance.DEFAULT:
                counts["exact"] += 1
        except Exception:
            pass
        return original(target, srcs, hub, *args, **kwargs)

    staging._run_stage = counting
    try:
        for sql in QUERIES.values():
            try:
                session.sql(sql).collect(adaptive=True)
                counts["queries"] += 1
            except Exception:
                continue
    finally:
        staging._run_stage = original
    return counts["queries"], counts["stages"], counts["exact"]


def main() -> None:
    con = duckdb.connect()
    con.execute("INSTALL tpch; LOAD tpch;")
    con.execute(f"CALL dbgen(sf={_SCALE})")
    session = bt.Session()
    tables = con.execute("select table_name from information_schema.tables order by 1").fetchall()
    for (name,) in tables:
        session.register(name, bt.from_arrow(con.execute(f"select * from {name}").arrow()))

    cold = held_up = missed = planned = 0
    flipped: list[str] = []
    for query, sql in QUERIES.items():
        try:
            ds = session.sql(sql)
        except Exception:
            continue
        planned += 1
        on_cold = resolve_adaptive("auto", ds._plan, ds._sources, _fresh_hub())

        accurate = _fresh_hub()
        _teach(accurate, ds._plan, ratio=1.0)
        on_accurate = resolve_adaptive("auto", ds._plan, ds._sources, accurate)

        wrong = _fresh_hub()
        _teach(wrong, ds._plan, ratio=8.0)
        on_wrong = resolve_adaptive("auto", ds._plan, ds._sources, wrong)

        cold += on_cold
        held_up += on_accurate
        missed += on_wrong
        if on_cold and not on_accurate:
            flipped.append(query)

    print(f"TPC-H queries planned:                    {planned}")
    print(f"routed to staging, cold hub:              {cold}")
    print(f"routed to staging, estimates held up:     {held_up}")
    print(f"routed to staging, estimates missed:      {missed}")
    print(f"flipped off by measured accuracy:         {len(flipped)}")
    print(f"  {sorted(flipped)}")

    queries, stages, exact = _stage_waste(session)
    print()
    print(f"queries executed adaptively:              {queries}")
    print(f"breaker stages executed:                  {stages}")
    print(f"  already exactly sized (measure nothing): {exact}")


if __name__ == "__main__":
    main()
