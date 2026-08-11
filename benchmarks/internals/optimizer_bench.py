"""Optimizer-time benchmark - guard planning latency as the rule set grows.

The Kyber rule engine is pattern-indexed: each rule declares the node types it
`matches`, and the driver only attempts rules whose types are present in the plan.
That is what lets the rule set grow toward thousands without every query paying for
every rule. This benchmark measures *planning* time (no execution) for plans of
increasing size, and - the key property - shows that adding hundreds of rules that
*cannot* fire on a plan leaves its optimization time essentially unchanged.

Run:
    source .venv/bin/activate
    python3 benchmarks/run.py --benchmark optimizer
    python3 benchmarks/internals/optimizer_bench.py    # the same, as a script
"""

from __future__ import annotations

import time

import batcher as bt
from batcher import col
from batcher.kyber.optimizer import Optimizer
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rule import Phase, node_rule
from batcher.plan.logical import Window


def filter_chain(n: int) -> bt.Dataset:
    """A deep plan: `n` stacked (filter → project) pairs over one scan."""
    ds = bt.from_pydict({"a": list(range(128)), "b": list(range(128))})
    for i in range(n):
        ds = ds.filter(col("a") > i).select("a", "b", c=col("a") + col("b"))
    return ds


def join_star(n: int) -> bt.Dataset:
    """An `n`-way inner join on a shared key - exercises cost-based join reordering."""
    ds = bt.from_pydict({"k": [1, 2, 3], "v0": [1, 2, 3]})
    for i in range(1, n):
        other = bt.from_pydict({"k": [1, 2, 3], f"v{i}": [i, i, i]})
        ds = ds.join(other, on="k")
    return ds


def _best_ms(opt: Optimizer, plan, iters: int = 25) -> float:
    opt.optimize(plan)  # warm
    best = float("inf")
    for _ in range(iters):
        t0 = time.perf_counter()
        opt.optimize(plan)
        best = min(best, (time.perf_counter() - t0) * 1e3)
    return best


def _noop_rules(count: int) -> list:
    # Rules that match a node type (Window) absent from the benchmark plans, so the
    # pattern index must skip all of them - they should add ~zero planning cost.
    return [
        node_rule(f"noop_{i}", Phase.REWRITE, lambda _node, _ctx: None, matches=(Window,))
        for i in range(count)
    ]


def main() -> int:
    base_rules = DEFAULT_REGISTRY.rules()
    print(f"{len(base_rules)} built-in rules\n")

    print(f"{'plan':<26}{'rules':>8}{'optimize_ms':>14}")
    print("-" * 48)

    cases = [
        ("filter_chain(10)", filter_chain(10)),
        ("filter_chain(40)", filter_chain(40)),
        ("join_star(4)", join_star(4)),
        ("join_star(8)", join_star(8)),
    ]
    for label, ds in cases:
        plan = ds._plan
        base = _best_ms(Optimizer(sources=ds._sources, rules=base_rules), plan)
        print(f"{label:<26}{len(base_rules):>8}{base:>14.3f}")

    # Scaling property: +500 inapplicable rules should not move the needle.
    print("\nPattern-indexing scaling (filter_chain(40)):")
    ds = filter_chain(40)
    plan = ds._plan
    for extra in (0, 100, 500, 1000):
        rules = base_rules + _noop_rules(extra)
        ms = _best_ms(Optimizer(sources=ds._sources, rules=rules), plan)
        print(f"  +{extra:>4} inapplicable rules ({len(rules):>4} total): {ms:.3f} ms")

    _bench_repeated_planning()


def _bench_repeated_planning() -> None:
    """The memoized `optimize_full` path — what a re-issued query (BI dashboard, benchmark
    harness, adaptive stage re-optimization) actually pays. The lookup key is now built from
    the plan's memoized `content_key()` instead of re-serializing the whole IR on every call.
    """
    from batcher.kyber.optimizer import optimize_full

    print("\nRepeated planning (memoized optimize_full, cache hit):")
    ds = join_star(6)
    plan = ds._plan
    for _ in range(50):  # warm the plan memo + content_key
        optimize_full(plan, sources=ds._sources)
    best = float("inf")
    for _ in range(6):
        t0 = time.perf_counter()
        for _ in range(2000):
            optimize_full(plan, sources=ds._sources)
        best = min(best, (time.perf_counter() - t0) / 2000 * 1e6)
    print(f"  cache-hit planning: {best:.2f} us/call")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
