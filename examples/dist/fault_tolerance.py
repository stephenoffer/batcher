"""What survives a worker failure, and what the mergeable algebra guarantees.

A partial aggregate is a value, not a process, so a lost worker costs its partition's work
and nothing else — the rest of the partials are still valid and still combine. That is why
`combine` has to be associative and commutative rather than merely correct in order.

    python examples/dist/fault_tolerance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import resolve_distributed, tpch
from batcher import col


def main() -> None:
    distributed = resolve_distributed()
    lineitem = tpch("lineitem")

    # The partials, as separate datasets — this is what each worker would produce.
    shards = [lineitem.slice(start, 50_000) for start in range(0, 200_000, 50_000)]
    assert sum(shard.count() for shard in shards) == lineitem.count()

    partials = [
        shard.group_by("l_shipmode").agg(
            lines=bt.count(), qty=col("l_quantity").sum()
        )
        for shard in shards
    ]

    def combine(pieces: list[bt.Dataset]) -> dict:
        merged = pieces[0]
        for piece in pieces[1:]:
            merged = merged.union(piece)
        return (
            merged.group_by("l_shipmode")
            .agg(lines=col("lines").sum(), qty=col("qty").sum())
            .sort("l_shipmode")
            .to_pydict()
        )

    reference = (
        lineitem.group_by("l_shipmode")
        .agg(lines=bt.count(), qty=col("l_quantity").sum())
        .sort("l_shipmode")
        .to_pydict()
    )

    # In order.
    forwards = combine(partials)
    assert forwards["lines"] == reference["lines"]

    # Reversed: `combine` is commutative, so the order the partials arrive in cannot
    # matter — which is what makes a retried worker's result safe to fold in late.
    backwards = combine(list(reversed(partials)))
    assert backwards["lines"] == reference["lines"]
    assert backwards["l_shipmode"] == reference["l_shipmode"]

    # Regrouped: associativity, so partials can be combined in any tree shape.
    nested = combine([bt.concat(partials[:2]), bt.concat(partials[2:])])
    assert nested["lines"] == reference["lines"]

    print("combine is order- and grouping-independent across", len(partials), "partials")

    # And the engine's own distributed path agrees.
    engine = (
        lineitem.group_by("l_shipmode")
        .agg(lines=bt.count(), qty=col("l_quantity").sum())
        .sort("l_shipmode")
        .collect(distributed=distributed, num_partitions=4)
        .to_pydict()
    )
    assert engine["lines"] == reference["lines"]


if __name__ == "__main__":
    main()
