"""Watching a query run: verbosity, logging, and execution statistics.

Observability is configuration, not instrumentation you sprinkle through the pipeline.
Turn it up for one block, read what happened, and turn it back down -- the query code is
unchanged either way.

    python examples/operations/observability.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col
from batcher.config import (
    disable_logging,
    enable_logging,
    get_logger,
    option_context,
    set_log_level,
    set_verbosity,
)


def main() -> None:
    ds = bt.from_pydict({"grp": ["a", "b", "a", "b"], "v": [1, 2, 3, 4]})

    query = ds.group_by("grp").agg(total=col("v").sum()).sort("grp")

    # Execution statistics for the last run: what the engine actually did.
    result = query.to_pydict()
    assert result["total"] == [4, 6]
    stats = query.stats()
    print("stats:", type(stats).__name__)
    assert stats is not None

    # Verbosity is a config knob, scoped like any other.
    with option_context("observability.verbosity", "quiet"):
        quiet = query.to_pydict()
    assert quiet == result

    # The named helpers do the same thing.
    set_verbosity("quiet")
    assert query.to_pydict() == result
    set_verbosity("normal")

    # Logging can be turned on for a diagnosis and off again.
    logger = get_logger()
    assert logger is not None
    enable_logging()
    set_log_level("WARNING")
    assert query.to_pydict() == result
    disable_logging()

    # Progress reporting off, for a script whose output is parsed by something else.
    # The option takes 'auto' / 'on' / 'off' (or None), not a bool.
    with option_context("observability.progress", "off"):
        assert query.to_pydict() == result

    # The plan and its estimates, which is where you look when a query is slow.
    plan = query.explain()
    print(plan)
    assert "aggregate" in plan

    # Profiling a plan reports per-operator detail.
    profile = ds.profile()
    print("profile:", type(profile).__name__)
    assert profile is not None

    # None of the above changed the answer -- observability is never semantic.
    assert query.to_pydict() == result


if __name__ == "__main__":
    main()
