"""The exception hierarchy: catching the failure you meant to catch.

Every error the engine raises descends from ``BatcherError``, so a pipeline can catch that
one type at its boundary. The specific subclasses let you distinguish a user mistake
(``PlanError``) from an environment problem (``IOError``) from a missing extra
(``MissingDependencyError``), which is the difference between retrying and giving up.

    python examples/operations/error_handling.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col


def main() -> None:
    ds = bt.from_pydict({"a": [1, 2, 3]})

    # Referencing a column that does not exist is a plan-time error, raised before any
    # data is read -- which is why it is cheap to catch.
    try:
        ds.select(col("nope")).to_pydict()
    except bt.ColumnNotFoundError as exc:
        print("ColumnNotFoundError:", str(exc)[:70], "| column:", exc.column)
        assert exc.column == "nope"
    else:
        raise AssertionError("expected a ColumnNotFoundError")

    # It is also a `PlanError`, a `BatcherError`, and a `KeyError` -- so a broad handler
    # at the pipeline boundary still catches it, and a mapping-style `except KeyError`
    # works too.
    assert issubclass(bt.ColumnNotFoundError, bt.PlanError)
    assert issubclass(bt.ColumnNotFoundError, KeyError)
    assert issubclass(bt.PlanError, bt.BatcherError)
    for cls in (
        bt.ExecutionError,
        bt.IOError,
        bt.CompileError,
        bt.OptimizationError,
        bt.MissingDependencyError,
        bt.TransportError,
    ):
        assert issubclass(cls, bt.BatcherError), cls

    # One handler at the pipeline boundary catches everything the engine raises.
    def run(dataset: bt.Dataset) -> str:
        try:
            dataset.select(col("missing")).to_pydict()
        except bt.BatcherError as exc:
            return type(exc).__name__
        return "ok"

    assert run(ds) == "ColumnNotFoundError"

    # A plan-time argument mistake is also a PlanError, with an actionable message.
    try:
        ds.select(bad=col("a").str.levenshtein(col("a"))).to_pydict()
    except bt.PlanError as exc:
        print("PlanError:", str(exc)[:80])
    else:
        raise AssertionError("expected a PlanError")

    # An optional backend that is not installed reports as a missing dependency rather
    # than an ImportError from somewhere deep in the stack.
    assert issubclass(bt.MissingDependencyError, bt.BatcherError)

    # Validation errors from the quality layer are catchable the same way.
    dirty = bt.from_pydict({"v": [1, -1]})
    try:
        dirty.dq.check(col("v") > 0, name="positive").fail().to_pydict()
    except bt.BatcherError as exc:
        print("dq failure:", type(exc).__name__)
    else:
        raise AssertionError("expected the contract to fail")


if __name__ == "__main__":
    main()
