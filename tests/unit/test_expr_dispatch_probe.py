"""The optimizer's expression-shape probe must never reach `Expr.__getattr__`.

`Expr.__getattr__` exists to tell a user which Batcher spelling replaces a pandas or Polars
idiom, and building that message loads the whole migration registry. The optimizer asks every
expression type whether it has an `op` or a `fn`; asked with `hasattr`, each miss built and
discarded that message, costing about 0.7 s on the first query of every process.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.kyber.optimizer import expr_dispatch
from batcher.plan.expr_ir.core import Expr

pytestmark = pytest.mark.unit


def test_the_probe_does_not_fall_back_to_getattr(monkeypatch):
    misses: list[str] = []
    original = Expr.__getattr__

    def spy(self, name):
        misses.append(name)
        return original(self, name)

    monkeypatch.setattr(Expr, "__getattr__", spy)
    monkeypatch.setattr(expr_dispatch, "_DISCRIMINATOR_ATTR", {})
    expr = (bt.col("a") > 1) & bt.col("b").is_null()

    for node in (expr, bt.col("a"), bt.lit(3)):
        expr_dispatch.discriminator(node)

    assert misses == []


def test_the_probe_still_reads_each_operator(monkeypatch):
    """Positive control: the probe finds `op` where a node has one and `None` where not."""
    monkeypatch.setattr(expr_dispatch, "_DISCRIMINATOR_ATTR", {})
    binary = bt.col("a") > 1
    assert expr_dispatch.discriminator(binary) == binary.op
    assert expr_dispatch.discriminator(bt.col("a")) is None


def test_a_user_facing_miss_still_carries_its_guidance():
    """The fix sits in the optimizer, so the user's error message is unchanged."""
    with pytest.raises(AttributeError, match="alias"):
        _ = bt.col("a").suffix
