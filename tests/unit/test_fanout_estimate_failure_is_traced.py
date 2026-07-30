"""A fan-out estimate that fails must leave a trace, not just a fallback.

`annotate._fanout` returns 1.0 when the cardinality estimator raises, and that fallback is
the right behaviour -- budgeting must never break a plan. But 1.0 is also the honest answer
for an operator that genuinely does not fan out, so the fallback is indistinguishable from
success. An estimator broken on this path would cap every downstream operator's parallelism
budget at a single morsel, forever, with every gate green.

The module already traces its outer abstention (`annotate resource bounds`); this is the
same decision one level down.
"""

from __future__ import annotations

import pytest

from batcher._internal.logging import _FIELDS_ATTR
from batcher.kyber.annotate import _fanout

pytestmark = pytest.mark.unit

_STEP = "estimate operator fan-out"


class _Node:
    """A plan node with a single input, which is all `_fanout` reads off it."""

    def __init__(self) -> None:
        self.input = object()


class _RaisingEstimator:
    def estimate(self, node):
        raise RuntimeError("estimator is broken")


class _WorkingEstimator:
    """Doubles its input: 100 rows in, 200 out."""

    def __init__(self) -> None:
        self._first = True

    def estimate(self, node):
        rows, self._first = (100, False) if self._first else (200, True)
        return type("Est", (), {"rows": rows})()


def test_a_failed_fanout_estimate_is_recorded(caplog) -> None:
    # `note_suppressed` logs at DEBUG on the `batcher.kyber` logger specifically, and
    # `log_kv` short-circuits if that logger is not enabled -- so a root-level at_level
    # would make this pass or fail on import order in a full-suite run.
    with caplog.at_level("DEBUG", logger="batcher.kyber"):
        assert _fanout(_Node(), _RaisingEstimator()) == 1.0

    fields = [getattr(r, _FIELDS_ATTR, {}) for r in caplog.records]
    noted = [f for f in fields if f.get("step") == _STEP]
    assert noted, (
        f"a broken estimator must be recorded, not silently budgeted at 1.0\n{caplog.text}"
    )
    assert noted[0]["error"] == "RuntimeError"
    assert "estimator is broken" in noted[0]["detail"]


def test_a_working_estimate_is_not_recorded(caplog) -> None:
    """The trace must mark failure only -- a normal fan-out is not a suppressed error."""
    with caplog.at_level("DEBUG", logger="batcher.kyber"):
        assert _fanout(_Node(), _WorkingEstimator()) == 2.0

    fields = [getattr(r, _FIELDS_ATTR, {}) for r in caplog.records]
    assert not [f for f in fields if f.get("step") == _STEP], caplog.text


def test_a_node_with_no_input_is_not_recorded(caplog) -> None:
    """No input is a normal shape (a scan), not an estimator failure."""

    class _Leaf:
        input = None

    with caplog.at_level("DEBUG", logger="batcher.kyber"):
        assert _fanout(_Leaf(), _RaisingEstimator()) == 1.0

    fields = [getattr(r, _FIELDS_ATTR, {}) for r in caplog.records]
    assert not [f for f in fields if f.get("step") == _STEP], caplog.text
