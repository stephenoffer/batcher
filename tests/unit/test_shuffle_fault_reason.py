"""A retryable shuffle fault must not lose the reason it was retryable.

The transport classifies a fetch failure as retryable (a lost/idle peer) or fatal, and a
reducer returns the retryable sources so the driver can recompute and try again. It used
to return only their *indices*. That is a lie by omission: a deterministic bug on the map
side reaches the driver as "worker N unreachable", the driver recomputes, and after
`recovery_max_attempts` the query dies with a worker-loss error on a cluster where every
worker is alive. A ticket collision surfaced exactly that way, three frames and one wrong
noun from its cause.

`_lost` is the seam. It still hands the driver plain indices — that is the recovery
protocol, and widening it would change every reducer — but the fault text goes to the log
on the worker that saw it, before the index is all that is left.
"""

from __future__ import annotations

import logging

import pytest

from batcher.dist.flight_worker import _lost

pytestmark = pytest.mark.unit


def test_the_indices_the_driver_recomputes_from_are_unchanged():
    assert _lost([(2, "connection refused"), (0, "idle timeout")]) == [0, 2]


def test_one_source_lost_on_several_replicas_is_recomputed_once():
    """Each copy of a source can fail separately; the driver recomputes the source, once."""
    assert _lost([(1, "peer gone"), (1, "replica gone")]) == [1]


def test_nothing_lost_is_an_empty_list():
    assert _lost([]) == []


def test_the_fault_text_reaches_the_log(caplog):
    """Structured, not interpolated: `log_kv` attaches the fields so both the human and
    the JSON formatter render them, and the dashboard reads them as typed data."""
    from batcher._internal.logging import _FIELDS_ATTR

    with caplog.at_level(logging.WARNING, logger="batcher.dist.shuffle"):
        _lost([(3, "status: Unavailable, message: transport error")])
    fields = [getattr(r, _FIELDS_ATTR, {}) for r in caplog.records]
    assert any(
        f.get("source") == 3 and "transport error" in str(f.get("fault", "")) for f in fields
    ), (
        "the reason a source was unreachable must survive; without it the driver "
        "reports worker loss for a deterministic fault"
    )
