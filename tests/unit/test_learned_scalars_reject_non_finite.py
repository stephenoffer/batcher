"""A non-finite observation must never reach a learned scalar, on any of its four folds.

Exponential smoothing is ``alpha*value + (1-alpha)*prior``. Feed it a NaN or an infinity once
and the *stored* value becomes non-finite, and from there every later update does too: the
entry is poisoned for the life of the store and nothing raises. `metadata.smoothed` documents
that argument and guards `math.isfinite`; three independent folds beside it had only caught
the NaN half, because `value != value` and `size < 1` both let an infinity through.

The second half is the read. A store outlives the build that wrote it, so a write-side guard
cannot be the only one — and every consumer here turns the entry into an integer, where an
infinity is not a bad estimate but an `OverflowError` raised out of planning or batch sizing.
"""

from __future__ import annotations

import math

import pytest

from batcher.metadata import MetadataHub
from batcher.metadata.backends import InProcessBackend

pytestmark = pytest.mark.unit

_NON_FINITE = [float("inf"), float("-inf"), float("nan")]


def _hub() -> MetadataHub:
    return MetadataHub(InProcessBackend())


@pytest.mark.parametrize("bad", _NON_FINITE)
def test_dist_sizing_ema_rejects_a_non_finite_observation(bad: float) -> None:
    from batcher.dist.adaptive_sizing.sizing import _ema, _read_ema

    hub = _hub()
    _ema(hub, "ns", "sig", bad)
    assert _read_ema(hub, "ns", "sig") is None, "a poisoned entry must not be readable"
    for _ in range(5):
        _ema(hub, "ns", "sig", 100.0)
    assert _read_ema(hub, "ns", "sig") == pytest.approx(100.0), "good writes still converge"


@pytest.mark.parametrize("bad", _NON_FINITE)
def test_autobatch_rejects_a_non_finite_size(bad: float) -> None:
    from batcher.ml.autobatch import learned_batch_size, record_batch_size

    hub = _hub()
    record_batch_size(hub, "sig", bad)
    assert learned_batch_size(hub, "sig") is None
    record_batch_size(hub, "sig", 512)
    assert learned_batch_size(hub, "sig") == 512


@pytest.mark.parametrize("bad", _NON_FINITE)
def test_a_poisoned_store_is_not_read_back(bad: float) -> None:
    """The read guard, reached by writing the bad value past the writer.

    This is the case the write guard cannot cover: an entry left by an older build. Both
    consumers coerce to `int`, and `int(inf)` raises rather than estimating badly.
    """
    from batcher.dist.adaptive_sizing.sizing import _MIN_SAMPLES, _read_ema
    from batcher.metadata.hardware_scope import scoped

    hub = _hub()
    hub.put_keyed_param(scoped("ns"), "sig", {"ema": bad, "n": _MIN_SAMPLES + 1})
    assert _read_ema(hub, "ns", "sig") is None
    with pytest.raises(OverflowError):
        int(float("inf"))  # what the unguarded read would have handed its caller


def test_the_alpha_floor_cannot_be_set_where_the_blend_diverges() -> None:
    """`alpha > 1` makes `(1 - alpha)` negative, so the estimate moves *past* the observation.

    Two of the floor's three consumers apply it with no upper clamp, so blending 100 toward
    200 at a floor of 3.0 yields 400 — divergence, not convergence. Its sibling
    `learning_smoothing_alpha` has always been range-checked; this one was not.
    """
    import dataclasses

    from batcher import Config, config_context
    from batcher._internal.errors import ConfigError

    base = Config()
    diverging = dataclasses.replace(
        base, optimizer=dataclasses.replace(base.optimizer, learned_scalar_alpha_floor=3.0)
    )
    with pytest.raises(ConfigError, match="learned_scalar_alpha_floor"), config_context(diverging):
        pass
    fine = dataclasses.replace(
        base, optimizer=dataclasses.replace(base.optimizer, learned_scalar_alpha_floor=0.25)
    )
    with config_context(fine):
        assert math.isfinite(0.25)
