"""Internal cache keys must not fail on a FIPS-enforcing host.

`hashlib.sha1()` *raises* when OpenSSL is in FIPS mode. Both call sites here are cache
keys — a plan's identity and a join shape's — not security claims, and both are on paths
every query takes, so a bare `sha1()` does not merely disable a cache there: it fails the
query. `usedforsecurity=False` declares the non-security use, which is what makes OpenSSL
allow it, and it changes no byte of the digest.
"""

from __future__ import annotations

import hashlib

import pytest

import batcher as bt
from batcher.dist.skew import join_skew_key
from batcher.kyber.signature import plan_signature


def _join_plan():
    left = bt.from_pydict({"k": [1, 2], "a": [1, 2]})
    right = bt.from_pydict({"k": [1, 2], "b": [3, 4]})
    return left.join(right, on="k")._plan


@pytest.fixture
def fips_hashlib(monkeypatch):
    """Model a FIPS build: `sha1()` without `usedforsecurity=False` raises."""
    real = hashlib.sha1

    def strict(data=b"", **kwargs):
        if not kwargs.get("usedforsecurity", True):
            return real(data, usedforsecurity=False)
        raise ValueError("[digital envelope routines] unsupported")

    monkeypatch.setattr(hashlib, "sha1", strict)
    return strict


def test_a_join_shape_key_survives_a_fips_host(fips_hashlib):
    plan = _join_plan()
    key = join_skew_key("{}", "{}", plan)
    assert len(key) == 16


def test_a_plan_signature_survives_a_fips_host(fips_hashlib):
    """Taken several times per plan, so a raise here fails every query, not just a cache."""
    assert len(plan_signature(_join_plan())) == 16


def test_declaring_the_non_security_use_does_not_change_the_digest():
    payload = b"whatever"
    assert (
        hashlib.sha1(payload, usedforsecurity=False).hexdigest()
        == hashlib.sha1(payload).hexdigest()
    )


def test_the_join_shape_key_still_distinguishes_shapes():
    left = bt.from_pydict({"k": [1], "a": [1]})
    right = bt.from_pydict({"k": [1], "b": [1]})
    inner = left.join(right, on="k")._plan
    outer = left.join(right, on="k", how="left")._plan
    assert join_skew_key("{}", "{}", inner) != join_skew_key("{}", "{}", outer)


def test_a_hub_that_cannot_be_read_is_noted_rather_than_silent(monkeypatch, caplog):
    """The write half already reported a failing hub; the read half swallowed it, so an
    unreachable hub disabled learned skew forever with nothing saying why."""
    import batcher.core as core
    from batcher.dist.skew import load_learned_hot_keys

    def boom():
        raise RuntimeError("hub unreachable")

    monkeypatch.setattr(core, "default_hub", boom)
    # Levelled on `batcher.dist` specifically: `ensure_configured` sets an explicit level
    # on the engine's loggers, which a root-level `at_level` does not override — so in a
    # full-suite run (where something else has already configured logging) `log_kv` would
    # short-circuit and this test would pass or fail depending on import order.
    with caplog.at_level("DEBUG", logger="batcher.dist"):
        assert load_learned_hot_keys("shape") is None
    # `note_suppressed` carries the step and the cause as structured fields hung off the
    # record, not in the message, so assert on those rather than on the rendered line.
    from batcher._internal.logging import _FIELDS_ATTR

    fields = [getattr(r, _FIELDS_ATTR, {}) for r in caplog.records]
    noted = [f for f in fields if f.get("step") == "load learned hot keys"]
    assert noted, caplog.text
    assert noted[0]["error"] == "RuntimeError"
