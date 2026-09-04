"""A governance setting that cannot be honored must fail at the point it is set.

This section is the one place in the config where a wrong value makes the system do
*less*, silently. `_refuse_ungoverned_read` selects on ``mode == "off"`` and
``mode == "strict"`` and lets everything else fall through to the advisory warning, so
``mode="Strict"`` -- a capitalization -- used to be accepted and turn a deployment that
refuses ungoverned reads into one that logs them and proceeds. Nothing downstream could
notice: an advisory warning on a read that should have raised looks exactly like an
advisory deployment working correctly.

`default_deny` is the same shape with the reader missing entirely.
"""

from __future__ import annotations

import dataclasses
import warnings

import pytest

import batcher as bt
from batcher._internal.errors import AccessDeniedError, ConfigError
from batcher.config import active_config, config_context
from batcher.config.validation.sections import GOVERNANCE_MODES

pytestmark = pytest.mark.unit


def _with_governance(**kw):
    base = active_config()
    return dataclasses.replace(base, governance=dataclasses.replace(base.governance, **kw))


@pytest.mark.parametrize("mode", GOVERNANCE_MODES)
def test_every_legal_mode_is_accepted(mode: str) -> None:
    """The positive control: the check must not reject a mode enforcement understands."""
    with config_context(_with_governance(mode=mode)):
        assert active_config().governance.mode == mode


@pytest.mark.parametrize("mode", ["Strict", "STRICT", "strcit", "enforced", ""])
def test_an_unrecognized_mode_is_refused_rather_than_downgraded(mode: str) -> None:
    with (
        pytest.raises(ConfigError, match=r"governance\.mode must be one of"),
        config_context(_with_governance(mode=mode)),
    ):
        pass


def test_default_deny_is_refused_while_nothing_implements_it() -> None:
    """It is declared and documented; no code path reads it. Accepting it grants nothing."""
    with (
        pytest.raises(ConfigError, match="default_deny is not implemented"),
        config_context(_with_governance(default_deny=True)),
    ):
        pass


def test_strict_actually_refuses_where_a_mistyped_mode_only_warned() -> None:
    """What the mistyped mode cost: the same read, refused under `strict`, allowed under it.

    This is the behaviour the validation protects, asserted directly rather than trusted --
    without it the checks above only prove that a string comparison works.
    """
    with config_context(_with_governance(mode="strict")), pytest.raises(AccessDeniedError):
        bt.from_pydict({"a": [1]}).collect()

    with config_context(_with_governance(mode="advisory")):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bt.from_pydict({"a": [1]}).collect()  # allowed, with a warning
        assert any("ungoverned" in str(w.message) for w in caught)
