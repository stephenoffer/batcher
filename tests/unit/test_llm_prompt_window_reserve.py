"""How much of the context window is held back for the generation itself.

A prompt is truncated to fit the window *minus a reserve*, and the reserve has to cover what
the caller asked to generate. A fixed 512 is right for a short answer and wrong for
`max_tokens=4096`: a prompt filling everything but the reserve leaves an eighth of the room
the request needs, and the generation is then cut short rather than refused -- which reads as
the model losing the thread rather than as a sizing mistake.
"""

from __future__ import annotations

import pytest

from batcher.ml.llm.sizing import prompt_window

pytestmark = pytest.mark.unit


def test_the_default_reserve_is_kept_when_nothing_was_asked_for() -> None:
    assert prompt_window(None, {"max_model_len": 8192}) == 8192 - 512


def test_a_larger_generation_budget_widens_the_reserve() -> None:
    assert prompt_window(None, {"max_model_len": 8192}, 4096) == 4096


def test_a_smaller_budget_does_not_narrow_the_reserve() -> None:
    # Shrinking below the default would let a prompt crowd out the decode on a request that
    # merely happens to declare a small budget; the floor is there for the general case.
    assert prompt_window(None, {"max_model_len": 8192}, 64) == 8192 - 512


def test_a_budget_that_swallows_the_window_still_leaves_room_for_a_prompt() -> None:
    # Degenerate, but it must not return zero or a negative: the truncation would then keep
    # nothing of the prompt.
    assert prompt_window(None, {"max_model_len": 1024}, 4096) == 1


def test_the_live_config_is_consulted_when_the_window_is_not_declared() -> None:
    class _Config:
        max_model_len = 4096

    class _Engine:
        model_config = _Config()

    class _Llm:
        llm_engine = _Engine()

    assert prompt_window(_Llm(), {}, 1024) == 4096 - 1024


def test_an_undeterminable_window_stays_undeterminable() -> None:
    assert prompt_window(None, {}) is None
    assert prompt_window(None, {"max_model_len": 0}) is None
    assert prompt_window(None, {"max_model_len": "auto"}) is None
