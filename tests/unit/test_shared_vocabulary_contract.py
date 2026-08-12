"""Vocabularies that exist once must stay one object, not two that agree today.

A string vocabulary — join strategies, window function names, truthy tokens, cache dtypes —
tends to get *restated* rather than imported, because restating is a two-line change and
importing needs someone to decide where the canonical copy lives. The copies always agree on
the day they are made. What they do not do is stay in step, and every failure in this family
is silent:

* a join strategy the engine gains that the bandit never ranges over is an arm only the static
  cost model can pick, so plans stay correct and merely stop improving;
* a window function `plan.ir_tags` gains that the device translator does not classify is a
  shape the tier claims and then mistranslates, which is a *wrong column* rather than a
  decline (`.claude/rules/device-tier.md`);
* a cache dtype config validates but the sizing math has no width for sizes at zero bytes, so
  admission waves through a load the device cannot hold;
* a boolean env knob read with a narrower token set is a flag an operator believes is on.

So each of these was collapsed to one object, and this file is what keeps it collapsed. Most
assertions are deliberately identity (`is`) rather than equality: equality passes for two lists
that happen to match, which is the state being ruled out.

The independence contract is why several of these live in `plan` rather than beside a caller:
`kyber`, `carbonite`, `core` and `governance` may not import one another, so for two of them
copy-paste is the *only* wrong way to share, and lifting into a neutral layer is the only right
one.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_the_bandit_ranges_over_exactly_the_engine_s_join_strategies() -> None:
    """Membership must match; order must not, and is asserted separately below."""
    from batcher.kyber.learned_tuning.bandit import JOIN_ARMS
    from batcher.plan.logical.join import JOIN_STRATEGIES

    assert set(JOIN_ARMS) == JOIN_STRATEGIES
    assert len(JOIN_ARMS) == len(JOIN_STRATEGIES), "an arm is listed twice"


def test_the_cold_signature_s_first_join_sample_is_spent_on_hash() -> None:
    """`ucb1_best_arm` takes the first untried arm, so the tuple's order is a policy.

    Pinned because the obvious "simplification" — deriving the arms with `sorted()` — would
    put `broadcast` first and silently change which algorithm every cold signature measures.
    """
    from batcher.kyber.learned_tuning.bandit import JOIN_ARMS

    assert JOIN_ARMS[0] == "hash"


def test_the_device_translator_shares_the_engine_s_window_vocabulary() -> None:
    """Not equal sets — the same objects, so a name added to one cannot miss the other."""
    from batcher.core.gpu_plan import windows
    from batcher.plan.ir_tags import WINDOW_FILL, WINDOW_OFFSET_VALUE, WINDOW_RANKING

    assert windows._RANKING is WINDOW_RANKING
    assert windows._VALUE is WINDOW_OFFSET_VALUE
    assert windows._FILL is WINDOW_FILL


def test_window_value_functions_are_the_offset_ones_plus_the_fills() -> None:
    """`WINDOW_VALUE` is derived, so the three cannot drift into an inconsistent partition."""
    from batcher.plan.ir_tags import WINDOW_FILL, WINDOW_OFFSET_VALUE, WINDOW_VALUE

    assert WINDOW_VALUE == WINDOW_OFFSET_VALUE | WINDOW_FILL
    assert not (WINDOW_OFFSET_VALUE & WINDOW_FILL), "a function selects by offset or by nullness"


def test_core_and_kyber_share_one_counting_aggregate_set() -> None:
    """The two subsystems cannot import each other, so `plan` is the only legal home."""
    from batcher.core.gpu_plan.aggs import _COUNTING
    from batcher.kyber.stats.aggregate_columns import _COUNTING_AGGS
    from batcher.plan.ir_tags import COUNTING_AGGS

    assert _COUNTING is COUNTING_AGGS
    assert _COUNTING_AGGS is COUNTING_AGGS


def test_every_counting_aggregate_is_a_real_aggregate() -> None:
    """A typo here would make an aggregate return null over an empty group instead of zero."""
    from batcher.plan.ir_tags import AGG_FNS, COUNTING_AGGS

    assert COUNTING_AGGS <= AGG_FNS


def test_the_kv_cache_validator_and_the_sizing_math_read_one_table() -> None:
    from batcher.carbonite.accel import kv_cache
    from batcher.config.accelerator import CACHE_DTYPE_BYTES

    assert kv_cache._DTYPE_BYTES is CACHE_DTYPE_BYTES
    assert all(width > 0 for width in CACHE_DTYPE_BYTES.values()), "a zero width sizes to nothing"


def test_a_validated_cache_dtype_can_always_be_sized() -> None:
    """The property the two-copy version could break: accepted but unsizable."""
    from batcher.carbonite.accel.kv_cache import kv_bytes_per_token
    from batcher.config.accelerator import CACHE_DTYPE_BYTES

    for dtype in CACHE_DTYPE_BYTES:
        assert kv_bytes_per_token(layers=2, kv_heads=4, head_dim=8, dtype=dtype) > 0


def test_the_three_device_visibility_callers_read_one_env_list() -> None:
    """The inventory, the pinner and the telemetry sampler must agree on which vars matter."""
    from batcher._internal.accelerators import VISIBLE_DEVICE_ENVS
    from batcher._internal.hardware.devices import scope
    from batcher.ml import gpu

    assert scope.VISIBLE_DEVICE_ENVS is VISIBLE_DEVICE_ENVS
    assert gpu.VISIBLE_DEVICE_ENVS is VISIBLE_DEVICE_ENVS
    assert VISIBLE_DEVICE_ENVS[0] == "CUDA_VISIBLE_DEVICES", "the first one set wins"


def test_masked_off_means_the_same_thing_to_the_inventory_and_the_pinner() -> None:
    """Both took the whole node on a `-1` once; the copies were kept in step by a comment."""
    from batcher._internal.accelerators import NO_DEVICE_TOKENS
    from batcher._internal.hardware.devices import scope

    assert scope.NO_DEVICE_TOKENS is NO_DEVICE_TOKENS
    assert "-1" in NO_DEVICE_TOKENS


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", " yes ", "on"])
def test_every_spelling_of_yes_is_read_as_yes(raw: str) -> None:
    from batcher.config.env import truthy

    assert truthy(raw)


@pytest.mark.parametrize("raw", ["0", "false", "no", "off", "", "auto", "maybe", None])
def test_nothing_else_is_read_as_yes(raw: str | None) -> None:
    from batcher.config.env import truthy

    assert not truthy(raw)


def test_truthy_and_falsy_are_not_each_other_s_negation() -> None:
    """The three-way distinction a `bool | str` config field needs: yes, no, and "auto"."""
    from batcher.config.env import falsy, truthy

    assert not truthy("auto") and not falsy("auto")


def test_the_expression_dispatch_diagnostic_accepts_more_than_the_literal_one() -> None:
    """It compared against ``"1"`` alone, so `BATCHER_VERIFY_EXPR_MATCHES=true` was silently off.

    The worst failure available to a *diagnostic* flag: the operator believes the cross-check
    is running, and the thing it would have caught stays uncaught.
    """
    from batcher.config.env import env_flag

    assert env_flag.__module__ == "batcher.config.env"
    for spelling in ("1", "true", "yes", "on"):
        assert env_flag("BATCHER_VERIFY_EXPR_MATCHES") is False  # unset in the test process
        assert _flag_with(spelling)


def _flag_with(value: str) -> bool:
    """`env_flag` against a temporarily-set variable, without leaking it to other tests."""
    import os

    from batcher.config.env import env_flag

    name = "BATCHER_VERIFY_EXPR_MATCHES"
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        return env_flag(name)
    finally:
        if previous is None:
            del os.environ[name]
        else:
            os.environ[name] = previous


def test_the_spot_vocabulary_extends_the_truthy_one_rather_than_restating_it() -> None:
    """A launcher writes `BATCHER_SPOT=1`; an orchestrator writes `NODE_LIFECYCLE=spot`."""
    from batcher.config.env import TRUE_TOKENS
    from batcher.config.profiles import _SPOT_TRUE

    assert TRUE_TOKENS <= _SPOT_TRUE
    assert "preemptible" in _SPOT_TRUE
