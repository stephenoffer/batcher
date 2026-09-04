"""The usage channel takes pairs, and says so rather than indexing into whatever it is given.

`_usage_columns` carries a message about the row *count* an engine reports, and nothing at all
about the shape. `p[0]`/`p[1]` took the first two of whatever arrived, which fails three
different ways and only one of them is loud enough to act on.

(The count message turns out to be unreachable through `ds.ml.generate`, which the last test
here pins rather than assumes — it was written expecting the guard to fire, and it did not.)

The one that matters is the three-element tuple, because it does not fail at all. An engine
reporting ``(total, prompt, completion)`` — an ordering nothing in the contract warns against —
wrote the *total* into `prompt_tokens` for every row and raised nothing, so a cost report
summed a column that was confidently wrong. Every number in these tests is one this module
measured against the pre-guard code rather than one it reasoned about.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import BackendError
from batcher.ml.llm.channels import usage_sink

pytestmark = pytest.mark.unit

#: Prompts of known, distinct lengths, so a misreported count is visible per row rather than
#: only as a total.
_PROMPTS = ["x" * 12, "y" * 34]


def _engine(report):
    """An engine that reports `report(prompt)` into the usage channel for each row."""

    class _E:
        def __call__(self, prompts):
            usage_sink().report([report(p) for p in prompts])
            return [p.upper() for p in prompts]

    return lambda: _E()


def _run(report):
    ds = bt.from_pydict({"prompt": _PROMPTS})
    return ds.ml.generate(_engine(report), prompt_column="prompt", usage=True).to_pydict()


def test_a_pair_of_integers_is_the_contract_and_still_works():
    out = _run(lambda p: (len(p), 1))
    assert out["prompt_tokens"] == [12, 34]
    assert out["completion_tokens"] == [1, 1]


def test_a_row_reporting_nothing_stays_null():
    """`None` is a legal per-row value: the engine generated, but reported no usage for it."""
    assert _run(lambda p: None)["prompt_tokens"] == [None, None]


def test_a_three_element_tuple_is_refused_rather_than_silently_truncated():
    """The silent one. Measured pre-guard at ``[999, 999]`` against real lengths ``[12, 34]``."""
    with pytest.raises(BackendError, match="prompt_tokens, completion_tokens"):
        _run(lambda p: (999, len(p), 1))


def test_the_other_three_element_ordering_is_refused_too():
    """``(prompt, completion, total)`` happens to give the right two, which is not a reason.

    Accepting it would make correctness depend on an ordering the contract never states, so
    both orderings are refused by the same length check.
    """
    with pytest.raises(BackendError):
        _run(lambda p: (len(p), 1, 999))


def test_a_dict_is_refused_and_the_message_names_the_fix():
    """The likeliest mistake: it is the shape the OpenAI-style APIs hand you."""
    with pytest.raises(BackendError, match="prompt_tokens"):
        _run(lambda p: {"prompt_tokens": len(p), "completion_tokens": 1})


def test_a_string_is_refused():
    with pytest.raises(BackendError, match="str"):
        _run(lambda p: str(len(p)))


def test_the_message_names_the_engine_the_shape_and_the_escape_hatch():
    with pytest.raises(BackendError) as err:
        _run(lambda p: {"prompt_tokens": 1})
    message = str(err.value)
    assert "_E" in message, "the engine that reported must be named"
    assert "dict" in message
    assert "usage=False" in message, "the reader needs a way to proceed"


def test_a_wrong_row_count_nulls_the_columns_rather_than_raising():
    """What actually happens, pinned because it is not what `_usage_columns`' message implies.

    That function carries a `BackendError` for a count mismatch — "they must correspond
    one-to-one and in prompt order" — and it is **unreachable through `ds.ml.generate`**.
    `_row_reported.resolve` compares the reported length against the request count first and
    substitutes `[None] * n` when they differ, deliberately, so that a *unique*-length list
    does not raise a spurious mismatch after dedup. The consequence is that a genuine
    mismatch is discarded just as quietly: the columns come back all-null and nothing says
    the engine miscounted.

    Written as an assertion on the nulls rather than left undocumented, so the next reader
    meets the behaviour rather than the message. Both channels do it — the per-call sink here
    and the legacy `last_usage` attribute — and neither raises.
    """

    class _WrongCount:
        def __call__(self, prompts):
            usage_sink().report([(1, 1)])  # one pair for two rows
            return [p.upper() for p in prompts]

    out = (
        bt.from_pydict({"prompt": _PROMPTS})
        .ml.generate(lambda: _WrongCount(), prompt_column="prompt", usage=True)
        .to_pydict()
    )
    assert out["prompt_tokens"] == [None, None]
    assert out["response"] == [p.upper() for p in _PROMPTS], "the generations still arrive"


def test_usage_false_skips_the_channel_entirely():
    """The escape hatch the message names has to work, or the message is not actionable."""
    ds = bt.from_pydict({"prompt": _PROMPTS})
    out = ds.ml.generate(_engine(lambda p: {"bad": 1}), prompt_column="prompt").to_pydict()
    assert out["response"] == [p.upper() for p in _PROMPTS]
    assert "prompt_tokens" not in out
