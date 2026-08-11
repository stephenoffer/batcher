"""The client-side token bucket in front of a hosted LLM endpoint.

The properties that matter are the ones a retry loop cannot provide: a burst is *held* rather
than sent and refused, both quota dimensions bind independently, and the limiter is a genuine
no-op when nothing was configured, so a local endpoint pays no lock per request.

Timings are asserted as bounds rather than equalities. The bucket refills on wall time, and a
test that pinned an exact wait would fail on a loaded machine.
"""

from __future__ import annotations

import threading
import time

import pytest

import batcher as bt
import batcher.ml as ml
from batcher._internal.errors import PlanError
from batcher.ml.llm.engines.limits import (
    RateLimiter,
    _estimated_tokens,
    build_limiter,
)

pytestmark = pytest.mark.unit


def test_no_configured_rate_builds_no_limiter():
    """The uncontended path must not pay for a lock it does not need."""
    assert build_limiter(None, None) is None


def test_a_configured_rate_builds_a_limiter():
    limiter = build_limiter(600, None)
    assert limiter is not None
    assert not limiter.unlimited


def test_an_unconfigured_limiter_never_waits():
    assert RateLimiter().unlimited
    assert RateLimiter().acquire(estimated_tokens=10_000) == 0.0


def test_a_full_bucket_admits_immediately():
    assert RateLimiter(requests_per_minute=6000).acquire() == 0.0


def test_the_request_bucket_holds_a_burst_instead_of_sending_it():
    """One minute's allowance is 60 requests; a bucket sized to ~1 must delay the second."""
    limiter = RateLimiter(requests_per_minute=60, burst=1 / 60)
    assert limiter.acquire() == 0.0
    waited = limiter.acquire()
    assert waited > 0.5  # one request per second at this rate


def test_the_token_bucket_binds_independently_of_the_request_bucket():
    """A generous request quota must not let a token-heavy call through the token quota."""
    limiter = RateLimiter(requests_per_minute=100_000, tokens_per_minute=60, burst=1 / 60)
    assert limiter.acquire(estimated_tokens=1) == 0.0
    assert limiter.acquire(estimated_tokens=1) > 0.5


def test_a_request_larger_than_the_bucket_is_admitted_rather_than_stuck():
    """The limiter smooths the send rate; it does not reject work already decided on."""
    limiter = RateLimiter(tokens_per_minute=60, burst=1 / 60)
    limiter.acquire(estimated_tokens=10_000)  # far larger than the bucket
    started = time.monotonic()
    limiter.acquire(estimated_tokens=10_000)
    assert time.monotonic() - started < 5.0


def test_the_bucket_refills_over_time():
    limiter = RateLimiter(requests_per_minute=600, burst=1 / 600)
    limiter.acquire()
    time.sleep(0.3)  # 600/min is 10/s, so 0.3s restores well over one request
    assert limiter.acquire() == 0.0


def test_the_limit_is_shared_across_the_threads_of_one_worker():
    """A per-thread limiter would multiply the quota by the pool size, which is the bug."""
    limiter = RateLimiter(requests_per_minute=60, burst=1 / 60)
    admitted: list[float] = []

    def call() -> None:
        admitted.append(limiter.acquire())

    threads = [threading.Thread(target=call) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert len(admitted) == 3
    # One went straight through; the others had to wait for the bucket to refill.
    assert sum(1 for w in admitted if w == 0.0) == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"requests_per_minute": 0}, "requests_per_minute"),
        ({"tokens_per_minute": -1}, "tokens_per_minute"),
        ({"burst": 0}, "burst"),
    ],
)
def test_a_non_positive_setting_is_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        RateLimiter(**kwargs)


def test_the_token_estimate_counts_the_reply_the_request_reserved():
    """A tokens-per-minute quota counts output too, so `max_tokens` is part of the charge."""
    prompt = "a" * 40  # ~10 tokens
    assert _estimated_tokens(prompt, {}) == 10
    assert _estimated_tokens(prompt, {"max_tokens": 100}) == 110


def test_both_hosted_engines_accept_the_limits():
    """The parameters have to reach the engines, not only exist on the limiter."""
    assert callable(ml.http_engine("http://x/v1", "m", requests_per_minute=100))
    assert callable(ml.http_engine("http://x/v1", "m", tokens_per_minute=1000))
    assert callable(ml.anthropic_engine("claude-opus-5", api_key="k", requests_per_minute=50))


# --- spend -------------------------------------------------------------------------


def test_token_spend_prices_input_and_output_separately():
    """Output is priced several times input, so the two must not be summed together first."""
    usage = bt.from_pydict({"pt": [1_000_000], "ct": [500_000]})
    got = usage.agg(c=bt.token_spend("pt", "ct", input_price=3.0, output_price=15.0))
    assert got.to_pydict()["c"][0] == pytest.approx(3.0 + 7.5)


def test_token_spend_totals_across_rows_and_splits_by_group():
    usage = bt.from_pydict(
        {"m": ["a", "a", "b"], "pt": [1_000_000, 1_000_000, 2_000_000], "ct": [0, 0, 0]}
    )
    got = usage.group_by("m").agg(c=bt.token_spend("pt", "ct", input_price=1.0, output_price=1.0))
    by_model = dict(zip(*(got.to_pydict()[k] for k in ("m", "c")), strict=True))
    assert by_model["a"] == pytest.approx(2.0)
    assert by_model["b"] == pytest.approx(2.0)


def test_token_spend_of_an_empty_corpus_is_not_an_error():
    """A run that generated nothing must report no spend, not raise."""
    empty = bt.from_pydict({"pt": [1], "ct": [1]}).filter(bt.col("pt") < bt.lit(0))
    got = empty.agg(c=bt.token_spend("pt", "ct", input_price=1.0, output_price=1.0)).to_pydict()
    assert got["c"][0] in (None, 0.0)


def test_token_spend_rejects_a_negative_price():
    with pytest.raises(PlanError):
        bt.token_spend("pt", "ct", input_price=-1.0, output_price=1.0)


def test_a_vision_request_charges_its_images_against_the_token_quota():
    """Counting only the text under-charged a vision batch by ~1400 tokens a row.

    An image is most of a vision request's input, so a limiter blind to it let the batch run
    far over the tokens-per-minute quota and the 429 it exists to prevent arrived anyway.
    Both wire shapes are covered: Anthropic spells `image`, OpenAI `image_url`.
    """
    from batcher.ml.llm.engines.limits import _estimated_tokens

    text_only = {"max_tokens": 100, "messages": [{"role": "user", "content": "hello there"}]}
    assert _estimated_tokens("hello there", text_only) < 200

    for kind in ("image", "image_url"):
        vision = {
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": [{"type": kind}, {"type": "text", "text": "hi"}]}
            ],
        }
        assert _estimated_tokens("hi", vision) > 1000

    two = {
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": [{"type": "image"}, {"type": "image"}]},
        ],
    }
    one = {"max_tokens": 100, "messages": [{"role": "user", "content": [{"type": "image"}]}]}
    assert _estimated_tokens("", two) - _estimated_tokens("", one) > 1000


def test_a_body_without_messages_still_estimates():
    """The completions wire shape carries `prompt`, not `messages`."""
    from batcher.ml.llm.engines.limits import _estimated_tokens

    assert _estimated_tokens("hi", {"max_tokens": 5, "prompt": "hi"}) == 5
