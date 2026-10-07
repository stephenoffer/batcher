"""Live smoke test: `http_engine` sends a row's request id to a real OpenAI-compatible API.

Skipped unless ``BATCHER_LIVE_OPENAI_BASE_URL`` (and ``BATCHER_LIVE_OPENAI_MODEL``) are set;
``BATCHER_LIVE_OPENAI_API_KEY`` is read when the endpoint needs one. A provider does not echo
``X-Client-Request-Id`` back, so this proves the request is accepted with the header (a
non-ASCII or over-long value is a 400 at OpenAI) and the id is recorded; matching it in the
provider's logs is the manual half of the check.
"""

from __future__ import annotations

import os

import pytest

import batcher as bt

pytestmark = pytest.mark.skipif(
    not os.environ.get("BATCHER_LIVE_OPENAI_BASE_URL"),
    reason="set BATCHER_LIVE_OPENAI_BASE_URL and BATCHER_LIVE_OPENAI_MODEL to run",
)


def test_request_ids_are_accepted_and_recorded():
    engine = bt.ml.http_engine(
        os.environ["BATCHER_LIVE_OPENAI_BASE_URL"],
        os.environ.get("BATCHER_LIVE_OPENAI_MODEL", "gpt-4o-mini"),
        api_key=os.environ.get("BATCHER_LIVE_OPENAI_API_KEY"),
        max_tokens=4,
        concurrency=1,
    )
    ds = bt.from_pydict({"q": ["Say ok.", "Say yes."]})
    out = ds.ml.generate(engine, prompt_column="q", request_id_column="rid").to_pydict()
    assert all(out["response"])
    assert len(set(out["rid"])) == 2
