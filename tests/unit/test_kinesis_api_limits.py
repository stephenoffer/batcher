"""Regression: the Kinesis source must respect two hard AWS API limits.

Both bugs were invisible without a real stream, and both were silent about it.

`GetRecords` caps `Limit` at 10,000 records and rejects anything larger with
`InvalidArgumentException`. The source sent the engine's 16,384-row morsel size straight
through as the default, so the default configuration was an illegal request.

`ListShards` returns at most 100 shards per call and hands back a `NextToken` for the rest.
The source read one page and stopped, which looks like it works and quietly skips every
shard past the hundredth — on a large stream, most of the data.

These use a fake boto3 client, so they need no AWS.
"""

from __future__ import annotations

from typing import Any

import pytest

from batcher.io.formats.streaming.kinesis import _GET_RECORDS_MAX_LIMIT, KinesisSource

pytestmark = pytest.mark.unit


class _FakeKinesis:
    """A boto3 `kinesis` client that records calls and paginates like the real one."""

    def __init__(self, shard_count: int, page_size: int = 100) -> None:
        self.shard_ids = [f"shard-{i:04d}" for i in range(shard_count)]
        self.page_size = page_size
        self.get_records_limits: list[int] = []
        self.list_shards_calls: list[dict[str, Any]] = []

    def list_shards(self, **kwargs: Any) -> dict[str, Any]:
        self.list_shards_calls.append(kwargs)
        # The real API rejects StreamName and NextToken together.
        assert not ("StreamName" in kwargs and "NextToken" in kwargs)
        start = int(kwargs["NextToken"]) if "NextToken" in kwargs else 0
        page = self.shard_ids[start : start + self.page_size]
        out: dict[str, Any] = {"Shards": [{"ShardId": s} for s in page]}
        nxt = start + self.page_size
        if nxt < len(self.shard_ids):
            out["NextToken"] = str(nxt)
        return out

    def get_shard_iterator(self, **kwargs: Any) -> dict[str, Any]:
        return {"ShardIterator": "iter-0"}

    def get_records(self, **kwargs: Any) -> dict[str, Any]:
        limit = kwargs["Limit"]
        # The real API raises InvalidArgumentException above 10,000.
        assert limit <= _GET_RECORDS_MAX_LIMIT, f"Limit={limit} exceeds the GetRecords cap"
        self.get_records_limits.append(limit)
        return {"Records": [], "NextShardIterator": "iter-1"}


def _source(fake: _FakeKinesis, **kwargs: Any) -> KinesisSource:
    """A source whose lazily-built boto3 client is already the fake."""
    src = KinesisSource("my-stream", **kwargs)
    src._client_obj = fake  # `_client()` returns this instead of building a real one
    return src


def test_default_poll_size_is_a_legal_get_records_limit():
    # The default used to be 16,384 — larger than the API allows.
    assert KinesisSource("s").poll_size <= _GET_RECORDS_MAX_LIMIT


def test_poll_clamps_an_oversized_poll_size():
    fake = _FakeKinesis(shard_count=1)
    src = _source(fake, poll_size=50_000)
    src._poll()
    assert fake.get_records_limits == [_GET_RECORDS_MAX_LIMIT]


def test_poll_passes_a_smaller_poll_size_through():
    fake = _FakeKinesis(shard_count=1)
    src = _source(fake, poll_size=500)
    src._poll()
    assert fake.get_records_limits == [500]


@pytest.mark.parametrize("shard_count", [1, 100, 101, 250])
def test_list_shards_is_paginated(shard_count):
    fake = _FakeKinesis(shard_count=shard_count)
    src = _source(fake)
    assert src._shards() == fake.shard_ids
    assert len(src._shards()) == shard_count


def test_shards_are_discovered_once_and_cached():
    fake = _FakeKinesis(shard_count=250)
    src = _source(fake)
    src._shards()
    calls_after_first = len(fake.list_shards_calls)
    src._shards()
    assert len(fake.list_shards_calls) == calls_after_first  # cached, not re-listed
    assert calls_after_first == 3  # 250 shards over 100-shard pages
