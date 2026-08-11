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
    # `_shards()` yields the whole shard *descriptor*, not a bare id: `ParentShardId` /
    # `AdjacentParentShardId` are what say which shards replaced one a reshard closed, and
    # `_adopt_children` needs them. The property under test is unchanged either way — every
    # page is followed, so nothing past the hundredth shard goes missing.
    fake = _FakeKinesis(shard_count=shard_count)
    src = _source(fake)
    assert [s["ShardId"] for s in src._shards()] == fake.shard_ids
    assert len(src._shards()) == shard_count


def test_shards_are_discovered_once_and_cached():
    fake = _FakeKinesis(shard_count=250)
    src = _source(fake)
    src._shards()
    calls_after_first = len(fake.list_shards_calls)
    src._shards()
    assert len(fake.list_shards_calls) == calls_after_first  # cached, not re-listed
    assert calls_after_first == 3  # 250 shards over 100-shard pages


def test_poll_retires_a_closed_shard_instead_of_reusing_a_stale_iterator():
    """A `GetRecords` with no `NextShardIterator` means the shard is drained/closed.

    Reusing its final iterator on the next poll raises `ExpiredIteratorException` forever;
    the source must instead retire the shard and stop polling it.
    """

    class _ClosingKinesis(_FakeKinesis):
        def __init__(self) -> None:
            super().__init__(shard_count=1)
            self.get_records_calls = 0

        def get_records(self, **kwargs: Any) -> dict[str, Any]:
            self.get_records_calls += 1
            # The shard hands back its last records and then closes (no next iterator).
            return {"Records": [], "NextShardIterator": None}

    fake = _ClosingKinesis()
    src = _source(fake)
    assert src._poll() == []  # drains and retires the shard
    assert fake.get_records_calls == 1
    assert src._poll() == []  # retired: no second GetRecords against a stale iterator
    assert fake.get_records_calls == 1


def test_is_throttle_matches_by_name_and_by_client_error_code():
    from batcher.io.formats.streaming.kinesis import _is_throttle

    class ProvisionedThroughputExceededException(Exception):
        pass

    class _ClientError(Exception):
        def __init__(self, code):
            self.response = {"Error": {"Code": code}}

    assert _is_throttle(ProvisionedThroughputExceededException()) is True
    assert _is_throttle(_ClientError("ProvisionedThroughputExceededException")) is True
    assert _is_throttle(_ClientError("AccessDenied")) is False
    assert _is_throttle(ValueError("nope")) is False


def test_poll_skips_a_throttled_shard_instead_of_failing():
    class _Throttle(Exception):
        pass

    _Throttle.__name__ = "ProvisionedThroughputExceededException"

    class _ThrottlingKinesis(_FakeKinesis):
        def __init__(self):
            super().__init__(shard_count=1)
            self.calls = 0

        def get_records(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise _Throttle()  # first poll is throttled
            return {"Records": [], "NextShardIterator": "iter-1"}

    fake = _ThrottlingKinesis()
    src = _source(fake)
    assert src._poll() == []  # throttled shard skipped, no exception
    assert fake.calls == 1
    assert src._poll() == []  # next poll succeeds
    assert fake.calls == 2
