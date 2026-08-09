"""Kinesis broker source — one Split per shard, via ``boto3`` shard iterators.

Backed by ``boto3`` (the optional ``kinesis`` extra). A :class:`KinesisSource`
polls a shard with ``get_records`` (after obtaining a shard iterator with
``get_shard_iterator``) and assembles each poll into one Arrow batch via the
shared ``_make_batch`` helper.

``splits()`` returns one split per shard (the shard id is the offset locator), so
a distributed reader assigns one consumer per shard. The Kinesis sequence number
is opaque text; it is hashed into the int64 ``offset`` column so the fixed broker
schema is preserved (the raw sequence is not needed downstream, only ordering /
de-dup within a shard, which the sequence-number-based iterator already provides).

The ``boto3`` import is deferred to construction; if the extra is missing a
:class:`BackendError` instructs the user to install it.
"""

from __future__ import annotations

import hashlib
from typing import Any

from batcher._internal.errors import BackendError
from batcher.io.formats.base import SOURCES
from batcher.io.formats.streaming.broker import BrokerMessage, BrokerSource

__all__ = ["KinesisSource"]


def _import_boto3() -> Any:
    """Import ``boto3`` or raise a guiding ``BackendError``."""
    try:
        import boto3
    except ImportError as exc:
        raise BackendError(
            "reading from Kinesis needs the kinesis extra: pip install 'batcher-engine[kinesis]'"
        ) from exc
    return boto3


def _is_throttle(exc: BaseException) -> bool:
    """Whether `exc` is Kinesis back-pressure (a `GetRecords` throughput limit).

    Kinesis caps `GetRecords` at 5 transactions/second/shard and rejects an over-rate poll
    with ``ProvisionedThroughputExceededException``. That is normal back-pressure, not a
    failure — the records are still there next poll. Matched by class name and by the boto
    ``ClientError`` error code so the optional ``botocore`` need not be importable to
    recognize it.
    """
    return _is_aws_error(exc, "ProvisionedThroughputExceededException")


def _is_expired_iterator(exc: BaseException) -> bool:
    """Whether `exc` is an expired Kinesis shard iterator.

    A shard iterator is valid for five minutes. Any trigger interval longer than that — and
    any shard that simply goes quiet while the query polls its siblings — outlives its
    iterator, and the next `GetRecords` fails with `ExpiredIteratorException`. That is a
    routine, expected condition on a long-cadence stream, and letting it escape killed the
    whole query. The cure is to re-obtain the iterator from the last delivered sequence,
    which is exactly what recovery does.
    """
    return _is_aws_error(exc, "ExpiredIteratorException")


def _is_aws_error(exc: BaseException, name: str) -> bool:
    """Match a boto exception by class name or by its `ClientError` error code.

    Both spellings are needed and neither is sufficient: `boto3` synthesizes per-service
    exception classes at runtime (so the class name matches) while a generic `ClientError`
    carries only the code in its response. Matching by name rather than by `isinstance`
    keeps the optional `botocore` out of the import path.
    """
    if type(exc).__name__ == name:
        return True
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        return response.get("Error", {}).get("Code") == name
    return False


# AWS caps `GetRecords`'s `Limit` at 10,000 records per call, and rejects anything larger
# with `InvalidArgumentException`. The engine's usual 16,384-row morsel is therefore not a
# legal request here, so every poll is clamped to the API's ceiling.
_GET_RECORDS_MAX_LIMIT = 10_000

# Ceiling on concurrent `GetRecords` calls from one reader. A shard is rate-limited on its
# own, so more threads than shards buys nothing; this only stops a thousand-shard stream from
# opening a thousand sockets in one worker.
_MAX_FETCH_THREADS = 16


@SOURCES.register("kinesis")
class KinesisSource(BrokerSource):
    """An unbounded Kinesis stream, consumed via ``boto3``.

    The ``topic`` is the Kinesis stream name. Options: ``region`` (AWS region),
    ``starting_position`` (``"earliest"`` / ``"latest"``, the name every broker here shares)
    or its Kinesis spelling ``iterator_type`` (``"TRIM_HORIZON"`` / ``"LATEST"``), and
    ``partitions`` (the specific shards to read — set by :class:`BrokerSplit` on a
    worker; the values are shard *indices* into the discovered shard list).

    ``poll_size`` is the records requested per ``GetRecords`` call. AWS caps that at
    10,000, so a larger value is clamped rather than sent (it would be rejected).
    """

    format_name = "kinesis"

    __slots__ = (
        "_client_obj",
        "_closed",
        "_iterators",
        "_partitions",
        "_pool_obj",
        "_shard_ids",
    )

    def __init__(
        self,
        topic: str,
        *,
        poll_size: int = _GET_RECORDS_MAX_LIMIT,
        partitions: list[int] | None = None,
        region: str = "us-east-1",
        iterator_type: str = "TRIM_HORIZON",
        starting_position: str | None = None,
        **options: Any,
    ) -> None:
        # One option name across every broker (`starting_position`), mapped here onto
        # Kinesis's own `ShardIteratorType`. The native spelling still works.
        if starting_position is not None:
            from batcher.io.formats.streaming.broker.schema import normalize_starting_position

            iterator_type = normalize_starting_position(
                starting_position, aliases={"earliest": "TRIM_HORIZON", "latest": "LATEST"}
            )
        super().__init__(
            topic,
            poll_size=poll_size,
            region=region,
            iterator_type=iterator_type,
            **options,
        )
        self._partitions = partitions
        self._client_obj: Any = None
        self._shard_ids: list[str] | None = None
        self._iterators: dict[str, str] = {}
        # Lazily built; only a multi-shard reader ever needs it. See `_get_records`.
        self._pool_obj: Any = None
        # Shards drained to their end (a `GetRecords` returning no `NextShardIterator`).
        # Their records already flowed through the children a reshard created, so they must
        # never be polled again — reusing their final iterator raises `ExpiredIterator`.
        self._closed: set[str] = set()

    def _client(self) -> Any:
        if self._client_obj is None:
            boto3 = _import_boto3()
            self._client_obj = boto3.client("kinesis", region_name=self._options["region"])
        return self._client_obj

    def _shards(self) -> list[str]:
        """Every shard of the stream, paginated.

        `list_shards` returns at most 100 shards per call and hands back a `NextToken` for
        the rest. Reading only the first page looked like it worked — it just silently
        skipped every shard past the hundredth, which on a large stream is most of the
        data. Note that `StreamName` and `NextToken` are mutually exclusive in the API.

        The result is cached for the life of the source. That is correct across a
        *restart* — a fresh source re-lists, and `_shard_map` keys everything on stable
        shard numbers so a reshard is handled (see `_shard_map`). It does mean a shard
        created **mid-run**, without a restart, is not discovered until the next planning
        cycle; a continuous consumer that must pick up splits live needs a periodic
        re-list, which is a driver-timed change left for when there is a live stream to
        validate it against.
        """
        if self._shard_ids is None:
            client = self._client()
            shard_ids: list[str] = []
            kwargs: dict[str, Any] = {"StreamName": self.topic}
            while True:
                resp = client.list_shards(**kwargs)
                shard_ids.extend(s["ShardId"] for s in resp.get("Shards", []))
                token = resp.get("NextToken")
                if not token:
                    break
                kwargs = {"NextToken": token}
            self._shard_ids = shard_ids
        return self._shard_ids

    def _shard_map(self) -> dict[int, str]:
        """The stream's shards keyed by their **stable** shard number, not list position.

        This is the whole of the reshard-correctness fix. A checkpoint is
        ``{partition: sequence_number}``, and a Kinesis sequence number is only valid
        against the shard that produced it. When `partition` was the shard's *position*
        in `list_shards`, a reshard — a split or merge, the routine event partitioning
        exists to handle — shifted every position, so a sequence checkpointed for
        position 2 was replayed against whatever shard now sat at position 2:
        ``AFTER_SEQUENCE_NUMBER`` with a foreign sequence, silently skipping or
        re-reading records across the restart.

        A Kinesis ShardId is ``shardId-`` followed by a zero-padded, monotonically
        assigned number that never moves once created, so parsing it gives an identity
        that survives a reshard. A non-standard id (a test stub, a future format) falls
        back to a deterministic `sha256` — stable across processes, unlike `hash()`.
        """
        return {_shard_number(shard_id): shard_id for shard_id in self._shards()}

    def _discover_partitions(self) -> list[int]:
        if self._partitions is not None:
            return list(self._partitions)
        return sorted(self._shard_map())

    def _active_shards(self) -> list[tuple[int, str]]:
        """``(shard_number, shard_id)`` for the shards this reader should poll.

        A requested partition whose shard is no longer in the stream is dropped — but
        that is not the silent loss the old positional code risked: a shard absent from
        ``list_shards`` has been closed by a reshard, its records already drained through
        its children, which appear under their own new numbers.
        """
        shard_map = self._shard_map()
        numbers = self._partitions if self._partitions is not None else sorted(shard_map)
        return [
            (n, shard_map[n])
            for n in numbers
            if n in shard_map and shard_map[n] not in self._closed
        ]

    def _iterator(self, shard_id: str, shard_number: int) -> str:
        """A shard iterator, resuming after a checkpointed sequence when present.

        On recovery ``seek`` records the raw sequence number in ``_resume_from``
        (keyed by the stable shard number); the iterator is then obtained with
        ``AFTER_SEQUENCE_NUMBER`` so no record is replayed or skipped. Otherwise
        the configured ``iterator_type`` (``TRIM_HORIZON`` / ``LATEST``) applies.

        The *live* position (`_positions`, updated after every poll) takes precedence over
        the recovery position, because this is also the path that rebuilds an iterator that
        expired mid-run. Falling back to `_resume_from` there would have re-read the whole
        micro-batch history since the restart, and falling back to ``TRIM_HORIZON`` — which
        is what an absent `_resume_from` means — would have replayed the entire shard.
        """
        if shard_id not in self._iterators:
            client = self._client()
            resume = self._positions.get(shard_number, self._resume_from.get(shard_number))
            if resume is not None:
                resp = client.get_shard_iterator(
                    StreamName=self.topic,
                    ShardId=shard_id,
                    ShardIteratorType="AFTER_SEQUENCE_NUMBER",
                    StartingSequenceNumber=str(resume),
                )
            else:
                resp = client.get_shard_iterator(
                    StreamName=self.topic,
                    ShardId=shard_id,
                    ShardIteratorType=self._options["iterator_type"],
                )
            self._iterators[shard_id] = resp["ShardIterator"]
        return self._iterators[shard_id]

    def _poll(self) -> list[BrokerMessage] | None:
        shards = self._active_shards()
        if not shards:
            return []
        responses = self._get_records(shards)
        messages: list[BrokerMessage] = []
        for (shard_number, shard_id), resp in zip(shards, responses, strict=True):
            if resp is None:
                continue  # throttled or expired: retried on the next poll
            self._advance(shard_id, resp)
            messages.extend(self._decode(shard_number, resp))
        return messages

    def _get_records(self, shards: list[tuple[int, str]]) -> list[dict | None]:
        """One `GetRecords` per shard — concurrently when there is more than one.

        This was a sequential loop, so a micro-batch on a 64-shard stream cost 64 serialized
        HTTPS round-trips before a single row reached the plan. Latency scaled with the shard
        count, which is exactly backwards: shards are the unit of parallelism, and Kinesis
        rate-limits each one *independently*, so the calls have no reason to queue behind each
        other. A botocore client is thread-safe, so the fan-out shares one client; results come
        back positionally and every mutation of `_iterators` happens back on this thread.

        Args:
            shards: The ``(shard_number, shard_id)`` pairs to poll, in output order.

        Returns:
            One response per shard, positionally aligned; ``None`` where the shard was skipped
            for back-pressure or a stale iterator.
        """
        if len(shards) == 1:
            shard_number, shard_id = shards[0]
            return [self._get_records_one(shard_number, shard_id)]
        pool = self._pool(len(shards))
        futures = [pool.submit(self._get_records_one, n, sid) for n, sid in shards]
        return [f.result() for f in futures]

    def _pool(self, shards: int) -> Any:
        """The shared fetch pool, sized to the shards this reader owns."""
        from concurrent.futures import ThreadPoolExecutor

        if self._pool_obj is None:
            self._pool_obj = ThreadPoolExecutor(
                max_workers=min(shards, _MAX_FETCH_THREADS),
                thread_name_prefix="batcher-kinesis",
            )
        return self._pool_obj

    def _get_records_one(self, shard_number: int, shard_id: str) -> dict | None:
        """`GetRecords` for one shard, tolerating the two routine AWS refusals.

        Back-pressure leaves the iterator untouched so the records are read next poll. An
        *expired* iterator cannot be reused at all, so it is dropped: the next poll rebuilds
        it with `AFTER_SEQUENCE_NUMBER` from the last delivered sequence, which resumes at
        exactly the right place. Both used to escape and kill the query.
        """
        client = self._client()
        try:
            return client.get_records(  # type: ignore[no-any-return]
                ShardIterator=self._iterator(shard_id, shard_number),
                Limit=min(self.poll_size, _GET_RECORDS_MAX_LIMIT),
            )
        except Exception as exc:
            if _is_throttle(exc):
                # Back-pressure on this shard: skip it for this poll (its iterator is
                # unchanged, so its records are read next poll) rather than failing the
                # whole query. No sleep here — the trigger cadence paces the retry, and a
                # blocking sleep would stall the loop's stop signal.
                return None
            if _is_expired_iterator(exc):
                self._iterators.pop(shard_id, None)
                return None
            raise

    def _advance(self, shard_id: str, resp: dict) -> None:
        """Carry the shard's iterator forward, or retire a shard that has been drained."""
        next_iter = resp.get("NextShardIterator")
        if next_iter is not None:
            self._iterators[shard_id] = next_iter
            return
        # No next iterator means this shard is closed and fully drained. Retire it: drop the
        # now-invalid iterator and stop `_active_shards` from polling it, rather than reusing
        # the stale token forever (an `ExpiredIterator` loop).
        self._closed.add(shard_id)
        self._iterators.pop(shard_id, None)
        # A shard closes for exactly one reason: a reshard replaced it with children. Those
        # children are absent from the cached shard list, and the cache lives for the whole
        # run — so retiring the parent without invalidating the cache made the reader go
        # permanently quiet on the resharded key range. Every poll returned nothing, the
        # empty-poll back-off made it look like an idle stream, and the records that flowed
        # into the children were never read. Dropping the cache costs one `list_shards` per
        # reshard, which is as rare as a reshard is.
        self._shard_ids = None

    def _decode(self, shard_number: int, resp: dict) -> list[BrokerMessage]:
        """One shard's `GetRecords` response as broker messages."""
        messages = []
        for rec in resp.get("Records", []):
            ts = rec.get("ApproximateArrivalTimestamp")
            messages.append(
                BrokerMessage(
                    value=rec["Data"],
                    partition=shard_number,
                    offset=_seq_to_offset(rec["SequenceNumber"]),
                    # The raw sequence is the resume token (the int64 offset is a
                    # lossy hash); `AFTER_SEQUENCE_NUMBER` needs the exact string.
                    resume_token=rec["SequenceNumber"],
                    timestamp=int(ts.timestamp() * 1000) if ts is not None else 0,
                    topic=self.topic,
                    key=(rec.get("PartitionKey") or "").encode("utf-8") or None,
                )
            )
        return messages

    def close(self) -> None:
        """Shut the fetch pool down; the boto client owns no socket worth closing here."""
        if self._pool_obj is not None:
            pool, self._pool_obj = self._pool_obj, None
            pool.shutdown(wait=False)


def _shard_number(shard_id: str) -> int:
    """The stable numeric identity of a Kinesis shard, from its ShardId.

    A ShardId is ``shardId-`` followed by a zero-padded number assigned once and never
    reused, so the number is a durable partition key — unlike the shard's position in a
    `list_shards` response, which a reshard reorders. A ShardId that does not fit the
    format (a test stub, a hypothetical future scheme) falls back to a deterministic
    `sha256`: stable across processes and workers, which `hash()` is not, so a checkpoint
    still round-trips to the same shard after a restart.
    """
    _, _, suffix = shard_id.partition("shardId-")
    if suffix.isdigit():
        return int(suffix)
    return int.from_bytes(hashlib.sha256(shard_id.encode("utf-8")).digest()[:8], "big") % (1 << 63)


def _seq_to_offset(sequence_number: str) -> int:
    """Map an opaque Kinesis sequence number to a stable int64 offset column.

    The raw sequence is a large decimal string; take it modulo 2**63 so it fits
    the fixed int64 ``offset`` column while preserving monotonic ordering within
    the precision of int64 (sequence numbers within a shard are increasing).

    A non-numeric sequence falls back to a `sha256` digest rather than `hash()`: Python salts
    `str` hashing per process, so the fallback produced a different `offset` for the same
    record on every run and on every worker — silently breaking the ordering and de-dup the
    column exists for, across exactly the restart and distributed boundaries that matter.
    """
    try:
        return int(sequence_number) % (1 << 63)
    except ValueError:
        return int.from_bytes(
            hashlib.sha256(sequence_number.encode("utf-8")).digest()[:8], "big"
        ) % (1 << 63)
