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

from typing import Any

from batcher._internal.optional import require
from batcher.io.formats.base import SOURCES
from batcher.io.formats.streaming.broker import BrokerMessage, BrokerSource, opaque_offset

__all__ = ["KinesisSource"]


def _import_boto3() -> Any:
    """Import ``boto3`` or raise a guiding ``BackendError``."""
    return require("boto3", feature="Kinesis support", provides="boto3", extra="kinesis")


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
        "_adopted",
        "_client_obj",
        "_closed",
        "_iterators",
        "_partitions",
        "_pool_obj",
        "_shards_cache",
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
        self._shards_cache: list[dict[str, Any]] | None = None
        self._iterators: dict[str, str] = {}
        # Lazily built; only a multi-shard reader ever needs it. See `_get_records`.
        self._pool_obj: Any = None
        # Shards drained to their end (a `GetRecords` returning no `NextShardIterator`).
        # Their records already flowed through the children a reshard created, so they must
        # never be polled again — reusing their final iterator raises `ExpiredIterator`.
        self._closed: set[str] = set()
        # Children this reader has taken over from a drained shard of its own. They are the
        # continuation of a key range it already owns, so they are read from the *beginning*
        # rather than from the configured `iterator_type`. See `_adopt_children`.
        self._adopted: set[str] = set()

    def _client(self) -> Any:
        if self._client_obj is None:
            boto3 = _import_boto3()
            self._client_obj = boto3.client("kinesis", region_name=self._options["region"])
        return self._client_obj

    def _shards(self) -> list[dict[str, Any]]:
        """Every shard of the stream, paginated, with its lineage.

        `list_shards` returns at most 100 shards per call and hands back a `NextToken` for
        the rest. Reading only the first page looked like it worked — it just silently
        skipped every shard past the hundredth, which on a large stream is most of the
        data. Note that `StreamName` and `NextToken` are mutually exclusive in the API.

        The whole descriptor is kept, not just the id, because ``ParentShardId`` /
        ``AdjacentParentShardId`` are what say which shards *replaced* a shard a reshard
        closed — the fact `_adopt_children` needs and that a bare id list threw away.

        The result is cached for the life of the source, and `_advance` invalidates the
        cache the moment a shard is drained, which is exactly when a reshard has happened.
        """
        if self._shards_cache is None:
            client = self._client()
            shards: list[dict[str, Any]] = []
            kwargs: dict[str, Any] = {"StreamName": self.topic}
            while True:
                resp = client.list_shards(**kwargs)
                shards.extend(resp.get("Shards", []))
                token = resp.get("NextToken")
                if not token:
                    break
                kwargs = {"NextToken": token}
            self._shards_cache = shards
        return self._shards_cache

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
        return {_shard_number(s["ShardId"]): s["ShardId"] for s in self._shards()}

    def _discover_partitions(self) -> list[int]:
        if self._partitions is not None:
            return list(self._partitions)
        return sorted(self._shard_map())

    def _active_shards(self) -> list[tuple[int, str]]:
        """``(shard_number, shard_id)`` for the shards this reader should poll.

        A reader pinned to a partition set — which is every reader on the distributed path,
        since :class:`BrokerSplit` rebuilds the source as ``partitions=[n]`` — used to poll
        *only* the shards whose numbers were in that set. A reshard is the routine event
        partitioning exists to absorb, and it replaces a shard with children carrying
        **new** numbers. Those numbers were in nobody's pinned set, so once the parent
        drained the reader went permanently quiet on that key range: every poll returned
        nothing, the empty-poll back-off made it look like an idle stream, and every record
        written to the children was silently lost. Nothing raised, and no count was wrong
        anywhere it could be compared.

        `_adopt_children` closes that by following the lineage `list_shards` reports, so a
        pinned reader takes over the children of its own drained shards.
        """
        shards = self._shards()
        self._adopt_children(shards)
        owned = self._owned_shard_ids(shards)
        return sorted((_shard_number(sid), sid) for sid in owned if sid not in self._closed)

    def _owned_shard_ids(self, shards: list[dict[str, Any]]) -> set[str]:
        """The shard ids this reader is responsible for: its pinned set, plus adoptions."""
        if self._partitions is None:
            return {s["ShardId"] for s in shards}  # this reader owns the whole stream
        wanted = set(self._partitions)
        return {
            s["ShardId"]
            for s in shards
            if _shard_number(s["ShardId"]) in wanted or s["ShardId"] in self._adopted
        }

    def _adopt_children(self, shards: list[dict[str, Any]]) -> None:
        """Take over the children of every shard of ours that has been drained.

        A drained shard (`_closed`) has been replaced by children a reshard created. Its
        records are already read; theirs are not, and they belong to the key range this
        reader owns — so it must take them over rather than wait for a replan.

        **Exactly one reader may adopt a merge child**, or its records are delivered twice.
        A merge child has two parents, which may sit on two different readers, and neither
        can see the other's state. The owner is defined as the reader holding the
        *lowest-numbered* parent: a rule each reader evaluates from the child's own
        lineage, with no coordination, that names the same reader everywhere. A split child
        has a single parent, so the rule reduces to "the parent's owner".

        Adoption is transitive — a child can itself be resharded — so this runs to a
        fixpoint rather than one level deep.
        """
        pending = [s for s in shards if s["ShardId"] not in self._adopted]
        while True:
            newly = [s for s in pending if self._should_adopt(s)]
            if not newly:
                return
            self._adopted.update(s["ShardId"] for s in newly)
            adopted_now = {s["ShardId"] for s in newly}
            pending = [s for s in pending if s["ShardId"] not in adopted_now]

    def _should_adopt(self, shard: dict[str, Any]) -> bool:
        """Whether this reader takes over `shard` from a parent it has drained."""
        parents = _parent_ids(shard)
        if not parents:
            return False
        primary = min(parents, key=_shard_number)
        # Not drained yet: the parent still holds records, and Kinesis orders a child
        # strictly after its parents. Reading the child now would deliver out of order.
        if primary not in self._closed:
            return False
        return self._owns(primary)

    def _owns(self, shard_id: str) -> bool:
        """Whether `shard_id` is one this reader is responsible for."""
        if self._partitions is None:
            return True
        return _shard_number(shard_id) in set(self._partitions) or shard_id in self._adopted

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

        A shard *adopted* from a drained parent (`_adopt_children`) starts at
        ``TRIM_HORIZON`` whatever `iterator_type` says. It is the continuation of a range
        this reader was already following, and its records begin at the reshard: under
        ``LATEST`` every record written between the reshard and the first poll of the child
        would be skipped, which is the same silent loss adoption exists to stop.
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
                start = (
                    "TRIM_HORIZON" if shard_id in self._adopted else self._options["iterator_type"]
                )
                resp = client.get_shard_iterator(
                    StreamName=self.topic, ShardId=shard_id, ShardIteratorType=start
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
        # reshard, which is as rare as a reshard is. `_adopt_children` then reads the fresh
        # listing's lineage and takes the children over.
        self._shards_cache = None

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
                    resume_token=rec["SequenceNumber"],  # exact; `offset` is the lossy one
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


def _parent_ids(shard: dict[str, Any]) -> list[str]:
    """The shards a `list_shards` descriptor names as this shard's parents.

    A shard created by a *split* has one ``ParentShardId``; one created by a *merge* has
    that plus an ``AdjacentParentShardId``. A shard that predates any reshard has neither.
    """
    return [
        shard[key]
        for key in ("ParentShardId", "AdjacentParentShardId")
        if shard.get(key) is not None
    ]


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
    return opaque_offset(suffix if suffix.isdigit() else shard_id)


#: Kinesis's spelling of the shared "opaque native position -> int64 offset column" rule.
#: The same digest-with-a-numeric-fast-path was pasted into four brokers; it lives in
#: `broker.schema` now, because four copies of one projection is four chances for one of
#: them to drift back to `hash()` — which is per-process salted, and so produced a different
#: offset for the same record on every worker.
_seq_to_offset = opaque_offset
