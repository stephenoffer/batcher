"""Arbitrary keyed state over a stream — the fold behind `transform_with_state`.

The relational operators cover the shapes that have an algebra: an aggregate has a
`combine`, a window has an eviction rule, a dedup has a seen-set. Sessionization with
custom rules, a running fraud score, a per-device state machine, "alert when this key has
been silent for ten minutes" — those have none, and every stream processor grows an
escape hatch for them. Spark's is `mapGroupsWithState` / `transformWithState`; this is
the same bargain: a user function owns one key's state, and the engine owns *when* it is
called, checkpointed, and expired.

Core's lane, exactly: this drives the user's function and measures what it retained. It
makes no optimization decision and owns no resource.

Two properties are the whole design.

**Bounded state.** A key's state lives until its TTL expires. Without one, the operator
retains a row per key seen since the query started — correct, and a leak measured in days
on any real key space. The TTL is checked once per micro-batch against the *engine's*
clock rather than per key on a timer, so there is no timer thread.

That check costs **O(keys expired)**, not O(keys retained), and the difference is what
decides whether the operator has a scale ceiling. A trigger that touches ten keys used to
walk all of them three times over — once to find the stale ones, twice more to size the
retained bytes for the budget and the metrics — so a ten-million-key space paid thirty
million dict steps per second to expire nothing. Insertion order *is* touch order here
(a touched key is reinserted, and the clock is held non-decreasing), so expiry walks the
stale prefix and stops at the first live key, and the byte estimate is a running maximum
rather than a scan.

**Checkpointable state.** State is a flat mapping of scalars, so the whole key space
serializes as one Arrow `RecordBatch` — the same thing `_AggFold` hands the `StateStore`.
That is why the constraint exists rather than accepting an arbitrary object: an operator
whose state cannot be written down cannot survive a restart, and a streaming operator that
cannot survive a restart is a demo.
"""

from __future__ import annotations

import json
import time
from itertools import pairwise
from typing import Any

import pyarrow as pa

from batcher._internal.native import engine
from batcher.config import active_config
from batcher.plan.logical import TransformWithState
from batcher.plan.streaming import StateOperatorProgress

__all__ = ["KeyedStateFold"]

#: The column the per-key last-touched timestamp rides in, inside the state snapshot.
#: Prefixed so it cannot collide with a user's state field, and stripped before the user
#: ever sees a state mapping.
_TOUCHED = "__bt_state_touched_us"

#: The column the group key rides in when the state is snapshotted. One JSON-encoded
#: column rather than one column per key: the keys may be any mix of types, and a snapshot
#: whose *shape* depends on the key types cannot be restored by a run that has not yet
#: seen a row. JSON is the same encoding the checkpoint's offset log already uses for an
#: opaque source position.
_KEY = "__bt_state_key"


class KeyedStateFold:
    """Per-key user state, folded across micro-batches and expired by a TTL."""

    __slots__ = (
        "_cap",
        "_cfg",
        "_clock",
        "_dropped",
        "_fn",
        "_input_ir",
        "_keys",
        "_nat",
        "_state",
        "_ttl",
        "_widest",
    )

    def __init__(self, node: TransformWithState) -> None:
        self._nat = engine()
        self._fn = node.fn
        self._keys = list(node.group_keys)
        self._ttl = node.ttl_micros
        self._input_ir = json.dumps(node.input.to_ir())
        # Constant for the query, so read and serialize it once rather than per micro-batch
        # (the same hoist `_AggFold` makes, and for the same reason).
        self._cfg = active_config().engine_config_json()
        #: ``{key_tuple: (state_mapping, last_touched_micros)}``, in **last-touched order**.
        #: A touched key is removed and reinserted so it moves to the end, which is what lets
        #: `_expire` stop at the first key that is still live instead of walking the rest.
        self._state: dict[tuple, tuple[dict[str, Any], int]] = {}
        self._cap = active_config().memory.streaming_state_budget_bytes()
        self._dropped = 0
        #: The most state fields any key has held. A running maximum rather than a scan, so
        #: the budget check is O(1); it never falls when a wide key is forgotten, which
        #: over-estimates the footprint and is the safe direction for a cap.
        self._widest = 0
        #: The last stamp handed out, held non-decreasing. `time.time()` can step backwards
        #: (an NTP correction), and a key stamped into the future would sit at the head of
        #: the order and stop expiry for every key behind it.
        self._clock = 0

    # --- the fold ---------------------------------------------------------
    def push(self, batch: pa.RecordBatch) -> list[pa.RecordBatch]:
        """Run one micro-batch through the user function, one call per key present.

        A key with no rows this micro-batch is *not* called. That is deliberate and it is
        Spark's rule too: calling every known key on every batch turns a million-key state
        into a million Python calls per trigger, which is the cost model of a timer service
        rather than of a stream operator. A key that must act without input acts when its
        TTL expires instead.

        Args:
            batch: One source micro-batch.

        Returns:
            The output batches the user function produced, empty ones dropped.
        """
        self._dropped = 0
        if batch.num_rows == 0:
            return self._expire()
        rows = self._nat.execute_plan(self._input_ir, [[batch]], self._cfg)
        out: list[pa.RecordBatch] = []
        now = self._tick()
        for key, group in _group_by(rows, self._keys):
            previous = self._state.get(key)
            produced, new_state = self._fn(key, group, previous[0] if previous else None)
            # Removed either way: a forgotten key is gone, and a retained one has to be
            # *reinserted* to move to the end of the last-touched order `_expire` reads.
            self._state.pop(key, None)
            if new_state is not None:
                checked = _check_state(new_state, key)
                self._widest = max(self._widest, len(checked))
                self._state[key] = (checked, now)
            emitted = _as_batch(produced)
            if emitted is not None and emitted.num_rows:
                out.append(emitted)
        out.extend(self._expire())
        self._check_bounded()
        return out

    def _tick(self) -> int:
        """The stamp for this micro-batch, never earlier than the one before it.

        `time.time()` is not monotonic — an NTP correction steps it backwards — and one
        backwards step would file a key ahead of keys touched before it. `_expire` reads the
        dict's order as touch order and stops at the first live key, so a single
        out-of-order entry near the head would pin every stale key behind it forever.
        """
        self._clock = max(self._clock, _now_micros())
        return self._clock

    def _expire(self) -> list[pa.RecordBatch]:
        """Forget every key untouched for longer than the TTL.

        Walks the **stale prefix only**. Keys sit in last-touched order (`push` reinserts a
        touched key at the end and `_tick` keeps the stamps non-decreasing), so the first key
        that is still live proves every key after it is too, and the scan stops there. That
        is the difference between paying for what expires and paying for what is retained:
        a trigger that expires nothing out of ten million keys does one dict step.

        Returns an empty list — expiry emits nothing. It is a list so the caller can
        `extend` it unconditionally, and so a future event-time expiry that *does* emit
        (Spark's `GroupState` timeout callback) is a change to this function alone.
        """
        self._dropped = 0
        if self._ttl <= 0 or not self._state:
            return []
        cutoff = self._tick() - self._ttl
        stale = []
        for key, (_, touched) in self._state.items():
            if touched >= cutoff:
                break  # and so is everything behind it — the order guarantees it
            stale.append(key)
        for key in stale:
            del self._state[key]
        self._dropped = len(stale)
        return []

    def _check_bounded(self) -> None:
        """Fail loudly when the retained state has outgrown its budget."""
        held = self.nbytes()
        if held <= self._cap:
            return
        from batcher._internal.errors import ResourceError

        hint = (
            "no state_ttl was set, so no key is ever forgotten"
            if self._ttl <= 0
            else "the key space is growing faster than the TTL expires it"
        )
        raise ResourceError(
            f"transform_with_state retained {held} bytes across {len(self._state)} keys "
            f"(cap {self._cap}): {hint}. Set or shorten state_ttl=, narrow the group keys, "
            "or raise memory.streaming_state_max_bytes."
        )

    # --- metrics and checkpointing ---------------------------------------
    def nbytes(self) -> int:
        """A conservative estimate of the retained state's footprint.

        The state is a Python dict of small mappings, so `sys.getsizeof` on the container
        undercounts it by the size of everything it points at. Charging a flat per-entry
        cost plus the key and value counts tracks the shape that actually grows — the
        number of keys — which is what the budget is defending against.

        The field count is the running maximum `push` maintains, not a scan of every value.
        This is called twice per micro-batch (the budget check and the metrics), so a scan
        here was two full walks of the key space per trigger to compute a number that
        changes only when a key holds more fields than any key ever has.
        """
        if not self._state:
            return 0
        per_field = 64  # a boxed scalar plus its dict slot, rounded up
        fields = 1 + self._widest
        return len(self._state) * (per_field * (fields + len(self._keys)))

    def metrics(self) -> StateOperatorProgress:
        """This operator's retained state after the last `push`."""
        return StateOperatorProgress(
            operator_name="transform_with_state",
            num_rows_total=len(self._state),
            num_rows_removed=self._dropped,
            memory_used_bytes=self.nbytes(),
        )

    def state(self) -> pa.RecordBatch | None:
        """The whole key space as one checkpointable batch, or None when empty.

        The key rides as JSON in one column rather than as one column per key, because a
        snapshot whose *shape* depends on the key types cannot be restored by a run that
        has not yet seen a row — and restore happens before the first row by construction.
        """
        if not self._state:
            return None
        keys = [json.dumps(list(key)) for key in self._state]
        touched = [stamp for _, stamp in self._state.values()]
        fields = sorted({name for value, _ in self._state.values() for name in value})
        columns: dict[str, Any] = {_KEY: pa.array(keys, type=pa.string())}
        for name in fields:
            columns[name] = pa.array([value.get(name) for value, _ in self._state.values()])
        columns[_TOUCHED] = pa.array(touched, type=pa.int64())
        return pa.record_batch(columns)

    def restore(self, state: pa.RecordBatch) -> None:
        """Rebuild the key space from a checkpoint snapshot.

        The snapshot's row order is the last-touched order `state()` wrote it in, and
        rebuilding in that order is what keeps `_expire`'s early stop correct across a
        restart. The derived counters are rebuilt with it: a restored `_widest` of zero
        would report a near-empty footprint and leave the state budget unenforced until
        some key happened to be touched.
        """
        self._state = {}
        self._widest = 0
        if state is None or state.num_rows == 0:
            return
        names = [n for n in state.schema.names if n not in (_KEY, _TOUCHED)]
        keys = state.column(_KEY).to_pylist()
        touched = state.column(_TOUCHED).to_pylist()
        values = {name: state.column(name).to_pylist() for name in names}
        ordered = sorted(range(len(keys)), key=lambda i: int(touched[i]))
        for i in ordered:
            key = tuple(json.loads(keys[i]))
            fields = {n: values[n][i] for n in names}
            self._widest = max(self._widest, len(fields))
            self._state[key] = (fields, int(touched[i]))
        self._clock = max(self._clock, int(touched[ordered[-1]]))


def _now_micros() -> int:
    """Processing-time now, in the microseconds every other streaming bound uses."""
    return int(time.time() * 1_000_000)


def _check_state(state: Any, key: tuple) -> dict[str, Any]:
    """Reject a state the checkpoint could not write down, naming the key that produced it.

    Failing here rather than at the next snapshot is the difference between a message that
    names the offending key and a `pa.lib.ArrowInvalid` from inside the checkpoint writer,
    an hour later, on a query that has been running the whole time.
    """
    from batcher._internal.errors import PlanError

    if not isinstance(state, dict):
        raise PlanError(
            f"transform_with_state: the state returned for key {key!r} is a "
            f"{type(state).__name__}, but state must be a flat mapping of scalars so it "
            "can be checkpointed. Keep a large payload elsewhere and hold a reference."
        )
    for name, value in state.items():
        if isinstance(value, (dict, list, tuple, set)):
            raise PlanError(
                f"transform_with_state: state field {name!r} for key {key!r} is a "
                f"{type(value).__name__}; state fields must be scalars so the key space "
                "serializes as one Arrow batch. Flatten it, or hold a reference to it."
            )
    return state


def _as_batch(produced: Any) -> pa.RecordBatch | None:
    """Normalize what the user function emitted into a `RecordBatch`, or None."""
    if produced is None:
        return None
    if isinstance(produced, pa.RecordBatch):
        return produced
    if isinstance(produced, pa.Table):
        batches = produced.combine_chunks().to_batches()
        return batches[0] if batches else None
    return pa.record_batch(produced)


def _group_by(batches: list[pa.RecordBatch], keys: list[str]):
    """Yield ``(key_tuple, rows)`` for each distinct key across `batches`.

    Sorted by the key columns and cut on the boundaries, so the only Python here runs once
    per *group* — never once per row. The whole premise of the operator is that the user
    function is the per-key cost; an O(rows) Python partition in front of it would make the
    engine the bottleneck before the user's code ran at all, and it is the exact shape
    `.claude/rules/architecture.md` forbids in the control plane.

    Null keys group together, as they do in `group_by`: `NULL = NULL` is null in SQL's
    three-valued logic, but grouping treats two nulls as the same group, and the boundary
    test has to say so explicitly or every null-keyed row becomes its own group.

    Args:
        batches: The micro-batch's rows, already through the input pipeline.
        keys: The group key column names.

    Yields:
        ``(key_values, rows)`` per distinct key, the rows as a zero-copy slice.
    """
    import pyarrow.compute as pc

    if not batches:
        return
    table = pa.Table.from_batches(batches).combine_chunks()
    if table.num_rows == 0:
        return
    order = pc.sort_indices(table, sort_keys=[(name, "ascending") for name in keys])
    table = table.take(order).combine_chunks()

    changed = None
    for name in keys:
        column = table.column(name).combine_chunks()
        if isinstance(column, pa.ChunkedArray):
            column = column.combine_chunks()
        previous = pa.concat_arrays([pa.nulls(1, column.type), column.slice(0, len(column) - 1)])
        same = pc.or_(
            pc.fill_null(pc.equal(column, previous), False),
            pc.and_(pc.is_null(column), pc.is_null(previous)),
        )
        step = pc.invert(same)
        changed = step if changed is None else pc.or_(changed, step)

    starts = pc.indices_nonzero(changed).to_pylist()
    if not starts or starts[0] != 0:
        starts.insert(0, 0)  # row 0 opens the first group whatever the boundary test said
    bounds = [*starts, table.num_rows]
    columns = [table.column(name) for name in keys]
    for start, end in pairwise(bounds):
        if end <= start:
            continue
        key = tuple(column[start].as_py() for column in columns)
        rows = table.slice(start, end - start).combine_chunks().to_batches()
        if rows:
            yield key, rows[0]
