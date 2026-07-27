"""`MetadataHub` — the façade over a `MetadataBackend`.

The Hub is the single seam the rest of the system uses to read learned state and
write feedback. It implements the `FeedbackSink` contract so Core can hand it
`OperatorFeedback` directly. Writes are best-effort and must never raise into the
hot path; reads return learned parameters for warm-starting plans.

Only a thin slice is implemented for the bootstrap engine (feedback recording +
generic param load/save). Sketch persistence, workload fingerprints, bandit
posteriors, and the compiled-artifact cache layer onto these same primitives.

**The Hub's derived views are maintained incrementally, never re-derived.** Kyber's
calibration, its cardinality correction, Carbonite's learned memory model, and the
CPU-share loop each want an *aggregate* over the feedback history, and each used to
get it by re-scanning the whole `op_stats` table and re-running `json.loads` over
every stored row. Because `record` fires once per operator per query, a read cache
keyed on "has anything been written" was invalidated by the very query that wanted
to read it — so every query paid a scan proportional to the session's cumulative
history, making a long-lived session (a notebook, a BI server) quadratic in the
number of queries it had already run. The Hub therefore owns the parsed rows and
folds each new one into the bucketed views as it arrives: a read is O(1), a record
is O(1), and the backend is scanned exactly once per view per process.
"""

from __future__ import annotations

import json
import threading
from typing import Any

from batcher._internal.errors import ConfigError
from batcher._internal.logging import get_logger
from batcher.metadata.hardware_scope import measured_here
from batcher.metadata.store import MetadataBackend, check_backend
from batcher.metadata.views import (
    PER_KIND_MAX,
    SIGNED_HISTORY_MAX,
    bucket_by_kind,
    chronological_signed,
    trimmed,
)
from batcher.plan.feedback import OperatorFeedback

__all__ = ["MetadataHub"]

_log = get_logger("metadata")

# Logical tables.
_OP_STATS = "op_stats"
_LEARNED_PARAMS = "learned_params"

# Cap on the `op_stats` rows a *forgettable* backend retains, and how often it is enforced.
# The views above are bounded; the table beneath them was not, so a long-lived session grew
# the store by one row per operator per query for its whole life with nothing reading the
# old ones. The cap sits far above what rebuilding either view needs, so pruning is
# invisible to consumers. A durable backend keeps everything — that is what it is for, and
# it simply does not offer the `delete` this uses.
_OP_STATS_MAX = 65_536
_OP_STATS_PRUNE_EVERY = 4_096

# Sentinel for "the parsed view has no entry under this key" — distinct from a stored `None`,
# which `_unchanged` must be able to recognize as already-written.
_MISSING = object()


# `OperatorFeedback`'s field names, resolved once. Every field is a scalar, so the row is
# a flat `{name: value}` — `dataclasses.asdict` would deep-copy recursively and re-derive
# the field tuple on each of the several calls a query makes.
_FEEDBACK_FIELDS: tuple[str, ...] = tuple(OperatorFeedback.__dataclass_fields__)


def _row_of(feedback: OperatorFeedback) -> dict[str, Any]:
    """One feedback row as the flat JSON-shaped dict both the store and the views hold."""
    return {name: getattr(feedback, name) for name in _FEEDBACK_FIELDS}


def _check_namespace(namespace: str) -> None:
    """Reject a namespace that cannot address a stored entry.

    The store's keys are tuples the backends JSON-encode, so a non-string namespace
    writes under one spelling and reads back under another — a silent "the learning
    loop never persists anything", not an error.
    """
    if not isinstance(namespace, str) or not namespace:
        raise ConfigError(
            f"A learned-parameter namespace must be a non-empty string, but got "
            f"{type(namespace).__name__} {namespace!r}.",
            hint="Namespaces are dotted names, e.g. 'kyber.cardinality'.",
        )


def _encoded(where: str, value: Any) -> bytes:
    """`value` as JSON bytes, or a typed error naming the entry that could not encode.

    `json.dumps` reports only the offending *type*, which in a map of learned stats is
    never enough to find the entry. Naming the namespace (and, for a dict, the key)
    turns a dead end into a one-line fix.
    """
    try:
        return json.dumps(value).encode()
    except TypeError as exc:
        culprit = ""
        if isinstance(value, dict):
            for key, item in value.items():
                try:
                    json.dumps(item)
                except TypeError:
                    culprit = f" (entry {key!r} has type {type(item).__name__})"
                    break
        raise ConfigError(
            f"Learned parameters for {where!r} are not JSON-serializable{culprit}: {exc}.",
            hint="Learned stats are stored as JSON, so use only str/int/float/bool/list/dict.",
        ) from exc


class MetadataHub:
    """Reads learned state and absorbs execution feedback."""

    def __init__(self, backend: MetadataBackend) -> None:
        """Wrap a persistence backend.

        Args:
            backend: Where learned state is stored. Validated here, not on first use:
                the hub is a process singleton read from several call sites, so a
                malformed backend would otherwise surface as an `AttributeError` inside
                whichever query happened to read learned stats first.

        Raises:
            ConfigError: If `backend` does not implement `MetadataBackend`.
        """
        self._backend = check_backend(backend)
        self._seq = 0
        # Parsed-read cache for the learned-parameter tables. `_params_generation` bumps
        # only on a *whole-blob* `save_params` (which the per-key view merges underneath
        # itself); a `put_keyed_param` refreshes the cached entry in place, and `record`
        # — which writes a different table entirely — no longer disturbs it at all. The
        # earlier single generation counter was bumped by every feedback row, so this
        # cache missed on every query it existed to serve.
        self._params_generation = 0
        self._keyed_cache: dict[str, tuple[int, dict[str, Any]]] = {}
        # Per namespace, the keys known to be backed by their own `(namespace, key)` backend
        # entry — as opposed to merged up from the legacy single-blob shape. `_unchanged`
        # only elides a redundant write for a key in here, so a legacy entry still migrates.
        self._keyed_stored: dict[str, set[str]] = {}
        # Bucketed-by-kind view of the feedback history: loaded from the backend once,
        # then folded forward by `record`. Consumers reduce each bucket to a median or a
        # regression, so buckets are bounded (`PER_KIND_MAX`) — the whole point is that
        # neither a read nor a record costs anything proportional to session history.
        self._by_kind: dict[str, list[dict[str, Any]]] | None = None
        # Chronological, bounded, in-memory view of the signature-carrying feedback rows.
        # Kyber's cardinality correction reads it on *every* optimize. Loaded once from
        # the backend on first read, then maintained incrementally by `record`.
        self._signed: list[dict[str, Any]] | None = None
        # Total rows ever appended to `_signed`, including any since trimmed away. Lets an
        # incremental consumer (Kyber's q-error fold) tell how many rows are new since it
        # last read, so it absorbs only those instead of re-folding the whole view.
        self._signed_appends = 0
        # `record` is the one writer, and several queries call it at once: the hub is a
        # process singleton (`core.default_hub`), so two concurrent pipelines both fold
        # their measurements in here. `self._seq += 1` is a read-modify-write, and `_seq`
        # is half the storage key — a lost update makes two rows collide and one query's
        # feedback silently overwrite another's. Measured with preemption forced: 124 of
        # 32,000 rows collided. Core measures and Kyber consumes, so a dropped row is not
        # a wrong answer, it is a plan that quietly stops improving. `record` runs once per
        # operator per query, so serializing it costs nothing on the hot path.
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        """Name the backend and how much has been learned.

        "Is my learned-stats store actually being written to?" is the question a user
        prints a hub to answer, and an address answers none of it.
        """
        return (
            f"MetadataHub(backend={self._backend!r}, recorded={self._seq}, "
            f"signed={self._signed_appends})"
        )

    # --- FeedbackSink ------------------------------------------------------
    def record(self, feedback: OperatorFeedback) -> None:
        """Persist one operator's feedback. Never raises into the caller.

        Thread-safe: concurrent pipelines share one hub (see `_lock`).
        """
        try:
            with self._lock:
                self._record_locked(feedback)
        except Exception:  # pragma: no cover - feedback must not break execution
            _log.warning("dropped operator feedback", exc_info=True)

    def _record_locked(self, feedback: OperatorFeedback) -> None:
        """The body of `record`, serialized against concurrent writers by `_lock`."""
        self._seq += 1
        key = (int(feedback.op_id), self._seq)
        row = _row_of(feedback)
        self._backend.put(_OP_STATS, key, json.dumps(row).encode())
        # Fold the row into whichever derived views have been materialized. A view
        # still `None` has not been read yet; its lazy load will pick this row up
        # from the backend, so there is nothing to do.
        # Same filter `_load_by_kind` applies; the two must agree or it leaks after a write.
        if self._by_kind is not None and measured_here(row):
            bucket = self._by_kind.setdefault(row["kind"], [])
            bucket.append(row)
            trimmed(bucket, PER_KIND_MAX)
        if self._signed is not None and row["signature"]:
            self._signed.append(row)  # keep the hot view current without a re-scan
            self._signed_appends += 1
            trimmed(self._signed, SIGNED_HISTORY_MAX)
        if self._seq % _OP_STATS_PRUNE_EVERY == 0:
            self._prune_op_stats()

    def _prune_op_stats(self) -> None:
        """Bound the stored operator feedback to its newest `_OP_STATS_MAX` rows.

        Amortized: the scan-and-drop costs O(stored) but runs once every
        `_OP_STATS_PRUNE_EVERY` records. Keys carry a monotonic sequence number, so "newest"
        is exact without parsing values. A backend with no `delete` is left alone.
        """
        delete = getattr(self._backend, "delete", None)
        if delete is None:
            return
        keys = [key for key, _value in self._backend.scan(_OP_STATS, ())]
        if len(keys) <= _OP_STATS_MAX:
            return
        keys.sort(key=lambda k: k[1])  # (op_id, seq) -> oldest sequence first
        delete(_OP_STATS, keys[: len(keys) - _OP_STATS_MAX])

    @property
    def signed_appends(self) -> int:
        """Count of signature-carrying rows ever appended to the `op_stats` view.

        A cursor for an incremental consumer of `op_stats_with_signature`: the difference
        against a previously observed value is exactly how many rows at the tail of that
        list are new, so a fold over the history can absorb only what it has not seen.
        """
        return self._signed_appends

    @property
    def version(self) -> int:
        """A monotonic counter that bumps on every recorded feedback row.

        A cheap change signal for caches built over the hub's `op_stats` (e.g. cost
        calibration): an unchanged version means the measured history this hub has
        absorbed is unchanged, so a derived computation can be reused instead of
        re-scanning the whole history. Resets only when a fresh hub is constructed.
        """
        return self._seq

    def operator_history(self, op_id: int) -> list[dict[str, Any]]:
        """All recorded feedback for an operator id, oldest first.

        Args:
            op_id: The operator's plan-local id.

        Returns:
            The recorded rows, oldest first. Empty when nothing was recorded.

        Raises:
            ConfigError: If `op_id` is not an integer. The store's keys are typed, so a
                string id matches nothing and would otherwise read as "never recorded".
        """
        if not isinstance(op_id, int) or isinstance(op_id, bool):
            raise ConfigError(
                f"operator_history needs an integer op_id, but got "
                f"{type(op_id).__name__} {op_id!r}.",
                hint="Operator ids are the plan-local integers on a PhysicalPlan's ops.",
            )
        out = [json.loads(value) for _key, value in self._backend.scan(_OP_STATS, (op_id,))]
        return out

    def op_stats_by_kind(self) -> dict[str, list[dict[str, Any]]]:
        """Operator feedback **measured on this machine**, bucketed by operator `kind`.

        The shape Kyber's cost calibration consumes: per-row/per-byte coefficients
        are fit per operator family (`scan`, `filter`, `hash_join`, ...), not per
        operator id.

        Restricted to rows measured on **this machine class** (`metadata.hardware_scope`):
        everything fit from this view is in machine units and none of it transfers. Its
        counterpart `op_stats_with_signature` is deliberately *not* restricted, because
        cardinality is a property of the data.

        The backend is scanned exactly once; `record` folds every later row straight
        into its bucket, so a steady-state read is O(1) rather than a re-parse of the
        session's whole history. Each bucket keeps its newest `PER_KIND_MAX` rows —
        far more than the median/regression its consumers fit needs, and enough to
        keep a long-lived session's planning cost flat.

        Best-effort; a malformed row is skipped, not raised."""
        if self._by_kind is None:
            self._by_kind = self._load_by_kind()
        return self._by_kind

    def _load_by_kind(self) -> dict[str, list[dict[str, Any]]]:
        """One-time bucketed load of this machine's feedback history from the backend."""
        return bucket_by_kind(self._backend.scan(_OP_STATS, ()))

    def op_stats_with_signature(self) -> list[dict[str, Any]]:
        """Signature-carrying operator feedback, **oldest first**.

        The shape Kyber's cardinality-correction loop consumes: the q-error
        (`n_actual / n_estimated`) is only meaningful when attributed to a *stable*
        operator identity across executions, which `op_id` is not. Rows without a
        signature are excluded — notably those a distributed worker reports for its
        sub-plan, whose `op_id`s live in their own space.

        Ordered by the hub's monotonic record sequence, **not** by the storage key
        `(op_id, seq)`: the same operator shape can appear at different positions in
        different plans, so an `op_id`-major order is not chronological. The consumer
        weights recent observations more heavily, so this ordering is load-bearing.

        Read on every optimize, so it must not cost the whole history: the backend is
        scanned exactly once, and `record` keeps the view current thereafter. The view is
        capped at the newest `SIGNED_HISTORY_MAX` rows, far above the handful of recent
        observations per signature the consumer averages — the persisted store keeps
        everything regardless.

        Best-effort; a malformed row is skipped, not raised.
        """
        if self._signed is None:
            self._signed = self._load_signed()
        return self._signed

    def _load_signed(self) -> list[dict[str, Any]]:
        """One-time chronological load of the signature-carrying rows from the backend."""
        rows = chronological_signed(self._backend.scan(_OP_STATS, ()))
        self._signed_appends = len(rows)
        return rows

    # --- learned parameters ------------------------------------------------
    def load_params(self, namespace: str) -> dict[str, Any]:
        """Every learned parameter under `namespace`, or an empty dict.

        Args:
            namespace: The learning loop's name, e.g. ``"kyber.calibration"``.

        Returns:
            The stored parameters.

        Raises:
            ConfigError: If `namespace` is not a non-empty string.
        """
        _check_namespace(namespace)
        raw = self._backend.get(_LEARNED_PARAMS, (namespace,))
        return json.loads(raw) if raw else {}

    def save_params(self, namespace: str, params: dict[str, Any]) -> None:
        """Replace every learned parameter under `namespace`.

        Args:
            namespace: The learning loop's name.
            params: The parameters to store. Must be JSON-serializable — the store
                holds opaque bytes so that any backend can serve it.

        Raises:
            ConfigError: If `namespace` is invalid, or `params` is not serializable.
                The offending key is named, because a `TypeError` reading "Object of
                type X is not JSON serializable" does not say *which* entry it was.
        """
        _check_namespace(namespace)
        self._backend.put(_LEARNED_PARAMS, (namespace,), _encoded(namespace, params))
        # A whole-blob write is the one thing the per-key view cannot patch in place: it
        # is the *legacy* layer that view merges underneath its own entries.
        self._params_generation += 1

    # --- per-key learned parameters ----------------------------------------
    # Learned stats are stored one backend key per entry — `(namespace, entry_key)`
    # — instead of a single `(namespace,)` blob, so a write touches only its own key
    # and concurrent writers learning different shapes can't clobber each other (the
    # lost-update race the whole-blob read-modify-write had). `load_keyed_params`
    # reassembles the same `{entry_key: value}` dict consumers expect, merging a
    # legacy single-blob value (length-1 key) underneath the per-key entries so an
    # older store migrates without losing what it learned.
    def load_keyed_params(self, namespace: str) -> dict[str, Any]:
        cached = self._keyed_cache.get(namespace)
        if cached is not None and cached[0] == self._params_generation:
            return cached[1]
        out: dict[str, Any] = {}
        legacy: dict[str, Any] = {}
        stored: set[str] = set()
        for key, value in self._backend.scan(_LEARNED_PARAMS, (namespace,)):
            if len(key) >= 2:
                out[key[1]] = json.loads(value)
                stored.add(str(key[1]))
            elif len(key) == 1:
                legacy = json.loads(value)
        for k, v in legacy.items():
            out.setdefault(k, v)  # per-key entries win over the legacy blob
        self._keyed_cache[namespace] = (self._params_generation, out)
        self._keyed_stored[namespace] = stored
        return out

    def get_keyed_param(self, namespace: str, key: str) -> Any | None:
        """The learned value under `(namespace, key)`, or `None`.

        Served from the same parsed view `load_keyed_params` builds — in which a per-key
        entry already shadows the legacy blob's — so the reads the tuning loops issue
        several times per query cost a dict lookup rather than a store round-trip and a
        `json.loads` (and, on a miss, a second round-trip for the legacy blob).
        """
        return self.load_keyed_params(namespace).get(key)

    def put_keyed_param(self, namespace: str, key: str, value: Any) -> None:
        """Store one learned entry under `(namespace, key)`.

        Args:
            namespace: The learning loop's name.
            key: The entry's name within the namespace.
            value: A JSON-serializable value.

        Raises:
            ConfigError: If `namespace` or `key` is not a non-empty string, or `value`
                is not serializable. A non-string key would round-trip through JSON as
                a string and silently stop matching the key it was written under.
        """
        _check_namespace(namespace)
        if not isinstance(key, str) or not key:
            raise ConfigError(
                f"A learned-parameter key must be a non-empty string, but got "
                f"{type(key).__name__} {key!r} in namespace {namespace!r}.",
                hint="Keys round-trip through JSON, so only strings survive unchanged.",
            )
        if self._unchanged(namespace, key, value):
            return
        blob = _encoded(f"{namespace}.{key}", value)
        self._backend.put(_LEARNED_PARAMS, (namespace, key), blob)
        self._keyed_stored.setdefault(namespace, set()).add(key)
        # Patch the parsed view rather than invalidating it: this write *is* the new value
        # of exactly one entry, and the tuning loops read the namespace back on the very
        # next query. Invalidating instead would re-scan and re-parse the namespace each
        # time. The blob is parsed back rather than caching `value` itself, so the view
        # holds exactly what a reader of the store would see (a tuple written is a list
        # read) and never aliases an object the caller still owns.
        cached = self._keyed_cache.get(namespace)
        if cached is not None:
            cached[1][key] = json.loads(blob)

    def _unchanged(self, namespace: str, key: str, value: Any) -> bool:
        """True when `(namespace, key)` already stores exactly `value` — so writing is a no-op.

        The learning loops **re-record what they already know on every query**: a query over
        the same source re-measures the same distinct counts, and merges them into the same
        map, and hands the same map back. Serving that write meant a `json.dumps` of the whole
        column map, a backend `put`, and a `json.loads` of the blob back — per column-stat
        table, per query — to arrive at the value already sitting in the parsed view. On the
        default in-process backend (a dict in this very process) the round-trip through JSON
        bytes was the *entire* cost. It was ~48% of a small query's control plane.

        The parsed view is by construction "what a reader of the store would see", so a value
        equal to it is a value already stored, and the write can be dropped. Two guards keep
        that inference honest:

        * the view must be current (`_params_generation`), and
        * the key must be backed by its own per-key backend entry (`_keyed_stored`) — an entry
          the view merged up from the *legacy* single-blob shape is readable but not yet
          migrated, and eliding its first write would defer that migration forever.

        Equality is `==` plus a top-level type check, which pins the int/float distinction
        JSON preserves. A nested int-vs-float drift under an equal value is not distinguished
        — it would require a deterministic producer to change a value's type while keeping it
        numerically equal, and every consumer of these learned stats does float arithmetic.
        """
        cached = self._keyed_cache.get(namespace)
        if cached is None or cached[0] != self._params_generation:
            return False
        if key not in self._keyed_stored.get(namespace, ()):
            return False
        current = cached[1].get(key, _MISSING)
        return type(current) is type(value) and current == value
