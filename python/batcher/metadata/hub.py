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
from batcher._internal.hardware import fingerprint
from batcher._internal.logging import get_logger
from batcher.metadata.params import LearnedParams
from batcher.metadata.store import MetadataBackend, check_backend
from batcher.metadata.views import (
    PER_KIND_MAX,
    SIGNED_HISTORY_MAX,
    build_views,
    trimmed,
)
from batcher.plan.feedback import OperatorFeedback

__all__ = ["MetadataHub"]

_log = get_logger("metadata")

# The logical table this module owns. Learned parameters live in their own (`params.py`).
_OP_STATS = "op_stats"

# Cap on the `op_stats` rows a backend retains, and how often it is enforced. The views above
# are bounded; the table beneath them was not, so a session grew the store by one row per
# operator per query for its whole life with nothing reading the old ones. The cap sits far
# above what rebuilding either view needs — the views themselves keep an order of magnitude
# less — so pruning is invisible to every consumer, and what it buys is that opening a store a
# served workload has been writing to for months does not begin with parsing all of it.
#
# A backend that offers no `delete` is left alone; that is the opt-out, and an archival store
# that wants the full history simply does not implement it.
_OP_STATS_MAX = 65_536
_OP_STATS_PRUNE_EVERY = 4_096


# `OperatorFeedback`'s field names, resolved once. Every field is a scalar, so the row is
# a flat `{name: value}` — `dataclasses.asdict` would deep-copy recursively and re-derive
# the field tuple on each of the several calls a query makes.
_FEEDBACK_FIELDS: tuple[str, ...] = tuple(OperatorFeedback.__dataclass_fields__)


def _row_of(feedback: OperatorFeedback) -> dict[str, Any]:
    """One feedback row as the flat JSON-shaped dict both the store and the views hold."""
    return {name: getattr(feedback, name) for name in _FEEDBACK_FIELDS}


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
        # Resolved once: a backend that can take a structured row (`InProcessBackend`, the
        # default) lets `_record_locked` skip serializing a flat dict of scalars purely so a
        # later read can parse it back. A backend that must produce real bytes does not
        # offer it and keeps the `put` path unchanged.
        self._put_row = getattr(backend, "put_row", None)
        self._seq = 0
        # The learned-parameter half of the store, with its own parsed-read cache. A separate
        # object because it is a separate job: this class absorbs measurements and maintains
        # views over them, `LearnedParams` holds what the tuning loops read back at plan time.
        self._params = LearnedParams(self._backend)
        # Feedback history bucketed by machine class, then by operator kind: loaded from the
        # backend once, then folded forward by `record`. Consumers reduce each bucket to a
        # median or a regression, so buckets are bounded (`PER_KIND_MAX`) — the whole point is
        # that neither a read nor a record costs anything proportional to session history.
        # Bucketed rather than filtered to the local class, so a driver can read what its
        # *workers* measured; see `op_stats_by_kind`.
        self._by_fp: dict[str, dict[str, list[dict[str, Any]]]] | None = None
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
        if self._put_row is not None:
            self._put_row(_OP_STATS, key, row)
        else:
            self._backend.put(_OP_STATS, key, json.dumps(row).encode())
        # Fold the row into whichever derived views have been materialized. A view
        # still `None` has not been read yet; its lazy load will pick this row up
        # from the backend, so there is nothing to do.
        # Same bucketing `build_views` applies; the two must agree or it leaks after a write.
        machine = str(row.get("hw_fingerprint", "") or "")
        if self._by_fp is not None and machine:
            bucket = self._by_fp.setdefault(machine, {}).setdefault(row["kind"], [])
            bucket.append(row)
            trimmed(bucket, PER_KIND_MAX)
        if self._signed is not None and row["signature"]:
            self._signed.append(row)  # keep the hot view current without a re-scan
            self._signed_appends += 1
            trimmed(self._signed, SIGNED_HISTORY_MAX)
        if self._seq % _OP_STATS_PRUNE_EVERY == 0:
            self._prune_op_stats()

    def _prune_op_stats(self, keys: list[Any] | None = None) -> None:
        """Bound the stored operator feedback to its newest `_OP_STATS_MAX` rows.

        Amortized: the scan-and-drop costs O(stored) but runs once every
        `_OP_STATS_PRUNE_EVERY` records, or once at the first view load of a process that
        finds an oversized store. Keys carry a monotonic sequence number, so "newest" is
        exact without parsing values. A backend with no `delete` is left alone.

        Args:
            keys: The stored keys, when the caller already has them (the view load does).
                Omit to read them, which costs a scan.
        """
        delete = getattr(self._backend, "delete", None)
        if delete is None:
            return
        if keys is None:
            keys = [key for key, _value in self._backend.scan(_OP_STATS, ())]
        if len(keys) <= _OP_STATS_MAX:
            return
        # `(op_id, seq)`, oldest sequence first. A key written by a build with a different
        # shape sorts as if it were the oldest, so it is dropped first — which is the right
        # answer for a row nothing can read anyway.
        keys = sorted(keys, key=lambda k: k[1] if len(k) > 1 and isinstance(k[1], int) else -1)
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

    def op_stats_by_kind(self, hw_fingerprint: str | None = None) -> dict[str, list[dict]]:
        """Operator feedback measured on **one machine class**, bucketed by operator `kind`.

        The shape Kyber's cost calibration consumes: per-row/per-byte coefficients
        are fit per operator family (`scan`, `filter`, `hash_join`, ...), not per
        operator id.

        Scoped to a single machine class, because everything fit from this view is in machine
        units and none of it transfers (`metadata.hardware_scope`). Its counterpart
        `op_stats_with_signature` is deliberately *unscoped*, because cardinality is a property
        of the data.

        **Which class, though, is the caller's question, not this hub's.** Defaulting to the
        local machine is right for a single-node run and wrong for every distributed one: Kyber
        runs on the driver, which executes none of the work, so a view of "what this process
        measured" excluded every worker row on any cluster whose driver was a different machine
        class. The rows arrived correctly attributed (a worker stamps its own fingerprint before
        shipping) and were then dropped at the reader, which silently disabled cost calibration
        and the CPU-share loop on exactly the deployment they exist for. A caller planning for a
        worker passes that worker's fingerprint.

        The backend is scanned exactly once; `record` folds every later row straight
        into its bucket, so a steady-state read is O(1) rather than a re-parse of the
        session's whole history. Each bucket keeps its newest `PER_KIND_MAX` rows —
        far more than the median/regression its consumers fit needs, and enough to
        keep a long-lived session's planning cost flat.

        Args:
            hw_fingerprint: The machine class to read, from `HardwareProfile.fingerprint`.
                `None` reads this process's own class, which is the single-node answer and
                what every caller without a cluster profile wants. An unknown class yields
                an empty view rather than another machine's rows.

        Returns:
            `{kind: rows}` for that machine class, empty when it has measured nothing.

        Best-effort; a malformed row is skipped, not raised."""
        if self._by_fp is None:
            self._load_views()
        assert self._by_fp is not None
        return self._by_fp.get(hw_fingerprint or fingerprint(), {})

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
            self._load_views()
        assert self._signed is not None
        return self._signed

    def _load_views(self) -> None:
        """Materialize both derived views from a **single** scan of the backend.

        The first read of either view builds both. They are read together on every optimize —
        cost calibration wants the by-kind buckets and cardinality correction wants the signed
        history — so loading them separately scanned the whole `op_stats` table twice and paid
        `json.loads` on every stored row twice, for the same rows. A view already materialized
        is left alone: `record` has been folding rows into it, and rebuilding it from the
        backend would be correct but would throw that work away.

        Serialized against `record` by the same lock. Without it a row recorded *during* the
        scan is dropped from the view: `record` sees the view still `None` and skips the fold,
        while the scan has already passed the position the new row would occupy. The row
        survives in the backend, so no result is wrong — but the measurement does not reach
        the model until the next process, and it is the newest measurement, which is exactly
        the one the learning loop weights most. The lock is held for one scan per process.
        """
        with self._lock:
            if self._by_fp is not None and self._signed is not None:
                return  # another thread finished the load while this one waited
            scanned = list(self._backend.scan(_OP_STATS, ()))
            by_fp, signed = build_views(scanned)
            if self._by_fp is None:
                self._by_fp = by_fp
            if self._signed is None:
                self._signed = signed
                self._signed_appends = len(signed)
            # The one point at which this process learns how large the *stored* history is.
            # `_record_locked` prunes every `_OP_STATS_PRUNE_EVERY` rows **it** wrote, which
            # a short-lived process never reaches — so a durable store written by a served
            # workload (a process per request, a handful of operators each) grew without
            # bound and was never trimmed by anyone, and every new process paid to scan and
            # parse the whole of it before its first plan. Pruning from the keys already in
            # hand costs no extra scan.
            if len(scanned) > _OP_STATS_MAX:
                self._prune_op_stats([key for key, _value in scanned])

    # --- learned parameters ------------------------------------------------
    # Delegated to `LearnedParams`, which owns the two storage shapes and the parsed-read
    # cache over them. The Hub keeps these names because they are the seam every tuning loop
    # in the tree calls through.
    def load_params(self, namespace: str) -> dict[str, Any]:
        """Every learned parameter under `namespace`, or an empty dict.

        Args:
            namespace: The learning loop's name, e.g. ``"kyber.calibration"``.

        Returns:
            The stored parameters, as the store's own parsed view. **Read-only** — see
            `LearnedParams.load`.

        Raises:
            ConfigError: If `namespace` is not a non-empty string.
        """
        return self._params.load(namespace)

    def save_params(self, namespace: str, params: dict[str, Any]) -> None:
        """Replace every learned parameter under `namespace`.

        Args:
            namespace: The learning loop's name.
            params: The parameters to store. Must be JSON-serializable.

        Raises:
            ConfigError: If `namespace` is invalid, or `params` is not serializable.
        """
        self._params.save(namespace, params)

    def load_keyed_params(self, namespace: str) -> dict[str, Any]:
        """The `{entry_key: value}` map for `namespace`, per-key entries over the legacy blob.

        Args:
            namespace: The learning loop's name.

        Returns:
            The store's parsed view. **Read-only** — see `LearnedParams.load`.
        """
        return self._params.load_keyed(namespace)

    def get_keyed_param(self, namespace: str, key: str) -> Any | None:
        """The learned value under `(namespace, key)`, or `None`.

        Args:
            namespace: The learning loop's name.
            key: The entry's name within the namespace.

        Returns:
            The stored value, or `None` when the namespace has no such entry.
        """
        return self._params.get_keyed(namespace, key)

    def put_keyed_param(self, namespace: str, key: str, value: Any) -> None:
        """Store one learned entry under `(namespace, key)`.

        This is the shape to write: it touches one backend key, so two pipelines learning
        different entries cannot lose each other's update.

        Args:
            namespace: The learning loop's name.
            key: The entry's name within the namespace.
            value: A JSON-serializable value.

        Raises:
            ConfigError: If `namespace` or `key` is not a non-empty string, or `value`
                is not serializable.
        """
        self._params.put_keyed(namespace, key, value)
