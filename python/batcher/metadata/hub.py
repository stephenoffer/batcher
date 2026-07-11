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
from typing import Any

from batcher._internal.logging import get_logger
from batcher.metadata.store import MetadataBackend
from batcher.plan.feedback import OperatorFeedback

__all__ = ["MetadataHub"]

_log = get_logger("metadata")

# Logical tables.
_OP_STATS = "op_stats"
_LEARNED_PARAMS = "learned_params"

# Cap on the in-memory view of signature-carrying feedback. The consumer averages the
# last handful of observations per signature, so this is orders of magnitude more than it
# reads; it exists only to bound a long-lived session's memory. Nothing is lost — the
# backend still holds the full history.
_SIGNED_HISTORY_MAX = 4096

# Cap on the retained rows *per operator family* in the `op_stats_by_kind` view. Every
# consumer of that view reduces a family's rows to a median or a regression coefficient,
# so the newest few thousand samples decide the fit and older ones only cost memory and
# time. Without a cap the view — and the per-query fit over it — grows for the life of
# the process. The backend still holds the full history.
_PER_KIND_MAX = 4096

# Bounded views are trimmed only once they exceed their cap by this factor, so a trim
# costs O(cap) once every O(cap) records rather than O(cap) on every record.
_TRIM_SLACK = 2


def _trimmed(rows: list[dict[str, Any]], cap: int) -> list[dict[str, Any]]:
    """`rows` bounded to its newest `cap` entries, trimming only past the slack factor."""
    if len(rows) > cap * _TRIM_SLACK:
        del rows[:-cap]
    return rows


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
        self._backend = backend
        self._seq = 0
        # Parsed-read cache for the learned-parameter tables. `_params_generation` bumps
        # only on a *whole-blob* `save_params` (which the per-key view merges underneath
        # itself); a `put_keyed_param` refreshes the cached entry in place, and `record`
        # — which writes a different table entirely — no longer disturbs it at all. The
        # earlier single generation counter was bumped by every feedback row, so this
        # cache missed on every query it existed to serve.
        self._params_generation = 0
        self._keyed_cache: dict[str, tuple[int, dict[str, Any]]] = {}
        # Bucketed-by-kind view of the feedback history: loaded from the backend once,
        # then folded forward by `record`. Consumers reduce each bucket to a median or a
        # regression, so buckets are bounded (`_PER_KIND_MAX`) — the whole point is that
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

    # --- FeedbackSink ------------------------------------------------------
    def record(self, feedback: OperatorFeedback) -> None:
        """Persist one operator's feedback. Never raises into the caller."""
        try:
            self._seq += 1
            key = (int(feedback.op_id), self._seq)
            row = _row_of(feedback)
            self._backend.put(_OP_STATS, key, json.dumps(row).encode())
            # Fold the row into whichever derived views have been materialized. A view
            # still `None` has not been read yet; its lazy load will pick this row up
            # from the backend, so there is nothing to do.
            if self._by_kind is not None:
                bucket = self._by_kind.setdefault(row["kind"], [])
                bucket.append(row)
                _trimmed(bucket, _PER_KIND_MAX)
            if self._signed is not None and row["signature"]:
                self._signed.append(row)  # keep the hot view current without a re-scan
                self._signed_appends += 1
                _trimmed(self._signed, _SIGNED_HISTORY_MAX)
        except Exception:  # pragma: no cover - feedback must not break execution
            _log.warning("dropped operator feedback", exc_info=True)

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
        """All recorded feedback for an operator id, oldest first."""
        out = [json.loads(value) for _key, value in self._backend.scan(_OP_STATS, (op_id,))]
        return out

    def op_stats_by_kind(self) -> dict[str, list[dict[str, Any]]]:
        """All recorded operator feedback bucketed by operator `kind`.

        The shape Kyber's cost calibration consumes: per-row/per-byte coefficients
        are fit per operator family (`scan`, `filter`, `hash_join`, ...), not per
        operator id.

        The backend is scanned exactly once; `record` folds every later row straight
        into its bucket, so a steady-state read is O(1) rather than a re-parse of the
        session's whole history. Each bucket keeps its newest `_PER_KIND_MAX` rows —
        far more than the median/regression its consumers fit needs, and enough to
        keep a long-lived session's planning cost flat.

        Best-effort; a malformed row is skipped, not raised."""
        if self._by_kind is None:
            self._by_kind = self._load_by_kind()
        return self._by_kind

    def _load_by_kind(self) -> dict[str, list[dict[str, Any]]]:
        """One-time bucketed load of the feedback history from the backend."""
        buckets: dict[str, list[dict[str, Any]]] = {}
        try:
            for _key, value in self._backend.scan(_OP_STATS, ()):
                row = json.loads(value)
                buckets.setdefault(row.get("kind", ""), []).append(row)
        except Exception:  # pragma: no cover - calibration must not break planning
            _log.warning("could not scan op_stats", exc_info=True)
        for bucket in buckets.values():
            _trimmed(bucket, _PER_KIND_MAX)
        return buckets

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
        capped at the newest `_SIGNED_HISTORY_MAX` rows, far above the handful of recent
        observations per signature the consumer averages — the persisted store keeps
        everything regardless.

        Best-effort; a malformed row is skipped, not raised.
        """
        if self._signed is None:
            self._signed = self._load_signed()
        return self._signed

    def _load_signed(self) -> list[dict[str, Any]]:
        """One-time chronological load of the signature-carrying rows from the backend."""
        ordered: list[tuple[int, dict[str, Any]]] = []
        try:
            for key, value in self._backend.scan(_OP_STATS, ()):
                row = json.loads(value)
                if not row.get("signature"):
                    continue
                seq = int(key[1]) if len(key) > 1 else 0
                ordered.append((seq, row))
        except Exception:  # pragma: no cover - learning must not break planning
            _log.warning("could not scan op_stats", exc_info=True)
            return []
        ordered.sort(key=lambda pair: pair[0])
        rows = [row for _seq, row in ordered[-_SIGNED_HISTORY_MAX:]]
        self._signed_appends = len(rows)
        return rows

    # --- learned parameters ------------------------------------------------
    def load_params(self, namespace: str) -> dict[str, Any]:
        raw = self._backend.get(_LEARNED_PARAMS, (namespace,))
        return json.loads(raw) if raw else {}

    def save_params(self, namespace: str, params: dict[str, Any]) -> None:
        self._backend.put(_LEARNED_PARAMS, (namespace,), json.dumps(params).encode())
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
        for key, value in self._backend.scan(_LEARNED_PARAMS, (namespace,)):
            if len(key) >= 2:
                out[key[1]] = json.loads(value)
            elif len(key) == 1:
                legacy = json.loads(value)
        for k, v in legacy.items():
            out.setdefault(k, v)  # per-key entries win over the legacy blob
        self._keyed_cache[namespace] = (self._params_generation, out)
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
        blob = json.dumps(value).encode()
        self._backend.put(_LEARNED_PARAMS, (namespace, key), blob)
        # Patch the parsed view rather than invalidating it: this write *is* the new value
        # of exactly one entry, and the tuning loops read the namespace back on the very
        # next query. Invalidating instead would re-scan and re-parse the namespace each
        # time. The blob is parsed back rather than caching `value` itself, so the view
        # holds exactly what a reader of the store would see (a tuple written is a list
        # read) and never aliases an object the caller still owns.
        cached = self._keyed_cache.get(namespace)
        if cached is not None:
            cached[1][key] = json.loads(blob)
