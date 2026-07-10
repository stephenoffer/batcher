"""`MetadataHub` — the façade over a `MetadataBackend`.

The Hub is the single seam the rest of the system uses to read learned state and
write feedback. It implements the `FeedbackSink` contract so Core can hand it
`OperatorFeedback` directly. Writes are best-effort and must never raise into the
hot path; reads return learned parameters for warm-starting plans.

Only a thin slice is implemented for the bootstrap engine (feedback recording +
generic param load/save). Sketch persistence, workload fingerprints, bandit
posteriors, and the compiled-artifact cache layer onto these same primitives.
"""

from __future__ import annotations

import dataclasses
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


class MetadataHub:
    """Reads learned state and absorbs execution feedback."""

    def __init__(self, backend: MetadataBackend) -> None:
        self._backend = backend
        self._seq = 0
        # Parsed-read cache. The optimizer re-reads the whole learned-stats and
        # op-stats history on *every* query (twice — main optimize + the
        # metadata-answer rewrite), re-running `json.loads` over every stored entry
        # each time (35+ parses/query on a warm store). Those reads only change when
        # this hub absorbs a write, so memoize the parsed result and invalidate it on
        # any write. `_generation` bumps on every write (not just `record`, which is
        # what `_seq`/`version` track); a cached read is valid while the generation is
        # unchanged. This is the single biggest fixed per-query overhead on small
        # queries — the control plane must not re-parse cold state it already holds.
        self._generation = 0
        self._keyed_cache: dict[str, tuple[int, dict[str, Any]]] = {}
        self._op_stats_cache: tuple[int, dict[str, list[dict[str, Any]]]] | None = None
        # Chronological, bounded, in-memory view of the signature-carrying feedback rows.
        # Kyber's cardinality correction reads it on *every* optimize, and a re-scan of
        # the whole persisted history there would make planning cost grow with the
        # session's cumulative query count (the O(queries²) trap the caches above exist to
        # avoid). Loaded once from the backend on first read, then maintained
        # incrementally by `record`, so a steady-state read is O(1).
        self._signed: list[dict[str, Any]] | None = None

    def _bump(self) -> None:
        """Invalidate the parsed-read cache after a write."""
        self._generation += 1

    # --- FeedbackSink ------------------------------------------------------
    def record(self, feedback: OperatorFeedback) -> None:
        """Persist one operator's feedback. Never raises into the caller."""
        try:
            self._seq += 1
            key = (int(feedback.op_id), self._seq)
            row = dataclasses.asdict(feedback)
            self._backend.put(_OP_STATS, key, json.dumps(row).encode())
            self._bump()
            if self._signed is not None and row.get("signature"):
                self._signed.append(row)  # keep the hot view current without a re-scan
                if len(self._signed) > _SIGNED_HISTORY_MAX:
                    del self._signed[:-_SIGNED_HISTORY_MAX]
        except Exception:  # pragma: no cover - feedback must not break execution
            _log.warning("dropped operator feedback", exc_info=True)

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
        operator id. Best-effort; a malformed row is skipped, not raised."""
        if self._op_stats_cache is not None and self._op_stats_cache[0] == self._generation:
            return self._op_stats_cache[1]
        buckets: dict[str, list[dict[str, Any]]] = {}
        try:
            for _key, value in self._backend.scan(_OP_STATS, ()):
                row = json.loads(value)
                buckets.setdefault(row.get("kind", ""), []).append(row)
        except Exception:  # pragma: no cover - calibration must not break planning
            _log.warning("could not scan op_stats", exc_info=True)
        self._op_stats_cache = (self._generation, buckets)
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
        return [row for _seq, row in ordered[-_SIGNED_HISTORY_MAX:]]

    # --- learned parameters ------------------------------------------------
    def load_params(self, namespace: str) -> dict[str, Any]:
        raw = self._backend.get(_LEARNED_PARAMS, (namespace,))
        return json.loads(raw) if raw else {}

    def save_params(self, namespace: str, params: dict[str, Any]) -> None:
        self._backend.put(_LEARNED_PARAMS, (namespace,), json.dumps(params).encode())
        self._bump()

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
        if cached is not None and cached[0] == self._generation:
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
        self._keyed_cache[namespace] = (self._generation, out)
        return out

    def get_keyed_param(self, namespace: str, key: str) -> Any | None:
        raw = self._backend.get(_LEARNED_PARAMS, (namespace, key))
        if raw is not None:
            return json.loads(raw)
        # Migration fallback: an entry still only in the legacy blob.
        return self.load_params(namespace).get(key)

    def put_keyed_param(self, namespace: str, key: str, value: Any) -> None:
        self._backend.put(_LEARNED_PARAMS, (namespace, key), json.dumps(value).encode())
        self._bump()
