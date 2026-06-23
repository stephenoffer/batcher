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
import logging
from typing import Any

from batcher.metadata.store import MetadataBackend
from batcher.plan.feedback import OperatorFeedback

__all__ = ["MetadataHub"]

_log = logging.getLogger("batcher.metadata")

# Logical tables.
_OP_STATS = "op_stats"
_LEARNED_PARAMS = "learned_params"


class MetadataHub:
    """Reads learned state and absorbs execution feedback."""

    def __init__(self, backend: MetadataBackend) -> None:
        self._backend = backend
        self._seq = 0

    # --- FeedbackSink ------------------------------------------------------
    def record(self, feedback: OperatorFeedback) -> None:
        """Persist one operator's feedback. Never raises into the caller."""
        try:
            self._seq += 1
            key = (int(feedback.op_id), self._seq)
            payload = json.dumps(dataclasses.asdict(feedback)).encode()
            self._backend.put(_OP_STATS, key, payload)
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
        buckets: dict[str, list[dict[str, Any]]] = {}
        try:
            for _key, value in self._backend.scan(_OP_STATS, ()):
                row = json.loads(value)
                buckets.setdefault(row.get("kind", ""), []).append(row)
        except Exception:  # pragma: no cover - calibration must not break planning
            _log.warning("could not scan op_stats", exc_info=True)
        return buckets

    # --- learned parameters ------------------------------------------------
    def load_params(self, namespace: str) -> dict[str, Any]:
        raw = self._backend.get(_LEARNED_PARAMS, (namespace,))
        return json.loads(raw) if raw else {}

    def save_params(self, namespace: str, params: dict[str, Any]) -> None:
        self._backend.put(_LEARNED_PARAMS, (namespace,), json.dumps(params).encode())

    # --- per-key learned parameters ----------------------------------------
    # Learned stats are stored one backend key per entry — `(namespace, entry_key)`
    # — instead of a single `(namespace,)` blob, so a write touches only its own key
    # and concurrent writers learning different shapes can't clobber each other (the
    # lost-update race the whole-blob read-modify-write had). `load_keyed_params`
    # reassembles the same `{entry_key: value}` dict consumers expect, merging a
    # legacy single-blob value (length-1 key) underneath the per-key entries so an
    # older store migrates without losing what it learned.
    def load_keyed_params(self, namespace: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        legacy: dict[str, Any] = {}
        for key, value in self._backend.scan(_LEARNED_PARAMS, (namespace,)):
            if len(key) >= 2:
                out[key[1]] = json.loads(value)
            elif len(key) == 1:
                legacy = json.loads(value)
        for k, v in legacy.items():
            out.setdefault(k, v)  # per-key entries win over the legacy blob
        return out

    def get_keyed_param(self, namespace: str, key: str) -> Any | None:
        raw = self._backend.get(_LEARNED_PARAMS, (namespace, key))
        if raw is not None:
            return json.loads(raw)
        # Migration fallback: an entry still only in the legacy blob.
        return self.load_params(namespace).get(key)

    def put_keyed_param(self, namespace: str, key: str, value: Any) -> None:
        self._backend.put(_LEARNED_PARAMS, (namespace, key), json.dumps(value).encode())
