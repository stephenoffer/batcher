"""Adaptive batch-size control for inference — what Ray Data makes you hand-tune.

Ray Data has no batch-size auto-tuning: users set `batch_size` per model/modality
by hand, and `batch_size=None` OOMs. Batcher tunes it online. There are two
*objectives*, and using the wrong one is the trap:

* **latency** (online serving) — drive a PID toward a per-batch latency target
  (`InferencePool`'s `_LatencyController`).
* **throughput** (offline batch — embeddings, LLM batch, the bulk of Ray Data
  workloads) — maximize rows/sec **subject to a VRAM cap**. A latency PID optimizes
  the wrong thing here. `ThroughputController` hill-climbs the batch size while
  throughput keeps rising and VRAM stays under the cap, then settles at the plateau
  — the automatic form of the guides' "increase batch size until throughput
  plateaus / VRAM ~80%" protocol.

Pure control logic (no GPU, no engine) so it is exhaustively unit-testable; the
inference pool feeds it measured throughput and (when available) VRAM.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.logging import note_suppressed
from batcher.metadata.hardware_scope import scoped

if TYPE_CHECKING:
    from batcher.metadata import MetadataHub

__all__ = ["ThroughputController", "learned_batch_size", "record_batch_size"]

# Hub namespace the settled per-model throughput-optimal batch size persists under, keyed by a
# stable model signature. Seeding the hill-climb from it lets a recurring inference job start at
# (or near) last run's plateau instead of climbing from the cold default every time. The batch
# size only shards rows, so a learned start is byte-identical to a cold one — only faster to
# converge.
_LEARN_NS = "udf_throughput_batch"


def learned_batch_size(hub: MetadataHub | None, signature: str | None) -> int | None:
    """The learned throughput-optimal batch size for a model signature, or `None` when unseen.

    Best-effort read of the persisted plateau; a cold store or hub error yields `None`, so the
    caller keeps its default initial size."""
    if hub is None or signature is None:
        return None
    try:
        s = hub.get_keyed_param(scoped(_LEARN_NS), signature) or {}
    except Exception as exc:  # pragma: no cover - learning must never break a query
        note_suppressed("ml", "read learned batch size", exc)
        return None
    v = s.get("size")
    return int(v) if isinstance(v, (int, float)) and v >= 1 else None


def record_batch_size(hub: MetadataHub | None, signature: str | None, size: int) -> None:
    """Persist a model's settled throughput-optimal batch `size`, exp-smoothed across runs.

    Best-effort — never raises into the inference loop."""
    if hub is None or signature is None or size < 1:
        return
    try:
        from batcher.config import active_config

        s = hub.get_keyed_param(scoped(_LEARN_NS), signature) or {}
        alpha = float(active_config().optimizer.learning_smoothing_alpha)
        prior = s.get("size")
        new = float(size)
        smoothed = new if prior is None else alpha * new + (1.0 - alpha) * float(prior)
        hub.put_keyed_param(scoped(_LEARN_NS), signature, {"size": smoothed})
    except Exception as exc:  # pragma: no cover - learning must never break a query
        note_suppressed("ml", "record autobatch size", exc)
        return


class ThroughputController:
    """Hill-climb the batch size toward maximum throughput under a VRAM cap.

    `update(throughput_rows_per_s, vram_fraction)` records one observation at the
    current size and returns the next size to try. VRAM is a hard *constraint*
    (over the cap → shrink); throughput is the *objective* (grow while it rises,
    settle at the plateau). Bounds-clamped to ``[min_rows, max_rows]``.

    When a `hub` and a model `signature` are supplied, the climb **warm-starts** from the
    plateau a prior run learned (`learned_batch_size`) and persists each new best back
    (`record_batch_size`), so a recurring inference job converges in a few batches instead of
    re-climbing from the cold default. Both are optional — omitted (the default), the controller
    is exactly the pure, hub-free hill-climb it was, so a caller that does not opt in is
    unchanged. The learned seed only changes the *starting* size, never the result.
    """

    def __init__(
        self,
        *,
        min_rows: int = 1,
        max_rows: int = 65_536,
        initial: int = 256,
        vram_cap: float = 0.85,
        grow: float = 1.5,
        shrink: float = 0.7,
        plateau_ratio: float = 1.02,
        stale_limit: int = 8,
        hub: MetadataHub | None = None,
        signature: str | None = None,
    ) -> None:
        if min_rows < 1 or max_rows < min_rows:
            raise ValueError("require 1 <= min_rows <= max_rows")
        if grow <= 1.0 or not (0.0 < shrink < 1.0):
            raise ValueError("require grow > 1 and 0 < shrink < 1")
        self._min = min_rows
        self._max = max_rows
        self._vram_cap = vram_cap
        self._grow = grow
        self._shrink = shrink
        self._plateau = plateau_ratio
        self._stale_limit = max(1, stale_limit)
        self._stale = 0
        self._hub = hub
        self._signature = signature
        # Warm-start the climb from the learned plateau when one exists (clamped to bounds).
        learned = learned_batch_size(hub, signature)
        seed = learned if learned is not None else initial
        self._cur = float(min(max(seed, min_rows), max_rows))
        self._best_throughput: float | None = None
        self._best_size: float | None = None

    def update(self, throughput_rows_per_s: float, vram_fraction: float | None = None) -> int:
        """Observe throughput (and optional VRAM) at the current size; return the next."""
        # VRAM is a hard cap: over it, shrink and restart the climb from here.
        if vram_fraction is not None and vram_fraction > self._vram_cap:
            self._cur = max(float(self._min), self._cur * self._shrink)
            self._best_throughput = None
            self._best_size = None
            return self.current()

        t = throughput_rows_per_s
        if t != t or t < 0:  # NaN / nonsense guard
            return self.current()

        improving = self._best_throughput is None or t > self._best_throughput * self._plateau
        if improving:
            self._stale = 0
            self._best_throughput = t
            self._best_size = self._cur
            # Persist the new plateau so the next run warm-starts here (best-effort, no-op
            # unless a hub + signature was supplied).
            record_batch_size(self._hub, self._signature, round(self._cur))
            # Predictive VRAM guard: a multiplicative grow scales the batch — and
            # roughly VRAM — by `grow`, which could overshoot the cap in a *single*
            # step before the reactive shrink (above) ever sees it. So only grow when
            # the predicted post-grow VRAM stays under the cap; otherwise hold at the
            # current (best) size, the safe ceiling. This makes the climb OOM-safe by
            # construction rather than relying on catching the OOM after the fact.
            if vram_fraction is None or vram_fraction * self._grow <= self._vram_cap:
                self._cur = min(float(self._max), self._cur * self._grow)
        elif self._best_size is not None:
            # Plateaued or regressed: settle back at the best size observed. But a *durable*
            # regression (a co-tenant landed, sequences got longer, a slower shard) makes
            # that stale optimum unreachable forever — `improving` can never fire again
            # against a `best` the environment no longer supports. After `stale_limit`
            # consecutive non-improving observations, forget the plateau and re-explore
            # from the current size so the controller can find the new optimum.
            self._stale += 1
            if self._stale >= self._stale_limit:
                self._stale = 0
                self._best_throughput = None
                self._best_size = None
            else:
                self._cur = self._best_size
        return self.current()

    def current(self) -> int:
        """The current batch-size target (clamped, rounded to a whole row count)."""
        return int(min(self._max, max(self._min, round(self._cur))))

    def best_size(self) -> int:
        """The best (throughput-optimal) size observed so far — the settled plateau.

        Falls back to the current size before any observation. This is the value the learned
        store persists so a future run can warm-start the climb."""
        best = self._best_size if self._best_size is not None else self._cur
        return int(min(self._max, max(self._min, round(best))))
