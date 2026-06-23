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

__all__ = ["ThroughputController"]


class ThroughputController:
    """Hill-climb the batch size toward maximum throughput under a VRAM cap.

    `update(throughput_rows_per_s, vram_fraction)` records one observation at the
    current size and returns the next size to try. VRAM is a hard *constraint*
    (over the cap → shrink); throughput is the *objective* (grow while it rises,
    settle at the plateau). Bounds-clamped to ``[min_rows, max_rows]``.
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
        self._cur = float(min(max(initial, min_rows), max_rows))
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
            self._best_throughput = t
            self._best_size = self._cur
            # Predictive VRAM guard: a multiplicative grow scales the batch — and
            # roughly VRAM — by `grow`, which could overshoot the cap in a *single*
            # step before the reactive shrink (above) ever sees it. So only grow when
            # the predicted post-grow VRAM stays under the cap; otherwise hold at the
            # current (best) size, the safe ceiling. This makes the climb OOM-safe by
            # construction rather than relying on catching the OOM after the fact.
            if vram_fraction is None or vram_fraction * self._grow <= self._vram_cap:
                self._cur = min(float(self._max), self._cur * self._grow)
        elif self._best_size is not None:
            # Plateaued or regressed: settle back at the best size observed.
            self._cur = self._best_size
        return self.current()

    def current(self) -> int:
        """The current batch-size target (clamped, rounded to a whole row count)."""
        return int(min(self._max, max(self._min, round(self._cur))))
