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

import math
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
    # `math.isfinite`, not just `>= 1`: an infinity satisfies `>= 1` and then `int(inf)` raises
    # `OverflowError` out of batch sizing. A store outlives the build that wrote it, so the
    # write guard below cannot be the only one. Reading a non-finite entry as "unseen" keeps
    # the caller's default initial size, which is the pre-learning behaviour.
    if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v < 1:
        return None
    return int(v)


def record_batch_size(hub: MetadataHub | None, signature: str | None, size: int) -> None:
    """Persist a model's settled throughput-optimal batch `size`, exp-smoothed across runs.

    Best-effort — never raises into the inference loop."""
    # `size < 1` rejects neither a NaN nor an infinity — both compare False against it — and
    # exponential smoothing then propagates the non-finite value into the stored entry and into
    # every later update, poisoning the key for the life of the store. The same half-guard as
    # `dist.adaptive_sizing._ema` and `core.udf.sizing._ema`; `metadata.smoothed` documents why.
    if hub is None or signature is None or not math.isfinite(size) or size < 1:
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
        # Typed, and naming the values: these are user-supplied autobatch bounds, and a
        # message that restates the rule without saying which number broke it makes the
        # caller re-read their own call to find out.
        from batcher._internal.errors import PlanError

        if min_rows < 1 or max_rows < min_rows:
            raise PlanError(
                f"autobatch needs 1 <= min_rows <= max_rows, got min_rows={min_rows}, "
                f"max_rows={max_rows}"
            )
        if grow <= 1.0 or not (0.0 < shrink < 1.0):
            raise PlanError(
                f"autobatch needs grow > 1 and 0 < shrink < 1 (it grows on success and "
                f"shrinks on OOM), got grow={grow}, shrink={shrink}"
            )
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
        # A ceiling learned from actual out-of-memory failures, distinct from `max_rows` (a
        # caller's declared bound). Without it the climb has no memory of an OOM: the batch
        # that died is halved, the halved batch succeeds and reports good throughput, and
        # `improving` promptly grows the size straight back into the same failure. That
        # oscillation is what makes an inference job spend its life at the edge of an OOM,
        # paying the retry cost on a large fraction of its batches.
        self._oom_ceiling: float | None = None
        # Consecutive OOMs. Each one backs the ceiling off further, so a device that keeps
        # failing converges downward instead of hovering just under a ceiling that is itself
        # too high (a co-tenant that grew, a longer-sequence shard).
        self._oom_streak = 0
        # Warm-start the climb from the learned plateau when one exists (clamped to bounds).
        learned = learned_batch_size(hub, signature)
        seed = learned if learned is not None else initial
        self._cur = float(min(max(seed, min_rows), max_rows))
        self._best_throughput: float | None = None
        self._best_size: float | None = None

    def note_oom(self, *, rows: int | None = None) -> int:
        """Record that a batch ran out of device memory; return the size to use next.

        The predictive VRAM guard in `update` prevents most out-of-memory failures and cannot
        prevent all of them: it sees this process's VRAM, so a co-tenant that grows between
        two batches, an unusually long sequence in one shard, or allocator fragmentation all
        produce an OOM at a size that measured safe. What made that expensive was that nothing
        told the controller. The failing batch was halved by the retry, the halved batch
        reported perfectly good throughput, and the climb grew straight back into the same
        failure — so a job could spend most of its life failing and retrying while every
        measurement said it was improving.

        This records a **ceiling** below the size that failed. The ceiling is permanent for
        the run (the climb may approach it but never exceed it) and ratchets down on each
        consecutive failure, so a device that keeps failing converges instead of hovering.
        A subsequent success clears the streak but not the ceiling: one batch fitting is not
        evidence that a size which already failed has become safe.

        The batch size only shards rows, so this never changes a result — only how many rows
        are handed to the model at once.

        Args:
            rows: The size that failed, when the caller knows it. Defaults to the current
                target, which is what it will be unless the caller re-batched underneath.

        Returns:
            The next batch-size target, always at least `min_rows`.
        """
        failed = float(rows if rows is not None and rows > 0 else self.current())
        self._oom_streak += 1
        # Back off harder the more consecutive failures there have been: the first OOM only
        # proves this size is too big, while a third in a row means the estimate of how much
        # too big is itself wrong.
        backoff = self._shrink**self._oom_streak
        ceiling = max(float(self._min), failed * backoff)
        current = self._oom_ceiling
        self._oom_ceiling = ceiling if current is None else min(current, ceiling)
        self._cur = self._oom_ceiling
        # The plateau was measured at a size that has now failed, so it is not a target to
        # settle back to. Forget it and let the climb re-find one under the new ceiling.
        self._best_throughput = None
        self._best_size = None
        self._stale = 0
        return self.current()

    def update(self, throughput_rows_per_s: float, vram_fraction: float | None = None) -> int:
        """Observe throughput (and optional VRAM) at the current size; return the next."""
        self._oom_streak = 0  # a batch completed, so the run of consecutive failures is over
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
            # Persist the size the run will actually use, not the raw internal target. After
            # an out-of-memory the two diverge: `current()` clamps to the ceiling while
            # `_cur` keeps being multiplied by `grow` on every improving observation, so it
            # runs away above a size already proven to fail. Recording that runaway value
            # handed the *next* run a warm start well above the ceiling this run learned —
            # measured at 5316 rows after an OOM at 1000 pinned the ceiling to 700. It is
            # the same trap `best_size` documents; this path simply bypassed it.
            record_batch_size(self._hub, self._signature, self.current())
            # Predictive VRAM guard: a multiplicative grow scales the batch — and
            # roughly VRAM — by `grow`, which could overshoot the cap in a *single*
            # step before the reactive shrink (above) ever sees it. So only grow when
            # the predicted post-grow VRAM stays under the cap; otherwise hold at the
            # current (best) size, the safe ceiling. This makes the climb OOM-safe by
            # construction rather than relying on catching the OOM after the fact.
            if vram_fraction is None or vram_fraction * self._grow <= self._vram_cap:
                # Bounded by the OOM ceiling as well as `max_rows`, so the internal target
                # cannot run away above the size the device has already refused. Left
                # unbounded it climbed indefinitely while `current()` held flat at the
                # ceiling, which also poisoned `_best_size` — the value the plateau settles
                # back to — with a size that was never actually run.
                self._cur = min(float(self._max), self._cur * self._grow)
                if self._oom_ceiling is not None:
                    self._cur = min(self._cur, self._oom_ceiling)
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
        """The current batch-size target (clamped, rounded to a whole row count).

        Clamped by the OOM ceiling as well as the caller's bounds, so every path that grows
        the size — the hill-climb, the settle-back, the learned warm start — is bounded by
        what has actually been observed to fit. Enforcing it here rather than at each of those
        sites is what makes it impossible for a new growth path to be added that forgets it.
        """
        target = self._cur if self._oom_ceiling is None else min(self._cur, self._oom_ceiling)
        return int(min(self._max, max(self._min, round(target))))

    def best_size(self) -> int:
        """The best (throughput-optimal) size observed so far — the settled plateau.

        Falls back to the current size before any observation. This is the value the learned
        store persists so a future run can warm-start the climb."""
        best = self._best_size if self._best_size is not None else self._cur
        if self._oom_ceiling is not None:
            # This is the figure the learned store persists for the *next* run to warm-start
            # from, so a plateau measured before an out-of-memory must not escape through it —
            # that would hand the failing size straight back on the following run.
            best = min(best, self._oom_ceiling)
        return int(min(self._max, max(self._min, round(best))))
