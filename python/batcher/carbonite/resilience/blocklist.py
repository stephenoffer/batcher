"""Learning which nodes and devices are bad from what actually happened on them.

Every other health signal in the engine is a *reading*: a temperature, an ECC counter, a driver
error code. Those catch the failures hardware knows how to report, and they miss the ones that
matter just as much and report nothing at all — a node whose CUDA runtime no longer matches its
driver, a container image that half-deployed, a NIC that drops one packet in ten thousand, a
disk that has started returning `EIO`. On those nodes every probe reads healthy and every task
dies, and a scheduler with a free slot there will keep offering it. One bad machine then walks
the entire queue onto itself, and the job's failure mode is not "one node was broken" but "the
job never finished".

The missing signal is the outcome. A node that has failed the last five tasks placed on it is
bad *whatever* the telemetry says, and this is the ledger that says so.

Four properties make it safe to act on, and each one is there because the naive version of this
is worse than nothing:

* **Decay.** Failures lose weight over a half-life, so a node that had a bad minute an hour ago
  is not still being punished for it. Without decay the ledger only ever grows and the fleet
  shrinks monotonically over a long run.
* **Hysteresis.** Quarantine takes several failures to enter and a cooling-off period to leave,
  and the period doubles for a repeat offender. A single failure never removes a node — most
  single failures are the workload's fault, not the node's.
* **Probation on expiry.** A quarantine does not end in "forgiven", it ends in "watched". The
  target becomes schedulable again, and the *next* failure re-quarantines it immediately at
  double the cooldown rather than waiting for the threshold to rebuild. Without an expiry a
  node quarantined during a transient blip is out for the rest of the job; without probation,
  a genuinely bad node is re-learned from scratch every time its cooldown lapses.
* **A blast-radius cap.** Never quarantine more than a bounded share of the fleet. This is the
  one that matters most, and it is the one nobody adds until after the incident: when the cause
  is *systemic* — an expired credential, a bad image, a model file that 404s — every node fails
  every task, the ledger condemns all of them in the first minute, and a degraded job becomes a
  dead cluster. Past the cap the ledger keeps only the worst offenders quarantined and reports
  that it did, because at that point the evidence points at the job and not at the fleet.

The ledger is keyed by an opaque string, so the same implementation serves node ids and device
UUIDs. Carbonite owns it (a protection concern); `dist` consults it when placing work.
Thread-safe — a distributed executor records outcomes from every in-flight task at once.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from batcher._internal import events

__all__ = [
    "FaultLedger",
    "QuarantinePolicy",
    "TargetHealth",
    "configured_policy",
    "default_ledger",
    "reset_default_ledger",
]


@dataclass(frozen=True, slots=True)
class QuarantinePolicy:
    """When repeated failures make a node or device unschedulable.

    Attributes:
        failure_threshold: Decayed failure weight at which a target is quarantined. Weights are
            per-failure, so with the default a target must fail roughly three times in quick
            succession — one failure is almost never the node's fault.
        half_life_s: How long a failure keeps half its weight. This is what lets a node come
            back on its own after a bad patch, and what stops the ledger from being a permanent
            record of everything that ever went wrong.
        cooldown_s: How long a first quarantine lasts before a probe is allowed through.
        max_cooldown_s: Ceiling on the doubling, so a repeatedly bad node is retried
            occasionally rather than never — a target that is out forever is a capacity loss
            nobody is tracking.
        max_blocked_fraction: Share of known targets that may be quarantined at once. The
            circuit breaker on the circuit breaker: past this the failures are systemic and
            blaming the fleet makes the outage worse.
        min_targets: Below this many known targets the cap is not applied, because a fraction
            of a two-node fleet rounds to something that either blocks nothing or blocks
            everything. Small fleets rely on the threshold and the cooldown instead.
    """

    failure_threshold: float = 3.0
    half_life_s: float = 300.0
    cooldown_s: float = 60.0
    max_cooldown_s: float = 900.0
    max_blocked_fraction: float = 0.34
    min_targets: int = 4


@dataclass(slots=True)
class TargetHealth:
    """One node's or device's standing in the ledger.

    Attributes:
        key: The target's identifier — a node id or a device UUID.
        weight: Decayed failure weight as of `updated_s`.
        updated_s: Monotonic time the weight was last recalculated.
        blocked_until_s: Monotonic time the quarantine expires; `0.0` when not quarantined.
        cooldown_s: The cooldown the *next* quarantine will use, doubled on each offense.
        offenses: How many times this target has been quarantined.
        probing: Whether the target is on probation — its cooldown has lapsed and it is
            schedulable again, but the next failure re-quarantines it at once instead of
            waiting for the threshold to rebuild.
        capped: Whether this target met the threshold but was left scheduled because the
            blast-radius cap was already reached. Held so the event announcing that is
            published once rather than on every subsequent failure — a systemically broken
            fleet produces failures continuously, and the one signal that matters would
            otherwise be buried under thousands of copies of itself.
        successes: Successful outcomes recorded, for a report.
        failures: Failed outcomes recorded, for a report.
        reasons: Distinct failure categories seen, most recent last.
    """

    key: str
    weight: float = 0.0
    updated_s: float = 0.0
    blocked_until_s: float = 0.0
    cooldown_s: float = 0.0
    offenses: int = 0
    probing: bool = False
    capped: bool = False
    successes: int = 0
    failures: int = 0
    reasons: tuple[str, ...] = field(default_factory=tuple)


#: Failure weight by `classify` category. A failure the classifier says must move blames the
#: placement, so it counts fully. One that does not — a device OOM, a throttled endpoint — is
#: about the work rather than the machine, and counting those at full weight would quarantine
#: the healthiest node in a fleet simply for having been given the most work.
_WEIGHTS: dict[str, float] = {
    "device_corruption": 3.0,
    "device_fault": 2.0,
    "storage": 2.0,
    "host_oom": 1.0,
    "worker_lost": 1.0,
    "network": 0.5,
    "timeout": 0.5,
    "device_oom": 0.0,
    "throttled": 0.0,
    "preemption": 0.0,  # a planned reclamation says nothing about the node's health
    "application": 0.0,  # the job's bug, and blaming the node walks it across the fleet
}


class FaultLedger:
    """Failure history per node or device, and the quarantine decisions it implies.

    Examples:
        .. doctest::

            >>> from batcher.carbonite.resilience import FaultLedger, QuarantinePolicy
            >>> ledger = FaultLedger(QuarantinePolicy(failure_threshold=2.0, min_targets=1))
            >>> ledger.observe(("node-a", "node-b", "node-c", "node-d"))
            >>> for _ in range(3):
            ...     ledger.record_failure("node-a", "device_fault")
            >>> ledger.is_blocked("node-a"), ledger.is_blocked("node-b")
            (True, False)
    """

    def __init__(self, policy: QuarantinePolicy | None = None, *, label: str = "") -> None:
        """Build a ledger.

        Args:
            policy: Thresholds to apply; permissive defaults when omitted.
            label: What this ledger covers (`"node"`, `"device"`), carried on its events so a
                fleet with both can tell which one quarantined something.
        """
        self._policy = policy or QuarantinePolicy()
        self._label = label
        self._lock = threading.Lock()
        self._targets: dict[str, TargetHealth] = {}
        self._known: set[str] = set()

    # -- recording -------------------------------------------------------------------------

    def observe(self, keys: tuple[str, ...] | list[str]) -> None:
        """Note that these targets exist, so the blast-radius cap knows the fleet size.

        Without this the cap is computed against only the targets that have already failed,
        which on a fleet where one node fails first means "one of one is blocked" and the cap
        engages immediately. Call it with the current placement group at the start of a stage.

        Args:
            keys: Node ids or device UUIDs currently available.
        """
        with self._lock:
            self._known.update(k for k in keys if k)

    def record_failure(self, key: str, category: str = "worker_lost") -> None:
        """Record that work placed on this target failed.

        Args:
            key: Node id or device UUID.
            category: A `classify.CATEGORIES` name. Categories that do not blame the placement
                — a device OOM, a throttled endpoint, a deterministic application bug — carry
                zero weight and only update the counters, because quarantining a node over the
                job's own bug takes out the next node the retry lands on too.
        """
        weight = _WEIGHTS.get(category, 0.5)
        with self._lock:
            self._known.add(key)
            health = self._decayed(key)
            health.failures += 1
            if category not in health.reasons:
                health.reasons = (*health.reasons, category)
            if weight <= 0.0:
                return
            health.weight += weight
            if health.probing:
                # It failed on probation, so the quarantine was right. Re-block at double the
                # cooldown without waiting for the threshold to rebuild — a target that has
                # now failed either side of a cooldown should not get the same short timer.
                health.probing = False
                self._block(health, doubled=True)
                return
            if not self._quarantined(health) and health.weight >= self._policy.failure_threshold:
                self._block(health, doubled=False)

    def record_success(self, key: str) -> None:
        """Record that work placed on this target succeeded.

        A success is the only evidence that clears a quarantine, which is why the half-open
        probe exists at all: a target that is never given work can never prove it recovered.

        Args:
            key: Node id or device UUID.
        """
        with self._lock:
            self._known.add(key)
            health = self._decayed(key)
            health.successes += 1
            if health.probing:
                # It worked on probation. Release fully — the quarantine did its job and the
                # target has now proved itself on real work, which is the only evidence that
                # can clear one.
                health.probing = False
                health.blocked_until_s = 0.0
                health.weight = 0.0
                self._publish("released", health)
                return
            # A success does not zero the ledger — a node failing one task in three is a real
            # problem that a "reset on success" rule would hide forever. It repays one
            # failure's worth of weight, so a target has to be *mostly* healthy to clear.
            health.weight = max(0.0, health.weight - 1.0)

    # -- decisions -------------------------------------------------------------------------

    def is_blocked(self, key: str) -> bool:
        """Whether work should be kept off this target.

        Asking is what moves an expired quarantine into probation, so the single-target and
        whole-fleet queries below must agree on the answer — a scheduler that used one of them
        would otherwise never let a quarantined target back, or never keep one out.

        Args:
            key: Node id or device UUID.

        Returns:
            True while the cooldown is running. False once it has expired, at which point the
            target is on probation: schedulable, and re-quarantined at double the cooldown by
            its next failure rather than by another threshold's worth of them.
        """
        with self._lock:
            return self._blocked_now(self._decayed(key))

    def blocked_keys(self) -> tuple[str, ...]:
        """Every target currently quarantined, sorted.

        Returns:
            The keys a scheduler should avoid. Already capped by the blast-radius rule, so a
            caller can subtract this from its placement set without checking the size.
        """
        with self._lock:
            return tuple(sorted(k for k in self._targets if self._blocked_now(self._decayed(k))))

    def health(self, key: str) -> TargetHealth:
        """One target's standing, decayed to now.

        Args:
            key: Node id or device UUID.

        Returns:
            A copy of the record, safe to read without the lock.
        """
        with self._lock:
            health = self._decayed(key)
            return TargetHealth(**{f: getattr(health, f) for f in TargetHealth.__slots__})

    def report(self) -> tuple[TargetHealth, ...]:
        """Every target with a recorded outcome, worst first.

        The list an operator reads after a job that limped. Sorted by decayed weight, so the
        node that caused the trouble is at the top whether or not it ended up quarantined.

        Returns:
            One record per target, worst first, then by key for a stable order.
        """
        with self._lock:
            records = [self._decayed(k) for k in sorted(self._targets)]
            return tuple(
                sorted(records, key=lambda h: (-h.weight, -h.failures, h.key)),
            )

    def reset(self) -> None:
        """Forget every outcome, for a new job on a reused process."""
        with self._lock:
            self._targets.clear()
            self._known.clear()

    # -- internals -------------------------------------------------------------------------

    def _decayed(self, key: str) -> TargetHealth:
        """The target's record with its weight aged to now. The lock must be held."""
        now = time.monotonic()
        health = self._targets.get(key)
        if health is None:
            health = TargetHealth(key=key, updated_s=now)
            self._targets[key] = health
            return health
        elapsed = now - health.updated_s
        half_life = self._policy.half_life_s
        if elapsed > 0.0 and half_life > 0.0 and health.weight > 0.0:
            health.weight *= 0.5 ** (elapsed / half_life)
            if health.weight < 1e-6:
                health.weight = 0.0
        health.updated_s = now
        return health

    @staticmethod
    def _quarantined(health: TargetHealth) -> bool:
        """Whether a quarantine has been entered at all, expired or not. Lock must be held."""
        return health.blocked_until_s > 0.0

    def _blocked_now(self, health: TargetHealth) -> bool:
        """Whether the target is inside its cooldown, moving it to probation if not.

        The state transition lives here rather than in the two public queries so both give the
        same answer and neither can leave a target quarantined forever by not being the one
        that was called. The lock must be held.
        """
        if not self._quarantined(health):
            return False
        if time.monotonic() < health.blocked_until_s:
            return True
        health.probing = True
        return False

    def _block(self, health: TargetHealth, *, doubled: bool) -> None:
        """Quarantine a target unless the blast-radius cap forbids it. The lock must be held."""
        if not self._may_block(health):
            # Past the cap the failures are systemic. Blaming the fleet at that point removes
            # capacity from a job that is going to fail on its own cause anyway, and it removes
            # it permanently, because nothing succeeds anywhere to clear the ledger.
            #
            # Announced on the transition only. A systemically broken fleet keeps failing, so
            # this branch is reached by every subsequent failure on every capped target, and
            # an event per visit would bury the one signal that matters under copies of itself.
            if not health.capped:
                health.capped = True
                self._publish("quarantine_capped", health)
            return
        health.capped = False
        base = health.cooldown_s or self._policy.cooldown_s
        if doubled:
            base = min(base * 2.0, self._policy.max_cooldown_s)
        health.cooldown_s = base
        health.blocked_until_s = time.monotonic() + base
        health.offenses += 1
        self._publish("quarantined", health)

    def _may_block(self, health: TargetHealth) -> bool:
        """Whether quarantining one more target stays inside the cap. The lock must be held."""
        known = len(self._known)
        if known < self._policy.min_targets:
            return True
        now = time.monotonic()
        blocked = sum(
            1
            for k, h in self._targets.items()
            if k != health.key and h.blocked_until_s > 0.0 and now < h.blocked_until_s
        )
        return (blocked + 1) <= int(known * self._policy.max_blocked_fraction)

    def _publish(self, event: str, health: TargetHealth) -> None:
        """Announce a quarantine transition. The lock must be held; publishing is cheap."""
        events.publish(
            events.RECOVERY,
            name=self._label,
            event=event,
            target=health.key,
            weight=round(health.weight, 3),
            offenses=health.offenses,
            reasons=list(health.reasons),
            cooldown_s=round(health.cooldown_s, 1),
        )


#: One ledger per kind, process-wide. A driver places work from one process, so a shared ledger
#: is what lets a stage benefit from what the previous stage learned — a per-stage ledger would
#: re-discover the same bad node at every shuffle.
_LEDGERS: dict[str, FaultLedger] = {}
_LEDGER_LOCK = threading.Lock()


def default_ledger(kind: str = "node") -> FaultLedger:
    """The process-wide ledger for a kind of target.

    Args:
        kind: `"node"` or `"device"` — or any label; ledgers are created on first use.

    Returns:
        The shared `FaultLedger`, built with the configured policy on first use.
    """
    with _LEDGER_LOCK:
        ledger = _LEDGERS.get(kind)
        if ledger is None:
            ledger = FaultLedger(configured_policy(), label=kind)
            _LEDGERS[kind] = ledger
        return ledger


def reset_default_ledger() -> None:
    """Drop every process-wide ledger, for a new job on a reused process (and for tests)."""
    with _LEDGER_LOCK:
        _LEDGERS.clear()


def configured_policy() -> QuarantinePolicy:
    """The quarantine policy the active configuration asks for.

    Keeps the mapping from `fault_tolerance.quarantine` to a policy in one place, so no caller
    restates it and the two cannot drift.

    Returns:
        The policy to apply on this deployment.
    """
    from batcher.config import active_config

    cfg = active_config().fault_tolerance.quarantine
    return QuarantinePolicy(
        failure_threshold=cfg.failure_threshold,
        half_life_s=cfg.half_life_s,
        cooldown_s=cfg.cooldown_s,
        max_cooldown_s=cfg.max_cooldown_s,
        max_blocked_fraction=cfg.max_blocked_fraction,
    )
