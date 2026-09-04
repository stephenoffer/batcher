"""Which capacity a stage runs on: spot where the work is recomputable, on-demand where it isn't.

A mixed fleet buys the same core at two prices with two reliability contracts, and Batcher's
own execution model already says which stages belong on which. A stateless map partition
*is* its own lineage — it recomputes idempotently from a durable partition descriptor, which
is precisely why `policies._barrier` can resubmit one after a preemption instead of failing
the stage. A shuffle worker cannot: it holds accumulated partial state and the mapped output
its peers will fetch, so reclaiming it costs the stage, not the task.

Ray can express that. `label_selector` holds a task or actor to nodes carrying a label, and
`fallback_strategy` names where it should go instead when the first choice cannot host it. The
pair is what makes the preference safe to state: a fleet asking for on-demand capacity on a
cluster that has run out of it lands on spot rather than pending forever.

**Every gate here fails toward today's behavior.** The selector is emitted only on a fleet
that is *actually mixed* — some spot capacity and some on-demand, both labelled, under one
label key — because on any other fleet it either matches everything (a no-op that costs a
scheduler round trip) or matches nothing (a hang). That makes it a no-op on every unlabelled,
single-market, and single-node cluster, which is nearly all of them.

The label vocabulary is `topology`'s: Ray's own `ray.io/market-type` plus the three keys the
major Kubernetes provisioners set, read in that order. The key travels with the value for the
reason `node_zone` states — a Karpenter-labelled fleet answers nothing to a selector on Ray's
key — which is why the census carries `market_label` beside `market_type`.
"""

from __future__ import annotations

import functools

from batcher._internal.logging import note_suppressed
from batcher.dist.executors.ray_runtime.fabric.topology import ON_DEMAND, SPOT
from batcher.plan.resource import CAPACITY_ON_DEMAND, CAPACITY_SPOT

__all__ = ["capacity_bundle_selector", "capacity_selector", "fleet_market_split"]

#: Preference -> the market-type label value that satisfies it, and the one to fall back to.
#: The fallback is the whole point of stating a preference at all: it is what makes "prefer
#: on-demand" a preference rather than a way to hang a query on a fleet that has none left.
_TARGETS = {
    CAPACITY_SPOT: (SPOT, ON_DEMAND),
    CAPACITY_ON_DEMAND: (ON_DEMAND, SPOT),
}


def _noop() -> None:  # pragma: no cover - never called, only decorated
    """The body of the probe remote function. Ray validates options without running it."""


@functools.cache
def _accepts(option: str, probe: str) -> bool:
    """Whether this Ray accepts `option` on a task, decided once per process.

    Both keys used here are newer than the rest of the scheduling API, and an older Ray
    rejects an unknown one at `.options()` time — at *every* task submission, which is the
    wrong place to discover it. `.options()` validates locally and does not need a live
    cluster, so the question is answerable once, at the cost of one decorated no-op.

    Args:
        option: The `.options(...)` keyword to test.
        probe: A minimal valid value for it, as a label value.

    Returns:
        True when the option validates, False on any rejection or when Ray is unimportable.
    """
    try:
        import ray

        value = (
            {"batcher.io/probe": probe}
            if option == "label_selector"
            else [{"label_selector": {"batcher.io/probe": probe}}]
        )
        ray.remote(_noop).options(**{option: value})
        return True
    except Exception as exc:
        note_suppressed("dist", f"probe Ray support for {option}", exc)
        return False


def fleet_market_split(census: list[dict] | None = None) -> tuple[str, dict[str, float]]:
    """The fleet's market label key and the schedulable cores behind each market type.

    Args:
        census: `scaling.node_class_census()` output, or `None` to read the live fleet. Passed
            in by the tests and by any caller that already holds the census, so the fleet is
            walked once per query rather than once per question.

    Returns:
        `(label_key, {market_type: cores})`. The key is `""` — and the mapping empty — when no
        node carries a market label, or when the fleet carries **more than one** market label
        key. Two keys are not a fleet this can select on: a selector names exactly one, so
        emitting one would silently exclude every node labelled under the other.
    """
    if census is None:
        try:
            from batcher.dist.executors.ray_runtime.scaling import node_class_census

            census = node_class_census()
        except Exception as exc:
            note_suppressed("dist", "read the fleet market split", exc)
            return "", {}
    keys = {row.get("market_label") or "" for row in census}
    keys.discard("")
    if len(keys) != 1:
        return "", {}
    cores: dict[str, float] = {}
    for row in census:
        market = row.get("market_type") or ""
        if not market:
            continue
        # Free cores, not nameplate: "can this market host the fleet" is a question about what
        # is unreserved now, and the nameplate answers a different one — the same distinction
        # `node_classes` draws between `cpus` and `free_cpus`.
        free = float(row.get("free_cpus") or row.get("cpus") or 0.0)
        cores[market] = cores.get(market, 0.0) + free * max(1, int(row.get("count") or 1))
    return next(iter(keys)), cores


def capacity_selector(
    preference: str,
    *,
    workers: int = 1,
    num_cpus: float = 1.0,
    census: list[dict] | None = None,
) -> dict:
    """Ray `.options(...)` label-selector fragment holding a stage to a market type, or `{}`.

    Emitted only when every one of the following holds, and `{}` otherwise — so a cluster that
    cannot express the preference keeps exactly the placement it has today:

    1. `preference` names a market type (`CAPACITY_ANY` states nothing).
    1. `distributed.capacity_aware_placement` is on.
    1. This Ray accepts `label_selector` on a task.
    1. The fleet labels its market type, under a single label key (`fleet_market_split`).
    1. The fleet is genuinely **mixed** — both market types present with schedulable cores.
       On a single-market fleet the selector matches either everything or nothing, and the
       second of those is a hang.
    1. The preferred market can host `workers x num_cpus` cores, **or** this Ray accepts
       `fallback_strategy`, which makes an unsatisfiable preference degrade instead of pend.

    Placement never changes which rows a task processes, so the result is identical either
    way. What changes is which capacity the work lands on, and therefore what a preemption
    costs.

    Args:
        preference: One of `CAPACITY_PREFERENCES`.
        workers: How many tasks or actors the stage will place.
        num_cpus: Cores one of them requests.
        census: A pre-read `node_class_census()`, or `None` to read the live fleet.

    Returns:
        `{"label_selector": {...}}`, optionally with `"fallback_strategy"`, or `{}`.
    """
    target = _TARGETS.get(preference)
    if target is None:
        return {}
    from batcher.config import active_config

    if not active_config().distributed.capacity_aware_placement:
        return {}
    if not _accepts("label_selector", SPOT):
        return {}
    key, cores = fleet_market_split(census)
    if not key:
        return {}
    want, other = target
    if cores.get(SPOT, 0.0) <= 0.0 or cores.get(ON_DEMAND, 0.0) <= 0.0:
        # Not a mixed fleet. Stating a preference on it can only narrow, never choose.
        return {}
    selector = {"label_selector": {key: want}}
    if _accepts("fallback_strategy", other):
        selector["fallback_strategy"] = [{"label_selector": {key: other}}]
        return selector
    # No fallback available: the preference is only safe where the preferred market can
    # actually hold the stage, because an unsatisfiable selector pends rather than degrades.
    return selector if cores[want] >= max(1, workers) * max(num_cpus, 1e-9) else {}


def capacity_bundle_selector(
    preference: str,
    *,
    workers: int,
    num_cpus: float,
    census: list[dict] | None = None,
) -> dict[str, str]:
    """The market-type **bundle** label selector for a gang-scheduled fleet, or `{}`.

    A placement group decides where its bundles go, and an actor pinned to a bundle inherits
    that decision. So a fleet states its capacity preference on the bundles, never on the
    actors: a `label_selector` on an actor whose bundle already landed on the other market
    contradicts the reservation and the actor never schedules — a hang, not a fallback.

    Bundles have no `fallback_strategy`, so the gate does that job instead. The selector is
    emitted only when the preferred market's free cores can hold the whole fleet, because a
    group that cannot form is not refused: it is waited on until `placement_timeout_s` and
    then abandoned to default scheduling, which costs the reservation's whole budget to end
    up where declining would have started.

    Args:
        preference: One of `CAPACITY_PREFERENCES`.
        workers: Bundles in the fleet.
        num_cpus: Cores one bundle reserves.
        census: A pre-read `node_class_census()`, or `None` to read the live fleet.

    Returns:
        A `{label_key: market_type}` mapping to merge into `bundle_label_selector`, or `{}`.
    """
    opts = capacity_selector(preference, workers=workers, num_cpus=num_cpus, census=census)
    selector: dict[str, str] = opts.get("label_selector", {})
    if not selector:
        return {}
    _key, cores = fleet_market_split(census)
    want = next(iter(selector.values()))
    return selector if cores.get(want, 0.0) >= max(1, workers) * max(num_cpus, 1e-9) else {}
