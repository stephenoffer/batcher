"""The middle stage of an N-stage streaming pipeline: fetch upstream, transform, republish.

A two-stage pipeline needs two kinds of actor — one that *produces* morsels from a partition
(`producers.ProducerActor`) and one that *consumes* them and returns rows
(`executors.map._MapActor`). A longer pipeline needs a third: a stage that is a consumer of
the stage below it and a producer for the stage above it. That is this actor, and it is the
whole difference between splitting a chain once and splitting it at every resource boundary.

It matters because the single cut put every stage above the first model in *one* actor. Two
chained models therefore shared a device and took turns, and a CPU postprocess ran on the GPU
actor — spending device time on host work and tying the two stages' pool sizes together. With
a relay between them each stage is its own pool, sized and placed on its own.

The hand-off is the same one the two-stage pipeline uses: the morsel moves over credit-bounded
Arrow Flight and only `(addr, ticket)` crosses Ray. What a relay adds is that the fetch and the
publish happen in the same call, so a morsel is never resident on the driver at any hop.
"""

from __future__ import annotations

from batcher.plan.logical import LogicalPlan

__all__ = ["RelayActor"]


try:
    import ray

    @ray.remote
    class RelayActor:
        """A middle stage: fetches one upstream morsel, maps it, publishes the result.

        Its model (a class UDF) builds once here, so a relay carrying a load-once stage
        reuses it across every morsel — the same property the producer and the terminal
        consumer already have, and the reason a stage gets a resident pool rather than a task.
        """

        def __init__(self, plan0: LogicalPlan, credits: int, target_rows: int = 0) -> None:
            from batcher.carbonite.transfer import ShuffleSession
            from batcher.dist.executors.map import _prebuild_factories, _sustained_utilization

            self._plan = _prebuild_factories(plan0)
            self._target_rows = max(0, int(target_rows))
            host = ray.util.get_node_ip_address()
            self.session = ShuffleSession(credits, advertise_host=host)
            self._peak = 0
            # The same utilization window the terminal consumer keeps, because a relay can be
            # a GPU stage too — a chain with two models has one in a relay and one at the end,
            # and sizing only the last one's pool from measurement is how the middle model
            # becomes the bottleneck nobody is looking at.
            self._util = _sustained_utilization()

        def addr(self) -> str:
            """This relay's Flight address, for the stage above to fetch from."""
            return self.session.addr

        def node_host(self) -> str:
            """The node this relay runs on, so the driver can hand it morsels from nearby."""
            import os

            return os.environ.get("BATCHER_ADVERTISE_HOST") or ray.util.get_node_ip_address()

        def gpu_stats(self) -> float | None:
            """This relay's sustained GPU utilization, drained — the same probe `_MapActor` has."""
            return self._util.drain()

        def consume(self, up_addr: str, up_ticket, plan_id: int, stage_id: int, morsel: int) -> int:
            """Fetch `(up_addr, up_ticket)`, run this stage, publish the output morsels.

            The published tickets are **derived**, not passed in: output `i` of this hand-off is
            published under `(plan_id, stage_id, morsel, i)`, so the driver can name every child
            from the count alone. That is what makes a re-run after a lost actor idempotent —
            the same upstream morsel replayed through a fresh relay publishes the same tickets
            and produces the same keys, so the driver's results overwrite rather than duplicate.

            Args:
                up_addr: The upstream stage's Flight address.
                up_ticket: The upstream morsel's ticket.
                plan_id: The query's plan id.
                stage_id: This stage's index, which scopes the tickets it publishes.
                morsel: The driver's stable id for the upstream morsel being consumed.

            Returns:
                How many morsels were published — `0` when this stage produced no rows, which
                is a complete answer and not a failure.
            """
            from batcher import core
            from batcher.carbonite.transfer import ShuffleTicket
            from batcher.carbonite.transfer.lifecycle import process_client
            from batcher.dist.streaming.producers import coalesce
            from batcher.io.source import InMemorySource

            # The pooled client, for the same reason `_MapActor.run_split` uses it: this is a
            # per-morsel fetch, and the one-shot form opens a fresh gRPC channel each time.
            rows = process_client().fetch(up_addr, str(up_ticket))
            if not rows:
                return 0
            self._util.begin_call()
            out = core.execute_with_udfs(self._plan, [InMemorySource(rows)])
            self._util.end_call()
            published = 0
            for batch in coalesce(out, self._target_rows):
                if batch.num_rows == 0:
                    continue
                self.session.publish(ShuffleTicket(plan_id, stage_id, morsel, published), [batch])
                published += 1
            self._peak = max(self._peak, self.session.partition_count)
            return published

        def release(self, ticket) -> None:
            """Evict a published morsel once the stage above has fetched it."""
            self.session.release(ticket)

        def peak_retained(self) -> int:
            """Peak published-but-unreleased morsels this relay ever held (a memory probe)."""
            return self._peak

except ImportError:  # pragma: no cover - ray optional
    RelayActor = None  # type: ignore
