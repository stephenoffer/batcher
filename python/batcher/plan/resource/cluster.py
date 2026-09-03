"""The *shape* of the fleet a plan is optimized for — nodes, devices, and the wires between.

`HardwareProfile` describes the **binding node**: the smallest VRAM, the weakest worker's
cores, one aggregate device count. That is exactly the right contract for sizing an operator
so it is valid wherever it lands, and it is the wrong one for every decision that depends on
where two pieces of work are *relative to each other*.

On the machines Batcher targets those are not the same question. Thirty-two devices on four
nodes and thirty-two devices on thirty-two nodes report an identical `gpu_count`, and a hash
shuffle across them differs by a factor of eight in bytes on the wire: on the first fleet a
quarter of every exchange never leaves its host, and inside a host it crosses NVLink at
something close to memory bandwidth rather than a NIC at a fortieth of it. A cost model that
cannot see the difference ranks the two fleets' plans identically, and the plan it picks is
right for at most one of them.

`ClusterShape` is that missing half: one record per node class with its devices, its coherent
fabric width, its rack, and its egress. It carries **no policy** — no weights, no thresholds,
no verdicts. It answers structural questions ("what fraction of a shuffle across `W` workers
stays inside a node?") and leaves what a byte costs on each tier to Kyber, which is the layer
allowed to decide.

It lives in `plan` for the same reason `HardwareProfile` does: `dist` can populate it from live
Ray topology and `kyber` can consume it without either importing the other.

**Every field is optional and every derivation degrades.** An empty `ClusterShape` — what a
single-node run, an unreadable topology, and every existing caller get — reports the flat
answers the engine has always used, so nothing about an unlabelled deployment moves.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from batcher.plan.resource.locality import (
    LocalityShares,
    _domain_split,
    _group_share_census,
    _spread_census,
)

__all__ = ["ClusterShape", "NodeShape"]


@dataclass(frozen=True, slots=True)
class NodeShape:
    """One node in the fleet, as much of it as the topology could report.

    Every field defaults to the "could not report it" sentinel (`0` or `""`) rather than to a
    plausible value, for the reason `HardwareProfile` states: a fabricated figure here does not
    stay here. It propagates into a locality share, which prices a shuffle, which picks a join
    order — and the resulting plan is wrong in a way no test can see, because the number it
    rests on was never measured.

    Attributes:
        node_id: Scheduler-assigned node identifier, `""` when unlabelled.
        cpu_cores: Usable cores the node advertises.
        memory_bytes: Usable host RAM.
        gpus: Accelerator devices the node advertises.
        accelerator_type: Device model (`"NVIDIA_H100"`), `""` when unlabelled or mixed.
        gpu_memory_bytes: Usable VRAM of one of this node's devices.
        nvlink_domain: Devices on this node that share one coherent fabric, `0` when unknown.
            Never larger than `gpus`, which `local_domain` enforces rather than trusting.
        fabric_gbps: The node's aggregate off-node rate in gigabits per second, `0.0` when
            unmeasured.
        rails: NICs carrying device traffic off the node, `0` when unknown. A device's egress
            is its rail's share, not the node's total, which is what makes an eight-device
            one-rail node behave nothing like an eight-device eight-rail one.
        rack: Physical enclosure identifier, `""` when unlabelled.
        zone: Availability zone, `""` when unlabelled.
        power_zone: Power domain (busway or PDU) the node draws from, `""` when unlabelled.
        unhealthy_gpus: Devices quarantined or degraded out of rotation. Subtracted from the
            schedulable count, never from `gpus` — a plan sized against the devices that exist
            and scheduled onto the devices that work is the pair of figures a fan-out needs.
    """

    node_id: str = ""
    cpu_cores: int = 0
    memory_bytes: int = 0
    gpus: int = 0
    accelerator_type: str = ""
    gpu_memory_bytes: int = 0
    nvlink_domain: int = 0
    fabric_gbps: float = 0.0
    rails: int = 0
    rack: str = ""
    zone: str = ""
    power_zone: str = ""
    unhealthy_gpus: int = 0

    @property
    def healthy_gpus(self) -> int:
        """Devices on this node that are actually schedulable, never below zero."""
        return max(0, self.gpus - max(0, self.unhealthy_gpus))

    @property
    def local_domain(self) -> int:
        """Devices here that exchange over one coherent fabric, at least 1.

        Capped by the node's own device count: an H100's domain is eight, and on a two-device
        node the domain is two whatever the datasheet says. A node with no reported domain
        reports its whole device count, which is the assumption every caller made before —
        everything on a node is local — rather than a fabricated fabric width.
        """
        devices = max(1, self.gpus)
        return devices if self.nvlink_domain <= 0 else min(self.nvlink_domain, devices)

    @property
    def domains(self) -> int:
        """How many coherent-fabric groups this node's devices fall into, at least 1."""
        devices = max(1, self.gpus)
        width = self.local_domain
        return max(1, -(-devices // width))

    @property
    def aggregate_gpu_memory_bytes(self) -> int:
        """VRAM across every device on this node, `0` when either figure is unknown."""
        return max(0, self.gpus) * max(0, self.gpu_memory_bytes)

    @property
    def per_device_egress_gbps(self) -> float:
        """The off-node rate one device here actually has, in gigabits per second.

        The node's aggregate rate divided by the devices contending for it. Sizing a
        cross-node transfer against the node total over-commits by the device count on
        exactly the dense nodes where the mistake costs most, and a rail assignment cannot
        fix arithmetic that never divided.

        `0.0` when the rate is unmeasured, which a caller must read as "no opinion" and not
        as "no bandwidth".
        """
        if self.fabric_gbps <= 0.0:
            return 0.0
        return self.fabric_gbps / max(1, self.gpus)


@dataclass(frozen=True, slots=True)
class ClusterShape:
    """The fleet's structure: one record per *class* of node, and what falls out of them.

    An empty shape — the default, and what a single-node run or an unreadable topology
    produces — reports the flat answers the engine used before this existed. Every consumer
    must therefore treat `0`, `""` and `unknown` as "keep what you had", never as a measurement.

    A fleet is described as a **census**: distinct node shapes, each with how many nodes have
    it. A hundred-thousand-node cluster is a handful of instance types across a handful of
    zones, and holding one record per node made every figure here O(nodes) — recomputed on
    each property access, since these are properties and not stored fields. Measured on a
    synthetic 100,000-node fleet: 106 ms for one `locality_shares` call, which the cost model
    makes once per pipeline breaker, and 28 ms to hash the shape, which happens on every plan
    cache lookup because the shape reaches the key.

    `multiplicity` is parallel to `nodes`. Empty means one node per entry, which is what every
    caller that builds a shape by hand gets and is bit-identical to the per-node form — the
    census is a compression of the description, never a change to it.

    Attributes:
        nodes: One entry per distinct node shape.
        multiplicity: How many nodes have each shape, or empty for one each.
    """

    nodes: tuple[NodeShape, ...] = field(default_factory=tuple)
    multiplicity: tuple[int, ...] = field(default_factory=tuple)

    @property
    def _weights(self) -> tuple[int, ...]:
        """How many nodes each entry stands for, defaulting to one each."""
        return self.multiplicity or (1,) * len(self.nodes)

    @property
    def _census(self) -> tuple[tuple[NodeShape, int], ...]:
        """`(shape, node count)` pairs — the fleet as this class actually reasons about it."""
        return tuple(zip(self.nodes, self._weights, strict=True))

    # ---- counts -------------------------------------------------------------------------

    @property
    def known(self) -> bool:
        """Whether the topology was readable at all."""
        return bool(self.nodes)

    @property
    def node_count(self) -> int:
        """Nodes in the fleet, `0` when the topology is unknown."""
        return sum(self._weights)

    @property
    def gpu_nodes(self) -> tuple[NodeShape, ...]:
        """The distinct accelerator-bearing node shapes.

        Shapes, not nodes: on a census fleet two entries here can stand for a thousand
        machines each. Every caller reads it for a `min`, a `max` or a set of models, all of
        which are indifferent to how many nodes share a shape. The two that are not — the
        device count and the exchange placement — use `gpu_census` instead.
        """
        return tuple(n for n in self.nodes if n.gpus > 0)

    @property
    def gpu_census(self) -> tuple[tuple[NodeShape, int], ...]:
        """`(shape, node count)` for the accelerator-bearing part of the fleet."""
        return tuple((n, w) for n, w in self._census if n.gpus > 0)

    @property
    def gpu_node_count(self) -> int:
        """Nodes carrying at least one accelerator."""
        return sum(w for _, w in self.gpu_census)

    @property
    def total_gpus(self) -> int:
        """Accelerator devices across the fleet."""
        return sum(n.gpus * w for n, w in self._census)

    @property
    def healthy_gpus(self) -> int:
        """Devices across the fleet that are schedulable right now."""
        return sum(n.healthy_gpus * w for n, w in self._census)

    @property
    def total_cores(self) -> int:
        """Usable cores across the fleet."""
        return sum(n.cpu_cores * w for n, w in self._census)

    @property
    def total_memory_bytes(self) -> int:
        """Usable host RAM across the fleet."""
        return sum(n.memory_bytes * w for n, w in self._census)

    @property
    def aggregate_gpu_memory_bytes(self) -> int:
        """VRAM across every device in the fleet; a node that could not report it adds nothing.

        A **partial** total, and it has to be read as one: a node whose VRAM is unknown
        contributes `0`, so on a partly-labelled fleet this under-states the real capacity
        rather than reporting `0` for the whole thing. Under-stating is the safe direction for
        a capacity figure — it declines work the fleet could have held, where over-stating
        admits work it cannot. Use `known` and `binding_gpu_memory_bytes` to tell "the fleet is
        small" from "the fleet did not say".
        """
        return sum(n.aggregate_gpu_memory_bytes * w for n, w in self._census)

    @property
    def device_models(self) -> tuple[str, ...]:
        """Distinct accelerator models present, sorted; empty on an unlabelled fleet."""
        return tuple(sorted({n.accelerator_type for n in self.gpu_nodes if n.accelerator_type}))

    @property
    def homogeneous_gpus(self) -> bool:
        """Whether every accelerator node reports the same device model.

        False on an unlabelled fleet as well as a genuinely mixed one: both mean "a
        model-specific decision has no single right answer here", which is the question every
        caller is really asking.
        """
        models = self.device_models
        return len(models) == 1 and all(n.accelerator_type for n in self.gpu_nodes)

    @property
    def binding_gpu_memory_bytes(self) -> int:
        """VRAM of the smallest device in the fleet, `0` when unknown.

        The figure a shard must fit, because a shard sized to the largest device out-of-memories
        on every other one.
        """
        sized = (n.gpu_memory_bytes for n in self.gpu_nodes if n.gpu_memory_bytes > 0)
        return min(sized, default=0)

    @property
    def binding_cpu_cores(self) -> int:
        """Cores on the smallest worker, `0` when no node reported any.

        The per-worker figure a plan must be valid at, for the reason `binding_gpu_memory_bytes`
        takes the smallest device: a task sized to the largest node's core count over-subscribes
        every smaller node it lands on.
        """
        sized = (n.cpu_cores for n in self.nodes if n.cpu_cores > 0)
        return min(sized, default=0)

    @property
    def binding_memory_bytes(self) -> int:
        """Usable RAM on the smallest worker, `0` when no node reported any."""
        sized = (n.memory_bytes for n in self.nodes if n.memory_bytes > 0)
        return min(sized, default=0)

    @property
    def largest_nvlink_domain(self) -> int:
        """Widest coherent device group anywhere in the fleet, `0` when there are no devices."""
        return max((n.local_domain for n in self.gpu_nodes), default=0)

    @property
    def max_gpus_per_node(self) -> int:
        """Devices on the densest accelerator node, `0` when there are none."""
        return max((n.gpus for n in self.gpu_nodes), default=0)

    @property
    def min_gpus_per_node(self) -> int:
        """Devices on the sparsest accelerator node, `0` when there are none."""
        return min((n.gpus for n in self.gpu_nodes), default=0)

    @property
    def racks(self) -> int:
        """Distinct labelled racks, `0` on an unlabelled fleet."""
        return len({n.rack for n in self.nodes if n.rack})

    @property
    def power_zones(self) -> int:
        """Distinct labelled power domains, `0` on an unlabelled fleet."""
        return len({n.power_zone for n in self.nodes if n.power_zone})

    @property
    def zones(self) -> int:
        """Distinct labelled availability zones, `0` on an unlabelled fleet."""
        return len({n.zone for n in self.nodes if n.zone})

    @property
    def fabric_gbps(self) -> float:
        """The binding node's off-node rate, `0.0` when unmeasured anywhere.

        The slowest measured node rather than the mean: a shuffle finishes when its last
        participant does, so the fleet's exchange runs at the rate of the node that carries it
        worst.
        """
        rated = [n.fabric_gbps for n in self.nodes if n.fabric_gbps > 0.0]
        return min(rated) if rated else 0.0

    # ---- locality -----------------------------------------------------------------------

    def exchange_width(self, unit: str = "cpu") -> int:
        """How many workers an exchange across this fleet actually fans out to.

        The fleet's schedulable capacity — cores for a relational fleet, devices for a device
        fan-out — because that is what the scheduler clamps a fan-out to, and so what a large
        plan reaches. It exists because `HardwareProfile.worker_count` is the **node** count,
        which is a fine proxy for the volume term and a bad one for the tier shares: at one
        worker per node an exchange has no intra-node tier at all, which erases the whole
        distinction on precisely the dense fleets the tier model exists for.

        Device capacity is counted in **schedulable** devices, not in devices that exist: a
        board the fleet has quarantined for uncorrectable ECC or a pending reset is one the
        scheduler will not place on, so a fan-out sized to it asks for a width the cluster
        cannot satisfy and the placement group pends. `total_gpus` is the right figure for
        sizing a shard (it says what hardware the plan may meet) and the wrong one for sizing a
        fan-out (which needs what hardware will actually take work), and this is the fan-out.

        Args:
            unit: `"cpu"` for a relational exchange, `"gpu"` for a device fan-out.

        Returns:
            The capacity, at least 1. `1` on an unknown fleet, which makes the caller fall back
            to whatever width it already had.
        """
        if not self.known:
            return 1
        # A fleet whose devices are *all* quarantined still has to run somewhere, and reporting
        # a width of zero would be read as "unknown" rather than as "nothing schedulable" —
        # so the physical count stands in, matching `_schedulable`'s survivors-or-nothing rule.
        capacity = (self.healthy_gpus or self.total_gpus) if unit == "gpu" else self.total_cores
        return max(1, capacity)

    def locality_shares(self, workers: int, *, unit: str = "cpu") -> LocalityShares:
        """How an all-to-all exchange across `workers` splits over the interconnect tiers.

        Args:
            workers: Workers the exchange runs across.
            unit: `"cpu"` places workers against each node's cores, `"gpu"` against its
                devices. A relational shuffle runs on the CPU fleet and a device fan-out on the
                accelerator nodes, and on a cluster whose CPU-only nodes outnumber its GPU
                nodes the two placements are nothing alike.

        Returns:
            The five shares. On an unknown topology, on a single worker, or with a `unit` the
            fleet reports nothing for, everything that is not local is charged to `cross_rack`
            — the flat model the cost axis used before this existed, and the pessimistic
            direction, so an unreadable fleet is never ranked as though it were fast.
        """
        count = max(1, int(workers))
        local = 1.0 / count
        census = self.gpu_census if unit == "gpu" else self._census
        classes = [((n.gpus if unit == "gpu" else n.cpu_cores), w) for n, w in census]
        if not census or sum(c * w for c, w in classes) <= 0 or count == 1:
            return LocalityShares(local=local, cross_rack=1.0 - local)

        # Per class: the workers it took as a whole, then the individual placements inside it.
        # `_spread` deals evenly among equal capacities, so a class's nodes differ by at most
        # one worker and `divmod` recovers the exact split — `taken` nodes hold `each + 1` and
        # the rest hold `each`. Working in classes is what keeps this O(classes) rather than
        # O(nodes); on a 100,000-node fleet the per-node form took 106 ms per call, and the
        # cost model makes one call per pipeline breaker.
        node_groups: list[tuple[int, int]] = []
        domain_groups: list[tuple[int, int]] = []
        solo_racks: list[tuple[int, int]] = []
        by_rack: dict[tuple[str, str], int] = {}
        for (node, weight), total in zip(census, _spread_census(classes, count), strict=True):
            if total <= 0:
                continue
            each, taken = divmod(total, weight)
            for placed, many in ((each + 1, taken), (each, weight - taken)):
                if placed <= 0 or many <= 0:
                    continue
                node_groups.append((placed, many))
                domain_groups.extend((size, many) for size in _domain_split(node, placed, unit))
                if node.rack:
                    # Qualified by zone, so a rack label is only ever compared within the
                    # availability zone that issued it. Rack identifiers are namespaced per
                    # zone by every scheduler that emits them, so two nodes in different zones
                    # sharing the string `"rack-3"` are in different buildings — and grouping
                    # them would report a cross-zone byte as rack-local, which is the largest
                    # single under-charge the tier model can make.
                    key = (node.zone, node.rack)
                    by_rack[key] = by_rack.get(key, 0) + placed * many
                else:
                    # An unlabelled node is its own rack: two nodes that never said they were
                    # adjacent are not evidence that they are, and assuming otherwise would
                    # under-charge every cross-host byte on an unlabelled fleet — the common
                    # case. So these stay `many` separate groups rather than one of `many`.
                    solo_racks.append((placed, many))
        if not node_groups:
            return LocalityShares(local=local, cross_rack=1.0 - local)

        same_node = _group_share_census(node_groups, count)
        same_domain = _group_share_census(domain_groups, count)
        same_rack = _group_share_census(
            [*solo_racks, *((placed, 1) for placed in by_rack.values())], count
        )

        # Clamped and ordered so rounding in the spread can never produce a negative share or
        # one tier claiming data another already took. The containment local <= domain <= node
        # <= rack is a property of the grouping, not of the arithmetic, so it is enforced.
        same_domain = max(local, min(same_domain, 1.0))
        same_node = max(same_domain, min(same_node, 1.0))
        same_rack = max(same_node, min(same_rack, 1.0))
        return LocalityShares(
            local=local,
            intra_domain=same_domain - local,
            intra_node=same_node - same_domain,
            intra_rack=same_rack - same_node,
            cross_rack=1.0 - same_rack,
        )

    def summary(self) -> dict:
        """A flat description of the fleet, for the decision log and `explain()`.

        Returns:
            Node and device counts, the widest coherent domain, the device models present, and
            how many racks and power zones the fleet spans. Every count is `0` on an unknown
            topology, which a reader distinguishes from a genuinely empty fleet by `known`.
        """
        return {
            "known": self.known,
            "nodes": self.node_count,
            "gpu_nodes": self.gpu_node_count,
            "gpus": self.total_gpus,
            "healthy_gpus": self.healthy_gpus,
            "cores": self.total_cores,
            "largest_nvlink_domain": self.largest_nvlink_domain,
            "device_models": list(self.device_models),
            "homogeneous_gpus": self.homogeneous_gpus,
            "racks": self.racks,
            "power_zones": self.power_zones,
            "zones": self.zones,
            "fabric_gbps": self.fabric_gbps,
        }
