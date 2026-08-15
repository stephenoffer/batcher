"""The machine Kyber is planning *for* — the binding node, and the fleet it sits in.

`HardwareProfile` is the neutral contract carrying real hardware figures into the optimizer, so
a threshold sized to cache, memory, or VRAM tracks the device instead of a constant tuned on
somebody's laptop. Its fields describe the **binding** node — the smallest VRAM, the weakest
worker — because a plan sized against the weakest node is valid on every node it might land on.

`cluster` carries the other half: the fleet's *shape*, which the binding node cannot express.
Thirty-two devices on four nodes and thirty-two devices on thirty-two nodes are the same
binding node and a different cluster, and a shuffle across them differs by a factor of eight.
See `plan.resource.cluster`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from batcher.plan.resource.cluster import ClusterShape, NodeShape

__all__ = ["HardwareProfile"]


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    """The hardware Kyber is planning *for* — detected, never assumed.

    Kyber otherwise plans against fixed constants tuned on one machine (a 4 MiB broadcast
    threshold, a 12 GB GPU, `target_rows_per_task` blind to core count), so the same plan is
    produced on a 4-core laptop and a 128-core server and is wrong on both. This is the neutral
    contract that carries the real numbers into the optimizer: the conductor resolves it once —
    from this machine single-node, from the cluster's topology when distributed — and threads it
    through `OptimizerContext`. It lives in `plan` so Kyber can read it and `api`/`dist` can
    populate it without any layer importing another.

    On a **heterogeneous cluster** the fields describe the *binding* node for each resource, not
    an average: `gpu_memory_bytes` is the **smallest** GPU (a working set sized to the largest
    would OOM every other one), and `memory_bytes` the representative worker. Sizing to the
    weakest node is what keeps a plan valid on every node it might land on.

    All fields default to `0` meaning "unknown", so a partial profile (a CPU-only driver that
    cannot see remote GPUs) degrades to the caller's own default rather than to a wrong number.

    * `cpu_cores`         — usable cores per worker (cgroup-quota aware, not host count).
    * `memory_bytes`      — usable RAM per worker (host RAM ∧ cgroup limit).
    * `l3_cache_bytes`    — last-level cache per cache domain; the broadcast-residency bound.
    * `gpu_count`         — GPU **devices** reachable by the plan (`0` on a CPU-only host):
                            this machine's devices single-node, the cluster's device total
                            distributed. Devices, never GPU-bearing *nodes* — it is consumed
                            as a multiplier for the whole-fleet VRAM budget
                            (`one_gpu_bytes * gpu_count`), which counting nodes would
                            under-state eightfold on an 8-GPU box.
    * `gpu_memory_bytes`  — usable VRAM of the *smallest* visible GPU; `0` when unknown.
    * `worker_count`      — workers the plan will run across (`1` single-node); lets Kyber
                            reason about total vs per-node budgets on a cluster.
    * `accelerator_type`  — the *model* of the binding GPU (`"NVIDIA_H100"`), `""` when
                            unknown or when the fleet is mixed. VRAM alone cannot answer
                            "how fast is the host link", "can this device be partitioned",
                            or "what does it draw" — every one of which changes a plan, and
                            none of which is derivable from a byte count. `""` is the
                            pre-existing behavior: every model-specific decision then reports
                            no opinion and the plan is sized exactly as it was before.
    * `fingerprint`       — the hardware-scoping key of the machine class the plan will run
                            on, or `""` when the fleet is mixed or unprobeable. This is what
                            lets a learned quantity measured in *machine units* be read back
                            for the machine it will be spent on: Kyber runs on the driver,
                            which on a cluster executes none of the work, so a cost
                            coefficient or a CPU share looked up under the local key describes
                            the wrong machine. `""` means "no single honest answer", and every
                            consumer then falls back to the local key, which is the behavior
                            that held before this field existed. The rule matches
                            `accelerator_type` above, for the same reason.
    * `storage_class`     — device class backing the spill volume of the node that will spill
                            (`nvme`, `ssd`, `rotational`, `network`, ...), `""` when unknown.
                            The spread across those classes is roughly thirtyfold and it runs
                            in the direction that decides whether an out-of-core plan is
                            acceptable, so a spilled byte priced against the *driver's* disk
                            is the same class of error as one priced against its memory. The
                            **worst** class across the fleet, for the reason every field here
                            is the binding one: a plan that is affordable on the slowest
                            volume it might land on is affordable everywhere.
    * `cluster`           — the fleet's *shape* (`ClusterShape`): one record per node with its
                            devices, coherent fabric width, rack, and egress. The binding-node
                            fields above cannot express it, and it is what decides whether an
                            exchange crosses a network or a NVLink. Empty by default, which
                            reports the flat single-tier answers the engine used before it
                            existed, so nothing about an unreadable topology moves.
    """

    cpu_cores: int = 0
    memory_bytes: int = 0
    l3_cache_bytes: int = 0
    gpu_count: int = 0
    gpu_memory_bytes: int = 0
    worker_count: int = 1
    accelerator_type: str = ""
    fingerprint: str = ""
    storage_class: str = ""
    cluster: ClusterShape = field(default_factory=ClusterShape)

    @property
    def gpus_per_node(self) -> int:
        """Devices on the densest accelerator node, or the whole device count when unknown.

        The figure that separates "eight devices on one host" from "eight hosts with one
        device", which `gpu_count` alone cannot. Falls back to `gpu_count` on an unreadable
        topology, which is the single-node truth and the assumption every caller made before.
        """
        dense = self.cluster.max_gpus_per_node
        return dense if dense > 0 else max(0, self.gpu_count)

    @property
    def gpu_node_count(self) -> int:
        """Accelerator-bearing nodes, `1` when the fleet has devices but no readable shape."""
        nodes = len(self.cluster.gpu_nodes)
        if nodes:
            return nodes
        return 1 if self.gpu_count > 0 else 0

    @property
    def nvlink_domain(self) -> int:
        """Widest coherent device group in the fleet, `0` when there are no devices.

        The hard ceiling on a fan-out that stays on the fast path: a collective wider than this
        leaves NVLink for PCIe or the network, and the drop is more than an order of magnitude.
        Falls back to `gpus_per_node` when no domain was reported, matching the pre-existing
        assumption that every device on a host is local to every other.
        """
        domain = self.cluster.largest_nvlink_domain
        return domain if domain > 0 else self.gpus_per_node

    @classmethod
    def local(cls) -> HardwareProfile:
        """Detect the profile of *this* machine — the single-node and driver default.

        Reads the neutral hardware layer only, so it is safe to call from anywhere and needs
        no cluster. A distributed run replaces this with a cluster-derived profile via
        [`for_cluster`]; every field a probe cannot answer stays `0` ("unknown").
        """
        from batcher._internal.device_specs import resolve_device_name
        from batcher._internal.hardware import (
            available_cpu_count,
            fingerprint,
            gpu_inventory,
            l3_cache_bytes,
            machine_memory_bytes,
        )

        gpus = gpu_inventory()
        vram = min((int(g.get("memory_bytes") or 0) for g in gpus), default=0)
        # The device *model*, resolved from whatever the driver called it. Only when every
        # local device is the same model: a mixed host has no single binding model, and
        # naming one of them would attach one device's power and host link to another's plan.
        names = {resolve_device_name(str(g.get("name") or "")) or "" for g in gpus}
        model = names.pop() if len(names) == 1 else ""
        cores = available_cpu_count()
        memory = machine_memory_bytes()
        # One node, described honestly. A single-node profile whose shape says "one host with
        # these devices" is what lets the same locality arithmetic serve both deployments: a
        # fan-out across four local devices is intra-node here and cross-node on a fleet of
        # four one-device workers, and before this both read as an anonymous `workers=4`.
        shape = ClusterShape(
            nodes=(
                NodeShape(
                    cpu_cores=cores,
                    memory_bytes=memory,
                    gpus=len(gpus),
                    accelerator_type=model,
                    gpu_memory_bytes=vram,
                    nvlink_domain=_local_nvlink_domain(model, len(gpus)),
                ),
            )
        )
        return cls(
            cpu_cores=cores,
            memory_bytes=memory,
            l3_cache_bytes=l3_cache_bytes(),
            gpu_count=len(gpus),
            gpu_memory_bytes=vram,
            worker_count=1,
            accelerator_type=model,
            # Single-node: the machine that will run the plan is the one asking, so the local
            # key is the honest answer rather than a fallback.
            fingerprint=fingerprint(),
            storage_class=_local_storage_class(),
            cluster=shape,
        )

    @classmethod
    def for_cluster(
        cls,
        *,
        cpu_cores: int,
        memory_bytes: int,
        worker_count: int,
        gpu_count: int = 0,
        gpu_memory_bytes: int = 0,
        l3_cache_bytes: int = 0,
        accelerator_type: str = "",
        fingerprint: str = "",
        storage_class: str = "",
        cluster: ClusterShape | None = None,
    ) -> HardwareProfile:
        """A profile for a distributed run, built by the conductor from live cluster topology.

        The caller passes the *binding* node's figures (smallest GPU VRAM, representative
        worker RAM/cores) so a plan sized against this profile is valid on every node it may
        land on. `l3_cache_bytes` is the binding worker's cache when the caller could probe the
        workers for it (Ray's topology omits cache), and `0` when it couldn't — which keeps a
        cache-sized threshold at its default rather than guessing from the driver's machine.

        `accelerator_type` is the model every GPU node shares, or `""` on a mixed fleet — the
        same rule `cluster_accelerator_type()` follows, because there is no honest single
        answer when the models differ. `fingerprint` follows that rule exactly: the
        hardware-scoping key every probed worker shares, `""` when they differ or none could be
        probed. A caller reading a machine-scoped learned value then gets the workers' key on a
        uniform fleet and falls back to its own on a mixed one, which is the pre-existing
        behavior and errs toward the model it already had rather than toward a blend.

        `cluster` is the fleet's shape. `None` — what a caller that cannot read per-node
        topology passes, and what every existing caller passed before it existed — leaves it
        empty, and every locality-aware decision then reports the flat single-tier answer it
        always did.

        **Every binding-node field a caller leaves unknown is derived from that shape**, not
        just `accelerator_type`. Deriving one field and not the others was an inconsistency
        with a real consequence: a caller that could read per-node topology but not the
        device-model labels (an unlabelled on-prem fleet, a Ray version whose node labels are
        absent) passed `gpu_memory_bytes=0` and got a profile reporting *no VRAM* for a cluster
        whose shape recorded it node by node — so every VRAM-sized decision took its default on
        a fleet that had, in the same object, said exactly how much it had. A value the caller
        *did* supply always wins; derivation only fills a stated unknown.
        """
        shape = cluster or ClusterShape()
        model = accelerator_type
        if not model and shape.homogeneous_gpus:
            model = shape.device_models[0]
        return cls(
            # `or` rather than `max`: `0` is this contract's "unknown", so an unsupplied field
            # falls through to the shape and a supplied one is kept even where the shape
            # disagrees — the caller measured the fleet directly and the shape may be partial.
            cpu_cores=max(0, cpu_cores) or shape.binding_cpu_cores,
            memory_bytes=max(0, memory_bytes) or shape.binding_memory_bytes,
            l3_cache_bytes=max(0, l3_cache_bytes),
            gpu_count=max(0, gpu_count) or shape.total_gpus,
            gpu_memory_bytes=max(0, gpu_memory_bytes) or shape.binding_gpu_memory_bytes,
            worker_count=max(1, worker_count),
            accelerator_type=model,
            fingerprint=fingerprint,
            storage_class=storage_class,
            cluster=shape,
        )


def _local_storage_class() -> str:
    """This machine's spill-device class, or `""` when it cannot be determined.

    Resolved through `spill_scratch_dir` — the one answer to "which disk does this process
    spill to" — so the class describes where bytes actually land rather than where the process
    happens to run. Best-effort: an unreadable device reports `""`, which every consumer treats
    as "no opinion" and prices exactly as it did before.
    """
    from batcher._internal.hardware.storage import device_class
    from batcher._internal.site import spill_scratch_dir

    found = device_class(spill_scratch_dir())
    return "" if found == "unknown" else found


def _local_nvlink_domain(model: str, devices: int) -> int:
    """The coherent fabric width of `devices` local devices of `model`, `0` when unknown.

    Read from the device specification rather than probed, and capped at what the host has:
    the datasheet domain of an eight-way part is eight, and on a two-device workstation the
    domain is two. `0` for an unrecognized model, which `NodeShape.local_domain` reads as "no
    narrower than the node", the assumption that held before any of this existed.
    """
    if devices <= 0 or not model:
        return 0
    from batcher._internal.device_specs import device_nvlink_domain

    domain = device_nvlink_domain(model)
    return min(domain, devices) if domain > 0 else 0
