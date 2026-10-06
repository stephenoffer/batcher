# The wires between GPUs

This page describes what Batcher reads about the interconnect on a multi-GPU node, and the scheduling decisions it makes from it.

A GPU node's throughput is set as much by its wires as by its devices. NVLink against host-memory staging, or the NIC next to a device against one across the socket, changes only the time a job takes, never its result. Three facts decide it, and Batcher reads all three.

## What Batcher reads

**Which pairs of devices can exchange directly.** `nvidia-smi topo -m` shows how far apart two devices sit on the PCI bus. That is the whole answer on a node with no coherent fabric and the wrong axis on a node that has one: two devices under different root complexes are the furthest apart the bus can express, and they exchange at full fabric rate if NVLink joins them. Batcher overlays the live NVLink pairs onto the bus matrix, so `nvlink` is a class alongside `pix`, `pxb`, `phb`, `node`, and `sys`, and a group is chosen on the wire the traffic actually uses.

The connected groups of that overlay are *islands*. An eight-device server is usually one island. Two four-device boards in one chassis are two, and a collective spanning them runs every step at the slower of the two links.

**Which NIC each device leaves through.** A dense node has several NICs, and a device paired with a NIC under a different root complex crosses the inter-socket link. Batcher assigns these *rails* node-wide, so eight devices don't all name the same nearest NIC and leave seven idle. A device takes its closest NIC unless that NIC already holds its share and an equally close one is free, and balance never overrides distance, because crossing a socket to even out a rail costs more than the imbalance saves.

**What the links actually negotiated.** A card that renegotiated to half width runs at half its host bandwidth, and that figure is the term a device decision is most sensitive to, so it is read rather than assumed.

Every probe degrades to nothing. Off Linux, without the driver, or inside a container that did not mount the relevant `/sys` tree, each reports an empty or neutral answer and every decision below keeps its default behavior.

![The fabric facts Batcher reads on a multi-GPU node, and the exchange schedule read off them. Peer islands are the connected groups of the NVLink-over-bus overlay: eight devices joined as two groups of four by a coherent fabric, with only the PCI bus between the groups, are two islands of four rather than one of eight. Two devices under different root complexes are the furthest apart the bus can express and still exchange at full fabric rate if NVLink joins them, so a group picked on bus distance alone picks the wrong four. Rails are which NIC each device leaves the node through, assigned node-wide rather than per device: asked one at a time, all eight devices can name the same NIC, and then one rail carries the whole shuffle while seven sit idle with every counter reporting a healthy fabric. The exchange schedule is n - 1 rounds of n / 2 disjoint pairs, so no device is the source of one copy and the destination of another in the same round: for four devices that is 0-3 and 1-2, then 0-2 and 1-3, then 0-1 and 2-3. A reduction ring is ordered by the fabric rather than by device index, because its rate is its worst hop. Three decisions follow: a collective is strict-packed inside one island and a plan covering fewer devices than the stage asked for is refused, because a partial gang hangs; shards are dealt by measured throughput under largest-remainder apportionment, with an unmeasured device treated as average and never as idle; and the device path is used only when it wins by 1.25x, so an unpriced link makes a plan refuse and a plan that merely ties loses.](/_static/diagrams/gpu_fabric_topology.svg)

## What changes because of it

Five scheduling decisions read those facts. None of them changes a result, and each one falls back to its earlier behavior when the facts are missing.

### The collective library is told, not left to guess

A multi-GPU stage that runs its own collective discovers the node's fabric by probing at initialization. Batcher has already measured it, so it hands over the answers in the GPU task's environment: the rail-aligned NIC list, the interfaces that actually carry the fabric, the real device-to-NIC distance for the GPUDirect threshold, and whether peer-to-peer can help on this node at all.

Two rules keep that safe. Nothing is set that a probe did not answer, so an unreadable node gets an empty block and the library probes exactly as before. And a variable already set in the environment is never replaced, because a deployment that pins its own NIC selection has a reason no probe can see.

### A collective is placed inside one fabric

A stage flagged as running its own collective is gang-scheduled with `STRICT_PACK`, so its workers are co-located. Co-location alone does not make a node wide enough. The bundle layout therefore comes from the fleet's topology: a node whose coherent domain already holds the whole world size is preferred, and the largest domain is filled first when none does. The planner, `plan_collective`, can also skip a node excluded by a data-residency rule or a power-zone budget before placement rather than after, but only when its caller names the stage's datasets and a zone budget. The scheduler passes neither, so on the live path neither filter removes a node.

A plan that covers fewer devices than the stage asked for is not used, because a partial gang would wait on a world size it never receives.

### Shards are dealt by what each device measured

Round-robin is right for a uniform fleet. For a mixed one, Batcher records each GPU run's rows per second per device model and deals shards in proportion, using largest-remainder apportionment so the counts sum exactly and no device is starved.

A device with no measurement is treated as average rather than as idle.

### A redistribution is scheduled, not serialized

When devices exchange with each other, transfers are paired so no device is the source or destination of two copies at once: an all-to-all over `n` devices runs as `n - 1` rounds of `n / 2` disjoint pairs, each at link rate. A reduction ring is ordered by the fabric rather than by device index, so it walks NVLink where NVLink exists.

The device path is used only when it predicts a clear gain over the host path. A plan whose links could not be priced is refused rather than assumed favorable, and a plan that merely ties loses, because the host path already moves those bytes correctly.

### The crossover is learned per workload, not just per device

Batcher learns where the GPU starts beating the CPU engine from measured runs. Two pipelines on one device have different crossovers: a wide projection is transfer-bound and a narrow group-by is not. The threshold is keyed by query shape as well as device model, with both lines of a crossover always taken from the same key, and a shape seen for the first time keeps exactly the threshold it had.

## What you can see

The accelerator report carries the rail layout and the peer topology beside the device rows:

```python
import batcher as bt

report = bt.accelerators()
fabric = report.get("fabric", {})
print(sorted(fabric.get("rails", {}).get("assignment", {})) or "no rails on this host")
print(fabric.get("peers", {}).get("largest_island", 0))
print(bt.accelerator_problems())  # [] when nothing is costing throughput
```

On a CPU-only host this prints `no rails on this host`, `0` and `[]`. Two conditions are called out in {py:func}`bt.accelerator_problems() <batcher.accelerator_problems>` because they cost throughput without costing correctness:

- devices unevenly spread over the rails, so a cross-node stage uses part of the port rate;
- no device pair able to copy directly, so every exchange stages through host memory.

A shuffle's own statistics carry the same measurement while it runs. Alongside the node-wide observed and capable fabric rates, `ShuffleSession.stats()` reports the busiest rail, how many rails carried nothing, and the spread between them.

## Practical limits

- **`/sys` access.** A container without the PCI tree, the InfiniBand tree, or NVML mounted reports an unreadable topology, and every decision here keeps its default behavior.
- **Control plane only.** Batcher places work, sizes it, and configures the collective library. The device-to-device copies themselves are carried out by the framework doing them, following the schedule in `carbonite/transfer/device_exchange.py`; the Arrow contract at every operator boundary is unchanged.
- **Nameplate or measured figures.** An unrecognized device model contributes no bandwidth figure, and an unpriced link makes a plan refuse.
- **NVIDIA fabrics.** AMD's XGMI fabric isn't read, so an Instinct node reports its bus topology and no coherent fabric.

## See also

- {doc}`GPU execution </architecture/deep-dives/distribution/gpu-execution>`: the two paths that run work on a device.
- {doc}`Shuffle over Arrow Flight </architecture/deep-dives/distribution/shuffle-flight>`: why bulk data bypasses the Ray object store.
- {doc}`Credit-based flow control </architecture/deep-dives/distribution/credit-flow-control>`: how a fast producer is kept from burying a slow consumer.
- {doc}`Hardware awareness </architecture/deep-dives/adaptive/hardware-awareness>`: what the optimizer knows about the machine it plans for.
- {doc}`GPU guide </ml/inference/gpu>`: the device knobs, from a user's side.
