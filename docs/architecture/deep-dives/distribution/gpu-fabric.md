# The wires between GPUs

This page describes what Batcher reads about the interconnect on a multi-GPU node, and the scheduling decisions it makes from it.

A GPU node's throughput is set as much by its wires as by its devices, and the wires are the part nothing reports. A device pair that exchanges over NVLink and one that stages through host memory both return correct results. A shuffle that leaves through the NIC next to a device and one that crosses the socket first both finish. The difference shows up only as time, so it is invisible to every check that asks whether the job worked.

Three facts decide it, and Batcher reads all three.

## What Batcher reads

**Which pairs of devices can exchange directly.** `nvidia-smi topo -m` shows how far apart two devices sit on the PCI bus. That is the whole answer on a node with no coherent fabric and the wrong axis on a node that has one: two devices under different root complexes are the furthest apart the bus can express, and they exchange at full fabric rate if NVLink joins them. Batcher overlays the live NVLink pairs onto the bus matrix, so `nvlink` is a class alongside `pix`, `pxb`, `phb`, `node`, and `sys`, and a group is chosen on the wire the traffic actually uses.

The connected groups of that overlay are *islands*. An eight-device server is usually one island. Two four-device boards in one chassis are two, and a collective spanning them runs every step at the slower of the two links.

**Which NIC each device leaves through.** A dense node has several NICs, and a transfer between a device and a NIC under a different root complex crosses the inter-socket link twice for a transfer whose whole purpose is to leave the node. Asking each device for its nearest NIC gives the right answer per device and the wrong one per node: eight devices asked independently can all name the same NIC, and then one rail carries the whole shuffle while seven sit idle. Batcher assigns rails node-wide instead. A device takes its closest NIC unless that NIC already holds its share and an equally close one is free, and balance never overrides distance, because crossing a socket to even out a rail costs more than the imbalance saves.

**What the links actually negotiated.** A card that renegotiated to half width enumerates, runs, and returns correct results at half its host bandwidth. That figure is the term a device decision is most sensitive to, so it is read rather than assumed.

Every probe degrades to nothing. Off Linux, without the driver, or inside a container that did not mount the relevant `/sys` tree, each reports an empty or neutral answer and every decision below keeps the behavior it had before the probe existed.

## What changes because of it

### The collective library is told, not left to guess

A multi-GPU stage that runs its own collective discovers the node's fabric by probing at initialization. Batcher has already measured it, so it hands over the answers in the GPU task's environment: the rail-aligned NIC list, the interfaces that actually carry the fabric, the real device-to-NIC distance for the GPUDirect threshold, and whether peer-to-peer can help on this node at all.

Two rules keep that safe. Nothing is set that a probe did not answer, so an unreadable node gets an empty block and the library probes exactly as before. And a variable already set in the environment is never replaced, because a deployment that pins its own NIC selection has a reason no probe can see.

### A collective is placed inside one fabric

A stage flagged as running its own collective is gang-scheduled with `STRICT_PACK`, so its workers are co-located. Co-location alone does not make a node wide enough, so the bundle layout comes from the fleet's topology: a node whose coherent domain already holds the whole world size is preferred, the largest domain is filled first when none does, and a node excluded by a data-residency rule or a power-zone budget is skipped before placement rather than after.

A plan that covers fewer devices than the stage asked for is not used. Reserving the partial gang succeeds and then hangs the stage on a world size it never receives, which is a worse failure than the pending request Ray reports on its own.

### Shards are dealt by what each device measured

Round-robin is right for a uniform fleet and wrong for every other kind. A node with one device twice as fast as its neighbour finishes half the work early and waits, so the stage runs at the slow device's rate with half the fleet idle. Batcher records each GPU run's rows per second per device model and deals shards in proportion, using largest-remainder apportionment so the counts sum exactly and no device is starved.

A device with no measurement is treated as average rather than as idle. Giving an unmeasured device nothing guarantees it stays unmeasured.

### A redistribution is scheduled, not serialized

When devices have to exchange with each other, the naive order copies one pair at a time through host memory. Two things fix it, and both are scheduling rather than semantics.

Pair the transfers so no device is the source or the destination of two copies at once. An all-to-all over `n` devices decomposes into `n - 1` rounds of `n / 2` disjoint pairs, and every round then runs at link rate instead of contending. Order the reduction ring by the fabric rather than by device index, so a ring walks NVLink where NVLink exists. An index-ordered ring on a two-board node crosses the bus twice per revolution for no reason but the numbering.

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
```

Two conditions are called out in {py:func}`bt.accelerator_problems() <batcher.accelerator_problems>` because they cost throughput without costing correctness, which is the class of fault a job's own timings never reveal:

- devices unevenly spread over the rails, so a cross-node stage uses part of the port rate;
- no device pair able to copy directly, so every exchange stages through host memory.

A shuffle's own statistics carry the same measurement while it runs. Alongside the node-wide observed and capable fabric rates, `ShuffleSession.stats()` reports the busiest rail, how many rails carried nothing, and the spread between them. The summed figure cannot tell a shuffle that used an eighth of the fabric from one that used one rail of eight at capacity, and those two have opposite fixes.

## Requirements and limitations

- **The probes need the host's `/sys` tree.** A container without the PCI tree, the InfiniBand tree, or NVML mounted reports an unreadable topology, and every decision here falls back to the behavior it had before. Nothing fails; the fleet is simply scheduled blind.
- **These are control-plane decisions.** Batcher places work, sizes it, and configures the collective library. It does not perform device-to-device copies itself: the Arrow contract at every operator boundary is unchanged, and the exchange schedule is a plan the framework doing the copying carries out.
- **The figures are nameplate or measured, never inferred.** A device model Batcher does not recognize contributes no bandwidth figure rather than a guessed one, and an unpriced link makes a plan refuse rather than proceed optimistically.
- **AMD's XGMI fabric is not read.** The sysfs names have moved between kernel releases, and a fabric figure that is wrong is worse than one that is absent, so an Instinct node reports its bus topology and no coherent fabric.

## See also

- {doc}`GPU execution </architecture/deep-dives/distribution/gpu-execution>`: the two paths that run work on a device.
- {doc}`Shuffle over Arrow Flight </architecture/deep-dives/distribution/shuffle-flight>`: why bulk data bypasses the Ray object store.
- {doc}`Credit-based flow control </architecture/deep-dives/distribution/credit-flow-control>`: how a fast producer is kept from burying a slow consumer.
