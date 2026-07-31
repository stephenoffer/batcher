# Distribution

Scaling out is a scheduling concern, not a second engine. These pages cover what moves, what
schedules it, and what keeps a fast producer from burying a slow consumer.

- {doc}`Shuffle over Arrow Flight </deep-dives/distribution/shuffle-flight>`: why bulk data bypasses the Ray object store.
- {doc}`Credit-based flow control </deep-dives/distribution/credit-flow-control>`: one credit is one batch slot, and the producer blocks at zero.
- {doc}`Distributed scheduling </deep-dives/distribution/distributed-scheduling>`: where work runs, how many pieces it runs in, and what does and doesn't travel through Ray.
- {doc}`GPU execution </deep-dives/distribution/gpu-execution>`: the two paths that run work on a device, and the scheduling that keeps it busy.
- {doc}`The wires between GPUs </deep-dives/distribution/gpu-fabric>`: rails, NVLink islands, and the placement and exchange decisions read off them.

```{toctree}
:hidden:

shuffle-flight
credit-flow-control
distributed-scheduling
gpu-execution
gpu-fabric
```
