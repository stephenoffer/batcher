# Distribution

Batcher scales out without becoming a second engine. The operator that runs on one core is the operator that runs on a cluster, so distribution is purely a question of where work runs, how many pieces it runs in, and how the bytes move between them. These pages cover each of those: the Arrow Flight shuffle that keeps bulk data out of the Ray object store, the credits that bound its memory, the scheduler that sizes and places the work, and the GPU paths that keep devices fed.

One equality holds the section together. A distributed result equals the single-node result, with three stated exceptions, and the diagram shows both in one view.

![What the single-node equals distributed equality promises, and its three stated exceptions. One operator serves three deployments, one core through bc-interp::execute, many cores through bc-interp::par and many machines through bc-interp::dist, because partial, combine and finalize are written once and combine is associative and commutative, so arrival order cannot change which rows come back. Three things are therefore exact however many partitions run: the multiset of rows, every column name, and every column type, with no tolerance and no qualification. Three places are carved out, and all three are places where the query itself fixes no answer. A float reduction is identical up to reassociation, because IEEE addition is not associative; Neumaier compensation and Chan's formula bound the error to near the last bits and nothing removes it while the partition count is free, and the row count and every type stay exact. A row_number over tied rows may break the tie differently, because SQL leaves it undefined and the two paths order rows physically by different routes, while the partitions and the rank set 1 to n stay exact and rank and dense_rank are unaffected. A LIMIT over an unordered relation, such as a group_by, may keep different rows, because a hash walk is not a property of the query, while the count stays n and every returned row is a row of the full result; sort(...).limit(n) does not diverge. A divergence that is not one of these three is a defect, however plausible its rows look.](/_static/diagrams/single_node_equals_distributed.svg)

Start with the page that answers your question:

- {doc}`Shuffle over Arrow Flight </architecture/deep-dives/distribution/shuffle-flight>`: why bulk data bypasses the Ray object store.
- {doc}`Credit-based flow control </architecture/deep-dives/distribution/credit-flow-control>`: one credit is one batch slot, and the producer blocks at zero.
- {doc}`Distributed scheduling </architecture/deep-dives/distribution/distributed-scheduling>`: where work runs, how many pieces it runs in, and what does and doesn't travel through Ray.
- {doc}`Planning on the layout a table already has </architecture/deep-dives/distribution/partition-aware-planning>`: when a partitioned table has already done the shuffle's work.
- {doc}`GPU execution </architecture/deep-dives/distribution/gpu-execution>`: the two paths that run work on a device, and the scheduling that keeps it busy.
- {doc}`The wires between GPUs </architecture/deep-dives/distribution/gpu-fabric>`: rails, NVLink islands, and the placement and exchange decisions read off them.

```{toctree}
:hidden:

shuffle-flight
credit-flow-control
distributed-scheduling
partition-aware-planning
gpu-execution
gpu-fabric
```
