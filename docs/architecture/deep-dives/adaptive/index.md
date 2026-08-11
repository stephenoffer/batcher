# The adaptive layer

Batcher re-optimizes at stage boundaries on measured cardinalities, the same mechanism and the
same granularity as Spark AQE, but available single-node too. It is also off for queries under
20M input rows, so most queries never reach it. What neither DuckDB nor Spark has is the second
half: a sketch-backed *cross-query* learned-stats and bandit loop, so a plan improves the more
a query runs. Read these pages for how both halves work and where each one stops.

- {doc}`Adaptive re-optimization </architecture/deep-dives/adaptive/adaptive-reoptimization>`: re-planning mid-query on measured cardinalities.
- {doc}`Cardinality estimation </architecture/deep-dives/adaptive/cardinality-estimation>`: how many rows a subtree will produce, how wrong that guess is, and how the engine tracks which.
- {doc}`The cost model </architecture/deep-dives/adaptive/cost-model>`: turning row counts into the one comparable number that ranks two plans.
- {doc}`Learned metadata </architecture/deep-dives/adaptive/learned-metadata>`: Core measures, Kyber consumes, and the plan improves the more a query runs.
- {doc}`Hardware awareness </architecture/deep-dives/adaptive/hardware-awareness>`: what the optimizer knows about the machine, which parts of it are measured, and which are not seen at all.

```{toctree}
:hidden:

adaptive-reoptimization
cardinality-estimation
cost-model
learned-metadata
hardware-awareness
```
