# The adaptive layer

These pages describe how Batcher learns from the data it runs on, within one query and across many.

The within-query half re-optimizes at stage boundaries on measured cardinalities. That is the same mechanism and the same granularity as Spark AQE, and Batcher runs it on a single node too. It engages where it can pay for itself. On one node a query needs a join and has to clear a floor charged per stage: 5,000,000 input rows, or roughly 320 MB, for each pipeline breaker the loop would cut at. On a cluster, a plan the one-shot dispatcher can't run correctly, such as a snowflake join or an aggregate over a `limit`, is staged at any size.

The cross-query half is what neither DuckDB nor Spark has. Sketches, calibrated cost coefficients and a bandit over equivalent strategies persist between runs, so a plan improves the more a query runs, at any size. Read these pages for how both halves work and where each one stops.

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
