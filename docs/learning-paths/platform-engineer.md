# Platform engineer learning path

This path is for whoever runs the engine. You control parallelism and memory, inject
defaults from the environment, and learn what the engine does when a query pushes past
its budget.

## Reading order

1. {doc}`Getting started <../getting-started/index>`: install and verify the build.
1. {doc}`Installation <../getting-started/installation>`: packaging and extras.
1. {doc}`Configuration <../configuration/index>`: the `Config` model and precedence.
1. {doc}`Configuration options <../configuration/options>`: every field and default.
1. {doc}`Environment variables <../configuration/environment>`: `BATCHER_*` and
   `BATCHER_CONFIG_FILE`.
1. {doc}`Configuration recipes <../configuration/profiles>`: configs for common goals.
1. {doc}`Cloud storage </user-guide/moving-data/cloud-storage>`: object-store access.
1. {doc}`Best practices </user-guide/operate/best-practices>` and
   {doc}`troubleshooting </user-guide/operate/troubleshooting>`.
1. {doc}`Configuration API reference </api/operations/configuration>`.

## Example: set process-wide defaults

```python
import dataclasses
import batcher as bt
from batcher import Config, set_config

base = Config()
cfg = base.replace(
    execution=dataclasses.replace(base.execution, parallelism=8),
    memory=dataclasses.replace(base.memory, soft_limit=0.75, hard_limit=0.85),
)
set_config(cfg)

out = bt.from_pydict({"x": [1, 2, 3]}).filter(bt.col("x") >= 2).to_pydict()
print(out)
# {'x': [2, 3]}
```

## Example: defaults from the environment

`Config.from_env` overlays `BATCHER_*` variables onto a base config. That is how a
deployment injects settings without touching code.

```python
from batcher import Config

cfg = Config.from_env(
    {"BATCHER_EXECUTION_PARALLELISM": "16", "BATCHER_MEMORY_SOFT_LIMIT": "0.70"}
)
print((cfg.execution.parallelism, cfg.memory.soft_limit))
# (16, 0.7)
```

## Runnable examples

- `performance_caching.py` caches a reused result, then spills under a tiny budget.
- `spill.py` runs out-of-core under a bounded memory budget.
- `adaptive_optimization.py` shows intra-query re-optimization producing an identical
  result.
- `distributed.py` compares a single node against a cluster and gets identical results.
  It needs the `[ray]` extra.

See also {doc}`performance and memory </user-guide/operate/performance>` and
{doc}`distributed fault tolerance <../architecture/fault-tolerance>`.


## How the engine spends your machine

If you operate Batcher, the {doc}`deep dives <../deep-dives/index>` are where the operational
behavior is explained: what spills and when, how the shuffle applies backpressure, and how a
plan re-tunes itself mid-query.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`database;1.1em` Spilling
:link: /deep-dives/memory/spilling
:link-type: doc
Staying alive when the data does not fit.
:::

:::{grid-item-card} {octicon}`broadcast;1.1em` Credit-based flow control
:link: /deep-dives/distribution/credit-flow-control
:link-type: doc
One credit is one batch slot, and the producer blocks at zero.
:::

:::{grid-item-card} {octicon}`gear;1.1em` Adaptive re-optimization
:link: /deep-dives/adaptive/adaptive-reoptimization
:link-type: doc
Re-planning on measured cardinalities, not estimates.
:::

:::{grid-item-card} {octicon}`meter;1.1em` Scaling benchmarks
:link: ../benchmarks/scaling
:link-type: doc
What actually happens when you add nodes.
:::
::::

:::{seealso}
- {doc}`Optimizing a slow query <../tutorials/optimizing-a-slow-query>`: a real diagnosis, start to finish.
- {doc}`Explain plans </user-guide/operate/explain-plans>`: reading what the optimizer decided.
- {doc}`Ray </integrations/compute/ray>`: scheduling only. The data plane goes over Arrow Flight.
:::


## See also

- {doc}`data-engineer`: the pipelines you are operating.
- {doc}`/user-guide/trust/hardening`: the trust boundaries to establish before a shared deployment.
- {doc}`../internals/index`: how the engine works, when an incident needs it.
