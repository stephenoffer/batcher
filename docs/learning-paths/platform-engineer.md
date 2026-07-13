# Platform engineer learning path

This path is for whoever runs the engine. You control parallelism and memory, inject
defaults from the environment, and learn what the engine does when a query pushes past
its budget.

## Reading order

1. [Getting started](../getting-started/index.md): install and verify the build.
2. [Installation](../getting-started/installation.md): packaging and extras.
3. [Configuration](../configuration/index.md): the `Config` model and precedence.
4. [Configuration options](../configuration/options.md): every field and default.
5. [Environment variables](../configuration/environment.md): `BATCHER_*` and
   `BATCHER_CONFIG_FILE`.
6. [Configuration recipes](../configuration/profiles.md): configs for common goals.
7. [Cloud storage](../user-guide/cloud-storage.md): object-store access.
8. [Best practices](../user-guide/best-practices.md) and
   [troubleshooting](../user-guide/troubleshooting.md).
9. [Configuration API reference](../api/configuration.md).

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

See also [performance and memory](../user-guide/performance.md) and
[distributed fault tolerance](../architecture/fault-tolerance.md).


## How the engine spends your machine

If you operate Batcher, the [deep dives](../deep-dives/index.md) are where the operational
behavior is explained: what spills and when, how the shuffle applies backpressure, and how a
plan re-tunes itself mid-query.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`database;1.1em` Spilling
:link: ../deep-dives/spilling
:link-type: doc
Staying alive when the data does not fit.
:::

:::{grid-item-card} {octicon}`broadcast;1.1em` Credit-based flow control
:link: ../deep-dives/credit-flow-control
:link-type: doc
One credit is one batch slot, and the producer blocks at zero.
:::

:::{grid-item-card} {octicon}`gear;1.1em` Adaptive re-optimization
:link: ../deep-dives/adaptive-reoptimization
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
- [Optimizing a slow query](../tutorials/optimizing-a-slow-query.md) — a real diagnosis, start to finish.
- [Explain plans](../user-guide/explain-plans.md) — reading what the optimizer decided.
- [Ray](../integrations/ray.md) — scheduling only; the data plane goes over Arrow Flight.
:::
