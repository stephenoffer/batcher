# Platform engineer learning path

For configuring, sizing, and operating the engine: control parallelism and memory,
set defaults from the environment, and understand how the engine bounds resources
under load.

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

`Config.from_env` overlays `BATCHER_*` variables onto a base config, which is how a
deployment injects settings without code changes.

```python
from batcher import Config

cfg = Config.from_env(
    {"BATCHER_EXECUTION_PARALLELISM": "16", "BATCHER_MEMORY_SOFT_LIMIT": "0.70"}
)
print((cfg.execution.parallelism, cfg.memory.soft_limit))
# (16, 0.7)
```

## Runnable examples

- `performance_caching.py` — cache a reused result and spill under a tiny budget.
- `spill.py` — out-of-core execution under a bounded memory budget.
- `adaptive_optimization.py` — intra-query re-optimization, result-identical.
- `distributed.py` — single-node vs. cluster, identical results (needs the `[ray]` extra).

See also [performance and memory](../user-guide/performance.md) and
[distributed fault tolerance](../architecture/fault-tolerance.md).
