# Configuration recipes

Batcher has no built-in named profiles and no `apply_profile`. A "profile" is just a
`Config` object you build for a goal and activate with `set_config` or
`config_context`. This page collects recipes for common goals. Each one derives from
`Config()` with `dataclasses.replace` on the sections it changes, leaving every
other field at its tuned default.

```python
import dataclasses
import batcher as bt
from batcher import Config, set_config, config_context
```

## Low-latency small queries

For many small, interactive queries, the priority is low fixed overhead. Use a
smaller morsel so a tiny input is not split into one oversized batch, and keep all
cores available.

```python
base = Config()
low_latency = base.replace(
    execution=dataclasses.replace(base.execution, parallelism=0, morsel_rows=4096),
)

with config_context(low_latency):
    out = bt.from_pydict({"x": [1, 2, 3, 4]}).filter(bt.col("x") > 2).to_pydict()

print(out)
# {'x': [3, 4]}
```

## Large, spill-heavy jobs

For jobs whose intermediate state exceeds memory, spill earlier and cap the budget
to the container limit so the OS does not over-commit. A larger morsel amortizes
per-batch overhead across more rows.

```python
base = Config()
large_job = base.replace(
    execution=dataclasses.replace(base.execution, morsel_rows=65536),
    memory=dataclasses.replace(
        base.memory,
        soft_limit=0.70,
        hard_limit=0.80,
        max_memory_bytes=16 * (1 << 30),  # 16 GiB cap
    ),
)

set_config(large_job)
```

## Constrained-memory container

When running inside a container whose limit the OS does not report, set
`max_memory_bytes` explicitly so spill decisions use the real ceiling.

```python
base = Config()
container = base.replace(
    memory=dataclasses.replace(base.memory, max_memory_bytes=4 * (1 << 30)),
)
print(container.memory.max_memory_bytes)
# 4294967296
```

## Conservative optimizer

To re-optimize more aggressively when estimates miss, lower `reoptimize_error`. To
spend more planning effort on many-way joins, raise the exact-DP threshold.

```python
base = Config()
aggressive = base.replace(
    optimizer=dataclasses.replace(
        base.optimizer,
        reoptimize_error=1.5,
        join_dp_max_tables=16,
    ),
)
print((aggressive.optimizer.reoptimize_error, aggressive.optimizer.join_dp_max_tables))
# (1.5, 16)
```

## OOM-resilient (bounded memory, spill instead of crash)

To make a single-node job survive an input far larger than memory, set
`max_memory_bytes`. That bounds the engine **and** opts it into out-of-core
spilling: the Rust runtime memory pool spills stateful operators that exceed the
budget rather than letting the process get OOM-killed. Point `spill_dir` at fast
local scratch (NVMe), and optionally overflow to object storage at PB scale.

```python
base = Config()
oom_resilient = base.replace(
    memory=dataclasses.replace(
        base.memory,
        max_memory_bytes=8 * (1 << 30),  # 8 GiB cap → spill budget = 8 GiB × hard_limit
        spill_dir="/mnt/nvme/batcher-spill",
        spill_remote_uri="s3://my-bucket/spill/",  # overflow when local disk fills
    ),
)

with config_context(oom_resilient):
    # A high-cardinality aggregate that would not fit in 8 GiB spills and still
    # returns the correct result.
    out = bt.from_pydict({"k": [1, 2, 1, 3], "v": [10, 20, 30, 40]})
    out = out.group_by("k").agg(s=bt.col("v").sum()).to_pydict()

print(sorted(zip(out["k"], out["s"])))
# [(1, 40), (2, 20), (3, 40)]
```

## Fault-tolerant cluster

For a long distributed job on a cluster where nodes may be lost, raise the Ray retry
budgets and the shuffle recompute attempts, and turn on keepalive so a dropped
connection is detected quickly. Speculation backs up stragglers.

```python
base = Config()
resilient_cluster = base.replace(
    distributed=dataclasses.replace(
        base.distributed,
        task_max_retries=4,  # rerun a transiently-failed shuffle task
        actor_max_restarts=2,  # respawn a crashed compute actor
        recovery_max_attempts=5,  # more recompute→retry rounds for a flaky cluster
        flight_idle_timeout_s=120.0,  # tolerate longer GC pauses before declaring death
        flight_keepalive_s=10.0,  # detect a dropped connection within ~10 s
        speculation_max_backups=2,  # back up the slowest stragglers
    ),
)

print(resilient_cluster.distributed.task_max_retries)
# 4
```

## Reusing a recipe

A recipe is an ordinary `Config`. Define it once, then activate it process-wide with
`set_config` or per block with `config_context`. Because `Config` is immutable, the
same object can be reused freely and combined by chaining `replace` calls. See
[options](options.md) for every field you can change.
