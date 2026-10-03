# Making it fast

These pages cover the levers that change how long a correct query takes, and the tools that tell you which lever to pull.

Most queries need none of them. Batcher already pushes filters and columns into the scan, sizes its shuffle from the data volume, spills to disk instead of running out of memory, and remembers what each query measured so its next plan starts from facts.

```python
import batcher as bt

orders = bt.from_pydict({"region": ["eu", "us", "eu"], "amount": [10, 20, 30]})
run = orders.group_by("region").agg(total=bt.col("amount").sum()).stats()
print(run.rows_out, run.spilled, run.bottleneck.kind)
# 2 False aggregate
```

The common levers are one line each: cache a subtree several consumers share, or cap the memory a query may use so it spills instead of failing.

```python
from batcher.config import MemoryConfig, active_config, config_context

paid = orders.filter(bt.col("amount") > 5).cache()
print(paid.count(), paid.agg(total=bt.col("amount").sum()).to_pydict())
# 3 {'total': [60]}

capped = active_config().replace(memory=MemoryConfig(max_memory_bytes=256 * 1024 * 1024))
with config_context(capped):
    print(orders.sort("amount", descending=True).to_pydict())
# {'region': ['eu', 'us', 'eu'], 'amount': [30, 20, 10]}
```

## Where to start

Read the plan before you tune anything. The operator you would have guessed at is usually not the one costing the time, and `explain(analyze=True)` names the one that is. Then work outward from what it shows:

| If the plan shows | Read |
|---|---|
| Nothing unexpected, and you want the general levers | {doc}`Performance and memory <performance>` |
| The same expensive subtree running for several consumers | {doc}`Caching results <caching>` |
| A filter that stayed above the join, or a scan with no `pushed[...]` note | {doc}`Filter and column pushdown <pushdown>` |
| Time spent planning a huge table before any row moves | {doc}`Reading a very large table <large-tables>` |
| One key carrying most of the rows, or a job that dies inside its budget | {doc}`Skewed keys and hostile data shapes <skew>` |
| A cluster scan bound by object-store latency | {doc}`Object storage and worker locality <object-storage>` |
| A large reducing query and a GPU on the cluster | {doc}`Running a query on the GPU <gpu>` |

{doc}`Reading query plans <explain-plans>` teaches the output itself, line by line, and {doc}`Best practices <best-practices>` collects the habits that keep a pipeline in the engine's fast path from the start.

## What stays the same while you tune

Every lever in this section changes *how* a query runs, never *what* it returns. Caching, morsel size, spilling, bucket counts, and the GPU backend are all result-invariant, so it's safe to turn a knob and measure.

## See also

- {doc}`/user-guide/operate/running/index`: keeping a job healthy once it is fast enough.
- {doc}`/benchmarks/index`: what these levers measure out at against DuckDB, Polars, and Daft.
- {doc}`/configuration/options`: the `Config` settings named in this section, with their defaults.

```{toctree}
:hidden:

performance
caching
explain-plans
best-practices
large-tables
skew
pushdown
object-storage
gpu
```
