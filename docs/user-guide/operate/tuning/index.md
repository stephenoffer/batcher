# Making it fast

These pages cover the levers that change how long a correct query takes, and the tools that tell you which lever to pull.

Most queries need none of them. Batcher already pushes filters and columns into the scan, sizes its shuffle from the data volume, spills to disk instead of running out of memory, and remembers what each query measured so its next plan starts from facts rather than guesses. A top-N that ran once starts from the cut it learned, so its second run skips the row groups that cannot hold a winner and decodes the other columns only for the rows that survive. The pages below are for the query that still isn't fast enough, and for the shapes where knowing the engine pays off.

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

Every lever in this section changes *how* a query runs, never *what* it returns. Caching, morsel size, spilling, bucket counts, the fast path, and the GPU backend are all result-invariant, and the runnable examples for spilling and caching print the comparison that proves it. That's what makes it safe to turn a knob and measure: if the numbers move, the plan moved, not the answer.

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
