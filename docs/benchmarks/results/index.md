# Results by workload

This section groups Batcher's measured results by workload family. Each page names the hardware, date and script behind every table, and reports the standing on every shape it measures, losses included.

Read these pages to see what Batcher does on work that looks like yours. For the standing against one named engine, {doc}`/benchmarks/comparisons/index` arranges the same numbers that way.

The following table lists the pages:

| Page | What it measures |
|---|---|
| {doc}`Analytics and I/O <analytics>` | TPC-H, ClickBench, JSON, scans and the operator mix, against DuckDB, Polars and Daft on identical input |
| {doc}`TPC-H <tpch>` | The 22-query suite at scale factors 1 and 10, the correctness record of every engine, and the planner work behind the results |
| {doc}`The engine matrix <engine-matrix>` | Every standard suite against every engine that can run it, with each gap labeled |
| {doc}`AI and GPU workloads <ai-and-gpu>` | Ten model families on 8xT4, and inference pipelines against Ray Data and Daft |
| {doc}`Multimodal ingest <multimodal-ingest>` | Camera frames, LiDAR sweeps, audio clips and video turned into model-ready tensors |
| {doc}`Scaling out <scaling>` | How results move with the data, with cores and across a cluster |

## See also

- {doc}`/benchmarks/index`: the summary across every family.
- {doc}`/benchmarks/methodology`: the correctness gate, the hardware per family, and the reproduce commands.

```{toctree}
:hidden:

analytics
tpch
engine-matrix
ai-and-gpu
multimodal-ingest
scaling
```
