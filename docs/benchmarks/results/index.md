# Results by workload

This section holds the measured results, one page per workload family. Each page carries the
hardware it ran on, the correctness gate it passed, and the shapes where Batcher loses.

Read these when you want to know what Batcher does on work that looks like yours. Read
{doc}`/benchmarks/comparisons/index` instead when you want the standing against one specific
engine.

| Page | What it measures |
|---|---|
| {doc}`Analytics and I/O <analytics>` | Relational operators, TPC-H, ClickBench, and the connectors, against DuckDB and Polars on identical input |
| {doc}`TPC-H <tpch>` | The full 22-query suite at scale factors 1 and 10, per query, against three engines and two DuckDB configurations |
| {doc}`AI and GPU workloads <ai-and-gpu>` | Ten families on 8xT4 with real models: inference, embeddings, decode, and training ingest |
| {doc}`Multimodal ingest <multimodal-ingest>` | Turning camera frames, LiDAR sweeps, and audio clips into model-ready tensors |
| {doc}`Scaling out <scaling>` | What distribution buys, what it costs at small sizes, and how much of each suite runs distributed at all |

## See also

- {doc}`/benchmarks/methodology`: the hardware per family and the commands that reproduce each number.
- {doc}`/benchmarks/index`: the one-table summary across every family here.

```{toctree}
:hidden:

analytics
tpch
ai-and-gpu
multimodal-ingest
scaling
```
