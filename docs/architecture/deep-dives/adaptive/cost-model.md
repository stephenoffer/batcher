# The cost model

This page describes how Kyber turns row counts into a cost it can compare, how the coefficients are calibrated from measured runs, and how hard it searches for a join order.

Every cost-based decision in Kyber, such as join order, build side, broadcast versus shuffle, or whether to split a filter, reduces to comparing two numbers. The cost model produces them. It doesn't predict milliseconds. It produces an abstract scalar, and the only property that matters is that a cheaper plan really is faster.

The weights and per-row coefficients are ordinary config:

```python
import batcher as bt

opt = bt.active_config().optimizer
print(opt.cost_weights)  # CostWeights(cpu=1.0, io=1.0, net=2.0)
c = opt.cost_coeffs
print(c.hash_build_row, c.hash_probe_row, c.jit_speedup)  # 2.0 1.0 4.0
```

## Four axes, three in the scalar

```python
# docs: skip
# python/batcher/kyber/cost/model.py
@dataclass(frozen=True)
class Cost:
    cpu: float
    mem: float
    io: float
    net: float

    def total(self, w) -> float:
        return w.cpu * self.cpu + w.io * self.io + w.net * self.net
```

:::{important}
`mem` is a *peak*, not a throughput cost, so it isn't in the scalar. As costs compose up the tree, `cpu`, `io`, and `net` **sum** over children while `mem` takes the **max**, the tallest single operator.
:::

That max is a floor on the real peak, and it isn't what admission reads. A hash join's build table stays resident while its probe subtree runs, so Carbonite walks the tree as a schedule instead (`carbonite/memory/estimator.py::_peak`):

```text
peak(join)  = max(peak(build), resident(join) + peak(probe))
peak(unary) = max(peak(input), resident(node))
```

On a four-way bushy join with hash tables of 18.2, 9.1 and 9.1 MB, the tallest operator is 18.2 MB and the concurrent peak is 27.4 MB. The {doc}`buffer pool </architecture/deep-dives/memory/buffer-pool>` page covers the envelope that figure is admitted against.

![What the cost model consumes and what it emits. Four inputs feed one fold over the plan tree in CostModel.cost(node): estimated rows per node from the estimator, a type-exact row width rather than a flat 64 bytes, machine terms such as L3 cache size, memory budget and spill device, and coefficients that ship as constants and are then calibrated from measured runs. It emits four axes, three of which enter the scalar. cpu, io and net combine as 1.0 times cpu plus 1.0 times io plus 2.0 times net, and that one comparable number ranks the alternatives: join order, join strategy, whether to spill. mem is the tallest single operator's state, a max along the tree and never summed. It is a floor on the peak rather than the peak, it ranks nothing, and Carbonite admits a plan on its own walk of the concurrent peak instead.](/_static/diagrams/cost_model_inputs.svg)

## Per-operator formulas

Coefficients are in `optimizer.cost_coeffs`, in abstract work-units per row:

| Coefficient | Default | Coefficient | Default |
|---|---:|---|---:|
| `scan_row` | 1.0 | `sort_row` | 1.0 |
| `filter_row` | 0.5 | `distinct_row` | 2.0 |
| `project_row` | 0.3 | `union_row` | 0.2 |
| `hash_build_row` | 2.0 | `map_row` | 5.0 |
| `hash_probe_row` | 1.0 | `bytes_per_row` | 64.0 |
| `output_row` | 0.5 | `jit_speedup` | 4.0 |

| Operator | cpu | mem |
|---|---|---|
| `Filter` | `filter_row × in_rows × expr_factor` | none |
| `Aggregate` | `hash_build_row × in_rows + output_row × out_rows` | `row_bytes × out_rows` |
| `Sort` | `sort_row × n × log2(max(2, heap))` | `row_bytes × heap` |
| `Join` | `hash_build_row × \|R\| + hash_probe_row × \|L\| + output_row × out_rows` | `row_bytes(right) × \|R\|` |
| `Window` | `sort_row × in_rows × log2(in_rows)` | `row_bytes × in_rows` |

Hash probes and hash aggregation state are also multiplied by `cache_factor` from `kyber/cost/terms.py`, which is 1.0 while the table fits the last-level cache and grows per doubling past it. State that won't fit the memory budget carries an `io` term priced by the spill device. A top-N's `heap = min(limit, n)`, so a `LIMIT` above the input is never costed above a full sort.

## Build-side selection

`hash_build_row` is twice `hash_probe_row`, which is why the build side matters. Write the small table on the left and Kyber swaps it into the build:

```python
import batcher as bt

small = bt.from_pydict({"k": list(range(100)), "name": [f"n{i}" for i in range(100)]})
big = bt.from_pydict({"k": [i % 100 for i in range(20000)], "w": [2.0] * 20000})
print(small.join(big, on="k").explain())
```

```text
query plan (planned)               3 operators
──────────────────────────────────────────────
OPERATOR                   ESTIMATE  NOTES
hash_join  [inner on k]  est≈20,000  (default)
├─ scan  [source 1]      est≈20,000  (exact)
└─ scan  [source 0]         est≈100  (exact)

decisions:
  - [kyber/selection] join build side: left≈100 right≈20,000 [exact] → swap build→left + broadcast
```

Join *ordering* uses `join_op_cost`, which for an inner join takes the cheaper of the two orientations, because that is what the SELECTION phase will pick later.

### Row width and the broadcast threshold

`CostModel.row_bytes` resolves per column: measured `avg_bytes` from the metadata hub, then the Arrow type's width, then the mean of the node's known columns, and only then the flat 64. That matters because the broadcast threshold is compared against the build side's bytes, and a two-`int64` key is 16 B/row.

The threshold is sized to cache, not memory: a broadcast table is probed from every core, so it wins only while it stays cache-resident. TPC-H sf1 puts the crossover between 4 and 10 MiB. `optimizer.broadcast_max_bytes` defaults to `0`, meaning auto, which resolves to a quarter of the detected last-level cache, or 4 MiB where the cache can't be read. Any positive value pins it.

```python
import batcher as bt

print(bt.active_config().optimizer.broadcast_max_bytes)  # 0
pinned = bt.Config().replace(optimizer=bt.OptimizerConfig(broadcast_max_bytes=8 << 20))
with bt.config_context(pinned):
    print(bt.active_config().optimizer.broadcast_max_bytes)  # 8388608
```

## Expressions have costs too

`filter_row × in_rows` prices running a filter, not *which* filter. `kyber/expr_cost/` prices the expression tree, and the per-node weights in `weights.py` were measured: each function ran as the sole expression of a projection over a million rows, with a bare column projection subtracted.

```text
eq / lt / add / sub / mul   1.0     (the unit: one interpreted numeric comparison, one row)
and / or                    0.5
div / mod                   3.0
concat                     12.0
len                        14.5
contains                   20.0
like                       28.0
regexp_matches             48.0
levenshtein               230.0
sha256                    325.0
image / audio / video     500.0    (media decode: estimated, not measured)
```

The high media cost is what makes Kyber push a filter *below* an image decode. `Case` costs `0.5 × (branches + 1)` plus its dearest arm, since one arm runs per row.

An expression the Cranelift tier can compile is divided by `jit_speedup` (4.0). `kyber/expr_cost/jit.py` mirrors [`crates/bc-codegen/src/analyze.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-codegen/src/analyze.rs) conservatively, answering `False` unless it can prove membership. The factor the operator cost uses is normalized so `col < 5` is always exactly 1.0 on its own tier, clamped to `[0.2, 1000]`.

## Calibration

The coefficients are priors. Once enough operators have run, `kyber/calibration.py` refits them from measured `op_stats`:

- **One family per coefficient.** `aggregate` fits `hash_build_row` and `hash_join` fits `hash_probe_row`. `output_row`, `map_row`, and `bytes_per_row` have no clean signal and keep their defaults.
- **Anchored globally.** The scale is chosen so the default model's total work equals total measured time, which makes calibration a no-op when reality already matches the defaults. `expr_factor` is divided back out, so a fitted coefficient describes the engine rather than the workload's expressions.
- **Shrunk, then clamped.** The measurement is blended toward the default with `weight = n / (n + 20)`, using `cost_calibration_min_samples`, and clamped to within `cost_calibration_clamp` (10x) of the default.
- **`jit_speedup`** is fitted from interpreted against JIT time per row on operators tagged `interp` or `jit`, and is clamped but not shrunk, since its measurement already anchors on the prior.
- **Throttled.** A refit runs only after 64 new feedback rows (`calibration._RECALIBRATE_AFTER`), so a small query doesn't rescan history on every {py:meth}`collect() <batcher.Dataset.collect>`.

## Join ordering

`kyber/rules/joins/order.py` dispatches on leaf count. Below three leaves the build-side rule handles orientation. From three up it runs a DPccp-style connected-subgraph DP over bushy trees, bounded by a per-query search budget, and falls back to greedy when the budget runs out or the join graph is disconnected.

![How a join order is chosen, and how hard it is looked for. The region is the connected inner joins of three leaves or more; with fewer, build-side selection handles it. The region is costed as written and a tenth of its estimated run time becomes a search budget, expressed in evaluated join pairs, clamped to between 512 and 200,000, and re-checked inside the loop rather than set once. Inside the budget, a connected-subset DP splits each subset into two connected halves, so the shapes it reaches are bushy, up to 20 leaves. When the budget is spent, the region runs past 20 leaves, or the join graph is disconnected, it falls back to a greedy search that starts from the smallest leaf and repeatedly adds the cheapest next join, which is left-deep by construction. Both searches rank every candidate by the same two things: cardinality, from left rows times right rows over the larger distinct count, refined by skew and range overlap using HLL, Misra-Gries and KLL, and cost, from build rows, probe rows and cache residency, priced at the cheaper build side. Both build the same relation and differ only in how much of the space they read. An exhaustive subset DP exists beside them as the test oracle and is never on the live path.](/_static/diagrams/join_order_search.svg)

The budget, set by `kyber/rules/joins/order_budget.py`, is a share of the region's own estimated execution cost: Kyber prices the region as written, grants a tenth of that time back as search, and divides by the measured ~150 µs cost of evaluating one join pair. The result is clamped to `[512, 200000]` pairs, and a graph too dense to search within budget declines to greedy before spending anything.

Measured planning time, over 1,000 rows with the plan cache off, against the flat cap it replaced:

| join graph | leaves | flat cap  | budgeted | speedup |
|------------|-------:|----------:|---------:|--------:|
| star       |     12 |  1.849 s  | 0.017 s  |  108x   |
| star       |     13 |  4.821 s  | 0.019 s  |  249x   |
| star       |     14 | 10.368 s  | 0.023 s  |  454x   |
| star       |     15 | 23.992 s  | 0.025 s  |  973x   |
| chain      |     12 |  0.042 s  | 0.043 s  |  1.0x   |
| chain      |     15 |  0.085 s  | 0.077 s  |  1.1x   |

Sparse graphs, the shape real queries have, keep their plans. Join order is semantics-preserving, so the budget only changes how much driver time goes into choosing one.

:::{dropdown} Rewrites gated on a ratio, not a cost
`eager_aggregation` and its siblings in `kyber/rules/agg_pushdown.py` pre-aggregate one side of a join and require a measured row reduction of at least 8x. A ratio can't see that a **global** aggregate above a join is already an `O(1)` streaming reduction, so for a global aggregate the push additionally requires that the join *amplifies*, meaning it duplicates the side being pre-aggregated. Grouped aggregates are unaffected.
:::

## Practical limits

- The per-row coefficients have no NUMA or memory-bandwidth term. Cache enters through the broadcast threshold and `cache_factor`.
- The model is only as good as the cardinalities feeding it. See {doc}`Cardinality estimation </architecture/deep-dives/adaptive/cardinality-estimation>`.

## See also

- {doc}`Architecture </architecture/index>`: Kyber decides, and the cost model is how.
- {doc}`Kyber optimizer </architecture/internals/kyber>`: the phases these costs run in.
- {doc}`Configuration options </configuration/options>`: `optimizer.cost_coeffs` and `cost_weights`.
- {doc}`Reading a plan </user-guide/operate/tuning/explain-plans>`: the decisions block these numbers produce.
- {doc}`TPC-H benchmarks </benchmarks/results/tpch>`: the join-order shapes the DP is for.
- {doc}`Cardinality estimation </architecture/deep-dives/adaptive/cardinality-estimation>`: the row counts every formula multiplies.
- {doc}`Learned metadata </architecture/deep-dives/adaptive/learned-metadata>`: where `op_stats` lives and what else reads it.
- {doc}`JIT compilation </architecture/deep-dives/query/jit-compilation>`: the tier `jit_speedup` is pricing.
