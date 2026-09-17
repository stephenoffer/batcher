# The cost model

This page describes how Kyber turns row counts into a cost it can compare, how the coefficients are calibrated from measured runs, and how hard it searches for a join order.

Given two plans that produce the same answer, which one runs faster? Every cost-based
decision in Kyber (join order, build side, broadcast versus shuffle, whether to split a
filter) reduces to comparing two numbers. The cost model produces those numbers.

It does not try to predict milliseconds. It produces an abstract, mutually-comparable
scalar, and the only property that matters is that a cheaper plan really is faster.

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
`mem` is deliberately absent from the scalar. It is a *peak*, not a throughput cost, so it gates
feasibility rather than speed. The question it answers is whether Carbonite can admit this plan.
When costs compose up the tree, `cpu`, `io`, and `net` **sum** over children while `mem`
accumulates as a **max**. Breakers run at different times, so peak memory is the tallest one
rather than the total.
:::

```text
              join            cpu = own + Σ children.cpu
             /    \           io  = own + Σ children.io
         scan      aggregate  net = own + Σ children.net
                       │      mem = max(own, max over children)   ← a PEAK
                     scan
                              total(w) = w.cpu·cpu + w.io·io + w.net·net
                                         mem is not in the scalar at all
```

Weights (`optimizer.cost_weights`):

```python
# docs: skip
cpu: float = 1.0
io: float = 1.0
net: float = 2.0  # a shuffled byte costs twice a local one
```

What feeds that fold, and what comes out of it:

![What the cost model consumes and what it emits. Four inputs feed one fold over the plan tree in CostModel.cost(node): estimated rows per node from the estimator, a type-exact row width rather than a flat 64 bytes, machine terms such as L3 cache size, memory budget and spill device, and coefficients that ship as constants and are then calibrated from measured runs. It emits four axes, three of which enter the scalar. cpu, io and net combine as 1.0 times cpu plus 1.0 times io plus 2.0 times net, and that one comparable number ranks the alternatives: join order, join strategy, whether to spill. mem is the peak working set, a max along the tree and never summed, so it gates feasibility rather than throughput, because a peak is not a quantity you can add up.](/_static/diagrams/cost_model_inputs.svg)

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

The interesting ones:

| Operator | cpu | mem |
|---|---|---|
| `Filter` | `filter_row × in_rows × expr_factor` | none |
| `Aggregate` | `hash_build_row × in_rows + output_row × out_rows` | `row_bytes × out_rows` |
| `Sort` | `sort_row × n × log2(max(2, heap))` | `row_bytes × heap` |
| `Join` | `hash_build_row × \|R\| + hash_probe_row × \|L\| + output_row × out_rows` | `row_bytes(right) × \|R\|` |
| `Window` | `sort_row × in_rows × log2(in_rows)` | `row_bytes × in_rows` |

The table shows the base terms. In `kyber/cost/model.py` the hash probe of a join and the hash
state of an aggregate or distinct are also multiplied by `cache_factor` from
`kyber/cost/terms.py`, which is 1.0 while the table fits the last-level cache of the node that
will run it and grows by a fixed penalty per doubling past it. The same operators carry an `io`
term from `spill_io` for the state that won't fit the memory budget, priced by the spill device.

`Sort`'s `heap = min(limit, n)` when there is a limit. A top-N heap can never hold more
rows than exist, so a `LIMIT` above the input degenerates to a full sort rather than being
costed *above* one.

`hash_build_row` is twice `hash_probe_row`, and that ratio is the whole reason build-side
selection matters. `explain()` prints the decision it drove:

```python
import batcher as bt

left = bt.from_pydict({"k": list(range(1000)), "v": [1.0] * 1000})
right = bt.from_pydict({"k": [i % 50 for i in range(20000)], "w": [2.0] * 20000})
print(left.join(right, on="k").group_by("k").agg(s=bt.sum("w")).explain())
```

:::{dropdown} The plan, and the decision the coefficients drove
```text
query plan (planned)                                                5 operators
───────────────────────────────────────────────────────────────────────────────
OPERATOR                            ESTIMATE  NOTES
aggregate  [by k · sum]               est≈50  (default)
└─ hash_join  [inner on k]        est≈20,000  (default)
   ├─ scan  [source 1]            est≈20,000  (exact)
   └─ filter  [k ≥ 0 AND k ≤ 49]      est≈50  (default)
      └─ scan  [source 0]          est≈1,000  (exact)  pushed[k ≥ 0 AND k ≤ 49]

decisions:
  - [kyber/selection] join build side: left≈1,000 right≈20,000 [exact] → swap build→left + broadcast
```

The filter nobody wrote is `rules.joins.runtime_join_filter`. The right side's `k` runs 0 to 49,
so a left row outside that range can never match, and the `[min, max]` bound is mirrored onto the
left scan. The build-side line at the bottom is the one this section is about.
:::

The join is written left-joins-right, and the two orientations do not cost the same:

::::{tab-set}
:::{tab-item} As written
```text
build on the 20,000-row side, probe with 1,000 rows

  hash_build_row × 20,000  +  hash_probe_row × 1,000
       2.0       × 20,000  +       1.0       × 1,000
```
:::

:::{tab-item} Swapped (what Kyber picks)
```text
build on the 1,000-row side, probe with 20,000 rows

  hash_build_row × 1,000   +  hash_probe_row × 20,000
       2.0       ×  1,000  +       1.0       × 20,000
```
The cheaper orientation wins, and because the small side is also under the resolved broadcast
threshold, it is broadcast rather than shuffled.
:::
::::

That asymmetry is also why join *ordering* uses a different entry point, `join_op_cost`,
which for an inner join takes the cheaper of the two build orientations, because that is
what the SELECTION phase will actually pick later. Costing a join as
written, when the build side is still up for grabs, prices a decision that has not been
made.

## Row width is type-exact, not 64 bytes

`bytes_per_row = 64.0` is the last resort, not the first. `CostModel.row_bytes` resolves
per column: measured `avg_bytes` from the metadata hub → the Arrow type's width → the mean
of this node's known columns → the flat 64.

:::{warning}
This decides broadcast eligibility. The broadcast threshold is compared against the build
side's *bytes*, and a two-`int64` key is 16 B/row, not 64. Reading it against a flat 64 made
the effective threshold roughly 4 times smaller than its nominal value, and the optimizer
declined broadcasts it should have taken.
:::

That threshold is sized to **cache, not memory**. A broadcast join builds one hash table and
probes it from every core, so each probe row is a random access into it. The strategy wins only
while the table stays cache-resident. Past that the partitioned join wins, because each of its
buckets probes a small L2-resident table. TPC-H sf1 puts the crossover between 4 and 10 MiB:
q3's 4.4 MB build over a 3.2M-row probe takes 52 ms partitioned against 83 ms broadcast.

Because it bounds a cache-resident table, the threshold is detected rather than fixed.
`optimizer.broadcast_max_bytes` defaults to `0`, meaning auto, and
`resolved_broadcast_max_bytes` then returns a quarter of the detected last-level cache. A
machine whose cache cannot be read, such as a non-Linux host, falls back to 4 MiB, which is the
historical default and the value a 16 MiB L3 resolves to anyway. Any positive
`broadcast_max_bytes` pins the threshold and wins over detection.

## Expressions have costs too

`filter_row × in_rows` prices "running a filter". It does not price *which* filter. A
`col > 5` and a {py:meth}`regexp_matches(col, '...') <batcher.plan.expr_ir.namespaces.strings._StrNamespace.regexp_matches>` are not the same work, and an optimizer that
cannot tell them apart will not bother pushing the expensive one anywhere useful.

`kyber/expr_cost/` prices the expression tree. `weights.py` carries the per-node table, and
these numbers were **measured**, not guessed. Each function ran as the sole expression of a
projection over a million rows in a fresh process, with a bare column projection subtracted:

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

The media functions are costed high on purpose. That is what makes Kyber push a filter
*below* an image decode rather than above it.

`Case` costs `0.5 × (branches + 1)` for the selection itself, one masked pick per branch.
The branches on top of that are charged at the dearest single arm rather than the sum,
because the engine evaluates one arm per row. `Aliased` costs 0, because it is transparent
in the IR.

### The JIT divisor

An expression the Cranelift tier can compile does not cost what the interpreter would charge
for it. `expr_cost` divides by `jit_speedup` (4.0) when `jit_compilable(expr)`.

`kyber/expr_cost/jit.py` is a conservative mirror of `crates/bc-codegen/src/analyze.rs`. It
answers `False` whenever it cannot *prove* membership in the supported subset. Costing errs
toward "interpreted", never toward a fast path that does not exist. Integer `div`/`mod`
compile only against a constant divisor that is neither 0 nor −1, because Cranelift's `sdiv`
traps. `round`, `cbrt`, `sign` stay interpreted for bit-for-bit parity with the oracle.

The multiplier the operator cost actually uses is normalized against the archetypal
predicate, priced *on its own tier*:

```python
# docs: skip
_BASELINE_RAW = own_cost(Col("x")) + own_cost(Lit(0)) + BINARY_COST["lt"]  # 0.2 + 0.2 + 1.0
baseline = _BASELINE_RAW / speedup
factor = clamp(expr_cost(expr, speedup) / baseline, 0.2, 1000.0)
```

So `col < 5` is always exactly 1.0 whatever the measured speedup is. Raising the speedup
makes *interpreted* expressions more expensive, which is precisely the signal
that should make the optimizer work harder to keep them off hot rows.

## Calibration

The coefficients are priors. Once enough operators have run, `kyber/calibration.py` refits
them from measured `op_stats`.

The method matters as much as the fit. Each operator family maps to the one coefficient its
dominant per-row term scales, so `aggregate` fits `hash_build_row` as the purest hash-build
signal and `hash_join` fits `hash_probe_row`. The remaining three, `output_row`, `map_row`, and
`bytes_per_row`, have no clean single-family signal and keep their defaults.

Everything is then anchored globally. `k = total_default_work / total_ms` is chosen so the
default model's total work over all samples equals their total measured time, which means
calibration is a no-op when reality already matches the defaults. That is the property you want
from a self-tuning system: it must not drift when it has nothing to say. Each coefficient
becomes `median(k × t / (basis × expr_factor))`. Dividing out `expr_factor` keeps the fitted
coefficient a property of the *engine* rather than of whichever expressions the workload
happened to contain. Without it a regex-heavy workload fits a huge `filter_row`, which the cost
model then multiplies by the regex's factor a second time.

The measurement is blended toward the shipped default by **shrinkage** rather than a fixed
ratio, with `weight = n / (n + prior_strength)` and `prior_strength` set to
`cost_calibration_min_samples` (20). A fixed `alpha=0.5` blend has a fixed point, so a
coefficient whose true value was 10 times the default converged to 5.5 times it and stayed
there. The result is finally clamped to within `cost_calibration_clamp` (10 times) of the
default, so timing noise cannot produce a degenerate model.

`jit_speedup` is fitted from the ratio of interpreted to JIT residual time per row on `filter`
and `project` operators, which `op_stats.backend` tags as `"interp"` or `"jit"`. Operators
tagged `"interp+jit"` are skipped, since they blend the tiers. Unlike the absolute
coefficients it is deliberately *not* shrunk. The absolute coefficients fit a value on the
engine's own scale, so the shipped default is a genuine independent prior, but the speedup is
measured as `prior × ratio`, which already anchors it on the prior. Shrinking would anchor it
twice and reintroduce the fixed point shrinkage exists to remove. The clamp still bounds how
far it may travel.

Refit is throttled: only after 64 new feedback rows accrue
(`calibration._RECALIBRATE_AFTER`). Profiling a small query once showed ~90% of its latency
was the planner, growing with the session's query count, because calibration rescanned the
entire `op_stats` history on every {py:meth}`collect() <batcher.Dataset.collect>`.

## Join ordering

`kyber/rules/joins/order.py` dispatches on leaf count. Under three leaves it skips, because a
two-way join has no ordering left to choose and its orientation is the build-side rule's
business. From three leaves up it runs a DPccp-style connected-subgraph DP over bushy trees,
bounded by a per-query search budget rather than a fixed cap. When the budget runs out, or the
join graph is disconnected and the DP has nothing to enumerate, it falls back to greedy: start
from the smallest leaf and repeatedly add whichever connected leaf costs least to add next.

Both searches and the budget that chooses between them, end to end:

![How a join order is chosen, and how hard it is looked for. The region is the connected inner joins of three leaves or more; with fewer, build-side selection handles it. The region is costed as written and a tenth of its estimated run time becomes a search budget, expressed in evaluated join pairs, clamped to between 512 and 200,000, and re-checked inside the loop rather than set once. Inside the budget, a connected-subset DP splits each subset into two connected halves, so the shapes it reaches are bushy, up to 20 leaves. When the budget is spent, the region runs past 20 leaves, or the join graph is disconnected, it falls back to a greedy search that starts from the smallest leaf and repeatedly adds the cheapest next join, which is left-deep by construction. Both searches rank every candidate by the same two things: cardinality, from left rows times right rows over the larger distinct count, refined by skew and range overlap using HLL, Misra-Gries and KLL, and cost, from build rows, probe rows and cache residency, priced at the cheaper build side. Both build the same relation and differ only in how much of the space they read. An exhaustive subset DP exists beside them as the test oracle and is never on the live path.](/_static/diagrams/join_order_search.svg)

### How hard to search is itself a decision

The search budget is set per query by `kyber/rules/joins/order_budget.py`, in the unit the DP
already counts: *evaluated join pairs*, a candidate split that gets built, estimated and
costed. A pair costs the planner ~150 us, measured across sixteen configurations spanning 6 to
14 leaves on star and chain graphs, with no trend in leaf count or graph density.

Neither of the axes a fixed cap can use predicts the right answer. Leaf count does not predict
search *work*, because density decides it: measured over 1,000 rows, a 14-leaf chain evaluates
455 pairs and a 14-leaf star evaluates 53,248. Same leaf count, 117 times the work. And
no static number predicts what a better order is *worth*, because that depends entirely on the
data: a 15-leaf star over a thousand rows spent 25.5 s searching for an order whose best and
worst cases are microseconds apart, while the same shape over a petabyte would repay far more
searching than any cap allows.

So the budget is a share of the region's own estimated execution cost. Kyber prices the join
region as written, converts that to seconds through a measured ~1.7e-10 seconds per cost unit,
grants a tenth of it back as search time, and divides by the measured cost of a pair. The
result is clamped to `[512, 200000]` pairs. The floor covers the full search for every star up
to 7 leaves and every chain up to 15, so small queries keep the plans they already get. The
ceiling is the same number as the flat cap it replaces, so no query that could afford a search
before gets a smaller one now. What changed is that the ceiling has to be earned, and only a
query estimated to run for minutes earns it.

Density is triaged before any of it is spent. The DP knows its connected-subset count before it
costs anything, and evaluated pairs run about 4 to 7 times that count across every density
measured, so a graph too dense to search within budget declines immediately instead of spending
the budget to discover the same thing and handing the answer to greedy regardless.

Measured effect on planning time, over 1,000 rows with the plan cache off, A/B against the
flat cap on one tree:

| join graph | leaves | flat cap  | budgeted | speedup |
|------------|-------:|----------:|---------:|--------:|
| star       |     12 |  1.849 s  | 0.017 s  |  108x   |
| star       |     13 |  4.821 s  | 0.019 s  |  249x   |
| star       |     14 | 10.368 s  | 0.023 s  |  454x   |
| star       |     15 | 23.992 s  | 0.025 s  |  973x   |
| chain      |     12 |  0.042 s  | 0.043 s  |  1.0x   |
| chain      |     15 |  0.085 s  | 0.077 s  |  1.1x   |

The two halves of that table are the point. Sparse graphs, which is the shape real queries
have, are unchanged to within noise and keep the plans they had. What the budget removes is
the search a dense graph could never repay.

Join order is semantics-preserving, so none of this can change a result. It changes only how
much of the driver's time is spent choosing one.

The DP recurrence adds only *this join's* op cost to the two halves' already-accumulated
costs. Using the full recursive `cost()` would re-walk and double-count children, penalizing
deep subtrees super-linearly.

:::{note}
`optimizer.join_dp_max_tables` (12) and `greedy_max_tables` (25) are declared and validated,
but the rule does not read them, so setting either changes nothing. They predate the search
budget above, which answers the question they were meant to answer and answers it per query
rather than per session. Greedy also has no upper leaf bound, so there is no table count above
which reordering stops.
:::

## Limits

The per-row coefficients have no notion of NUMA or memory bandwidth, and cache enters only
through the broadcast threshold and `cache_factor`.
`hash_build_row = 2.0` is still a single number for an operation whose real cost varies by an
order of magnitude with hash table size. What partly rescues this is that the decisions it drives are comparative, and
the errors are usually in the same direction on both sides of the comparison.

It is only as good as the cardinalities feeding it. A cost model applied to a row count that
is 80× low produces a confident, precise, wrong answer. That is the cold-start join failure
described in {doc}`Cardinality estimation </architecture/deep-dives/adaptive/cardinality-estimation>`.

### A ratio is not a cost

Some rewrites are gated outside the model, on a **row-reduction ratio** rather than a cost
comparison, because the model prices a hash aggregate linearly in input rows and is blind to what
a large group count does to cache. `eager_aggregation` and its siblings in
`kyber/rules/agg_pushdown.py` are the case: they pre-aggregate one side of a join and require a
measured reduction of at least 8x before firing.

A ratio answers "how much smaller does this get?" and never "what was the alternative?". For an
aggregate with a `GROUP BY` above the join the two questions have the same answer, because the
outer aggregate builds a hash table either way and only its input size changes. For a **global**
aggregate they do not: without the push the outer aggregate is a streaming reduction with `O(1)`
state, the cheapest thing the engine can do, and the push replaces it with a *grouped* hash
aggregate over the same rows.

Measured on a 10 million row probe joined against a unique 10,000-row build side, varying only
which column the aggregate reads so that only the push changes:

| Query over the same join | Pushed? | Time |
| --- | --- | --- |
| `count()` of a right-side column | no | 7.6 ms |
| `count()` of the join key | yes | 19.6 ms |
| `sum()` of a left-side column | yes | 20.3 ms |
| `min()` of a left-side column | yes | 21.0 ms |

The reduction there is about 1000x, so it clears the 8x bar by two orders of magnitude and is
still a 2.1x to 2.3x pessimization. The push pays a full ten-million-row grouped hash aggregate
to save a broadcast probe that was nearly free.

The gate that fixes it asks the question the ratio cannot: **does the join amplify?** Eager
aggregation pays when the join duplicates the side being pre-aggregated, because collapsing rows
early avoids paying for the copies. When the join emits no more rows than it reads there is
nothing downstream to save, so for a global aggregate the push is refused. Grouped aggregates are
untouched.

One consequence is worth stating because it looks like a bug. The additive form of the rule
(`sum`/`count`) requires the *other* side to be unique on the join key for correctness, and that
is exactly the condition under which a join cannot amplify. So the gate withdraws that rule from
global aggregates entirely, which is correct rather than over-eager: there was never a fan-out
for it to save. It keeps firing wherever the outer aggregate groups.

## See also

- {doc}`Architecture </architecture/index>`: Kyber decides, and the cost model is how.
- {doc}`Kyber optimizer </architecture/internals/kyber>`: the phases these costs run in.
- `docs/architecture/internals/mathematical_foundations.md` (in the repo, not a site page): the shrinkage estimator and its fixed point.
- {doc}`Configuration options </configuration/options>`: `optimizer.cost_coeffs` and `cost_weights`.
- {doc}`Reading a plan </user-guide/operate/tuning/explain-plans>`: the decisions block these numbers produce.
- {doc}`TPC-H benchmarks </benchmarks/results/tpch>`: the join-order shapes the DP is for.
- {doc}`Cardinality estimation </architecture/deep-dives/adaptive/cardinality-estimation>`: the row counts every formula multiplies.
- {doc}`Learned metadata </architecture/deep-dives/adaptive/learned-metadata>`: where `op_stats` lives and what else reads it.
- {doc}`JIT compilation </architecture/deep-dives/query/jit-compilation>`: the tier `jit_speedup` is pricing.
