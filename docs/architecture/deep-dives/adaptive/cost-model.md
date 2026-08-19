# The cost model

Given two plans that produce the same answer, which one runs faster? Every cost-based
decision in Kyber (join order, build side, broadcast versus shuffle, whether to split a
filter) reduces to comparing two numbers. The cost model produces those numbers.

It does not try to predict milliseconds. It produces an abstract, mutually-comparable
scalar, and the only property that matters is that a cheaper plan really is faster.

## Four axes, three in the scalar

```python
# docs: skip
# python/batcher/kyber/cost.py
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
io:  float = 1.0
net: float = 2.0   # a shuffled byte costs twice a local one
```

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
| `Join` | `hash_build_row × |R| + hash_probe_row × |L| + output_row × out_rows` | `row_bytes(right) × |R|` |
| `Window` | `sort_row × in_rows × log2(in_rows)` | `row_bytes × in_rows` |

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
aggregate                       est≈2,000 (default)
  hash_join                     est≈20,000 (default)
    scan                        est≈20,000 (exact)
    scan                        est≈1,000 (exact)

decisions:
  - [kyber/selection] join build side: left≈1,000 right≈20,000 [exact] → swap build→left + broadcast
```
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

`Case` costs `0.5 × (branches + 1)` because the engine's `CASE` does not short-circuit. It
evaluates every branch over every row and selects. `Aliased` costs 0, because it is
transparent in the IR.

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

`kyber/rules/joins/order.py` dispatches on leaf count:

- fewer than 3 leaves: skip (a two-way join is the build-side rule's business)
- up to 12: exhaustive subset DP, bushy trees, O(3ⁿ)
- above 12: DPccp-style connected-subgraph DP, bailing above 20 leaves or 200,000 pairs
- on a bail, or a disconnected graph: greedy. Start from the smallest leaf, repeatedly add
  the connected leaf minimizing the incremental cost

The DP recurrence adds only *this join's* op cost to the two halves' already-accumulated
costs. Using the full recursive `cost()` would re-walk and double-count children, penalizing
deep subtrees super-linearly.

:::{note}
`optimizer.join_dp_max_tables` (12) and `greedy_max_tables` (25) are declared and validated,
but the rule reads its own module constants (`_MAX_EXHAUSTIVE_LEAVES = 12`, `_MAX_DP_LEAVES
= 20`, `_MAX_DP_PAIRS = 200_000`). Setting the config knobs changes nothing today. Greedy
also has no upper leaf bound, so there is no table count above which reordering stops.
:::

## Limits

The model has no notion of cache, NUMA, or memory bandwidth. `hash_build_row = 2.0` is a
single number for an operation whose real cost varies by an order of magnitude with hash
table size. What partly rescues this is that the decisions it drives are comparative, and
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
