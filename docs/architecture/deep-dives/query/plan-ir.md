# Plan IR

Python builds the plan; Rust runs it. They meet at one JSON document, and that document is
a wire contract in the ordinary sense: two independent programs agree on a set of tags, and
a disagreement is a bug that no compiler will catch for you.

## Why JSON, and why it is cheap

A plan is one document per execution, a few kilobytes, parsed once. Execution then runs for
milliseconds to minutes over gigabytes. Serialization format is not on the hot path, so the
choice was made for the property that matters at 3 a.m.: you can print the plan, diff it,
paste it into a bug report, and hand it to `serde_json` in a test. Nothing about the design
depends on JSON specifically, and nothing about performance argues against it either.

The data does *not* travel this way. Arrow batches cross the boundary zero-copy through the
C Data Interface. The IR carries the plan; the Arrow pointers carry the rows.

## The two levels

The document nests two independent trees, each defined by one Rust type.

`RelOp`, in `crates/bc-ir/src/lib.rs`, is the relational plan. Its serde attributes are `tag = "op"`, `rename_all = "snake_case"`, and `deny_unknown_fields`, so every node in the document announces itself with an `op` key holding a snake_case variant name. The fifteen variants are `scan`, `filter`, `project`, `aggregate`, `sort`, `limit`, `hash_join`, `asof_join`, `distinct`, `window`, `union`, `unnest`, `unpivot`, `row_id`, and `sample`.

{py:class}`Expr <batcher.plan.expr_ir.core.Expr>`, in `crates/bc-expr/src/lib.rs`, is the scalar expression tree carried inside `RelOp` nodes. It uses the same attributes with `tag = "e"`, and its variants include `col`, `lit`, `binary`, `not`, `cast`, `is_null`, `is_not_null`, `is_nan`, `is_inf`, `case`, `str`, and `date`.

:::{important}
There is exactly one of each. The interpreter, the JIT, the runtime primitives, and the
distributed path all consume the same `Expr` and the same `RelOp`. A new backend consumes the
existing IR; it never forks a second representation. That shared source is what makes semantic
parity between tiers a structural property rather than a promise.
:::

Python's side of the contract lives in `python/batcher/plan/ir_tags.py`, which holds the tag
*strings* as constants rather than scattering literals across thirty `to_ir()` methods. A
typo is then an `AttributeError` (`Op.SCNA`) instead of a silently wrong tag that only a
differential test would find.

## What a plan looks like

```python
import batcher as bt
import json

ds = bt.from_pydict({"g": ["a", "b"], "x": [1, 2]})
q = ds.filter(bt.col("x") > 1).select("g", "x")

# `_plan` is internal. This is the document Core ships across the FFI boundary.
print(json.dumps(q._plan.to_ir(), indent=2))
```

The document is two nested trees. The relational one is tagged `op`; the scalar one hanging
off the filter's `predicate` is tagged `e`:

```text
  RelOp tree  (serde tag: "op")             Expr tree  (serde tag: "e")
  ────────────────────────────────          ────────────────────────────────
  project { exprs: [g, x] }
      │
  filter  { predicate: ───────────────────► binary { op: "gt" }
      │                                       ├── left:  col { name: "x" }
      │                                       └── right: lit { int: 1 }
      │
  scan    { source_id: 0 }
      │
      └── binds to sources[0], a list of pyarrow RecordBatches
          handed across separately, zero-copy. The IR names no file.
```

:::{dropdown} The document that actually crosses the boundary
```text
{
  "op": "project",
  "input": {
    "op": "filter",
    "input": { "op": "scan", "source_id": 0 },
    "predicate": {
      "e": "binary",
      "op": "gt",
      "left":  { "e": "col", "name": "x" },
      "right": { "e": "lit", "value": { "int": 1 } }
    }
  },
  "exprs": [
    { "expr": { "e": "col", "name": "g" }, "alias": "g" },
    { "expr": { "e": "col", "name": "x" }, "alias": "x" }
  ]
}
```
:::

Three things to notice.

**`source_id` is an index, not a path.** `Scan { source_id: 0 }` binds to `sources[0]`, the
list of pyarrow `RecordBatch`es passed alongside the plan. The IR never names a file. The Python `io` layer resolves splits, pushes predicates down to the reader, and applies schema evolution, then hands the engine batches.

**There is no schema in the document.** The engine infers types from the Arrow input, which
already carries them. This keeps the IR small and makes it impossible for a declared schema
and an actual batch to disagree.

**The planner has already resolved names.** A `hash_join` node carries an explicit `output:
Vec<JoinOutputCol>`: which side each output column comes from, its name there, and its name
in the result. The planner knows both schemas, so the engine does not have to
re-derive them.

## Expressions

An expression lowers on its own:

```python
import batcher as bt
import json

print(json.dumps(((bt.col("x") * 2 + 1) > bt.col("y")).to_ir()))
```

```text
{"e": "binary", "op": "gt",
 "left": {"e": "binary", "op": "add",
          "left": {"e": "binary", "op": "mul",
                   "left": {"e": "col", "name": "x"},
                   "right": {"e": "lit", "value": {"int": 2}}},
          "right": {"e": "lit", "value": {"int": 1}}},
 "right": {"e": "col", "name": "y"}}
```

Literals are tagged by type (`{"int": 2}`, `{"float": 1.5}`, `{"str": "a"}`) so the engine
never has to guess whether `1` means `1i64` or `1.0f64`.

## Physical hints ride along

Some fields are the optimizer talking to the executor, not the user talking to either:

| Hint | Set by | Effect |
|---|---|---|
| `Sort { limit }` | fusing a downstream `Limit` | the sort becomes a top-N, computed by a partial sort rather than a full one |
| `HashJoin { strategy }` | Kyber, from cardinality | `hash` (shuffle), `broadcast`, or `sort_merge`. All three produce the same relation; only the data movement differs |
| `Window { rank_limit }` | fusing `QUALIFY rn <= k` | a per-partition top-N instead of a full ranking |
| `Aggregate { group_keys: [] }` | the planner | an empty key list is a global aggregate, not an error |

A wrong `strategy` is slow, never wrong. Each field has a `#[serde(default)]`, so a Python
side that does not emit it still produces a valid document, which is what lets the optimizer
learn to emit a new hint without a coordinated flag day.

## The rule that keeps this honest

:::{important}
Changing the IR is a **two-sided change in a single commit**. `deny_unknown_fields` means a
field Python emits and Rust does not know about is a loud parse error at the boundary rather
than a silently ignored instruction, and a silently ignored instruction is how a filter
disappears and a query quietly returns too many rows.
:::

::::{tab-set}
:::{tab-item} The Python side
```text
python/batcher/plan/ir_tags.py       the tag vocabulary, as constants
python/batcher/plan/logical/         one to_ir() per node
python/batcher/plan/physical.py      document assembly
```
:::

:::{tab-item} The Rust side
```text
crates/bc-ir/src/lib.rs      RelOp: serde tag "op", snake_case, deny_unknown_fields
crates/bc-expr/src/lib.rs    Expr:  serde tag "e"
crates/bc-py/src/lib.rs      deserialization at the boundary
```
:::
::::

If you add a variant to `bc_ir::RelOp`, you add its tag to `plan/ir_tags.py`, its `to_ir()`
to the corresponding `LogicalPlan` node, and a test that the Python shape deserializes in
Rust. All in the same commit.

Two invariants make this a hard gate rather than a convention:

- `CLAUDE.md` invariant 8: the JSON IR is a stable wire contract; Python `to_ir()` tags and
  Rust `serde` tags stay in lockstep.
- `CLAUDE.md` invariant 6: one `Expr`, one `RelOp`, shared across tiers. A new backend
  consumes the existing IR; it does not fork a second representation.

## The costs

:::{note}
The IR carries no schema and no stage. Both absences earn their place: types come from the
Arrow input, which already has them, and stages are composed in Python out of ordinary plans,
so the engine sees the same document shape whether it is running one node or a hundred.
:::

The IR is a tree, not a DAG with sharing. A common sub-plan referenced twice is serialized
twice and executed twice; there is no `WITH`-style CTE node and no result reuse. For the
shapes the engine targets that has not been worth the machinery, but it is a real limit and
it is the first thing to look at if you find yourself scanning a source twice.

The IR also has no notion of a *stage*. The distributed path composes stages in Python
(`python/batcher/dist/`) out of the same mergeable primitives, shipping a sub-plan per task.
The engine sees an ordinary plan every time, which is exactly why single-node and distributed
cannot drift apart.

## Where the code lives

Each piece of the IR has one home. These are the files to open when a tag, a literal, or a
config field is not behaving as this page describes:

| Piece | File |
|---|---|
| `RelOp` + physical hints | `crates/bc-ir/src/lib.rs` |
| `Expr` + literals | `crates/bc-expr/src/lib.rs` |
| `EngineConfig` (morsel size, parallelism, tuning) | `crates/bc-ir/src/engine_config.rs` |
| Python tag vocabulary | `python/batcher/plan/ir_tags.py` |
| Per-node `to_ir()` | `python/batcher/plan/logical/` |
| Document assembly | `python/batcher/plan/physical.py` |
| Deserialization at the boundary | `crates/bc-py/src/lib.rs` |

## See also

- {doc}`Architecture </architecture/index>`: why the control plane and the data plane meet at a document.
- {doc}`Execution engine </architecture/internals/execution>`: what happens to the `RelOp` tree after it lands.
- {doc}`Kyber </architecture/internals/kyber>`: the passes that set the physical hints above.
- {doc}`Reading a plan </user-guide/operate/tuning/explain-plans>`: the same tree, rendered for humans.
- {doc}`Performance </user-guide/operate/tuning/performance>`: what to do when the plan is not the one you wanted.
- {doc}`TPC-H benchmarks </benchmarks/results/tpch>`: the query shapes these hints are tuned against.
- {doc}`Query lifecycle </architecture/deep-dives/query/query-lifecycle>`: where the document is produced and consumed.
- {doc}`Expression evaluation </architecture/deep-dives/query/expression-evaluation>`: what the engine does with an `Expr`.
- {doc}`Join algorithms </architecture/deep-dives/operators/join-algorithms>`: what the `strategy` hint actually selects.
