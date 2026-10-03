# Plan IR

Python builds the plan and Rust runs it. They meet at one JSON document, and that document is a wire contract: two independent programs agree on a set of tags, and the engine rejects any tag or field it doesn't know.

You can read the document for any query from {py:meth}`explain(format="json") <batcher.Dataset.explain>`, both as written and as Kyber left it:

```python
import batcher as bt
import json

ds = bt.from_pydict({"g": ["a", "b"], "x": [1, 2]})
q = ds.filter(bt.col("x") > 1).select("g", "x")
doc = json.loads(q.explain(format="json"))
print(doc["logical_ir"]["op"], "->", doc["optimized_ir"]["op"])  # project -> filter
print(doc["optimized_ir"]["predicate"]["e"])                      # binary
```

The optimizer dropped the `project`, because selecting every column in order is the identity.

![The lowering chain with the plane boundary drawn across it. Above the line, in Python, nothing touches a row: a lazy immutable Dataset returns a new node per operation, those nodes form a validated LogicalPlan, Kyber rewrites plan to plan through seven phases from NORMALIZE to ENFORCE, and the rewritten tree lowers through to_ir() into a PhysicalPlan carrying the IR and its resource bounds. Below the line, in Rust, nothing chooses a plan: bc-py's execute_plan is the one FFI entry, serde deserializes the document into a bc-ir RelOp tree and hard-errors on an unknown tag, and bc-interp and bc-runtime walk that tree once over 16,384-row Arrow morsels. Exactly one edge crosses the boundary, carrying to_json() output plus the Arrow input batches, and nothing else crosses at all.](/_static/diagrams/plan_lowering.svg)

## Why JSON

A plan is one document per execution, a few kilobytes, parsed once, while execution runs for milliseconds to minutes over gigabytes. Serialization format isn't on the hot path, so the choice was made for debuggability: you can print the plan, diff it, paste it into a bug report, and hand it to `serde_json` in a test. The data doesn't travel this way. Arrow batches cross zero-copy through the C Data Interface.

## The two levels

The document nests two trees, each defined by one Rust type.

- **`RelOp`**, in [`crates/bc-ir/src/lib.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-ir/src/lib.rs), is the relational plan. Its serde attributes are `tag = "op"`, `rename_all = "snake_case"`, and `deny_unknown_fields`. The sixteen variants are `scan`, `filter`, `project`, `aggregate`, `sort`, `limit`, `hash_join`, `asof_join`, `range_join`, `distinct`, `window`, `union`, `unnest`, `row_id`, `unpivot`, and `sample`.
- **`Expr`**, in [`crates/bc-expr/src/lib.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-expr/src/lib.rs), is the scalar tree carried inside `RelOp` nodes, the Rust type the Python {py:class}`Expr <batcher.plan.expr_ir.core.Expr>` lowers to. It uses `tag = "e"` and has fifty-five variants, from `col`, `lit`, and `binary` through `case`, `str`, `date`, and the rest of the function surface.

:::{important}
There is exactly one of each. The interpreter, the JIT, the runtime primitives, and the distributed path all consume the same `Expr` and the same `RelOp`, which makes semantic parity between tiers structural rather than promised.
:::

Python's tag strings live as constants in [`python/batcher/plan/ir_tags.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/plan/ir_tags.py), so a typo is an `AttributeError` rather than a silently wrong tag.

## What a plan looks like

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

:::{dropdown} The logical document for that query
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

Three things to notice:

- **`source_id` is an index, not a path.** It binds to `sources[0]`, the batches passed alongside the plan. The Python `io` layer resolves splits, pushes predicates to the reader, and applies schema evolution before the engine sees anything.
- **There is no schema in the document.** Types come from the Arrow input, so a declared schema and an actual batch can't disagree.
- **Names are already resolved.** A `hash_join` carries an explicit `output: Vec<JoinOutputCol>` naming each output column's side, source name, and result name.

![The same plan before and after Kyber's pushdown phase. As written: Scan orders and Scan customers each read every row and every column, a Join on c_id consumes every matching row, a Filter on o_date then drops rows above the join, and a Project keeps o_id and c_name. As Kyber leaves it: the orders scan reads three columns, o_id, c_id and o_date, and carries the predicate as a source hint so the reader may skip row groups; the customers scan reads two, c_id and c_name; and the Filter now sits below the join, so the discarded rows are gone before the build. Two rules the picture pins. The Filter node is still in the plan afterwards, because source_predicates is only a hint to the connector and a source that translates none of the predicate, or only part of it, must still be correct. And only a conjunct naming one side of the join moves below it; one naming both stays above.](/_static/diagrams/pushdown_before_after.svg)

## Expressions

An expression lowers on its own, and literals are tagged by type so the engine never guesses whether `1` means `1i64` or `1.0f64`:

```python
import batcher as bt

ir = ((bt.col("x") * 2 + 1) > bt.col("y")).to_ir()
print(ir["e"], ir["op"])          # binary gt
print(ir["left"]["right"])        # {'e': 'lit', 'value': {'int': 1}}
print(bt.lit(1.5).to_ir())        # {'e': 'lit', 'value': {'float': 1.5}}
```

## Physical hints ride along

Some fields are the optimizer talking to the executor:

| Hint | Set by | Effect |
|---|---|---|
| `Sort { limit }` | fusing a downstream `Limit` | the sort becomes a top-N, computed by a partial sort rather than a full one |
| `HashJoin { strategy }` | Kyber, from cardinality | `hash` (shuffle), `broadcast`, or `sort_merge`. All three produce the same relation; only the data movement differs |
| `Window { rank_limit }` | fusing `QUALIFY rn <= k` | a per-partition top-N instead of a full ranking |
| `Aggregate { group_keys: [] }` | the planner | an empty key list is a global aggregate, not an error |

A wrong `strategy` is slow, never wrong. Each hint field has `#[serde(default)]`, so the optimizer can learn to emit a new hint without a coordinated flag day.

## Keeping the two sides in step

:::{important}
Changing the IR is a **two-sided change in a single commit**. `deny_unknown_fields` turns a field Rust doesn't know into a loud parse error at the boundary, rather than a silently ignored instruction.
:::

![One JSON document, written once and read once. Python's to_ir() writes the tag string held in plan/ir_tags.py, the wire carries it as an "op" key holding hash_join, and Rust serde must accept it into bc_ir::RelOp under deny_unknown_fields. Sixteen RelOp tags and fifty-five Expr tags today, plus nineteen function vocabularies underneath them. Below the spine, tools/lint_ir_contract.py reads the Python class on one side and the Rust enum body on the other and compares whole vocabularies. It runs no query, so a tag that no differential test names is still checked. The two drift directions fail in completely different ways: a tag Python emits that Rust rejects makes the plan fail to deserialize, which is loud but only for a query that uses that tag, while a tag Rust accepts that Python never emits leaves an engine capability unreachable, which is silent and raises nothing. Both sides change in one commit, or neither does.](/_static/diagrams/ir_wire_contract.svg)

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

Adding a variant to `bc_ir::RelOp` means adding its tag to `plan/ir_tags.py`, its `to_ir()` to the `LogicalPlan` node, and a test that the Python shape deserializes in Rust, all in the same commit. [`tools/lint_ir_contract.py`](https://github.com/stephenoffer/batcher/blob/main/tools/lint_ir_contract.py) compares the two vocabularies without running a query.

## What the IR leaves out

- **No schema and no stage.** Types come from Arrow, and the distributed path composes stages in Python ([`python/batcher/dist/`](https://github.com/stephenoffer/batcher/tree/main/python/batcher/dist)) out of ordinary plans, so the engine sees the same document shape on one node or a hundred.
- **No shared subtrees.** The IR is a tree with no CTE node. On the single-node path, `kyber/common_subplan.py` picks repeated subtrees worth computing once and `api/subplan_reuse.py` rewrites each appearance into a `Scan` over the result. A distributed plan that references a subplan twice computes it twice.

:::{dropdown} Where the code lives
| Piece | File |
|---|---|
| `RelOp` + physical hints | [`crates/bc-ir/src/lib.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-ir/src/lib.rs) |
| `Expr` + literals | [`crates/bc-expr/src/lib.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-expr/src/lib.rs) |
| `EngineConfig` (morsel size, parallelism, tuning) | [`crates/bc-ir/src/engine_config.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-ir/src/engine_config.rs) |
| Python tag vocabulary | [`python/batcher/plan/ir_tags.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/plan/ir_tags.py) |
| Per-node `to_ir()` | [`python/batcher/plan/logical/`](https://github.com/stephenoffer/batcher/tree/main/python/batcher/plan/logical) |
| Document assembly | [`python/batcher/plan/physical.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/plan/physical.py) |
| Deserialization at the boundary | [`crates/bc-py/src/lib.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-py/src/lib.rs) |
:::

## See also

- {doc}`Architecture </architecture/index>`: why the control plane and the data plane meet at a document.
- {doc}`Execution engine </architecture/internals/execution>`: what happens to the `RelOp` tree after it lands.
- {doc}`Kyber </architecture/internals/kyber>`: the passes that set the physical hints above.
- {doc}`Reading a plan </user-guide/operate/tuning/explain-plans>`: the same tree, rendered for humans.
- {doc}`Performance </user-guide/operate/tuning/performance>`: what to do when the plan isn't the one you wanted.
- {doc}`TPC-H benchmarks </benchmarks/results/tpch>`: the query shapes these hints are tuned against.
- {doc}`Query lifecycle </architecture/deep-dives/query/query-lifecycle>`: where the document is produced and consumed.
- {doc}`Expression evaluation </architecture/deep-dives/query/expression-evaluation>`: what the engine does with an `Expr`.
- {doc}`Join algorithms </architecture/deep-dives/operators/join-algorithms>`: what the `strategy` hint selects.
