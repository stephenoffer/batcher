# The query, end to end

Follow one query from Python down to Arrow and back.

- {doc}`Query lifecycle </architecture/deep-dives/query/query-lifecycle>`: what happens between {py:meth}`collect() <batcher.Dataset.collect>` and your rows.
- {doc}`The plan IR </architecture/deep-dives/query/plan-ir>`: the JSON wire contract between the control plane and the engine.
- {doc}`Expression evaluation </architecture/deep-dives/query/expression-evaluation>`: one {py:class}`Expr <batcher.plan.expr_ir.core.Expr>`, vectorized over Arrow.
- {doc}`JIT compilation </architecture/deep-dives/query/jit-compilation>`: the Cranelift fast path, and why it must fall back rather than diverge.
- {doc}`Physical properties </architecture/deep-dives/query/physical-properties>`: the ordering and partitioning a plan delivers, and the sorts and shuffles knowing them removes.

```{toctree}
:hidden:

query-lifecycle
plan-ir
expression-evaluation
jit-compilation
physical-properties
```
