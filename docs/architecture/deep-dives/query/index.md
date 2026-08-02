# The query, end to end

Follow one query from Python down to Arrow and back.

- {doc}`Query lifecycle </architecture/deep-dives/query/query-lifecycle>`: what happens between `collect()` and your rows.
- {doc}`The plan IR </architecture/deep-dives/query/plan-ir>`: the JSON wire contract between the control plane and the engine.
- {doc}`Expression evaluation </architecture/deep-dives/query/expression-evaluation>`: one `Expr`, vectorized over Arrow.
- {doc}`JIT compilation </architecture/deep-dives/query/jit-compilation>`: the Cranelift fast path, and why it must fall back rather than diverge.

```{toctree}
:hidden:

query-lifecycle
plan-ir
expression-evaluation
jit-compilation
```
