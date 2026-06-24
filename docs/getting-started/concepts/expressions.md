# Expressions run in Rust

Column work is expressed with `Expr` values built from
{py:obj}`bt.col(...) <batcher.col>` and {py:obj}`bt.lit(...) <batcher.lit>`. An
expression is a *description* of a computation, not a Python loop. When the plan
executes, the expression is evaluated in the Rust data plane over whole Arrow
batches — vectorized, and compiled to machine code where possible — never row by row
in Python.

```python
import batcher as bt

ds = bt.from_pydict({"x": [1, 2, 3, 4]})

total = bt.col("x") * bt.lit(10)
print(ds.select(scaled=total).to_pydict())
# {'scaled': [10, 20, 30, 40]}
```

This is why there are no per-row Python callbacks in the hot path: the control plane
never touches a tuple. Operators (`+`, `==`, `&`), methods (`.sum()`, `.cast(...)`),
and the accessor namespaces (`.str`, `.dt`, `.list`) all build up the same `Expr`
tree that the engine evaluates.

The one place user Python sees data is `map_batches`, which hands you a whole Arrow
batch (not a row) so the work still happens in bulk, off the per-row path.
