# Transforming inside a list column, without exploding it first

``explode`` then ``group_by`` re-collects is the expensive way to map over list elements. ``.list.transform`` and ``.list.filter`` do it in place, which keeps the row count fixed and avoids the shuffle a regroup would cost. Both take an *expression* over ``bt.element()`` -- the current element -- not a Python lambda, so the body runs in Rust.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/expressions/lists_transforms.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/lists_transforms.py
```
