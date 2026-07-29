# Branching inside an expression: when/then/otherwise, and the SQL null helpers

``bt.when(...).then(...).otherwise(...)`` is the columnar ``if``. Chain ``.when()`` for more branches; the first matching branch wins, exactly like SQL ``CASE``. Because it is an expression it runs in Rust, so a five-way bucketing is still one pass.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/expressions/conditionals.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/conditionals.py
```
