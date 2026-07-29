# Reading a plan, timing a query, and checking what the engine actually ran

``explain()`` shows the optimized plan, which is where you confirm a predicate really was pushed into the scan. Reading the plan is faster than guessing, and it is the only way to tell a fused pipeline from three separate passes.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/operations/inspecting_a_query.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/operations/inspecting_a_query.py
```
