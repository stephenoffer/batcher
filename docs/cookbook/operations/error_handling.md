# The exception hierarchy: catching the failure you meant to catch

Every error the engine raises descends from ``BatcherError``, so a pipeline can catch that one type at its boundary. The specific subclasses let you distinguish a user mistake (``PlanError``) from an environment problem (``IOError``) from a missing extra (``MissingDependencyError``), which is the difference between retrying and giving up.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/operations/error_handling.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/operations/error_handling.py
```
