# Configuring the engine: options, scoped overrides, and profiles

Configuration is a value, not global mutable state you have to remember to undo. ``option_context`` and ``config_context`` scope an override to a block, so a memory-tight step cannot leak its settings into the rest of the program.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/operations/configuration.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/operations/configuration.py
```
