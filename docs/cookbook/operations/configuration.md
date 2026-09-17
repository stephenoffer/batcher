# Configuration

Configuration in Batcher is a value you pass. Reach for `option_context` when one step needs a different setting and `config_context` when a whole block does. Both restore the previous value on the way out, so a memory-tight step cannot leak its settings into the rest of the program.

The script lists and reads options, scopes an override, and builds a variant of the whole configuration with `active_config().replace(...)`. `set_option` is the global escape hatch, and it needs a matching `reset_option`.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/operations/configuration.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/operations/configuration.py
```

## See also

- {doc}`environment`: what is installed, what the engine sees, and what to paste into a bug report.
- {doc}`error_handling`: catching the failure you meant to catch.
- {doc}`/user-guide/operate/tuning/performance`: measuring and tuning a query that is correct but slow.
- {doc}`/user-guide/operate/running/observability`: what the engine records about a run, and where.
