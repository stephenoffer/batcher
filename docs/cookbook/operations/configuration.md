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


## The config the engine is handed, per query

{py:meth}`Config.engine_config_json_with <batcher.Config>` renders the JSON configuration
the Rust data plane receives, with two per-query overrides folded in. It is the boundary
document: everything the engine knows about how to run a plan is in it, so printing it is
the direct way to check that a setting reached the other side.

`op_budgets` assigns a memory budget to individual operators by id, which is how Carbonite
gives a specific pipeline breaker more room than the query-wide envelope would.
`prefer_materializing_aggregate` asks the executor for the materializing aggregate rather
than the streaming one.

```python
import json

from batcher.config import Config

config = Config()
base = json.loads(config.engine_config_json_with({}))
print(sorted(base)[:4])

tuned = json.loads(
    config.engine_config_json_with({0: 1 << 20}, prefer_materializing_aggregate=True)
)
print(tuned["op_budgets"], tuned["prefer_materializing_aggregate"])
print(sorted(set(tuned) - set(base)))
```

An override you do not pass is **absent** from the rendering rather than present and null,
so the two calls differ in exactly the two fields you set and in nothing else. That is the
property worth checking after changing a setting: if a field you expected to appear is
missing from both renderings, the setting is not reaching the engine.

## See also

- {doc}`environment`: what is installed, what the engine sees, and what to paste into a bug report.
- {doc}`error_handling`: catching the failure you meant to catch.
- {doc}`/user-guide/operate/tuning/performance`: measuring and tuning a query that is correct but slow.
- {doc}`/user-guide/operate/running/observability`: what the engine records about a run, and where.
