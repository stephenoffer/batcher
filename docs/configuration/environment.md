# Environment variables

Batcher reads two environment-driven config layers once when the package is
imported: `BATCHER_*` variables and an optional JSON file at `BATCHER_CONFIG_FILE`.
Both overlay onto the built-in defaults and can be reproduced explicitly with
`Config.from_env` and `Config.from_file`.

## BATCHER_ variables

Each variable maps to one config field by path:
`BATCHER_<SECTION>_<FIELD>`. Nested sub-sections compose by appending another
segment. Names are uppercased section and field names.

| Variable | Field |
|----------|-------|
| `BATCHER_EXECUTION_PARALLELISM` | `config.execution.parallelism` |
| `BATCHER_EXECUTION_MORSEL_ROWS` | `config.execution.morsel_rows` |
| `BATCHER_MEMORY_SOFT_LIMIT` | `config.memory.soft_limit` |
| `BATCHER_MEMORY_MAX_MEMORY_BYTES` | `config.memory.max_memory_bytes` |
| `BATCHER_FLOW_CONTROL_DEFAULT_CREDITS` | `config.flow_control.default_credits` |
| `BATCHER_OPTIMIZER_REOPTIMIZE_ERROR` | `config.optimizer.reoptimize_error` |
| `BATCHER_OPTIMIZER_CARDINALITY_EQ_SELECTIVITY` | `config.optimizer.cardinality.eq_selectivity` |
| `BATCHER_PID_KP` | `config.pid.kp` |
| `BATCHER_METADATA_BACKEND` | `config.metadata.backend` |

Values are coerced to the field's type. Integers and floats are parsed directly;
booleans accept `1`, `true`, `yes`, or `on` (case-insensitive) as true.

```bash
# docs: skip
export BATCHER_EXECUTION_PARALLELISM=8
export BATCHER_MEMORY_SOFT_LIMIT=0.75
python my_pipeline.py
```

You can reproduce the overlay in code by passing an explicit environment mapping.
`Config.from_env` returns a new `Config` and does not mutate its base.

```python
from batcher import Config

cfg = Config.from_env(
    {"BATCHER_EXECUTION_PARALLELISM": "8", "BATCHER_MEMORY_SOFT_LIMIT": "0.75"}
)
print((cfg.execution.parallelism, cfg.memory.soft_limit))
# (8, 0.75)
```

A nested sub-section field composes its path the same way:

```python
from batcher import Config

cfg = Config.from_env({"BATCHER_OPTIMIZER_CARDINALITY_EQ_SELECTIVITY": "0.05"})
print(cfg.optimizer.cardinality.eq_selectivity)
# 0.05
```

## BATCHER_CONFIG_FILE

Set `BATCHER_CONFIG_FILE` to the path of a JSON document whose structure mirrors the
section layout. It is overlaid below the `BATCHER_*` variables.

```bash
# docs: skip
export BATCHER_CONFIG_FILE=/etc/batcher/config.json
python my_pipeline.py
```

```json
{
  "execution": { "morsel_rows": 4096, "parallelism": 4 },
  "memory": { "soft_limit": 0.80 },
  "optimizer": { "cardinality": { "eq_selectivity": 0.05 } }
}
```

`Config.from_file` applies the same overlay programmatically:

```python
# docs: skip
from batcher import Config

cfg = Config.from_file("/etc/batcher/config.json")
```

## Precedence

The two layers here sit in the middle of the resolution order (highest first):

1. `config_context(...)`
2. `set_config(...)`
3. `BATCHER_*` environment variables
4. `BATCHER_CONFIG_FILE` JSON
5. Built-in defaults

So a `BATCHER_*` variable overrides a value set in `BATCHER_CONFIG_FILE`, and a
runtime `set_config` or `config_context` overrides both. See
[index](index.md) for the runtime entry points.
