# Configuration

Batcher's runtime behavior is controlled by a single frozen `Config` object. It is
an immutable dataclass made of typed sections, one per concern: `execution`,
`memory`, `flow_control`, `optimizer`, `pid`, and `metadata`. There are no global
mutable settings and no dict of loose keys; you build a `Config`, then make it
active.

```python
import batcher as bt
from batcher import Config, set_config, config_context

cfg = Config()
print(cfg.execution.morsel_rows)
# 16384
```

## Building a config

`Config` and its sections are frozen dataclasses, so you derive new ones rather than
mutating in place. `Config.replace(...)` swaps whole sections; `dataclasses.replace`
changes a field within a section.

```python
import dataclasses
from batcher import Config

base = Config()
cfg = base.replace(
    execution=dataclasses.replace(base.execution, parallelism=4),
    memory=dataclasses.replace(base.memory, soft_limit=0.75),
)

print((cfg.execution.parallelism, cfg.memory.soft_limit))
# (4, 0.75)
```

The individual sections have no `.replace` method of their own; use
`dataclasses.replace(section, field=value)` for field-level edits.

## Making a config active

`set_config(Config(...))` installs a `Config` process-wide until it is changed
again. `config_context(Config(...))` activates one only for the duration of a `with`
block and restores the previous config on exit. Both take a `Config` object, not
keyword fields.

```python
from batcher import Config, set_config, config_context

set_config(cfg)  # process-wide

with config_context(Config()):
    result = bt.from_pydict({"x": [1, 2, 3]}).to_pydict()

print(result)
# {'x': [1, 2, 3]}
```

## Loading from the environment or a file

`Config.from_env()` overlays `BATCHER_*` environment variables onto a base config.
`Config.from_file(path)` overlays a JSON document. Both return a new `Config` and
leave their input untouched. See [environment](environment.md) for variable naming
and the file format.

## Precedence

When the engine resolves the active config, layers apply highest first:

1. `config_context(...)` (the innermost active context)
2. `set_config(...)` (process-wide)
3. `BATCHER_*` environment variables
4. A JSON file named by `BATCHER_CONFIG_FILE`
5. Built-in defaults

The environment and file layers are read once when `batcher` is imported.
`set_config` and `config_context` override them at runtime.

## In this section

```{toctree}
:maxdepth: 1

options
environment
profiles
```
