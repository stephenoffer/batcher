# Configuration API reference

The public configuration surface. For the full field-by-field reference of every
section, see [configuration/options](../configuration/options.md).

```python
from batcher import Config, set_config, config_context
```

## Config

`Config()` is a frozen dataclass composed of typed sections, one per concern.

```python
import batcher as bt
from batcher import Config

cfg = Config()
print(list(cfg.__dataclass_fields__))
# ['execution', 'memory', 'flow_control', 'optimizer', 'pid', 'metadata', 'distributed']
```

Sections are themselves frozen dataclasses. Read fields directly; derive new configs
to change them.

| Section | Concern |
|---------|---------|
| `config.execution` | Parallelism, morsel size, file splits |
| `config.memory` | Buffer-pool envelope and spill thresholds |
| `config.flow_control` | Credit-based shuffle backpressure |
| `config.optimizer` | Kyber planning thresholds, cost model, cardinality |
| `config.pid` | Adaptive batch-size controller gains |
| `config.metadata` | Learned-stats backend and decay |

### Config.replace

`Config.replace(**section_overrides)` returns a new `Config` with whole sections
replaced. To change a single field within a section, pass a `dataclasses.replace` of
that section. The individual sections do not expose a `.replace` method.

```python
import dataclasses
from batcher import Config

base = Config()
cfg = base.replace(
    execution=dataclasses.replace(base.execution, parallelism=4),
)
print(cfg.execution.parallelism)
# 4
```

### Config.from_env

`Config.from_env(environ=None, base=None)` overlays `BATCHER_*` environment variables
onto `base` (defaults when omitted) and returns a new `Config`. Pass an explicit
mapping to overlay specific variables.

```python
from batcher import Config

cfg = Config.from_env({"BATCHER_EXECUTION_PARALLELISM": "8"})
print(cfg.execution.parallelism)
# 8
```

### Config.from_file

`Config.from_file(path, base=None)` overlays a JSON document of nested section
overrides onto `base` and returns a new `Config`. The JSON mirrors the section
structure. See [configuration/environment](../configuration/environment.md) for the
format.

```python
# docs: skip
from batcher import Config

cfg = Config.from_file("/etc/batcher/config.json")
```

## set_config

`set_config(config)` installs a `Config` as the process-wide active configuration.
It takes a `Config` object, not keyword fields, and sits above the environment and
file layers but below `config_context`.

```python
from batcher import Config, set_config

set_config(Config())
```

## config_context

`config_context(config)` is a context manager that activates a `Config` for the
duration of a `with` block and restores the previous one on exit. It is the highest
precedence layer.

```python
from batcher import Config, config_context

with config_context(Config()):
    out = bt.from_pydict({"x": [1, 2, 3]}).to_pydict()

print(out)
# {'x': [1, 2, 3]}
```

## Precedence

Highest first: `config_context` > `set_config` > `BATCHER_*` env vars >
`BATCHER_CONFIG_FILE` JSON > defaults. The environment and file layers are read once
at import; the runtime entry points override them. Full discussion in
[configuration/index](../configuration/index.md).
