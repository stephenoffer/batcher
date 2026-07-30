# Configuration API reference

This page covers the public configuration surface: the `Config` dataclass, the two entry points that install one, and how the layers combine. For the field-by-field reference of every section, see {doc}`configuration/options <../configuration/options>`.

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
# ['execution', 'memory', 'flow_control', 'optimizer', 'pid', 'metadata', 'distributed', 'observability']
```

Sections are themselves frozen dataclasses. Read fields directly, and derive new configs to change them. Each section covers one concern:

| Section | Concern |
|---------|---------|
| `config.execution` | Parallelism, morsel size, file splits |
| `config.memory` | Buffer-pool envelope and spill thresholds |
| `config.flow_control` | Credit-based shuffle backpressure |
| `config.optimizer` | Kyber planning thresholds, cost model, cardinality |
| `config.pid` | Adaptive batch-size controller gains |
| `config.metadata` | Learned-stats backend and decay |

### Section classes

The section dataclasses are exported so you can construct one and slot it into
`Config.replace`. Their fields are documented in
{doc}`configuration/options <../configuration/options>`; each is summarized here.

| Class | Configures |
| --- | --- |
| `ExecutionConfig` | parallelism, morsel size, file-split size, CPUs per task |
| `MemoryConfig` | buffer-pool envelope, soft/hard limits, and spill thresholds |
| `FlowControlConfig` | credit-based shuffle backpressure and AIMD credit tuning |
| `OptimizerConfig` | Kyber planning thresholds, cost model, and cardinality defaults |
| `PIDConfig` | gains for the adaptive batch-size PID controller |
| `MetadataConfig` | learned-stats backend, URI, and decay rate |
| `GovernanceConfig` | whether row/column policy is advisory or mandatory |
| `TenantConfig` | which tenant a scope's work belongs to, and its share |

### Config.replace

`Config.replace(**section_overrides)` returns a new `Config` with whole sections
replaced. To change a single field within a section, pass a `dataclasses.replace` of
that section. The individual sections don't expose a `.replace` method.

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
structure. See {doc}`configuration/environment <../configuration/environment>` for the
format.

```python
# docs: skip
from batcher import Config

cfg = Config.from_file("/etc/batcher/config.json")
```

### Config.validate and Config.engine_config_json

`Config.validate()` checks the configuration and raises `ConfigError` on a bad value. `Config.engine_config_json()` serializes the Rust-relevant execution knobs for the data plane, which is the JSON the engine receives.

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
duration of a `with` block and restores the previous one on exit. It's the highest-precedence layer.

```python
from batcher import Config, config_context

with config_context(Config()):
    out = bt.from_pydict({"x": [1, 2, 3]}).to_pydict()

print(out)
# {'x': [1, 2, 3]}
```

## Precedence

Highest first: `config_context` > `set_config` > `BATCHER_*` env vars > `BATCHER_CONFIG_FILE` JSON > defaults. The environment and file layers are read once at import, and the runtime entry points override them. See {doc}`configuration/index <../configuration/index>` for the full discussion.

## See also

- {doc}`../configuration/index`: the runtime entry points, in prose.
- {doc}`../configuration/options`: every field, with its default and unit.
- {doc}`../configuration/environment`: the `BATCHER_*` spelling of the same settings.
- {doc}`../configuration/profiles`: ready-made configurations for common machine shapes.
- {doc}`../user-guide/performance`: which fields are worth changing for a slow query.
