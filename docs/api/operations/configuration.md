# Configuration API reference

This page covers the public configuration surface: the {py:class}`Config <batcher.Config>` dataclass, the two entry points that install one, and how the layers combine. For the field-by-field reference, see {doc}`configuration/options </configuration/options>`. The `distributed`, `accelerator`, and `fault_tolerance` sections are large enough to have pages of their own: {doc}`/configuration/distributed-options`, {doc}`/configuration/accelerator`, and {doc}`/configuration/fault-tolerance`.

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
# ['execution', 'memory', 'flow_control', 'streaming', 'optimizer', 'pid', 'metadata',
#  'distributed', 'observability', 'governance', 'tenant', 'accelerator', 'fault_tolerance']
```

Sections are frozen dataclasses too. Read their fields directly, and derive a new config
to change one. Each section is exported under its own class name, so you can build one and
slot it into {py:meth}`Config.replace <batcher.Config.replace>`. The table below gives the
attribute, the class, and the concern. The {doc}`configuration section </configuration/index>`
gives the fields.

| Section | Class | Covers |
| --- | --- | --- |
| `config.execution` | {py:class}`ExecutionConfig <batcher.ExecutionConfig>` | parallelism, morsel size, file-split size, CPUs per task |
| `config.memory` | {py:class}`MemoryConfig <batcher.MemoryConfig>` | buffer-pool envelope, soft/hard limits, and spill thresholds |
| `config.flow_control` | {py:class}`FlowControlConfig <batcher.FlowControlConfig>` | credit-based shuffle backpressure and AIMD credit tuning |
| `config.streaming` | {py:class}`StreamingConfig <batcher.StreamingConfig>` | the micro-batch loop's idle cadence and progress history |
| `config.optimizer` | {py:class}`OptimizerConfig <batcher.OptimizerConfig>` | Kyber planning thresholds, cost model, and cardinality defaults |
| `config.pid` | {py:class}`PIDConfig <batcher.PIDConfig>` | gains for the adaptive batch-size PID controller |
| `config.metadata` | {py:class}`MetadataConfig <batcher.MetadataConfig>` | learned-stats backend, URI, and decay rate |
| `config.distributed` | `DistributedConfig` | how the engine attaches to and shuffles across a Ray cluster |
| `config.observability` | `ObservabilityConfig` | the `batcher.*` loggers and the per-query event log |
| `config.governance` | {py:class}`GovernanceConfig <batcher.GovernanceConfig>` | whether row/column policy is advisory or mandatory |
| `config.tenant` | {py:class}`TenantConfig <batcher.TenantConfig>` | which tenant a scope's work belongs to, and its share |
| `config.accelerator` | {py:class}`AcceleratorConfig <batcher.config.AcceleratorConfig>` | GPU placement, VRAM headroom, MIG preference, and KV-cache sizing |
| `config.fault_tolerance` | {py:class}`FaultToleranceConfig <batcher.config.FaultToleranceConfig>` | the retry budget and what happens when a device corrupts rather than loses |

Two of those sections nest further, and the inner classes are exported as well.

| Section | Class | Covers |
| --- | --- | --- |
| `config.accelerator.energy` | {py:class}`EnergyConfig <batcher.config.EnergyConfig>` | the site's power budget, energy price, grid carbon intensity, and PUE |
| `config.accelerator.health` | {py:class}`DeviceHealthConfig <batcher.config.DeviceHealthConfig>` | when a device is derated or taken out of rotation |
| `config.accelerator.memory` | {py:class}`DeviceMemoryConfig <batcher.config.DeviceMemoryConfig>` | the device allocator, its pool sizing, and host spilling |
| `config.fault_tolerance.quarantine` | {py:class}`QuarantineConfig <batcher.config.QuarantineConfig>` | when repeated task failures take a node or device out of rotation |

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

{py:meth}`Config.from_env(environ=None, base=None) <batcher.Config.from_env>` overlays `BATCHER_*` environment variables
onto `base` (defaults when omitted) and returns a new `Config`. Pass an explicit
mapping to overlay specific variables.

```python
from batcher import Config

cfg = Config.from_env({"BATCHER_EXECUTION_PARALLELISM": "8"})
print(cfg.execution.parallelism)
# 8
```

### Config.from_file

{py:meth}`Config.from_file(path, base=None) <batcher.Config.from_file>` overlays a JSON document of nested section
overrides onto `base` and returns a new `Config`. The JSON mirrors the section
structure. See {doc}`configuration/environment </configuration/environment>` for the
format.

```python
# docs: skip
from batcher import Config

cfg = Config.from_file("/etc/batcher/config.json")
```

### Config.validate and Config.engine_config_json

{py:meth}`Config.validate() <batcher.Config.validate>` checks the configuration and raises {py:exc}`ConfigError <batcher.ConfigError>` on a bad value. {py:meth}`Config.engine_config_json() <batcher.Config.engine_config_json>` serializes the execution knobs the data plane needs. That JSON is what the Rust engine receives.

## set_config

{py:func}`set_config(config) <batcher.set_config>` installs a `Config` as the process-wide active configuration.
It takes a `Config` object, not keyword fields, and sits above the environment and
file layers but below {py:func}`config_context <batcher.config_context>`.

```python
from batcher import Config, set_config

set_config(Config())
```

## config_context

`config_context(config)` is a context manager that activates a `Config` for the
duration of a `with` block and restores the previous one on exit. It's the
highest-precedence layer. Both it and `set_config` validate before they install and write
the same `ContextVar`, so a bad tunable raises `ConfigError` at the call that set it
rather than surfacing later as a confusing runtime failure, and a scoped override nests
correctly under threads and asyncio instead of leaking past its block.

```python
from batcher import Config, config_context

with config_context(Config()):
    out = bt.from_pydict({"x": [1, 2, 3]}).to_pydict()

print(out)
# {'x': [1, 2, 3]}
```

## Precedence

Highest first: `config_context` > `set_config` > `BATCHER_*` env vars > `BATCHER_CONFIG_FILE` JSON > defaults. The environment and file layers are read once, at import. The two runtime entry points override whatever they found. For the reasoning behind that order, see {doc}`configuration/index </configuration/index>`.

## See also

- {doc}`/configuration/index`: the runtime entry points, in prose.
- {doc}`/configuration/options`: every field, with its default and unit.
- {doc}`/configuration/environment`: the `BATCHER_*` spelling of the same settings.
- {doc}`/configuration/profiles`: ready-made configurations for common machine shapes.
- {doc}`/user-guide/operate/tuning/performance`: which fields are worth changing for a slow query.
- {doc}`/cookbook/operations/configuration`: options, scoped overrides, and profiles, as a runnable script.
