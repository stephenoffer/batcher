# Configuration

This page describes how to build a Batcher {py:class}`Config <batcher.Config>`, make it active, and load one from the environment or a file.

The pages beneath it are the reference: {doc}`options` is the field-by-field listing, {doc}`distributed-options` covers the cluster and shuffle knobs, {doc}`accelerator` covers the GPU and device settings, {doc}`fault-tolerance` covers what happens when nodes and devices fail underneath a running job, {doc}`environment` covers the `BATCHER_*` variables and the JSON file format, and {doc}`profiles` shows worked configurations for common deployments.

Most of the time you don't configure Batcher at all. The defaults are tuned to saturate your cores and stay within memory on their own. When you do need to tune a memory limit, the thread count, or how aggressively the engine spills, every knob lives on one `Config` object. It's a typed, immutable dataclass grouped by concern: `execution`, `memory`, `flow_control`, `optimizer`, `pid`, `metadata`, `distributed`, `observability`, `accelerator`, and `fault_tolerance`. There's no global mutable state and no dict of loose keys. You build a `Config`, then make it active.

```python
import batcher as bt
from batcher import Config, set_config, config_context

cfg = Config()
print(cfg.execution.morsel_rows)
# 16384
```

## Building a config

`Config` and its sections are frozen dataclasses, so you derive new ones rather than
mutating in place. {py:meth}`Config.replace(...) <batcher.Config.replace>` swaps whole sections; `dataclasses.replace`
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

{py:func}`set_config(Config(...)) <batcher.set_config>` installs a `Config` process-wide until it is changed
again. {py:func}`config_context(Config(...)) <batcher.config_context>` activates one only for the duration of a `with`
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

## Setting one option by name

Building a whole `Config` to change one number is a lot of ceremony. Every tunable also has a dotted name, and `set_option` / `get_option` address it directly. This is the same API shape as `pandas.set_option` and `spark.conf.set`, and it goes through the same validation as `set_config`.

```python
from batcher.config import get_option, reset_option, set_option

set_option("execution.morsel_rows", 4096)
print(get_option("execution.morsel_rows"))
# 4096

reset_option("execution.morsel_rows")
print(get_option("execution.morsel_rows"))
# 16384
```

A trailing segment works too when it is unambiguous, so `get_option("morsel_rows")` finds `execution.morsel_rows`. Misspell a name and the error suggests the closest real ones rather than failing silently.

`option_context` is the scoped form. It restores the previous values on exit, including when the block raises, and it nests:

```python
from batcher.config import get_option, option_context

with option_context("execution.morsel_rows", 1024, "optimizer.build_bloom_index", True):
    print(get_option("execution.morsel_rows"))
# 1024

print(get_option("execution.morsel_rows"))
# 16384
```

`reset_option` takes a glob, so `reset_option("execution.*")` clears a section and `reset_option()` on its own resets everything.

To find an option without reading the source, search the names. `option_names` returns the matching paths and `describe_options` prints them with their current values, flagging anything that differs from the default:

```python
from batcher.config import describe_options, option_names

print(len(option_names()) > 50)
# True

print("memory.spill_dir" in describe_options("spill"))
# True
```

Finally, {py:meth}`Config.non_defaults() <batcher.Config.non_defaults>` answers "what is actually set here?" when a job behaves differently on two machines. Its `repr` shows the same thing, so printing a config is useful rather than a wall of 180 fields.

```python
import dataclasses
from batcher import Config

cfg = Config()
cfg = cfg.replace(execution=dataclasses.replace(cfg.execution, morsel_rows=4096))
print(cfg.non_defaults())
# {'execution.morsel_rows': 4096}
```

## Loading from the environment or a file

{py:meth}`Config.from_env() <batcher.Config.from_env>` overlays `BATCHER_*` environment variables onto a base config.
{py:meth}`Config.from_file(path) <batcher.Config.from_file>` overlays a document, choosing the parser from the suffix:
JSON, TOML, or YAML. {py:meth}`Config.from_toml <batcher.Config.from_toml>` and {py:meth}`Config.from_yaml <batcher.Config.from_yaml>` force a format when
the filename doesn't carry one. All of them return a new `Config` and leave their
input untouched. See {doc}`environment` for variable naming and the file format.

`env_var_names()` prints the mapping the other direction, from every environment
variable to the option it sets, which is what you want when writing a deployment
manifest:

```python
from batcher.config import env_var_names

print(env_var_names()["BATCHER_EXECUTION_MORSEL_ROWS"])
# execution.morsel_rows
```

{py:meth}`Config.from_dict <batcher.Config.from_dict>` and {py:meth}`Config.to_dict <batcher.Config.to_dict>` are the in-memory pair. {py:meth}`to_dict <batcher.Dataset.to_dict>` produces
plain JSON-encodable data, so a config travels as part of a job manifest, and
`only_non_default=True` emits the smallest document that reproduces it. The
standalone `config_to_dict` function does the same for callers that would rather not
reach through the object.

```python
from batcher import Config
from batcher.config import config_to_dict

resolved = Config.from_dict(Config().to_dict())
print(Config.from_dict(resolved.to_dict()) == resolved)
# True

print(config_to_dict(Config())["execution"]["morsel_rows"])
# 16384
```

{py:func}`from_dict <batcher.from_dict>` re-runs the same environment resolution every entry point does, which
auto-detects a spot node or an autoscaling cluster. That means a config captured on
one machine can legitimately differ from raw defaults when reloaded on another.
Reloading an already-resolved config is idempotent, which is the property to rely on.

## Precedence

When the engine resolves the active config, the layers apply highest first:

1. `config_context(...)`, the innermost active context.
1. `set_config(...)`, process-wide.
1. `BATCHER_*` environment variables.
1. A JSON file named by `BATCHER_CONFIG_FILE`.
1. Built-in defaults.

The environment and file layers are read once when `batcher` is imported.
`set_config` and `config_context` override them at runtime.

## See also

- {doc}`/user-guide/operate/tuning/performance`: which of these options matter when a query is slow.
- {doc}`/user-guide/operate/tuning/caching`: the result cache and the options that bound it.
- {doc}`/architecture/deep-dives/memory/buffer-pool`: what the memory options actually govern.
- {doc}`/api/operations/configuration`: the configuration objects as an API surface.

```{toctree}
:hidden:

options
distributed-options
accelerator
fault-tolerance
environment
profiles
```
