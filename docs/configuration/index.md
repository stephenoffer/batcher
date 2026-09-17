# Configuration

This page describes how to build a Batcher {py:class}`Config <batcher.Config>`, make it active, and load one from the environment or a file.

Most of the time you don't configure Batcher at all. The engine senses its cores, its memory envelope, and the cluster it's attached to, and sizes itself from what it finds. When you do want a memory cap, a thread count, or a different spill directory, every knob lives on one typed, immutable `Config` object with validated fields. It's grouped by concern into `execution`, `memory`, `flow_control`, `streaming`, `optimizer`, `pid`, `metadata`, `distributed`, `observability`, `governance`, `tenant`, `accelerator`, and `fault_tolerance`. There's no dict of loose keys, and a typo fails when you set it rather than being silently ignored.

The pages in this section are the reference:

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`list-unordered;1.1em` Options
:link: /configuration/options
:link-type: doc
Every field of the general sections, with its default.
:::

:::{grid-item-card} {octicon}`server;1.1em` Distributed options
:link: /configuration/distributed-options
:link-type: doc
Attaching to Ray, the shuffle, inference stages, and the GPU backend.
:::

:::{grid-item-card} {octicon}`cpu;1.1em` Accelerator
:link: /configuration/accelerator
:link-type: doc
Device placement, energy budgets, device memory, and device health.
:::

:::{grid-item-card} {octicon}`shield;1.1em` Fault tolerance
:link: /configuration/fault-tolerance
:link-type: doc
Retry budgets and quarantine when nodes and devices fail mid-job.
:::

:::{grid-item-card} {octicon}`terminal;1.1em` Environment
:link: /configuration/environment
:link-type: doc
The `BATCHER_*` variables, the config file, and what Batcher detects.
:::

:::{grid-item-card} {octicon}`stack;1.1em` Profiles
:link: /configuration/profiles
:link-type: doc
Worked configurations for common deployments.
:::
::::

```python
import batcher as bt
from batcher import Config, set_config, config_context

cfg = Config()
print(cfg.execution.morsel_rows)
# 16384
```

## Building a config

`Config` and its sections are frozen dataclasses, so you derive new ones rather than mutating in place. {py:meth}`Config.replace(...) <batcher.Config.replace>` swaps whole sections, and `dataclasses.replace` changes a field within a section.

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

The individual sections have no `.replace` method of their own. Use `dataclasses.replace(section, field=value)` for field-level edits.

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

Building a whole `Config` to change one number is a lot of ceremony. Every tunable also has a dotted name, and `set_option` and `get_option` address it directly. This is the same API shape as `pandas.set_option` and `spark.conf.set`, and it goes through the same validation as `set_config`.

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

Finally, {py:meth}`Config.non_defaults() <batcher.Config.non_defaults>` answers "what is actually set here?" when a job behaves differently on two machines. Its `repr` shows the same thing, so printing a config is useful rather than a wall of nearly 300 options.

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

{py:meth}`Config.from_dict <batcher.Config.from_dict>` and {py:meth}`Config.to_dict <batcher.Config.to_dict>` are the in-memory pair. {py:meth}`to_dict <batcher.Config.to_dict>` produces
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

{py:meth}`from_dict <batcher.Config.from_dict>` re-runs the same environment resolution every entry point does, which
auto-detects a spot node or an autoscaling cluster. That means a config captured on
one machine can legitimately differ from raw defaults when reloaded on another.
Reloading an already-resolved config is idempotent, which is the property to rely on.

## Precedence

When the engine resolves the active config, the layers apply highest first:

1. `config_context(...)`, the innermost active context.
1. `set_config(...)`, process-wide.
1. `BATCHER_*` environment variables.
1. A config file named by `BATCHER_CONFIG_FILE`.
1. Built-in defaults.

The environment and file layers are read once when `batcher` is imported.
`set_config` and `config_context` override them at runtime.

The following diagram shows the same five layers as a stack, grouped by when each one is set:

![Five layers are stacked from highest precedence at the top to lowest at the bottom. The top two are set at runtime. config_context, which option_context and tenant are built on, applies to the innermost with block and is restored on exit. set_config, which set_option goes through, is process-wide until changed. The bottom three are read once at import. BATCHER_* environment variables, loaded with Config.from_env, overlay the file named by BATCHER_CONFIG_FILE, loaded with Config.from_file, which overlays the built-in defaults. The defaults are the dataclass field values, and they are what reset_option restores rather than the values the environment produced.](/_static/diagrams/config_precedence.svg)

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
