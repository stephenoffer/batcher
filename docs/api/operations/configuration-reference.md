# Configuration reference

This page lists every option function, every configuration dataclass behind them, and the
whole-cache controls, straight from the source docstrings. {doc}`configuration` is the same surface
as prose, with the precedence order the layers resolve in, and {doc}`/configuration/index` is where
each field's default and unit live.

## Configuration functions

Two entry points install a {py:class}`Config <batcher.Config>`: one for the process, one
for a `with` block. `active_config` reports whichever is in force.
{doc}`/api/operations/configuration` explains how the two layer with the environment
variables and the config file, and which setting wins when they disagree.

```{eval-rst}
.. currentmodule:: batcher

.. autosummary::
   :toctree: generated
   :nosignatures:

   set_config
   config_context
```

```{eval-rst}
.. currentmodule:: batcher.config

.. autosummary::
   :toctree: generated
   :nosignatures:

   active_config
```

### Options by name

Address any tunable by its dotted path, the way `pandas.set_option` and `spark.conf.set`
do. {doc}`/configuration/options` lists every path with its default and its unit.

```{eval-rst}
.. currentmodule:: batcher.config

.. autosummary::
   :toctree: generated
   :nosignatures:

   get_option
   set_option
   reset_option
   option_context
   option_names
   describe_options
```

### Serialization

Turn the active configuration into plain data, or find out what environment variable
spells a given field.

```{eval-rst}
.. currentmodule:: batcher.config

.. autosummary::
   :toctree: generated
   :nosignatures:

   config_to_dict
   env_var_names
```

### Logging and verbosity

One-line switches over {py:class}`ObservabilityConfig <batcher.config.config.ObservabilityConfig>`. See
{doc}`observability </user-guide/operate/running/observability>`.

```{eval-rst}
.. currentmodule:: batcher.config

.. autosummary::
   :toctree: generated
   :nosignatures:

   set_log_level
   enable_logging
   disable_logging
   set_verbosity
   set_progress
   get_logger
```

### Metrics export

Process-wide counters as plain data, ready for Prometheus, OpenTelemetry, or a log line.

```{eval-rst}
.. currentmodule:: batcher.observe

.. autosummary::
   :toctree: generated
   :nosignatures:

   metrics_snapshot
   prometheus_text
   start_metrics
   reset_metrics
```

### Query history

The completed queries the event log recorded, as a `Dataset` of their measurements.

```{eval-rst}
.. currentmodule:: batcher

.. autosummary::
   :toctree: generated
   :nosignatures:

   query_history
```

## Result cache

Whole-cache control and measurement, for the results {py:meth}`Dataset.cache <batcher.Dataset.cache>` stores. See {doc}`caching results </user-guide/operate/tuning/caching>`.

```{eval-rst}
.. currentmodule:: batcher

.. autosummary::
   :toctree: generated
   :nosignatures:

   cache_stats
   clear_cache
   StorageLevel
```

## Configuration classes

The tunables themselves, one dataclass per subsystem. All of them are frozen, so changing a
setting means deriving a new config rather than mutating the live one.
{doc}`/api/operations/configuration` works that pattern through and says how the layers
combine. {doc}`/configuration/options` gives every field a default and a unit.

```{eval-rst}
.. currentmodule:: batcher

.. autosummary::
   :toctree: generated
   :nosignatures:

   Config
   ExecutionConfig
   MemoryConfig
   FlowControlConfig
   StreamingConfig
   OptimizerConfig
   PIDConfig
   MetadataConfig
   GovernanceConfig
   TenantConfig
   tenant
```

```{eval-rst}
.. currentmodule:: batcher.config.config

.. autosummary::
   :toctree: generated
   :nosignatures:

   CardinalityConfig
   CostWeights
   CostCoefficients
   DistributedConfig
   ObservabilityConfig
   ShuffleTlsConfig
```

```{eval-rst}
.. currentmodule:: batcher.config

.. autosummary::
   :toctree: generated
   :nosignatures:

   AcceleratorConfig
   EnergyConfig
   DeviceHealthConfig
   DeviceMemoryConfig
   FaultToleranceConfig
   QuarantineConfig
```

## See also

- {doc}`configuration`: the same surface as prose, with the precedence order the layers resolve in.
- {doc}`/configuration/options`: every field, its default, and its unit.
- {doc}`/configuration/environment`: the `BATCHER_*` spelling of the same settings.
- {doc}`/configuration/profiles`: the named bundles that set many of these fields at once.
