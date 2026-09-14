# Configuration and caching

The functions that read and set options, the configuration dataclasses behind them, and
the whole-cache controls.

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
.. autofunction:: batcher.config.active_config
```

### Options by name

Address any tunable by its dotted path, the way `pandas.set_option` and `spark.conf.set`
do. {doc}`/configuration/options` lists every path with its default and its unit.

```{eval-rst}
.. autofunction:: batcher.config.get_option
.. autofunction:: batcher.config.set_option
.. autofunction:: batcher.config.reset_option
.. autofunction:: batcher.config.option_context
.. autofunction:: batcher.config.option_names
.. autofunction:: batcher.config.describe_options
```

### Serialization

Turn the active configuration into plain data, or find out what environment variable
spells a given field.

```{eval-rst}
.. autofunction:: batcher.config.config_to_dict
.. autofunction:: batcher.config.env_var_names
```

### Logging and verbosity

One-line switches over {py:class}`ObservabilityConfig <batcher.config.config.ObservabilityConfig>`. See
{doc}`observability </user-guide/operate/running/observability>`.

```{eval-rst}
.. autofunction:: batcher.config.set_log_level
.. autofunction:: batcher.config.enable_logging
.. autofunction:: batcher.config.disable_logging
.. autofunction:: batcher.config.set_verbosity
.. autofunction:: batcher.config.set_progress
.. autofunction:: batcher.config.get_logger
```

### Metrics export

Process-wide counters as plain data, ready for Prometheus, OpenTelemetry, or a log line.

```{eval-rst}
.. autofunction:: batcher.observe.metrics_snapshot
.. autofunction:: batcher.observe.prometheus_text
.. autofunction:: batcher.observe.start_metrics
.. autofunction:: batcher.observe.reset_metrics
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
```

```{eval-rst}
.. autoclass:: batcher.StorageLevel
   :members:
   :undoc-members:
```

## Configuration classes

The tunables themselves, one dataclass per subsystem. All of them are frozen, so changing a
setting means deriving a new config rather than mutating the live one.
{doc}`/api/operations/configuration` works that pattern through and says how the layers
combine; {doc}`/configuration/options` gives every field a default and a unit.

```{eval-rst}
.. autoclass:: batcher.Config
   :members:

.. autoclass:: batcher.ExecutionConfig
   :members:

.. autoclass:: batcher.MemoryConfig
   :members:

.. autoclass:: batcher.FlowControlConfig
   :members:

.. autoclass:: batcher.StreamingConfig
   :members:

.. autoclass:: batcher.OptimizerConfig
   :members:

.. autoclass:: batcher.config.config.CardinalityConfig
   :members:

.. autoclass:: batcher.config.config.CostWeights
   :members:

.. autoclass:: batcher.config.config.CostCoefficients
   :members:

.. autoclass:: batcher.config.config.DistributedConfig
   :members:

.. autoclass:: batcher.PIDConfig
   :members:

.. autoclass:: batcher.MetadataConfig
   :members:

.. autoclass:: batcher.config.AcceleratorConfig
   :members:

.. autoclass:: batcher.config.EnergyConfig
   :members:

.. autoclass:: batcher.config.DeviceHealthConfig
   :members:

.. autoclass:: batcher.config.DeviceMemoryConfig
   :members:

.. autoclass:: batcher.config.FaultToleranceConfig
   :members:

.. autoclass:: batcher.config.QuarantineConfig
   :members:

.. autoclass:: batcher.GovernanceConfig
   :members:

.. autoclass:: batcher.TenantConfig
   :members:

.. autofunction:: batcher.tenant

.. autoclass:: batcher.config.config.ObservabilityConfig
   :members:

.. autoclass:: batcher.config.config.ShuffleTlsConfig
   :members:
```

## See also

- {doc}`/api/operations/configuration`: the same surface as prose, with the precedence order the layers resolve in.
- {doc}`/configuration/options`: every field, its default, and its unit.
- {doc}`/configuration/environment`: the `BATCHER_*` spelling of the same settings.
- {doc}`/configuration/profiles`: the named bundles that set many of these fields at once.
