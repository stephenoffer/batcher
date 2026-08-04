"""Batcher — a native, JIT-compiling, adaptive data engine.

The public surface is intentionally small and fluent. Everything in this package
is the *control plane*: it builds and optimizes plans and hands them to the Rust
engine (`batcher._native`). No tuple is ever processed in Python on the hot path.

    import batcher as bt

    ds = bt.from_pydict({"x": [1, 2, 3], "y": [10, 20, 30]})
    out = ds.filter(bt.col("x") > 1).select("x", xy=bt.col("x") * bt.col("y")).collect()

This module is a re-export façade: the full expression/Dataset/IO surface comes
from `batcher.api`, plus the tunable dataclasses of `batcher.config`, so a script
that wants to size the buffer pool or pin the optimizer never imports a second
module. The config names are listed explicitly rather than star-imported: the top
level is a curated surface, and adding to it is a deliberate act.
"""

from __future__ import annotations

from batcher import api as _api
from batcher.api import *  # noqa: F403  (re-export the api surface; governed by api.__all__)
from batcher.api.dataset.callbacks import udf as udf
from batcher.config import CardinalityConfig as CardinalityConfig
from batcher.config import Config as Config
from batcher.config import CostCoefficients as CostCoefficients
from batcher.config import CostWeights as CostWeights
from batcher.config import DistributedConfig as DistributedConfig
from batcher.config import ExecutionConfig as ExecutionConfig
from batcher.config import FlowControlConfig as FlowControlConfig
from batcher.config import GovernanceConfig as GovernanceConfig
from batcher.config import MemoryConfig as MemoryConfig
from batcher.config import MetadataConfig as MetadataConfig
from batcher.config import ObservabilityConfig as ObservabilityConfig
from batcher.config import OptimizerConfig as OptimizerConfig
from batcher.config import PIDConfig as PIDConfig
from batcher.config import ShuffleTlsConfig as ShuffleTlsConfig
from batcher.config import StreamingConfig as StreamingConfig
from batcher.config import TenantConfig as TenantConfig
from batcher.config import active_config as active_config
from batcher.config import config_context as config_context
from batcher.config import set_config as set_config
from batcher.config import tenant as tenant

__version__ = "0.1.0"

# The config names exposed at the top level: every tunable dataclass, plus the three
# ways to read or change the active one (`active_config`, `set_config`,
# `config_context`). Listed here so the export is a decision, not a side effect of
# whatever `batcher.config` happens to declare.
_CONFIG_EXPORTS = [
    "CardinalityConfig",
    "Config",
    "CostCoefficients",
    "CostWeights",
    "DistributedConfig",
    "ExecutionConfig",
    "FlowControlConfig",
    "GovernanceConfig",
    "MemoryConfig",
    "MetadataConfig",
    "ObservabilityConfig",
    "OptimizerConfig",
    "PIDConfig",
    "StreamingConfig",
    "ShuffleTlsConfig",
    "TenantConfig",
    "tenant",
    "active_config",
    "config_context",
    "set_config",
]

__all__ = [*_api.__all__, *_CONFIG_EXPORTS, "udf", "__version__"]


def __getattr__(name: str) -> object:
    """Turn a failed ``bt.<name>`` lookup into migration guidance (PEP 562).

    A migrant types the top-level name they know from pandas, Polars, or PySpark
    (``bt.DataFrame``, ``bt.SparkSession``, ``bt.scan_csv``); the traceback names the
    Batcher spelling instead of a bare ``module 'batcher' has no attribute``. Only
    reached for names not already bound above, so it never shadows the real surface.
    """
    # Dunder and private probes (import machinery, IPython, copy) must fail plainly.
    if name.startswith("_"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from batcher.api.session.onboarding import top_level_attribute_error

    raise top_level_attribute_error(name, __all__)
