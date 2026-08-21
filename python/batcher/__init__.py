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


#: The public subpackages a user reaches as ``bt.<name>``, resolved lazily on first access.
#:
#: They are not imported at package load on purpose: `ml` pulls in the whole model surface,
#: and every ``import batcher`` would pay for it whether or not the script does inference.
#: But they were not reachable *at all* — ``bt.ml.vllm_engine(...)`` raised
#: ``AttributeError``, which is the spelling every docstring and documentation page uses,
#: because `io`, `config` and `governance` happen to be imported transitively by `api` and
#: `ml` and `graph` do not. Nothing caught it: the examples that use this spelling all need a
#: GPU or a model, so every one of them carries `+SKIP` and none has ever run.
_PUBLIC_SUBPACKAGES = ("config", "governance", "graph", "io", "ml")


def __getattr__(name: str) -> object:
    """Resolve a public subpackage lazily, or turn the lookup into guidance (PEP 562).

    ``bt.ml``/``bt.graph`` import on first touch, so the documented spelling works in a fresh
    interpreter without every ``import batcher`` paying for the ML stack. Anything else is a
    migrant typing the top-level name they know from pandas, Polars, or PySpark
    (``bt.DataFrame``, ``bt.SparkSession``, ``bt.scan_csv``), and the traceback names the
    Batcher spelling instead of a bare ``module 'batcher' has no attribute``. Only reached for
    names not already bound above, so it never shadows the real surface.

    Args:
        name: The attribute that was not found on the module.

    Returns:
        The imported subpackage when `name` is one of the public subpackages.

    Raises:
        AttributeError: For any other name, carrying the migration hint.
    """
    # Dunder and private probes (import machinery, IPython, copy) must fail plainly.
    if name.startswith("_"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name in _PUBLIC_SUBPACKAGES:
        import importlib

        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module  # bind it, so the next lookup never reaches here
        return module
    from batcher.api.session.onboarding import top_level_attribute_error

    raise top_level_attribute_error(name, __all__)
