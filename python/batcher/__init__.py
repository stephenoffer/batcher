"""Batcher — a native, JIT-compiling, adaptive data engine.

The public surface is intentionally small and fluent. Everything in this package
is the *control plane*: it builds and optimizes plans and hands them to the Rust
engine (`batcher._native`). No tuple is ever processed in Python on the hot path.

    import batcher as bt

    ds = bt.from_pydict({"x": [1, 2, 3], "y": [10, 20, 30]})
    out = ds.filter(bt.col("x") > 1).select("x", xy=bt.col("x") * bt.col("y")).collect()

This module is a re-export façade: the full expression/Dataset/IO surface comes
from `batcher.api`, plus a curated subset of `batcher.config`.
"""

from __future__ import annotations

from batcher import api as _api
from batcher.api import *  # noqa: F403  (re-export the api surface; governed by api.__all__)
from batcher.api.dataset.callbacks import udf as udf
from batcher.config import Config as Config
from batcher.config import ExecutionConfig as ExecutionConfig
from batcher.config import FlowControlConfig as FlowControlConfig
from batcher.config import MemoryConfig as MemoryConfig
from batcher.config import MetadataConfig as MetadataConfig
from batcher.config import OptimizerConfig as OptimizerConfig
from batcher.config import PIDConfig as PIDConfig
from batcher.config import config_context as config_context
from batcher.config import set_config as set_config

__version__ = "0.1.0"

# The curated config names exposed at the top level (a subset of batcher.config).
_CONFIG_EXPORTS = [
    "Config",
    "ExecutionConfig",
    "FlowControlConfig",
    "MemoryConfig",
    "MetadataConfig",
    "OptimizerConfig",
    "PIDConfig",
    "config_context",
    "set_config",
]

__all__ = [*_api.__all__, *_CONFIG_EXPORTS, "udf", "__version__"]
