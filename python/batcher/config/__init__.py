"""Configuration: one frozen, typed `Config` object.

Replaces v1's twelve fragmented, mutable config submodules with a single
immutable `Config` composed of typed sections. Nothing mutates in place;
`config_context()` derives a new frozen copy and pushes it onto a `ContextVar`.
"""

from __future__ import annotations

from batcher.config.config import (
    CardinalityConfig,
    Config,
    CostCoefficients,
    CostWeights,
    DistributedConfig,
    ExecutionConfig,
    FlowControlConfig,
    MemoryConfig,
    MetadataConfig,
    OptimizerConfig,
    PIDConfig,
    active_config,
    config_context,
    set_config,
)

__all__ = [
    "CardinalityConfig",
    "Config",
    "CostCoefficients",
    "CostWeights",
    "DistributedConfig",
    "ExecutionConfig",
    "FlowControlConfig",
    "MemoryConfig",
    "MetadataConfig",
    "OptimizerConfig",
    "PIDConfig",
    "active_config",
    "config_context",
    "set_config",
]
