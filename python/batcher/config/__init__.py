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
    GovernanceConfig,
    MemoryConfig,
    MetadataConfig,
    ObservabilityConfig,
    OptimizerConfig,
    PIDConfig,
    ShuffleTlsConfig,
    TenantConfig,
    active_config,
    config_context,
    set_config,
    tenant,
)
from batcher.config.logs import (
    disable_logging,
    enable_logging,
    get_logger,
    set_log_level,
    set_progress,
    set_verbosity,
)
from batcher.config.options import (
    describe_options,
    get_option,
    option_context,
    option_names,
    reset_option,
    set_option,
)
from batcher.config.serde import config_to_dict, env_var_names

__all__ = [
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
    "ShuffleTlsConfig",
    "TenantConfig",
    "active_config",
    "config_context",
    "config_to_dict",
    "describe_options",
    "disable_logging",
    "enable_logging",
    "env_var_names",
    "get_logger",
    "get_option",
    "option_context",
    "option_names",
    "reset_option",
    "set_config",
    "set_log_level",
    "set_option",
    "set_progress",
    "set_verbosity",
    "tenant",
]
