"""Config range/consistency validation, applied at every `Config` entry point."""

from __future__ import annotations

from batcher.config.validation.gate import validate_config

__all__ = ["validate_config"]
