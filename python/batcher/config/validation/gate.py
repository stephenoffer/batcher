"""The validation gate: run every section check once per distinct `Config` object.

Lives next to `config.py` (not inside it) so the single-source-of-truth `Config`
module stays focused on the dataclasses and the active-config plumbing. `Config`
calls `validate_config(self)` at every entry point (`set_config`, `config_context`,
`from_env`, `from_file`) so an out-of-range or inconsistent tunable raises
`ConfigError` early and clearly instead of surfacing as a confusing runtime failure.

**Each distinct `Config` object is validated at most once.** `config_context` is one of
those entry points, and the conductor enters it *twice per query* — once for the sensed
memory envelope, once for Carbonite's adapted morsel target. So the ~60 range checks ran
twice on every `collect()`, ~24 µs a time: about 6% of a small query's entire control
plane, spent re-deriving that a config which passed a moment ago still passes.

A `Config` is a frozen dataclass, and validation is a pure function of its value, so
"this exact object already passed" is a sound answer — memoized on object identity, with
the config held as the memo's value so its `id()` cannot be recycled underneath the entry.
A config built by `dataclasses.replace` is a *new* object and is validated normally; the
hit comes from the conductor handing the same resolved object back each query.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher.config.validation.sections import run_checks

if TYPE_CHECKING:
    from batcher.config.config import (
        Config,
    )

__all__ = ["validate_config"]

# id(config) -> the config itself. The value is the keep-alive: holding a strong reference
# makes it impossible for the id to be reused by a different object while the entry lives.
_VALIDATED: dict[int, Config] = {}
# Bound on the memo. A process activates a handful of distinct configs; anything that
# churns past this is building configs per query, where re-validating is the lesser cost.
_VALIDATED_MAX = 64


def validate_config(cfg: Config) -> None:
    """Raise `ConfigError` if any tunable is out of range or inconsistent.

    Covers the memory envelope, execution sizing, the distributed fault-tolerance
    budgets/timeouts, and flow-control credits. A `Config` object that has already passed
    is not re-checked (see the module docstring); the checks themselves are pure.

    Args:
        cfg: The configuration to validate.
    """
    if id(cfg) in _VALIDATED:
        return
    run_checks(cfg)
    if len(_VALIDATED) >= _VALIDATED_MAX:
        _VALIDATED.clear()
    _VALIDATED[id(cfg)] = cfg
