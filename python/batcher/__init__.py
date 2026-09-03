"""Batcher — a native, JIT-compiling, adaptive data engine.

The public surface is intentionally small and fluent. Everything in this package
is the *control plane*: it builds and optimizes plans and hands them to the Rust
engine (`batcher._native`). No tuple is ever processed in Python on the hot path.

    import batcher as bt

    ds = bt.from_pydict({"x": [1, 2, 3], "y": [10, 20, 30]})
    out = ds.filter(bt.col("x") > 1).select("x", xy=bt.col("x") * bt.col("y")).collect()

This module is a re-export façade over `batcher.api` and the tunable dataclasses of
`batcher.config`, and it resolves **every one of those names lazily** (PEP 562).
Importing the surface eagerly cost 545 ms and pulled in 609 modules plus pyarrow and
numpy, for a script that may only want `bt.col`; the closure of `batcher.config` alone
is 13 modules and no third-party package at all. That price is paid once per process,
and the engine's scaling target is millions of them, so it was also the largest fixed
cost in a distributed run's startup.

`_exports.EXPORTS` routes each name to the module that *defines* it rather than to
`batcher.api`, so `bt.col` imports the expression package without the IO registry or
the SQL session coming with it. The table is generated (`just gen-exports`) and gated
against the eager surface by `tests/unit/test_lazy_exports.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._exports import EXPORTS as _EXPORTS
from batcher._exports import ROOT_SHADOWED as _ROOT_SHADOWED
from batcher._lazy import install as _install

if TYPE_CHECKING:
    # The declaration of this façade's surface, and the generator's input: `just
    # gen-exports` derives the routing table from exactly these imports. Eager only for
    # type checkers and editors, which is what preserves completion and `pyright`
    # resolution across the whole surface without importing any of it at runtime.
    #
    # The config names are listed one by one rather than star-imported because the top
    # level is a *curated* surface: `batcher.config.__all__` has 40 names and only these
    # 20 are public here. A star import would silently widen the API by twenty names.
    from batcher.api import *  # noqa: F403
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

#: The public surface, in the order the eager façade declared it. Rebuilt from the
#: generated routing table so the two cannot disagree about what is public.
__all__ = [*_EXPORTS, "__version__"]


#: The public subpackages a user reaches as ``bt.<name>``, resolved lazily on first access.
#:
#: They are not imported at package load on purpose: `ml` pulls in the whole model surface,
#: and every ``import batcher`` would pay for it whether or not the script does inference.
#: But they were not reachable *at all* — ``bt.ml.vllm_engine(...)`` raised
#: ``AttributeError``, which is the spelling every docstring and documentation page uses,
#: because `io`, `config` and `governance` happen to be imported transitively by `api` and
#: `ml` and `graph` do not. Nothing caught it: the examples that use this spelling all need a
#: GPU or a model, so every one of them carries `+SKIP` and none has ever run.
#:
#: Now that the surface is lazy none of them is imported transitively either, so this tuple
#: is the only thing making ``bt.io`` and ``bt.config`` resolve at all.
_PUBLIC_SUBPACKAGES = ("config", "governance", "graph", "io", "ml")


def _migration_hint(name: str, surface: list[str]) -> Exception:
    """Turn an unknown top-level name into migration guidance.

    A miss here is almost always a migrant typing the name they know from pandas,
    Polars, or PySpark (``bt.DataFrame``, ``bt.SparkSession``, ``bt.scan_csv``), so the
    traceback names the Batcher spelling instead of a bare "has no attribute".

    Args:
        name: The attribute that was not found.
        surface: The façade's `__all__`, searched for the nearest Batcher spelling.

    Returns:
        The `AttributeError` to raise, carrying the hint.
    """
    from batcher.api.session.onboarding import top_level_attribute_error

    return top_level_attribute_error(name, surface or __all__)


_lazy_getattr, __dir__ = _install(
    __name__,
    _EXPORTS,
    subpackages=_PUBLIC_SUBPACKAGES,
    shadowed=_ROOT_SHADOWED,
    on_missing=_migration_hint,
)


def __getattr__(name: str) -> object:
    """Resolve a public name lazily, plus the one private attribute worth keeping.

    ``bt._native`` is a *diagnostic*, not an API: `.claude/rules/concurrent-agents.md`
    tells an agent to confirm which engine a scratchpad sandbox is running with
    ``bt._native.__file__``. The eager façade answered it by accident — something in the
    import chain always reached the engine — and laziness took that away, so a bare
    ``import batcher`` followed by ``bt._native`` raised `AttributeError` and read as a
    broken sandbox rather than an unloaded engine.

    Resolved through `_internal.native`, the one sanctioned accessor, so this adds no
    static ``import batcher._native`` for the import graph to mis-attribute — which is the
    thing the contract actually forbids, and what once forged a phantom
    ``core -> batcher -> api -> kyber`` cycle. Every other private name still fails plainly.

    Args:
        name: The attribute that was not found on the module.

    Returns:
        The public object, subpackage, or the compiled engine module for ``_native``.

    Raises:
        AttributeError: For any other name, carrying the migration hint.
    """
    if name == "_native":
        from batcher._internal.native import engine

        module = engine()
        globals()["_native"] = module
        return module
    return _lazy_getattr(name)
