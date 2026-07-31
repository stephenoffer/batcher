"""Translate a Batcher plan to a GPU dataframe execution (cuDF) — many operators, not one.

Extends the GPU backend from a single group-by to whole relational chains — filter, project /
with_columns, group-by aggregate, sort, distinct, limit, window, equi/semi/anti join, union —
by walking the plan's `RelOp` IR and its `Expr` IR and replaying each on a cuDF DataFrame (the
approach Polars-GPU takes to the same engine). The executor is dataframe-library-parameterized:
it runs on **cuDF** on a GPU worker (the accelerated backend) and on **pandas** for the
head-runnable correctness check against the native CPU engine, so a GPU is *where* a plan runs
and never *what* it computes.

`gpu_plan_ops`/`gpu_join_spec`/`gpu_union_spec` match a plan and return the pieces to replay,
or `None` so the caller falls back to the CPU engine. `execute_cudf_*` replays them. Any
untranslatable operator or expression raises `Unsupported`, which the dispatcher turns into
the same fallback — an accelerator is never a requirement.
"""

from __future__ import annotations

from batcher.core.gpu_plan.backend import DfBackend, Unsupported
from batcher.core.gpu_plan.eligibility import gpu_join_spec, gpu_plan_ops, gpu_union_spec
from batcher.core.gpu_plan.execute import (
    execute_cudf_join,
    execute_cudf_plan,
    execute_cudf_union,
    run_chain,
    run_join,
    run_union,
)
from batcher.core.gpu_plan.ops import SUPPORTED_OPS, apply_op, supported_op

__all__ = [
    "SUPPORTED_OPS",
    "DfBackend",
    "Unsupported",
    "apply_op",
    "execute_cudf_join",
    "execute_cudf_plan",
    "execute_cudf_union",
    "gpu_join_spec",
    "gpu_plan_ops",
    "gpu_union_spec",
    "run_chain",
    "run_join",
    "run_union",
    "supported_op",
]
