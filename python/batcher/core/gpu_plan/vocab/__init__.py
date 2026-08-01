"""The translator's *named* vocabularies, one module per family — package façade.

`exprs` and `scalar_fns` are the operators and the four original function families. This
package is where the families that need a construction of their own live: list and vector
work, which no dataframe library exposes as a single call and which is exactly what a device
is bought for.

Kept as a subpackage rather than as more modules beside `exprs` because `gpu_plan/` is at its
twelve-file ceiling, and because these families genuinely group: each is a vocabulary whose
entries share one construction, not a set of unrelated cases.
"""

from __future__ import annotations

from batcher.core.gpu_plan.vocab.dates import date_typed, eval_make_temporal, eval_window_start
from batcher.core.gpu_plan.vocab.lists import (
    LIST_BINARY_FNS,
    LIST_REDUCTIONS,
    eval_list_binary,
    eval_list_contains,
    eval_list_fn,
    eval_list_get,
    eval_list_position,
    supported_list_binary,
    supported_list_fn,
)
from batcher.core.gpu_plan.vocab.regex import REGEX_FNS, eval_regex, portable

__all__ = [
    "LIST_BINARY_FNS",
    "LIST_REDUCTIONS",
    "REGEX_FNS",
    "date_typed",
    "eval_list_binary",
    "eval_list_contains",
    "eval_list_fn",
    "eval_list_get",
    "eval_list_position",
    "eval_make_temporal",
    "eval_regex",
    "eval_window_start",
    "portable",
    "supported_list_binary",
    "supported_list_fn",
]
