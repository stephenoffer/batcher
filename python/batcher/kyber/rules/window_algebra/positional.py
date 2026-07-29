"""`nth_value(x, 1)` is `first_value(x)`.

Both are positional value functions over the same frame: `nth_value` returns the frame's
n-th row and `first_value` returns its first, so at `n = 1` they name the same row by
construction — with the default frame and with any explicit one, because the frame is what
both are positioned within. Verified against the engine on a partitioned, ordered window.

The rewrite pays twice. `first_value` is the specialized kernel, which reads the frame's
head instead of counting into it. And it is the *canonical* spelling, which is what lets
`dedupe_window_functions` merge the result with a sibling spec that was already written
that way — two specs computing the same column in two spellings otherwise reach the data
plane as two windows over the same partition.

Only `n = 1` qualifies. `nth_value(x, 2)` is a genuinely different row, and there is no
`second_value` to name it.
"""

from __future__ import annotations

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase
from batcher.plan.logical import LogicalPlan, Window, WindowFuncSpec

__all__ = ["nth_value_at_one_to_first_value"]


@rule(name="nth_value_at_one_to_first_value", phase=Phase.NORMALIZE, matches=(Window,))
def nth_value_at_one_to_first_value(node: Window, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`nth_value(x, 1) OVER w` -> `first_value(x) OVER w`.

    The frame carries over untouched, which is the whole correctness argument: both
    functions are positioned *within* the frame, so the frame decides which rows are
    candidates and `n = 1` picks the same one either way. The alias and input are
    preserved, so nothing downstream of the window sees a change.
    """
    rewritten: list[WindowFuncSpec] = []
    for spec in node.functions:
        if spec.func == "nth_value" and spec.offset == 1:
            rewritten.append(
                WindowFuncSpec("first_value", spec.input, spec.alias, frame=spec.frame)
            )
        else:
            rewritten.append(spec)
    if all(new is old for new, old in zip(rewritten, node.functions, strict=True)):
        return None
    return Window(
        node.input,
        node.partition_keys,
        node.order_keys,
        tuple(rewritten),
        rank_limit=node.rank_limit,
    )
