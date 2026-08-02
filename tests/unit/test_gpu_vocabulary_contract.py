"""The device tier must classify every IR tag — supported or declined, never merely absent.

`CLAUDE.md` invariant #6 is "one `Expr`, one `RelOp`, across tiers", and the Cranelift JIT
honours it by consuming the same Rust `Expr` the interpreter does: a new expression cannot
silently bypass it, because there is only one definition. The device tier cannot do that. cuDF
has no maintained Rust binding, so `core/gpu_plan/` is a *translator* from the same JSON IR
onto a dataframe library, with its own hand-written dispatch table keyed by tag strings.

That leaves two drift modes a correctness suite cannot see, because both end in a **silent
fallback to the CPU engine** — the right answer, arriving slower, with no error anywhere:

* **A new tag nobody classified.** Add a `RelOp` or an `Expr` to the engine and the device
  tier declines it forever. Nothing records that this was a decision rather than an oversight,
  and the accelerated path quietly narrows with every release.
* **A renamed tag orphaning its handler.** The IR tags are a wire contract changed on the Rust
  and Python sides in one commit (invariant #8). `_HANDLERS` is a third place keyed by those
  same strings and is not part of that commit — so a rename leaves the handler pointing at a
  dead string while the tier stops translating the expression.

These tests make both loud. They are cheap and pure-Python: no engine, no device, no GPU.
"""

from __future__ import annotations

import pytest

from batcher.core.gpu_plan.exprs import _HANDLERS, DECLINED_EXPRS
from batcher.core.gpu_plan.ops import DECLINED_OPS, SUPPORTED_OPS
from batcher.plan.ir_tags import ExprTag, Op

pytestmark = pytest.mark.unit


def _tags(namespace: type) -> set[str]:
    return {
        value
        for name, value in vars(namespace).items()
        if not name.startswith("_") and isinstance(value, str)
    }


CANONICAL_OPS = _tags(Op)
CANONICAL_EXPRS = _tags(ExprTag)


# --- relational operators ------------------------------------------------------


def test_every_relop_tag_is_either_supported_or_declined():
    classified = set(SUPPORTED_OPS) | set(DECLINED_OPS)
    unclassified = CANONICAL_OPS - classified
    assert not unclassified, (
        f"{sorted(unclassified)} are `bc_ir::RelOp` tags the device tier neither translates "
        "nor declines. Add a handler, or record the decline in `gpu_plan.ops.DECLINED_OPS` "
        "with a reason — an unlisted tag is declined silently and forever."
    )


def test_no_relop_is_both_supported_and_declined():
    both = set(SUPPORTED_OPS) & set(DECLINED_OPS)
    assert not both, f"{sorted(both)} are listed as supported and declined"


def test_the_device_tier_claims_no_relop_the_engine_does_not_have():
    """A supported tag that is not canonical is a handler keyed on a dead string."""
    unknown = set(SUPPORTED_OPS) - CANONICAL_OPS
    assert not unknown, (
        f"{sorted(unknown)} are in SUPPORTED_OPS but are not `plan.ir_tags.Op` tags — most "
        "likely a tag was renamed and this list was not part of that commit."
    )


def test_declined_relops_are_real_tags_with_reasons():
    unknown = set(DECLINED_OPS) - CANONICAL_OPS
    assert not unknown, f"{sorted(unknown)} are declined but are not real RelOp tags"
    assert all(reason.strip() for reason in DECLINED_OPS.values()), "a decline needs a reason"


# --- scalar expressions --------------------------------------------------------


def test_every_expr_tag_is_either_handled_or_declined():
    classified = set(_HANDLERS) | set(DECLINED_EXPRS)
    unclassified = CANONICAL_EXPRS - classified
    assert not unclassified, (
        f"{sorted(unclassified)} are `bc_expr::Expr` tags the device tier neither translates "
        "nor declines. Add a handler to `gpu_plan.exprs._HANDLERS`, or record the decline in "
        "`DECLINED_EXPRS` with a reason."
    )


def test_no_expr_is_both_handled_and_declined():
    both = set(_HANDLERS) & set(DECLINED_EXPRS)
    assert not both, f"{sorted(both)} have a handler and are also listed as declined"


def test_no_handler_is_keyed_on_a_tag_the_engine_does_not_have():
    """The orphaned-handler case: a renamed tag stops reaching its translation, silently."""
    orphaned = set(_HANDLERS) - CANONICAL_EXPRS
    assert not orphaned, (
        f"{sorted(orphaned)} have device handlers but are not `plan.ir_tags.ExprTag` tags. "
        "The handler can never fire: the tier declines the expression and falls back to the "
        "CPU engine with no error. Re-key the handler, or drop it."
    )


def test_declined_exprs_are_real_tags_with_reasons():
    unknown = set(DECLINED_EXPRS) - CANONICAL_EXPRS
    assert not unknown, f"{sorted(unknown)} are declined but are not real Expr tags"
    assert all(reason.strip() for reason in DECLINED_EXPRS.values()), "a decline needs a reason"


# --- the tier's safety property ------------------------------------------------


def test_the_device_tier_is_opt_in():
    """`backend` must default to CPU: the tier is never reached unless asked for.

    This is the strongest safety property the device tier has — it is verified only on pandas,
    so a user who never types `backend=` must never meet it — and until now it was an accident
    of a default rather than a stated contract. Nothing failed if someone changed it.
    """
    import inspect

    from batcher.api.terminal.core import _collect

    default = inspect.signature(_collect).parameters["backend"].default
    assert default == "cpu", (
        f"the terminal op now defaults to backend={default!r}. The device tier is a second "
        "implementation of the engine's semantics verified on a pandas stand-in, never on "
        "cuDF; reaching it without an explicit request is not a safe default."
    )
