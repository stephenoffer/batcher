"""One reader for "is this a plain integer literal", not six.

`bool` subclasses `int` in Python, so `isinstance(Lit(True).value, int)` is true. Any Kyber
rule that reads an integer literal must therefore exclude `bool` explicitly, or it treats
`WHERE flag = TRUE` as `WHERE flag = 1` and folds a boolean comparison into arithmetic.

That one-line guard was copy-pasted into **six** rule modules under five different names --
`_int_lit` three times, plus `_int_literal`, `_seconds_literal`, and a sixth module importing
one of them across family boundaries. All byte-identical. Six copies of one subtlety is five
chances for a fix to be applied to one and missed on the rest, and nothing would have failed:
each copy is correct today, so no test could see the risk. They now share
`plan.expr_ir.core.int_literal`.

The shared home is the neutral `plan` layer rather than a Kyber helpers module, and that is
load-bearing rather than tidy. Every caller already imports `plan.expr_ir`, so sharing adds no
import edge at all -- whereas routing them through `kyber.rules.exprs.guards` would have made
two families import the `exprs` package for the first time, and that package's `__init__`
runs every module's `@rule` decorator. Rules run in the order their modules are imported, so
that refactor would have silently reordered the rule set. `just surface-diff` reports EMPTY
across rule order, IR tags, public API and the FFI for the change as landed.
"""

from __future__ import annotations

import ast
import hashlib
import pathlib

import pytest

from batcher.plan.expr_ir import Lit
from batcher.plan.expr_ir.core import int_literal

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (5, 5),
        (0, 0),
        (-3, -3),
        (True, None),
        (False, None),
        (1.0, None),
        ("7", None),
        (None, None),
    ],
    ids=["int", "zero", "negative", "true", "false", "float", "str", "null"],
)
def test_only_a_plain_integer_literal_reads_as_one(value, expected):
    """`True` and `False` must not read as 1 and 0, though Python says they are ints."""
    assert int_literal(Lit(value)) == expected


def test_a_non_literal_expression_reads_as_none():
    from batcher.plan.expr_ir import col

    assert int_literal(col("a")) is None


def _body_fingerprint(node: ast.FunctionDef) -> str:
    """The function's body, minus its docstring, as a structural hash."""
    body = list(node.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    dumped = ast.dump(ast.Module(body=body, type_ignores=[]), annotate_fields=False)
    return hashlib.sha1(dumped.encode()).hexdigest()


def test_no_module_grows_its_own_copy_again():
    """The anti-regression for the dedup itself.

    The six copies were added one at a time, each by someone who needed an integer literal
    and wrote the obvious four lines. Nothing objected, because each copy was correct. This
    is the thing that objects: any function anywhere in the control plane whose body is
    structurally identical to `int_literal`'s is a seventh copy.
    """
    root = pathlib.Path(__file__).resolve().parents[2] / "python" / "batcher"
    canonical_src = (root / "plan" / "expr_ir" / "core.py").read_text()
    canonical = next(
        n
        for n in ast.walk(ast.parse(canonical_src))
        if isinstance(n, ast.FunctionDef) and n.name == "int_literal"
    )
    fingerprint = _body_fingerprint(canonical)

    copies = []
    for path in root.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:  # a file mid-edit by another session
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or node.name == "int_literal":
                continue
            if _body_fingerprint(node) == fingerprint:
                copies.append(f"{path.relative_to(root)}::{node.name}")

    assert not copies, (
        "these re-implement `plan.expr_ir.core.int_literal` verbatim; import it instead so "
        f"the bool guard cannot drift: {sorted(copies)}"
    )
