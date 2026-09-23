"""A zero-argument accessor must reject a positional argument, not reinterpret it.

The `.str`, `.list` and `.dt` namespaces bind most of their methods from a table: one
generated function per name, closing over the IR tag it should build. The generated function
used to take that tag as an ordinary *positional* parameter with a default, which made it
reachable from a public call:

    col("s").str.upper("lower")     -> ['abc', 'def']      (lowercase)
    col("s").str.upper("reverse")   -> ['CbA', 'fEd']      (reversed)

The caller's first argument landed on the tag and redirected the method to a different
function in the same family. Nothing raised, and the answer was a perfectly good answer to a
question nobody asked.

Two things made it hard to notice. These accessors publish a synthetic `__signature__` that
reads `upper() -> Expr`, so a type checker, `inspect`, and the rendered reference page all
agreed the method took no arguments while the runtime accepted one. And a *wrong* tag does
raise -- `upper("to_lowercase")` reports an unknown function -- so the failure only stayed
silent when the injected string happened to name a real sibling, which is the case a user
mistyping a real method name lands on.

The fix is one `*`. These tests are here because the `*` looks like a style preference and
would be an easy casualty of a future refactor of the binding code.
"""

from __future__ import annotations

import inspect

import pytest

import batcher as bt
from batcher._sql.parser.expressions.lowering.accessors import accessor_namespaces

pytestmark = pytest.mark.unit

#: (accessor, method, a *real* sibling in the same family). The sibling matters: the bug was
#: only silent when the injected string named a function that exists, so a case whose
#: sibling is fictional would pass even with the bug present.
_CASES = [
    ("str", "upper", "lower"),
    ("str", "lower", "upper"),
    ("str", "reverse", "upper"),
    ("list", "max", "mean"),
    ("dt", "year", "month"),
]


def _accessor(namespace: str):
    return getattr(bt.col("x"), namespace)


@pytest.mark.parametrize(("namespace", "method", "sibling"), _CASES)
def test_a_positional_argument_is_refused(namespace, method, sibling):
    with pytest.raises(TypeError, match="positional argument"):
        getattr(_accessor(namespace), method)(sibling)


@pytest.mark.parametrize(("namespace", "method", "sibling"), _CASES)
def test_the_sibling_is_a_real_method(namespace, method, sibling):
    """The control. If the sibling did not exist the injection would have raised anyway, and
    the test above would pass for the wrong reason -- it would be checking that a nonsense
    tag is rejected rather than that a *valid* one cannot be injected."""
    assert callable(getattr(_accessor(namespace), sibling, None)), (
        f"{namespace}.{sibling} is not a real method, so {namespace}.{method} could not have "
        "been silently redirected to it and this case proves nothing"
    )


@pytest.mark.parametrize(("namespace", "method", "sibling"), _CASES)
def test_the_method_still_works_with_no_arguments(namespace, method, sibling):
    """The fix must not break the call the method exists for."""
    assert getattr(_accessor(namespace), method)() is not None


def test_the_published_signature_still_advertises_no_arguments():
    """The synthetic signature is what made the bug invisible; it must stay accurate rather
    than start advertising the private parameters now that they are keyword-only."""
    signature = inspect.signature(bt.col("x").str.upper)
    assert [p for p in signature.parameters if not p.startswith("_")] == []


def _every_zero_argument_accessor() -> list[tuple[str, str]]:
    """(namespace, method) for every accessor that takes no arguments, read off the live
    objects rather than a list, so a namespace added later is covered."""
    column = bt.col("x")
    out: list[tuple[str, str]] = []
    holders = [(ns, getattr(column, ns)) for ns in _NAMESPACES if getattr(column, ns, None)]
    holders.append(("Expr", column))
    for label, holder in holders:
        for name in dir(holder):
            if name.startswith("_"):
                continue
            function = getattr(holder, name, None)
            if not callable(function):
                continue
            try:
                signature = inspect.signature(function)
            except (TypeError, ValueError):
                continue
            if [p for p in signature.parameters if not p.startswith("_")]:
                continue
            out.append((label, name))
    return out


#: Read off `Expr` rather than written down. The hardcoded nine this replaced omitted
#: `.seq` and `.meta`, so 14 zero-argument accessors -- `.seq`'s 11 and `.meta`'s 3 -- were
#: never called by the sweep below while it reported itself complete. That is the same
#: failure `accessor_namespaces` was written to fix in the SQL vocabulary, one file over,
#: and the fix is to use it here too rather than to keep a second list in step by hand.
_NAMESPACES = accessor_namespaces()
_ALL_ZERO_ARG = _every_zero_argument_accessor()


def test_the_sweep_found_the_surface():
    """A sweep that enumerates nothing passes while checking nothing.

    The floor is a ratchet and it has already fired once for real: removing the duplicate
    spellings of each capability took the count from 377 to 291, below the 300 written when
    the sweep was nine namespaces wide. Deriving the namespace list restores the margin by
    covering `.seq` and `.meta`, which is the right way to clear it -- lowering the floor
    would have left those 14 accessors uncalled.
    """
    assert len(_ALL_ZERO_ARG) >= 300, f"only {len(_ALL_ZERO_ARG)} zero-argument accessors found"
    # And the namespaces themselves, so a derivation that silently returns () is not a pass.
    assert {"seq", "meta"} <= set(_NAMESPACES), _NAMESPACES


def test_no_zero_argument_accessor_accepts_a_positional_argument():
    """The whole surface, not the five cases above.

    This is the check that would have caught the original defect, and the signature-based
    equivalent is the one that would not: these accessors publish a synthetic
    `__signature__` reading `upper() -> Expr`, so `inspect` reported no parameters while the
    runtime accepted one. Scanning signatures for a private positional parameter finds
    nothing here even with the bug present. Only calling them does.
    """
    accepted = []
    for label, name in _ALL_ZERO_ARG:
        holder = bt.col("x") if label == "Expr" else getattr(bt.col("x"), label)
        try:
            getattr(holder, name)("an_unexpected_positional_argument")
        except TypeError:
            continue
        except Exception:
            continue
        accepted.append(f"{label}.{name}")
    assert accepted == [], (
        "these accessors silently accepted a positional argument, which for a table-bound "
        f"accessor redirects it to another function in its family: {accepted}"
    )


class TestTheBehaviourItProtects:
    """The redirect, spelled out on real data, so a regression is legible rather than a
    `TypeError` disappearing from a parametrized list."""

    def test_upper_is_upper_and_nothing_else(self):
        data = bt.from_pydict({"s": ["AbC", "dEf"]})
        assert data.select(r=bt.col("s").str.upper()).to_pydict()["r"] == ["ABC", "DEF"]
        assert data.select(r=bt.col("s").str.lower()).to_pydict()["r"] == ["abc", "def"]
        assert data.select(r=bt.col("s").str.reverse()).to_pydict()["r"] == ["CbA", "fEd"]

    def test_the_redirect_is_gone(self):
        """Before the fix this returned `['abc', 'def']` -- `lower` under the name `upper`."""
        with pytest.raises(TypeError):
            bt.col("s").str.upper("lower")
