"""Every error the engine raises has to be catchable as a Batcher error.

`BatcherError` is documented as "the root every other Batcher error subclasses, so catching
it covers them all", and for five types that was false in every *built* install. They were
declared in Rust with `create_exception!`, whose base is `RuntimeError`, so
`except bt.BatcherError` did not catch a cancelled query, a memory-budget refusal, a shuffle
failure, or an over-deep plan.

The reason it went unnoticed for so long is the shape worth remembering: `_internal.errors`
carried pure-Python fallbacks with the *right* bases for the case where the extension is not
built, and the unit suite runs without it. So the contract held exactly when the engine was
absent, and broke exactly when it was present. A type built by `create_exception!` also
cannot be repaired from Python afterwards -- `__bases__` assignment refuses on a layout
mismatch, and ABC registration satisfies `issubclass` while `except` still misses, which
would have been worse than the bug.

The fix inverts the ownership: Python defines these classes and `bc_py::errors` looks them
up by name on the error path.
"""

from __future__ import annotations

import pytest

from batcher._internal.errors import (
    BatcherError,
    ExecutionError,
    FatalShuffleError,
    MemoryBudgetExceededError,
    PlanError,
    PlanTooDeepError,
    QueryCancelledError,
    ResourceError,
    RetryableShuffleError,
    TransportError,
)

pytestmark = pytest.mark.unit

#: Each engine-raised type and the family a caller is told it belongs to.
FAMILIES = {
    "QueryCancelledError": (QueryCancelledError, ExecutionError),
    "MemoryBudgetExceededError": (MemoryBudgetExceededError, ResourceError),
    "RetryableShuffleError": (RetryableShuffleError, TransportError),
    "FatalShuffleError": (FatalShuffleError, TransportError),
    "PlanTooDeepError": (PlanTooDeepError, PlanError),
}


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_it_is_a_batcher_error(name):
    """The promise one `except` is supposed to keep."""
    cls, _ = FAMILIES[name]
    assert issubclass(cls, BatcherError)


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_it_is_in_the_family_the_documentation_names(name):
    """A caller catching `TransportError` must actually catch a shuffle failure."""
    cls, family = FAMILIES[name]
    assert issubclass(cls, family)


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_except_really_catches_it(name):
    """`issubclass` and `except` do not agree for every way of relating two classes.

    ABC registration makes `issubclass` true while `except` still misses, because exception
    matching uses a real subtype check. Asserting the subclass relation alone would
    therefore pass on a repair that does not work, so this raises one and catches it.
    """
    cls, family = FAMILIES[name]
    with pytest.raises(BatcherError):
        raise cls("boom")
    with pytest.raises(family):
        raise cls("boom")


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_the_engine_and_the_control_plane_agree_on_the_type(name):
    """One class, not two that share a name.

    While Rust declared its own, `batcher._native.X` and `batcher._internal.errors.X` were
    different objects, so which one a caller caught depended on where they imported it from.
    Skipped rather than failed without the extension: there is nothing to disagree with.
    """
    native = pytest.importorskip("batcher._native")
    exported = getattr(native, name, None)
    if exported is None:
        pytest.skip(f"{name} is not re-exported by this engine build")
    assert exported is FAMILIES[name][0]


def test_the_root_really_is_the_root():
    """The claim `docs/api/operations/exceptions.md` makes, as an assertion."""
    from batcher._internal import errors

    public = [
        getattr(errors, n)
        for n in errors.__all__
        if isinstance(getattr(errors, n, None), type)
        and issubclass(getattr(errors, n), BaseException)
        and not issubclass(getattr(errors, n), Warning)
    ]
    assert public, "no exception types found to check"
    missing = [c.__name__ for c in public if not issubclass(c, BatcherError)]
    assert not missing, f"not catchable as BatcherError: {missing}"
