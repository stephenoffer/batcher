"""`is_recoverable_task_failure` — a UDF bug must not be laundered into a resource error.

The combiner tree fetches from its upstreams *inside* the Ray task, so a genuinely lost
peer reaches the driver as a `RetryableShuffleError` wrapped in a `RayTaskError` — the
same type a user's failing UDF produces. The tree path used to treat every `RayTaskError`
as worker loss, so a deterministic bug burned all `recovery_max_attempts` re-running
itself and then surfaced as `ResourceError("shuffle did not recover after N attempts")`
with the original traceback discarded. A user got a resource error for a Python bug, and
on the `spot` profile paid six rounds of recompute to get it.

These pin both directions: transport-classified losses stay recoverable (or recovery
breaks), and everything else propagates untouched (or the bug stays buried).
"""

from __future__ import annotations

import pytest

from batcher._internal.errors import (
    FatalShuffleError,
    PlanError,
    ResourceError,
    RetryableShuffleError,
)
from batcher.dist.executors.ray_runtime import is_recoverable_task_failure

pytestmark = pytest.mark.unit


class _WrappedTaskError(Exception):
    """Stands in for Ray's `RayTaskError`, which carries the original in `cause`."""

    def __init__(self, cause: BaseException) -> None:
        super().__init__(str(cause))
        self.cause = cause


def test_an_unreachable_peer_is_recoverable() -> None:
    assert is_recoverable_task_failure(RetryableShuffleError("peer gone"))


def test_a_vanished_spill_file_is_recoverable() -> None:
    """An ephemeral/spot NVMe reclaimed under a spill file is a relocation, not a bug."""
    assert is_recoverable_task_failure(ResourceError("spill file vanished"))


def test_a_udf_bug_is_not_recoverable() -> None:
    """The regression: retrying this cannot help, and hiding it costs the real traceback."""
    assert not is_recoverable_task_failure(ZeroDivisionError("division by zero"))


def test_a_plan_error_is_not_recoverable() -> None:
    assert not is_recoverable_task_failure(PlanError("no such column"))


def test_a_fatal_shuffle_error_is_not_recoverable() -> None:
    """The transport itself said retrying will not help; recovery must not override it."""
    assert not is_recoverable_task_failure(FatalShuffleError("schema mismatch"))


def test_the_wrapped_cause_is_consulted() -> None:
    """Ray usually fuses the original type in, but not on every version — cover both."""
    assert is_recoverable_task_failure(_WrappedTaskError(RetryableShuffleError("peer gone")))
    assert not is_recoverable_task_failure(_WrappedTaskError(ZeroDivisionError("boom")))


def test_a_bare_exception_with_no_cause_is_not_recoverable() -> None:
    """`getattr(exc, "cause", None)` must not make a plain error look retryable."""
    assert not is_recoverable_task_failure(RuntimeError("something else"))
