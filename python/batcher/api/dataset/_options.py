"""Check a forwarded bag of `map_batches` options before it is forwarded.

Two surfaces take ``**config`` and hand it to `map_batches` (or, for a per-row callback,
`map`): the `@udf` decorator and `register_function`. A catch-all is the right shape for them
— the option set belongs to `map_batches`, not to each wrapper — but it moves every mistake to
the point of *use*. A misspelled `output_column` on a decorator surfaced as
``TypeError: DatasetML.map_batches() got an unexpected keyword argument``, raised from a method
the user never called, at whatever line finally applied the transform.

So the check happens where the option was written, names the surface the user actually used,
and lists what it could have been. The valid set is read off the live signature rather than
listed here, so a new `map_batches` option is accepted the day it is added.
"""

from __future__ import annotations

from typing import Any

__all__ = ["validate_map_options"]


def _parameters(*, per_row: bool) -> set[str]:
    """The keyword-only parameters of the method the options are forwarded to."""
    import inspect

    from batcher.api.dataset.ml import DatasetML

    method = DatasetML.map if per_row else DatasetML.map_batches
    return {
        name
        for name, p in inspect.signature(method).parameters.items()
        if p.kind is inspect.Parameter.KEYWORD_ONLY
    }


def validate_map_options(caller: str, options: dict[str, Any], *, per_row: bool = False) -> None:
    """Reject an option the forwarding target does not have, naming `caller`.

    Args:
        caller: How the user spelled the surface, e.g. ``"@udf"`` — this appears in the
            message, because the method the option is forwarded to is not one they called.
        options: The extra keywords collected by the surface's ``**config``.
        per_row: Whether the options are forwarded to `map` rather than `map_batches`.

    Raises:
        PlanError: If any option is not a parameter of the target.
    """
    if not options:
        return
    allowed = _parameters(per_row=per_row)
    unknown = sorted(set(options) - allowed)
    if not unknown:
        return
    from batcher._internal.errors import PlanError

    target = "a per-row callback" if per_row else "map_batches"
    raise PlanError(
        f"{caller}: {unknown} " + ("is" if len(unknown) == 1 else "are") + f" not "
        f"{'an option' if len(unknown) == 1 else 'options'} of {target}. "
        f"Valid options: {sorted(allowed)}."
    )
