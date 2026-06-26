"""The public function façade stays in sync — no silent re-export drift.

A function is declared in a family module (`plan/functions/<family>.py`), then
re-exported through `api/functions.py` and surfaced on the top-level `batcher`
namespace. That is three places a new function's name must appear; forgetting one
makes the function invisible (defined but not exported) or the façade broken
(exported but unresolvable). These checks turn either drift into a failing test
instead of a silent gap, so the surface stays trustworthy as functions scale.
"""

from __future__ import annotations

import batcher
import batcher.api.functions as api_functions
from batcher.plan import functions as plan_functions


def test_family_functions_are_all_re_exported() -> None:
    # Every public function declared in a plan/functions family module must appear in
    # the api façade — otherwise it is defined but unreachable by users.
    facade = set(api_functions.__all__)
    missing = sorted(name for name in plan_functions.__all__ if name not in facade)
    assert not missing, f"in plan/functions but missing from api/functions.__all__: {missing}"


def test_facade_names_resolve_on_top_level() -> None:
    # Every name the façade promises must resolve both on api.functions and on the
    # top-level `batcher` namespace — a name in __all__ that does not import is a
    # broken public surface.
    for name in api_functions.__all__:
        assert hasattr(api_functions, name), f"{name!r} in __all__ but not importable"
        assert hasattr(batcher, name), f"{name!r} in __all__ but not exposed on `batcher`"


def test_facade_all_is_sorted_and_unique() -> None:
    # A curated façade is a sorted, duplicate-free list — keeps diffs reviewable and
    # makes an accidental double-export obvious.
    names = api_functions.__all__
    assert names == sorted(names), "api/functions.__all__ should be sorted"
    assert len(names) == len(set(names)), "api/functions.__all__ has duplicates"
