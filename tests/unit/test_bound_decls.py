"""The typing declarations for runtime-bound methods match the runtime, and cover all of it.

`tools/gen_bound_decls.py` writes `plan/expr_ir/declared/bound.py` from the live classes so
a type checker can see the methods bound with `setattr`. These tests fail when that file is
stale, and when a public method is bound at runtime onto a class the generator does not
cover, so a new table-generated accessor cannot go invisible to editors again.
"""

from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import gen_bound_decls  # noqa: E402


def test_committed_declarations_are_current() -> None:
    assert gen_bound_decls.TARGET.read_text() == gen_bound_decls.render(), (
        "plan/expr_ir/declared/bound.py is stale; run `just gen-decls`"
    )


@pytest.mark.parametrize("target", gen_bound_decls.TARGETS, ids=lambda t: t[1])
def test_declared_names_equal_the_runtime_bound_names(target: tuple) -> None:
    module, cls_name, decl_name, _ = target
    cls = getattr(importlib.import_module(module), cls_name)
    from batcher.plan.expr_ir import declared

    decl = getattr(declared, decl_name)
    declared_names = sorted(n for n, v in vars(decl).items() if inspect.isfunction(v))
    bound = gen_bound_decls.runtime_bound(cls)
    assert bound, f"{cls_name} binds nothing at runtime; drop it from TARGETS"
    assert declared_names == bound


def test_the_declarations_are_inherited_only_for_type_checkers() -> None:
    """Each runtime class keeps `object` as its base: the declarations never run."""
    for module, cls_name, _, _ in gen_bound_decls.TARGETS:
        cls = getattr(importlib.import_module(module), cls_name)
        assert cls.__bases__ == (object,), cls_name


def test_every_runtime_bound_public_method_is_declared() -> None:
    """No receiver class gains a public method at runtime without a declaration."""
    from tools.parity.batcher_targets import receivers

    covered = {(m, c) for m, c, _, _ in gen_bound_decls.TARGETS}
    seen = 0
    for cls in receivers().values():
        if not inspect.isclass(cls):
            continue
        seen += 1
        public = [n for n in gen_bound_decls.runtime_bound(cls) if not n.startswith("_")]
        if public:
            assert (cls.__module__, cls.__name__) in covered, (cls.__qualname__, public)
    assert seen > 20  # the receiver table was actually walked


def test_runtime_bound_detects_a_setattr_method() -> None:
    """Positive control: a function bound after the class body is reported, a body one is not."""

    class Probe:
        def written(self) -> int:
            return 1

    def bound(self: Probe) -> int:
        return 2

    Probe.bound = bound  # type: ignore[attr-defined]
    assert gen_bound_decls.runtime_bound(Probe) == ["bound"]
