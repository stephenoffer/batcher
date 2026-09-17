"""The alias detector tells a second spelling from a method that merely calls another one.

`tools/lint_aliases.py` gates "one spelling per capability", so both of its failure modes
cost something real. A missed alias lets a second spelling back onto the surface; a false
positive blocks a commit on a method with a meaning of its own (`month_start` is not
`truncate`, it is `truncate("month")`). These tests plant each shape on a throwaway class and
check the classification, so the detector is tested on inputs whose answer is known rather
than only on the live surface, whose answer is what it is being used to find.
"""

from __future__ import annotations

import types

from tools.lint_aliases import _delegates, _public_members, _same_object

_PATTERN = "[a-z]+@[a-z]+"


def column_name(name: str, *, arg: str) -> str:
    return name


class Planted:
    def with_row_index(self, name: str, *, offset: int = 0) -> str:
        return f"{name}{offset}"

    def with_row_count(self, name: str, *, offset: int = 0) -> str:
        """Second spelling, forwarding every parameter through a normalizer."""
        return self.with_row_index(column_name(name, arg="name"), offset=offset)

    def truncate(self, unit: str) -> str:
        return unit

    def month_start(self) -> str:
        """A preset: a constant, so a meaning of its own."""
        return self.truncate("month")

    def has_email(self) -> str:
        """A wrapper: its argument comes from a module constant, not a parameter."""
        return self.truncate(_PATTERN)

    def head(self, n: int = 5, verbose: bool = False) -> str:
        """Not a second spelling: it accepts a parameter it does not forward."""
        return self.truncate(n)

    def limit(self, n: int) -> int:
        return n

    def take(self, n: int) -> int:
        return self.limit(n)

    count_distinct = limit


def _forward(self: Planted, *args: object, _t: str = "limit", **kwargs: object) -> object:
    return getattr(self, _t)(*args, **kwargs)


def _private_forward(self: Planted, *args: object, _t: str = "_x", **kwargs: object) -> object:
    return getattr(self, _t)(*args, **kwargs)


# What a table-driven binder produces: one copy of a template per row, target in a default.
Planted.first_n = types.FunctionType(_forward.__code__, globals(), "first_n")  # type: ignore[attr-defined]
Planted.first_n.__kwdefaults__ = {"_t": "limit"}  # type: ignore[attr-defined]
Planted.micros = types.FunctionType(_private_forward.__code__, globals(), "micros")  # type: ignore[attr-defined]
Planted.micros.__kwdefaults__ = {"_t": "_x"}  # type: ignore[attr-defined]


def _kinds() -> dict[str, tuple[str, str]]:
    members = _public_members(Planted)
    found = _same_object("Planted", members) + _delegates("Planted", members)
    return {f.name: (f.kind, f.target) for f in found}


def test_forwarding_every_parameter_is_a_second_spelling() -> None:
    kinds = _kinds()
    assert kinds["with_row_count"] == ("delegate", "with_row_index")
    assert kinds["take"] == ("delegate", "limit")


def test_the_same_function_object_under_two_names_is_a_second_spelling() -> None:
    kind, target = _kinds()["count_distinct"]
    assert kind == "same-object"
    assert target == "limit"


def test_a_constant_argument_is_a_preset_not_an_alias() -> None:
    assert _kinds()["month_start"] == ("preset", "truncate")


def test_computed_arguments_and_dropped_parameters_are_wrappers() -> None:
    kinds = _kinds()
    assert kinds["has_email"] == ("wrapper", "truncate")
    assert kinds["head"] == ("wrapper", "truncate")


def test_a_generated_forwarder_is_a_second_spelling_of_its_named_target() -> None:
    kinds = _kinds()
    assert kinds["first_n"] == ("delegate", "limit")
    # Forwarding to a private helper is the only public spelling, not a second one.
    assert "micros" not in kinds


def test_a_method_with_its_own_body_is_not_reported() -> None:
    kinds = _kinds()
    for own in ("with_row_index", "truncate", "limit"):
        assert own not in kinds
