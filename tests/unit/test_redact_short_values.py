"""`Redact` must not hand back a value shorter than what it reveals.

``Redact(show_last=4)`` is the "card ending 1234" policy. Applied to a value of four
characters or fewer it used to return that value **unchanged**, because revealing the last
four characters of a three-character string reveals all three. Measured before the fix, on
``['abcdefgh', 'abcd', 'abc', 'ab', 'a', '', None]``:

* ``Redact(show_last=4)`` returned ``abcd``, ``abc``, ``ab`` and ``a`` raw;
* ``Redact(show_first=2)`` returned ``ab`` and ``a`` raw;
* ``Redact(show_first=2, show_last=2)`` returned everything up to four characters raw.

The literal semantics were right and the security primitive was wrong, which is why the fix
is in `Redact` rather than in `batcher.mask` -- the expression is a general string utility
and stays as it is. `Redact.__post_init__` already states the principle this broke, while
refusing a negative count: under-masking "is the one direction a redaction policy must never
be wrong in".

The columns where this matters are not edge cases. A masking policy is written about names,
postcodes, national IDs and country codes, and a majority of the values in such a column are
shorter than the four characters a card policy reveals.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.governance import Redact

pytestmark = pytest.mark.unit

#: Long enough to reveal from, then every length down to empty, then null.
VALUES = ["abcdefgh", "abcd", "abc", "ab", "a", "", None]


def _apply(policy: Redact, values: list[str | None] = VALUES) -> list[str | None]:
    return bt.from_pydict({"v": values}).select(m=policy(bt.col("v"))).to_pydict()["m"]


POLICIES = [
    Redact(show_last=4),
    Redact(show_first=2),
    Redact(show_first=2, show_last=2),
    Redact(),
    Redact(show_last=1),
    Redact(show_first=3, show_last=3, char="*"),
]


class TestNoValueSurvivesRedaction:
    """The deny half: nothing comes back as itself."""

    @pytest.mark.parametrize("policy", POLICIES, ids=repr)
    def test_no_non_empty_value_is_returned_unchanged(self, policy):
        out = _apply(policy)
        raw = [(v, o) for v, o in zip(VALUES, out, strict=True) if v and o == v]
        assert raw == [], f"{policy!r} returned these values unmasked: {raw}"

    @pytest.mark.parametrize("policy", POLICIES, ids=repr)
    def test_a_value_no_longer_than_the_reveal_is_masked_completely(self, policy):
        reveal = policy.show_first + policy.show_last
        short = ["a" * n for n in range(1, max(reveal, 1) + 1)]
        out = _apply(policy, short)
        assert out == [policy.char * len(v) for v in short], (
            f"{policy!r} left part of a short value visible: {out}"
        )


class TestRedactionStillReveals:
    """The allow half. A mask that refuses everything passes every test above, which is the
    shape a security test hides its failure in, so the revealing behavior is pinned too."""

    def test_a_long_value_still_shows_its_tail(self):
        assert _apply(Redact(show_last=4), ["4111111111111234"]) == ["XXXXXXXXXXXX1234"]

    def test_a_long_value_still_shows_its_head(self):
        assert _apply(Redact(show_first=2), ["abcdefgh"]) == ["abXXXXXX"]

    def test_a_long_value_shows_both_ends(self):
        assert _apply(Redact(show_first=2, show_last=2), ["abcdefgh"]) == ["abXXXXgh"]

    def test_the_boundary_length_reveals_nothing_and_one_more_reveals(self):
        """Exactly at the reveal length is masked; one character longer is not. This is the
        pair that fails in *both* directions if the comparison drifts by one."""
        assert _apply(Redact(show_last=4), ["abcd"]) == ["XXXX"]
        assert _apply(Redact(show_last=4), ["abcde"]) == ["Xbcde"]

    def test_the_custom_char_is_used(self):
        assert _apply(Redact(show_last=2, char="*"), ["abcdef"]) == ["****ef"]


class TestTheShapesThatAreNotValues:
    """Null and empty are not disclosures and must not become one."""

    @pytest.mark.parametrize("policy", POLICIES, ids=repr)
    def test_null_stays_null(self, policy):
        assert _apply(policy, [None]) == [None]

    @pytest.mark.parametrize("policy", POLICIES, ids=repr)
    def test_empty_stays_empty(self, policy):
        assert _apply(policy, [""]) == [""]


class TestTheFullMaskPaysNothing:
    """`show_first == show_last == 0` reveals nothing, so no value can be short enough to
    escape and the conditional is unnecessary. Keeping the bare `mask` there is what stops
    this fix adding a length test and a branch per row to the common full-redaction case."""

    def test_a_full_redaction_lowers_to_a_bare_mask(self):
        assert repr(Redact()(bt.col("ssn"))) == "col('ssn').cast('string').str.mask('X', 0, 0)"

    def test_a_partial_redaction_lowers_to_a_conditional(self):
        """The positive control for the test above: it is only evidence that the full mask
        is *unconditional* if a partial one is visibly conditional."""
        rendered = repr(Redact(show_last=4)(bt.col("card")))
        assert rendered.startswith("when(")
        assert "str.mask('X', 0, 4)" in rendered
        assert "str.mask('X', 0, 0)" in rendered
