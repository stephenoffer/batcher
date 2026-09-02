"""Properties every zero-argument `.str` metric must hold, swept over the namespace.

These 132 accessors are text-quality features -- `alpha_ratio`, `word_char_ratio`,
`sentence_count`, `has_url` -- and most have no DuckDB equivalent, so the differential oracle
the rest of the engine leans on does not reach them. What is left is the properties they
must satisfy whatever they compute, which is worth having because the alternative is a
surface of this size with no automated check at all.

Three properties, and the null one is the reason to bother: a metric that returned `0.0` for
a null string rather than null would read as "this document has no letters" instead of "there
is no document", and a pretraining filter thresholding on it would silently drop or keep
every null row. All 132 preserve null today.

The ratio bound is split deliberately and the split is the interesting part. A name ending
`_ratio` is a *fraction of characters* and belongs in ``[0, 1]``. A name ending
`_to_word_ratio` is a count *per word* -- `digit_to_word_ratio` on ``"123"`` is 3.0, and
`symbol_to_word_ratio` on ``"...!?"`` is 5.0 -- which is a Gopher-style filter threshold and
correctly unbounded. Asserting ``[0, 1]`` across both was this file's first draft and it
failed on exactly those two, which is a wrong invariant rather than a defect. The names carry
the distinction; the docstrings state it.
"""

from __future__ import annotations

import inspect

import pytest

import batcher as bt

pytestmark = pytest.mark.unit

#: Empty, whitespace-only, null, unicode, very long, control characters, markup, contact
#: details, punctuation-only, and multi-sentence -- the inputs a text metric meets in a
#: crawl and the ones most likely to divide by zero or index off the end.
VALUES = [
    "Hello World",
    "",
    "  ",
    None,
    "123",
    "ééé",
    "a" * 300,
    "\n\t\n",
    "<b>x</b> http://a.b",
    "a@b.com #tag @me",
    "...!?",
    "one. two. three.",
]

#: A count *per word*, not a fraction of the string, so it is legitimately unbounded.
_PER_WORD_SUFFIX = "_to_word_ratio"


def _zero_argument_metrics() -> list[str]:
    """Every `.str` accessor taking no arguments, read off the live namespace."""
    accessor = bt.col("s").str
    names = []
    for name in sorted(dir(accessor)):
        if name.startswith("_"):
            continue
        function = getattr(accessor, name, None)
        if not callable(function):
            continue
        try:
            signature = inspect.signature(function)
        except (TypeError, ValueError):
            continue
        if [p for p in signature.parameters if not p.startswith("_")]:
            continue
        names.append(name)
    return names


def _evaluate(name: str) -> list:
    column = getattr(bt.col("s").str, name)()
    return bt.from_pydict({"s": VALUES}).select(r=column).to_pydict()["r"]


METRICS = _zero_argument_metrics()
RATIOS = [n for n in METRICS if n.endswith("_ratio") and not n.endswith(_PER_WORD_SUFFIX)]
PER_WORD = [n for n in METRICS if n.endswith(_PER_WORD_SUFFIX)]
COUNTS = [n for n in METRICS if n.endswith("_count")]


def test_the_sweep_found_the_surface():
    """A namespace-driven sweep that enumerates nothing passes while checking nothing."""
    assert len(METRICS) >= 100, f"only {len(METRICS)} zero-argument .str accessors found"
    assert len(RATIOS) >= 10, f"only {len(RATIOS)} fraction-style ratios found"
    assert len(COUNTS) >= 10, f"only {len(COUNTS)} count-style metrics found"
    assert PER_WORD, "no per-word ratios found, so the bound split below is untested"


@pytest.mark.parametrize("name", METRICS)
def test_a_null_string_yields_null(name):
    """Not zero. `0.0` for a null reads as "this document has no letters" rather than
    "there is no document", and a filter thresholding on it decides the wrong way."""
    result = _evaluate(name)
    assert result[VALUES.index(None)] is None, f"{name} turned a null string into a value"


@pytest.mark.parametrize("name", METRICS)
def test_no_input_in_the_set_raises(name):
    """Empty, whitespace-only and punctuation-only are where a metric divides by zero."""
    assert len(_evaluate(name)) == len(VALUES)


@pytest.mark.parametrize("name", RATIOS)
def test_a_fraction_ratio_is_between_zero_and_one(name):
    out = [v for v in _evaluate(name) if v is not None]
    assert out, f"{name} returned null for every input, so this bound is vacuous"
    assert all(0.0 <= v <= 1.0 for v in out), f"{name} left [0,1]: {out}"


@pytest.mark.parametrize("name", COUNTS)
def test_a_count_is_never_negative(name):
    out = [v for v in _evaluate(name) if v is not None]
    assert out, f"{name} returned null for every input, so this bound is vacuous"
    assert all(v >= 0 for v in out), f"{name} returned a negative count: {out}"


class TestThePerWordRatiosAreDeliberatelyUnbounded:
    """The control for the split above. If these were also in [0,1] the exclusion would be
    unnecessary, and a future reader would be right to delete it."""

    def test_digits_per_word_exceeds_one(self):
        by_value = dict(zip(VALUES, _evaluate("digit_to_word_ratio"), strict=True))
        assert by_value["123"] == pytest.approx(3.0)

    def test_symbols_per_word_exceeds_one(self):
        by_value = dict(zip(VALUES, _evaluate("symbol_to_word_ratio"), strict=True))
        assert by_value["...!?"] > 1.0

    def test_a_fraction_ratio_on_the_same_input_stays_bounded(self):
        """Same input, the other naming convention, to show the two really do differ."""
        by_value = dict(zip(VALUES, _evaluate("digit_ratio"), strict=True))
        assert by_value["123"] == pytest.approx(1.0)
