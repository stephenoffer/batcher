r"""What the text metrics compute, checked against independent Python implementations.

`test_text_metric_invariants.py` covers the *properties* of these 132 metrics: null in, null
out; a ratio inside `[0, 1]`; a count that is not negative. Every one of those passes for a
metric that computes the wrong number, and one of them did -- `mean_line_length` counts the
line separators and reads up to 80% high on the short-line documents it exists to find
(`docs/architecture/internals/text_metric_audit.md`).

So this file checks values, and against a *second* implementation rather than against the
engine's own. The Python on the right-hand side is deliberately naive and written from each
metric's documented definition: a re-derivation, not a transcription. Where the two agree
across eighteen inputs -- unicode, empty, whitespace-only, URLs, emails, hashtags, tabs,
newlines, punctuation-only -- the metric computes what it says.

Four metrics are checked against their *documented* definition rather than the obvious one,
because the obvious one is wrong about them and reading the name alone reproduces the error:

* `sentence_count` counts sentence-ending *punctuation*, so ``"Hello World"`` is 0.
* `avg_word_length` averages *letters* per word, so ``"abc123"`` is 3.0 rather than 6.0.
* `alnum_ratio` is ASCII, matching `alpha_ratio`, so ``"café"`` is 0.75 rather than 1.0. So
  is `avg_word_length`: the whole letter-counting family is ASCII-only, and only
  `alpha_ratio`'s docstring says so.
* `digit_to_word_ratio` is digits *per word*, so ``"123"`` is 3.0 and not a fraction.
"""

from __future__ import annotations

import base64
import hashlib
import re
import urllib.parse
import zlib
from collections.abc import Callable

import pytest

import batcher as bt

pytestmark = pytest.mark.unit

#: Ordinary prose, digits, whitespace-only, all-caps, empty, single character, null,
#: punctuation-joined, multi-sentence, multi-line, a URL and an email, social markers,
#: non-ASCII, tabs, punctuation-only, and a longer sentence.
VALUES = [
    "Hello World",
    "abc123",
    "  ",
    "ABC",
    "",
    "x",
    None,
    "a-b_c",
    "one. two. three.",
    "line1\nline2\nline3",
    "  padded  ",
    "CAPS LOCK ON",
    "http://a.b and x@y.com",
    "#tag @me",
    "café",
    "a\tb\tc",
    "...!?",
    "The quick brown fox jumps",
]

_EMAIL = r"[\w.+-]+@[\w-]+\.[\w.]+"


def _ratio(numerator: Callable[[str], int], denominator: Callable[[str], int]):
    """A fraction that is null rather than a division by zero, as these metrics define it."""

    def compute(s: str) -> float | None:
        below = denominator(s)
        return None if below == 0 else numerator(s) / below

    return compute


#: metric name -> an independent implementation of its documented definition.
REFERENCE: dict[str, Callable[[str], object]] = {
    # counts
    "digit_count": lambda s: sum(c.isdigit() for c in s),
    "space_count": lambda s: s.count(" "),
    "tab_count": lambda s: s.count("\t"),
    "newline_count": lambda s: s.count("\n"),
    "line_count": lambda s: s.count("\n") + 1,
    "non_ascii_count": lambda s: sum(ord(c) > 127 for c in s),
    "paren_count": lambda s: s.count("(") + s.count(")"),
    "quote_count": lambda s: s.count('"') + s.count("'"),
    "hashtag_count": lambda s: len(re.findall(r"#\w+", s)),
    "mention_count": lambda s: len(re.findall(r"@\w+", s)),
    "url_count": lambda s: len(re.findall(r"https?://\S+", s)),
    "email_count": lambda s: len(re.findall(_EMAIL, s)),
    "uppercase_word_count": lambda s: sum(1 for w in s.split() if w.isupper()),
    "word_count": lambda s: len(s.split()),
    # predicates
    "has_digits": lambda s: any(c.isdigit() for c in s),
    "has_url": lambda s: bool(re.search(r"https?://", s)),
    "has_email": lambda s: bool(re.search(_EMAIL, s)),
    "is_alpha": lambda s: s.isalpha(),
    "is_alnum": lambda s: s.isalnum(),
    "is_numeric": lambda s: s.isdigit(),
    "is_space": lambda s: len(s) > 0 and s.isspace(),
    # fractions of the string
    "alpha_ratio": _ratio(lambda s: sum(c.isalpha() and c.isascii() for c in s), len),
    "digit_ratio": _ratio(lambda s: sum(c.isdigit() for c in s), len),
    "whitespace_ratio": _ratio(lambda s: sum(c.isspace() for c in s), len),
    # transforms
    "remove_digits": lambda s: re.sub(r"\d", "", s),
    "remove_urls": lambda s: re.sub(r"https?://\S+", "", s),
    "remove_emails": lambda s: re.sub(_EMAIL, "", s),
    "remove_html_tags": lambda s: re.sub(r"<[^>]*>", "", s),
}

#: Interop-critical: these must equal the standard implementation byte for byte, because
#: something downstream is going to compare the result against one. A `md5` that is merely
#: self-consistent is worthless -- the point of a digest is that another system computes the
#: same one. Checked against `hashlib`, `zlib`, `base64` and `urllib`, not against the engine.
INTEROP: dict[str, Callable[[str], object]] = {
    "md5": lambda s: hashlib.md5(s.encode()).hexdigest(),
    "sha1": lambda s: hashlib.sha1(s.encode()).hexdigest(),
    "sha256": lambda s: hashlib.sha256(s.encode()).hexdigest(),
    "crc32": lambda s: zlib.crc32(s.encode()),
    "base64": lambda s: base64.b64encode(s.encode()).decode(),
    "url_encode": lambda s: urllib.parse.quote(s, safe=""),
    "len_chars": len,
    "octet_length": lambda s: len(s.encode()),
    "bit_length": lambda s: len(s.encode()) * 8,
    "upper": str.upper,
    "lower": str.lower,
    "capitalize": str.capitalize,
    "reverse": lambda s: s[::-1],
    "is_blank": lambda s: len(s.strip()) == 0,
    "normalize_whitespace": lambda s: " ".join(s.split()),
}


#: The four whose documented definition is not the one the name suggests. Written from the
#: docstring, and listed apart so the distinction is visible rather than buried in the table.
DOCUMENTED: dict[str, Callable[[str], object]] = {
    "sentence_count": lambda s: sum(c in ".!?" for c in s),
    # ASCII letters, like `alpha_ratio` and `alnum_ratio`. The whole letter-counting family
    # is ASCII-only; using Python's unicode-aware `isalpha` here reports "café" as 4.0.
    "avg_word_length": _ratio(
        lambda s: sum(sum(c.isalpha() and c.isascii() for c in w) for w in s.split()),
        lambda s: len(s.split()),
    ),
    "alnum_ratio": _ratio(lambda s: sum(c.isalnum() and c.isascii() for c in s), len),
    "digit_to_word_ratio": _ratio(lambda s: sum(c.isdigit() for c in s), lambda s: len(s.split())),
}


def _engine(name: str) -> list:
    column = getattr(bt.col("s").str, name)()
    return bt.from_pydict({"s": VALUES}).select(r=column).to_pydict()["r"]


def _python(reference: Callable[[str], object]) -> list:
    return [None if v is None else reference(v) for v in VALUES]


def _equal(got: object, want: object) -> bool:
    if got is None or want is None:
        return got is want
    if isinstance(got, float) or isinstance(want, float):
        return abs(float(got) - float(want)) < 1e-9
    return got == want


@pytest.mark.parametrize("name", sorted(REFERENCE))
def test_a_metric_matches_an_independent_implementation(name):
    got, want = _engine(name), _python(REFERENCE[name])
    mismatches = [(v, g, w) for v, g, w in zip(VALUES, got, want, strict=True) if not _equal(g, w)]
    assert mismatches == [], f"{name} differs from its definition on: {mismatches}"


@pytest.mark.parametrize("name", sorted(DOCUMENTED))
def test_a_metric_matches_its_documented_definition(name):
    """These four differ from what their names suggest, and the difference is deliberate."""
    got, want = _engine(name), _python(DOCUMENTED[name])
    mismatches = [(v, g, w) for v, g, w in zip(VALUES, got, want, strict=True) if not _equal(g, w)]
    assert mismatches == [], f"{name} differs from its documented definition on: {mismatches}"


def test_the_fixture_exercises_each_reference():
    """A reference that returns the same thing for every input would agree with an engine
    that did too. Each must produce at least two distinct answers over the fixture."""
    flat = {}
    for name, reference in {**REFERENCE, **DOCUMENTED}.items():
        answers = {repr(v) for v in _python(reference)}
        if len(answers) < 2:
            flat[name] = answers
    assert flat == {}, (
        f"these references are constant over the fixture, so they check nothing: {flat}"
    )


class TestTheFourThatLookWrong:
    """Spelled out on the exact values that mislead, so the intent survives a reader who
    checks only the name. Each is asserted against Python's *different* answer too, so the
    difference is recorded rather than implied."""

    def test_sentence_count_needs_a_terminator(self):
        by_value = dict(zip(VALUES, _engine("sentence_count"), strict=True))
        assert by_value["Hello World"] == 0
        assert by_value["one. two. three."] == 3

    def test_avg_word_length_counts_letters_not_characters(self):
        by_value = dict(zip(VALUES, _engine("avg_word_length"), strict=True))
        assert by_value["abc123"] == pytest.approx(3.0)
        assert len("abc123") == 6, "the point is that the character count is not the answer"

    def test_the_letter_family_is_ascii_throughout(self):
        """`avg_word_length` counts ASCII letters too, so an accented word is short by one.
        Grouped with the others because the restriction is family-wide and stated on only
        one member of the family."""
        by_value = dict(zip(VALUES, _engine("avg_word_length"), strict=True))
        assert by_value["café"] == pytest.approx(3.0)

    def test_alnum_ratio_is_ascii_only(self):
        by_value = dict(zip(VALUES, _engine("alnum_ratio"), strict=True))
        assert by_value["café"] == pytest.approx(0.75)
        assert "café".isalnum() is True, "Python counts the accented letter; this metric does not"

    def test_digits_per_word_is_not_a_fraction(self):
        by_value = dict(zip(VALUES, _engine("digit_to_word_ratio"), strict=True))
        assert by_value["abc123"] == pytest.approx(3.0)


@pytest.mark.parametrize("name", sorted(INTEROP))
def test_an_interop_function_matches_the_standard_implementation(name):
    """These leave the process. A digest or an encoding that is only self-consistent is
    worthless: the whole point is that another system computes the same one."""
    got, want = _engine(name), _python(INTEROP[name])
    mismatches = [(v, g, w) for v, g, w in zip(VALUES, got, want, strict=True) if not _equal(g, w)]
    assert mismatches == [], f"{name} differs from the standard implementation on: {mismatches}"


class TestHexIsSqlNotPython:
    """`hex` is uppercase, which is SQL's convention and not Python's.

    Python's `bytes.hex()` is lowercase, so using it as the oracle reports a defect that is
    not one. DuckDB's `hex('hello')` is `68656C6C6F`, and so is Batcher's. This is recorded
    because reaching for Python here is the obvious move and it gives the wrong answer.
    """

    def test_hex_is_uppercase(self):
        by_value = dict(zip(VALUES, _engine("hex"), strict=True))
        assert by_value["Hello World"] == b"Hello World".hex().upper()
        assert b"Hello World".hex().islower(), "Python's is lowercase; the engine's is not"

    def test_unhex_round_trips_through_non_ascii(self):
        """The round trip is the property that matters, and it has to survive multi-byte
        characters -- `café` is five characters and six bytes."""
        data = bt.from_pydict({"s": ["hello", "café", ""]})
        out = data.select(r=bt.col("s").str.hex().str.unhex()).to_pydict()["r"]
        assert out == ["hello", "café", ""]
