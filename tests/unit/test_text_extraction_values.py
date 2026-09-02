"""Soundex against Knuth, the extractors against their documented shape, and `strip_html`
against the regex idiom it claims to beat.

The third batch of value checks over the `.str` surface, after the metrics (820f5552) and the
interop functions (a93b45f3). These are the text-*ingestion* functions: what a crawl or a
scrape is turned into before it is chunked, embedded, or trained on. A defect here does not
raise, it degrades a corpus.

`strip_html` is the one with a claim to check. Its docstring says it is "strictly more
correct than the ``regexp_replace('<[^>]*>', '')`` idiom", and that is measured here rather
than repeated: on the four cases where the two differ, the regex is wrong in ways that matter
to a corpus, and fusing `<p>Hello</p><p>World</p>` into `HelloWorld` is the one that would be
hardest to notice downstream.

`soundex` is checked against the published algorithm on the names its literature uses. It is
fiddly -- the H/W rule and the "same code, adjacent, collapse" rule are where implementations
diverge -- so the reference here implements Knuth's version directly rather than trusting a
library.
"""

from __future__ import annotations

import re

import pytest

import batcher as bt

pytestmark = pytest.mark.unit


def _knuth_soundex(word: str) -> str:
    """The published algorithm, implemented from the definition rather than from a library.

    Letters map to digits; adjacent letters with the same code collapse to one; `H` and `W`
    are transparent, so letters either side of them still count as adjacent, while a vowel
    resets the run. Pad or truncate to four characters.
    """
    letters = "".join(c for c in word.upper() if c.isalpha())
    if not letters:
        return ""
    codes = {
        **dict.fromkeys("BFPV", "1"),
        **dict.fromkeys("CGJKQSXZ", "2"),
        **dict.fromkeys("DT", "3"),
        **dict.fromkeys("L", "4"),
        **dict.fromkeys("MN", "5"),
        **dict.fromkeys("R", "6"),
    }
    out = letters[0]
    previous = codes.get(letters[0], "")
    for character in letters[1:]:
        code = codes.get(character, "")
        if code and code != previous:
            out += code
        if character not in "HW":
            previous = code
        if len(out) == 4:
            break
    return (out + "000")[:4]


#: The names soundex literature uses, chosen because each exercises a different rule.
SOUNDEX_NAMES = [
    "Robert",  # the canonical R163
    "Rupert",  # must equal Robert
    "Ashcraft",  # the H rule
    "Tymczak",  # adjacent same-code letters
    "Pfister",  # a leading digraph mapping to one code
    "Honeyman",  # H between vowels
    "Smith",
    "Smyth",  # must equal Smith
]


def test_soundex_matches_the_published_algorithm():
    got = bt.from_pydict({"s": SOUNDEX_NAMES}).select(r=bt.col("s").str.soundex()).to_pydict()["r"]
    want = [_knuth_soundex(n) for n in SOUNDEX_NAMES]
    disagreements = [
        (name, g, w) for name, g, w in zip(SOUNDEX_NAMES, got, want, strict=True) if g != w
    ]
    assert disagreements == [], f"soundex differs from Knuth's algorithm on: {disagreements}"


def test_soundex_agrees_on_the_pairs_it_exists_to_equate():
    """The control on the fixture. If every name coded differently, the assertion above
    would hold for an implementation that just returned the first four characters."""
    coded = dict(
        zip(
            SOUNDEX_NAMES,
            bt.from_pydict({"s": SOUNDEX_NAMES})
            .select(r=bt.col("s").str.soundex())
            .to_pydict()["r"],
            strict=True,
        )
    )
    assert coded["Robert"] == coded["Rupert"], "the canonical pair must collide"
    assert coded["Smith"] == coded["Smyth"], "the canonical pair must collide"
    assert coded["Robert"] != coded["Smith"], "and distinct names must not"


#: (input, expected). The marker is part of the result -- the docstrings say "every
#: ``#hashtag``" and "every ``@mention``", written with the marker.
_EXTRACTORS = [
    ("extract_hashtags", "#one #two @me", ["#one", "#two"]),
    ("extract_mentions", "#one #two @me", ["@me"]),
    ("extract_emails", "a@b.com and c@d.org", ["a@b.com", "c@d.org"]),
    ("extract_urls", "see http://x.y and https://z.w now", ["http://x.y", "https://z.w"]),
]


@pytest.mark.parametrize(("name", "text", "expected"), _EXTRACTORS, ids=[c[0] for c in _EXTRACTORS])
def test_an_extractor_returns_the_marked_tokens(name, text, expected):
    got = bt.from_pydict({"s": [text]}).select(r=getattr(bt.col("s").str, name)()).to_pydict()["r"]
    assert got == [expected]


def test_an_extractor_returns_an_empty_list_when_there_is_nothing():
    """Empty, not null: the text exists and contains no matches."""
    for name, _, _ in _EXTRACTORS:
        got = (
            bt.from_pydict({"s": ["nothing here"]})
            .select(r=getattr(bt.col("s").str, name)())
            .to_pydict()["r"]
        )
        assert got == [[]], f"{name} returned {got} where nothing matched"


#: Cases where the regex idiom is wrong, with what each should produce. The last three are
#: cases where the two agree, kept so this is not only a list of the regex's failures.
_HTML = [
    ("<p>Hello</p><p>World</p>", "Hello World", True),
    ("a &amp; b &lt;tag&gt; &quot;q&quot;", 'a & b <tag> "q"', True),
    ("<script>var x = '<b>';</script>visible", "visible", True),
    ("<style>p{color:red}</style>text", "text", True),
    ("<!-- comment -->after", "after", False),
    ("<a href='#'>link</a> text", "link text", False),
    ("unclosed <b>bold", "unclosed bold", False),
]


@pytest.mark.parametrize(("markup", "expected", "regex_differs"), _HTML)
def test_strip_html_recovers_the_readable_text(markup, expected, regex_differs):
    got = bt.from_pydict({"s": [markup]}).select(r=bt.col("s").str.strip_html()).to_pydict()["r"]
    assert got == [expected]


@pytest.mark.parametrize(("markup", "expected", "regex_differs"), _HTML)
def test_the_regex_idiom_is_wrong_exactly_where_claimed(markup, expected, regex_differs):
    """The docstring claims `strip_html` is "strictly more correct" than
    ``regexp_replace('<[^>]*>', '')``. This measures that claim instead of repeating it: on
    four of these the regex gives a different and worse answer, and on three it agrees.

    Without the agreeing cases this would only show that the two differ, which a *worse*
    implementation would also satisfy."""
    naive = re.sub(r"<[^>]*>", "", markup)
    assert (naive != expected) == regex_differs, (
        f"the regex idiom now returns {naive!r} for {markup!r}; the claim this file measures "
        "has changed"
    )


class TestWhatTheRegexGetsWrong:
    """Named individually, because each is a distinct corpus defect rather than a formatting
    difference, and the first is the one that would survive review unnoticed."""

    def test_block_tags_become_a_word_boundary(self):
        """The regex fuses two paragraphs into `HelloWorld`, inventing a word that is then
        tokenized, embedded and trained on."""
        assert re.sub(r"<[^>]*>", "", "<p>Hello</p><p>World</p>") == "HelloWorld"
        got = (
            bt.from_pydict({"s": ["<p>Hello</p><p>World</p>"]})
            .select(r=bt.col("s").str.strip_html())
            .to_pydict()["r"]
        )
        assert got == ["Hello World"]

    def test_entities_are_decoded(self):
        assert "&amp;" in re.sub(r"<[^>]*>", "", "a &amp; b")
        got = (
            bt.from_pydict({"s": ["a &amp; b"]})
            .select(r=bt.col("s").str.strip_html())
            .to_pydict()["r"]
        )
        assert got == ["a & b"]

    def test_script_and_style_bodies_are_removed_not_kept(self):
        """The regex strips the tags and keeps the JavaScript, so the corpus gains source
        code where the page had none visible."""
        assert "var x" in re.sub(r"<[^>]*>", "", "<script>var x = 1;</script>visible")
        got = (
            bt.from_pydict({"s": ["<script>var x = 1;</script>visible"]})
            .select(r=bt.col("s").str.strip_html())
            .to_pydict()["r"]
        )
        assert got == ["visible"]
