"""The cleaning, masking and truncation half of ``.str``, against Python and Polars.

Where ``test_diff_string_corpus_quality.py`` covers the filters that *score* a document,
this module covers the ones that *rewrite* it: strip markdown and fenced code, scrub PII,
collapse repeated punctuation, drop stopwords, normalize an answer, and cut to a budget
in characters, words or sentences. None had a test.

Two oracles, chosen per method:

* **Polars**, where Batcher's docstring names it as the reference (``to_titlecase`` is
  documented as the Polars spelling of ``initcap``), so agreement is checkable rather
  than asserted.
* **Python**, recomputed from the documented rule, for the rest.

The truncation family is checked against an invariant as well as a value: a truncation
must be a prefix of its input and must not grow it, which is the property that survives a
change to the tokenizer and catches the off-by-one a fixed expected string would hide.
"""

from __future__ import annotations

import re

import pytest

import batcher as bt

pytestmark = pytest.mark.differential

#: Documents chosen so each rewrite has something to do in exactly one or two of them,
#: and nothing to do in the rest -- a cleaner that rewrites unrelated text fails here.
DOCS = [
    "See [the docs](https://ex.com/a) and [more](http://b.io).",
    "Text before\n```python\ncode here\n```\nText after",
    "* first\n- second\n  * indented",
    "Wow!!! Really??? Yes... ok.",
    "Call +1-555-123-4567 or (555) 987-6543 now",
    "Contact a@b.com and c.d@e.co.uk today",
    "the quick brown fox jumps over the lazy dog",
    (
        "Caf\N{LATIN SMALL LETTER E WITH ACUTE} "
        "na\N{LATIN SMALL LETTER I WITH DIAERESIS}ve "
        "r\N{LATIN SMALL LETTER E WITH ACUTE}sum\N{LATIN SMALL LETTER E WITH ACUTE}"
    ),
    None,
]
NULL_ROW = len(DOCS) - 1


@pytest.fixture(scope="module")
def ds():
    """The document fixture as a Batcher dataset."""
    return bt.from_pydict({"s": DOCS})


def _column(ds, expr) -> list:
    return ds.select(v=expr).to_pydict()["v"]


def _reference(fn) -> list:
    return [None if d is None else fn(d) for d in DOCS]


def _prefix_of_words(text: str, n: int) -> str:
    """The prefix of `text` ending at its n-th whitespace-separated word."""
    end = 0
    for count, match in enumerate(re.finditer(r"\S+", text), start=1):
        end = match.end()
        if count == n:
            break
    return text[:end]


def _squad_reference(text: str) -> str:
    """SQuAD answer normalization in the order Batcher applies it.

    The published implementation deletes punctuation and *then* articles; Batcher deletes
    articles first. The two agree on every ordinary answer and part company only where an
    article is glued to punctuation, which `test_squad_normalize_deletes_articles_before_
    punctuation` pins with the case that separates them.
    """
    lowered = text.lower()
    no_articles = re.sub(r"\b(a|an|the)\b", " ", lowered)
    no_punctuation = re.sub(r"[^\w\s]", "", no_articles)
    return re.sub(r"\s+", " ", no_punctuation).strip()


def test_markdown_links_become_their_link_text(ds):
    """``[text](url)`` collapses to ``text``, leaving everything else alone."""
    got = _column(ds, bt.col("s").str.remove_markdown_links())
    assert got == _reference(lambda d: re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", d))
    assert got[0] == "See the docs and more."
    for i in (1, 2, 3, 4, 5, 6, 7):
        assert got[i] == DOCS[i], f"row {i} has no markdown link and must be untouched"


def test_fenced_code_blocks_are_deleted_and_the_prose_around_them_kept(ds):
    """The whole fence, not just its markers -- a half-removed fence is worse than none."""
    got = _column(ds, bt.col("s").str.remove_code_blocks())
    assert got[1] == "Text before\n\nText after"
    assert "```" not in got[1]
    assert "code here" not in got[1]
    for i in (0, 2, 3, 4, 5, 6, 7):
        assert got[i] == DOCS[i], f"row {i} has no fence and must be untouched"


def test_remove_bullets_strips_the_leading_marker_only(ds):
    """Documented as "a leading list bullet", so the second line's marker stays.

    Pinned because the plausible reading is per-line. The method is a per-*value* cleaner
    used after a split, not a document reflower, and a test that assumed otherwise would
    read as a bug report rather than as coverage.
    """
    got = _column(ds, bt.col("s").str.remove_bullets())
    assert got[2] == "first\n- second\n  * indented"
    per_item = bt.from_pydict({"s": ["* first", "- second", "+ third", "no bullet"]})
    assert _column(per_item, bt.col("s").str.remove_bullets()) == [
        "first",
        "second",
        "third",
        "no bullet",
    ]


def test_repeated_punctuation_collapses_to_a_single_mark(ds):
    """``!!!`` becomes ``!``, and a single mark is left alone."""
    got = _column(ds, bt.col("s").str.remove_repeated_punctuation())
    assert got[3] == "Wow! Really? Yes. ok."
    assert got[0] == DOCS[0], "one mark in a row is not repetition"
    detected = _column(ds, bt.col("s").str.has_repeated_punctuation())
    for i in range(len(DOCS) - 1):
        assert (got[i] != DOCS[i]) == bool(detected[i]), (
            f"row {i}: the detector and the remover disagree about whether there is repetition"
        )


def test_pii_scrubbers_remove_what_their_detectors_find(ds):
    """``remove_phones`` and ``mask_emails`` against ``has_phone`` and ``extract_emails``."""
    got = ds.select(
        no_phone=bt.col("s").str.remove_phones(),
        masked=bt.col("s").str.mask_emails("<E>"),
        found=bt.col("s").str.extract_emails(),
        has_phone=bt.col("s").str.has_phone(),
    ).to_pydict()
    assert got["masked"][5] == "Contact <E> and <E> today"
    assert got["found"][5] == ["a@b.com", "c.d@e.co.uk"]
    for i in range(len(DOCS) - 1):
        assert "@b.com" not in (got["masked"][i] or ""), f"row {i} still carries an address"
        if got["has_phone"][i]:
            assert got["no_phone"][i] != DOCS[i], f"row {i} was detected but not scrubbed"


def test_masking_replaces_with_the_token_it_is_given(ds):
    """The replacement is a parameter, and the default is the documented placeholder."""
    got = ds.select(
        default_url=bt.col("s").str.mask_urls(),
        custom_url=bt.col("s").str.mask_urls("<U>"),
        default_email=bt.col("s").str.mask_emails(),
        custom_email=bt.col("s").str.mask_emails("<E>"),
    ).to_pydict()
    assert "[URL]" in got["default_url"][0]
    assert "<U>" in got["custom_url"][0]
    assert "[EMAIL]" in got["default_email"][5]
    assert "<E>" in got["custom_email"][5]
    assert got["default_url"][6] == DOCS[6], "a document with no URL is untouched"


def test_remove_stopwords_deletes_whole_words_in_either_case(ds):
    """Whole-word only, so ``the`` in ``theory`` survives -- the usual bug in this function."""
    got = _column(ds, bt.col("s").str.remove_stopwords(["the", "over"]))
    assert "quick brown fox jumps" in got[6]
    assert " the " not in f" {got[6]} "
    tricky = bt.from_pydict({"s": ["theory of the theatre", "The end", "another"]})
    cleaned = _column(tricky, bt.col("s").str.remove_stopwords(["the"]))
    assert "theory" in cleaned[0], "a word containing the stopword must survive"
    assert "theatre" in cleaned[0]
    assert "The" not in cleaned[1], "the capitalized form is removed too"
    assert cleaned[2] == "another"


def test_remove_non_ascii_drops_exactly_the_characters_non_ascii_ratio_counts(ds):
    """The scrubber and the ratio must agree on what "non-ASCII" means."""
    got = ds.select(
        cleaned=bt.col("s").str.remove_non_ascii(), ratio=bt.col("s").str.non_ascii_ratio()
    ).to_pydict()
    assert got["cleaned"] == _reference(lambda d: "".join(c for c in d if ord(c) <= 127))
    for i, doc in enumerate(DOCS):
        if doc is None:
            continue
        removed = len(doc) - len(got["cleaned"][i])
        assert removed == round(got["ratio"][i] * len(doc)), f"row {i}"


def test_titlecase_matches_polars(ds):
    """The docstring names Polars as the reference spelling, so it is checked against it."""
    polars = pytest.importorskip("polars")
    got = _column(ds, bt.col("s").str.to_titlecase())
    want = (
        polars.DataFrame({"s": DOCS}).select(polars.col("s").str.to_titlecase()).to_series()
    ).to_list()
    assert got == want, f"{got}\nvs polars\n{want}"


def test_squad_normalize_is_the_published_answer_normalization(ds):
    """Lowercase, drop standalone articles, delete punctuation, collapse spaces, trim."""

    got = _column(ds, bt.col("s").str.squad_normalize())
    assert got == _reference(_squad_reference), f"{got}"


def test_squad_normalize_deletes_articles_before_punctuation(ds):
    """A departure from the published order, pinned with the input that shows it.

    SQuAD's own ``normalize_answer`` is
    ``white_space_fix(remove_articles(remove_punc(lower(s))))`` -- punctuation first. Doing
    articles first, as Batcher does, changes the answer only when an article sits against
    punctuation, because removing the punctuation would otherwise glue it to its
    neighbour. On the answers this normalization exists for the two agree exactly, which
    is asserted here alongside the case where they do not, so the choice is visible rather
    than discovered later by a metric that moved.
    """
    ordinary = ["The Eiffel Tower", "an apple", "Paris, France", "  A  Cat.  "]
    published = []
    for text in ordinary:
        lowered = text.lower()
        no_punctuation = re.sub(r"[^\w\s]", "", lowered)
        published.append(
            re.sub(r"\s+", " ", re.sub(r"\b(a|an|the)\b", " ", no_punctuation)).strip()
        )
    got = _column(bt.from_pydict({"s": ordinary}), bt.col("s").str.squad_normalize())
    assert got == published, "the two orders must agree on ordinary answers"

    glued = "see https://ex.com/a and /an/ here"
    ours = _column(bt.from_pydict({"s": [glued]}), bt.col("s").str.squad_normalize())[0]
    theirs = re.sub(
        r"\s+",
        " ",
        re.sub(r"\b(a|an|the)\b", " ", re.sub(r"[^\w\s]", "", glued.lower())),
    ).strip()
    assert ours != theirs, (
        "the fixture must contain an article against punctuation, or this proves nothing"
    )
    assert ours == _squad_reference(glued)


def test_squad_normalize_makes_two_spellings_of_one_answer_equal():
    """The property the normalization exists for, which the value comparison cannot show."""
    variants = ["The Answer!", "answer", "  an ANSWER  ", "answer."]
    got = _column(bt.from_pydict({"s": variants}), bt.col("s").str.squad_normalize())
    assert len(set(got)) == 1, f"variants of one answer did not normalize together: {got}"
    assert got[0] == "answer"


#: ``(method, argument, Python reference)`` for the truncation family.
TRUNCATIONS = [
    ("truncate_chars", 10, lambda d: d[:10]),
    # The reference keeps the original spacing rather than re-joining on a single space:
    # `truncate_words` returns a prefix of its input, so a newline between two words
    # survives. Re-joining would assert a normalization the method does not perform.
    ("truncate_words", 3, lambda d: _prefix_of_words(d, 3)),
]


@pytest.mark.parametrize(("method", "argument", "reference"), TRUNCATIONS)
def test_truncation_matches_its_definition(ds, method, argument, reference):
    """Cutting on the boundary the method names."""
    got = _column(ds, getattr(bt.col("s").str, method)(argument))
    assert got == _reference(reference), f"{method}: {got}"


def test_every_truncation_shortens_without_inventing_characters(ds):
    """The invariant behind all three, which no fixed expected string can express.

    A truncation must never lengthen its input and never introduce a character that was
    not there. Asserted for all three so a future change to the word or sentence splitter
    still has to satisfy it.
    """
    got = ds.select(
        chars=bt.col("s").str.truncate_chars(10),
        words=bt.col("s").str.truncate_words(3),
        sentences=bt.col("s").str.truncate_sentences(1),
    ).to_pydict()
    for i, doc in enumerate(DOCS):
        if doc is None:
            for key in got:
                assert got[key][i] is None
            continue
        assert got["chars"][i] == doc[:10]
        for key in ("words", "sentences"):
            assert len(got[key][i]) <= len(doc), f"{key} grew row {i}"
            assert doc.startswith(got[key][i]) or got[key][i] == "", (
                f"{key} on row {i} is not a prefix of the input"
            )


def test_truncate_sentences_is_empty_when_there_is_no_sentence_mark(ds):
    """The documented behaviour of the sentence family, shared with ``first_sentence``."""
    got = ds.select(
        one=bt.col("s").str.truncate_sentences(1), first=bt.col("s").str.first_sentence()
    ).to_pydict()
    marked = bt.from_pydict({"s": ["One. Two! Three?", "no mark here"]})
    assert _column(marked, bt.col("s").str.truncate_sentences(2)) == ["One. Two!", ""]
    for i in range(len(DOCS) - 1):
        if not re.search(r"[.!?]", DOCS[i]):
            assert got["one"][i] == "", f"row {i} has no sentence mark"
            assert got["first"][i] == ""


def test_a_truncation_argument_below_one_is_refused_rather_than_guessed(ds):
    """``truncate_sentences(0)`` has no sensible answer, so it raises."""
    from batcher import PlanError

    with pytest.raises(PlanError):
        ds.select(v=bt.col("s").str.truncate_sentences(0))


def test_every_cleaner_leaves_a_null_null(ds):
    """One null document through the whole rewriting family."""
    cleaners = {
        "links": bt.col("s").str.remove_markdown_links(),
        "code": bt.col("s").str.remove_code_blocks(),
        "bullets": bt.col("s").str.remove_bullets(),
        "punctuation": bt.col("s").str.remove_repeated_punctuation(),
        "phones": bt.col("s").str.remove_phones(),
        "non_ascii": bt.col("s").str.remove_non_ascii(),
        "stopwords": bt.col("s").str.remove_stopwords(["the"]),
        "emails": bt.col("s").str.mask_emails(),
        "urls": bt.col("s").str.mask_urls(),
        "titled": bt.col("s").str.to_titlecase(),
        "squad": bt.col("s").str.squad_normalize(),
        "chars": bt.col("s").str.truncate_chars(5),
        "words": bt.col("s").str.truncate_words(2),
        "sentences": bt.col("s").str.truncate_sentences(1),
    }
    got = ds.select(**cleaners).to_pydict()
    for key, values in got.items():
        assert values[NULL_ROW] is None, f"{key} did not null on a null document"


def test_the_cleaners_compose_into_one_projection(ds):
    """Chained in a single ``select``, which is how a corpus pipeline actually spells this."""
    cleaned = ds.select(
        v=bt.col("s")
        .str.remove_code_blocks()
        .str.remove_markdown_links()
        .str.mask_emails()
        .str.mask_urls()
        .str.remove_repeated_punctuation()
        .str.truncate_chars(200)
    ).to_pydict()["v"]
    assert cleaned[NULL_ROW] is None
    assert "```" not in cleaned[1]
    assert "@b.com" not in cleaned[5]
    assert "!!!" not in cleaned[3]
    for value in cleaned[:-1]:
        assert len(value) <= 200


def test_streaming_agrees_with_collect(ds):
    """The rewriting family through ``iter_batches``."""
    projected = ds.select(
        a=bt.col("s").str.squad_normalize(),
        b=bt.col("s").str.mask_urls(),
        c=bt.col("s").str.truncate_words(2),
    )
    collected = projected.to_pydict()
    streamed: dict[str, list] = {k: [] for k in collected}
    for batch in projected.iter_batches():
        for key in streamed:
            streamed[key].extend(batch.column(key).to_pylist())
    assert streamed == collected
