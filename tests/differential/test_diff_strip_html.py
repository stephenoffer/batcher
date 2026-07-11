"""`.str.strip_html` — text extraction from markup, and why the regex idiom is not it.

DuckDB has no HTML extractor, so there is no oracle to compare digit-for-digit. What is
oracle-able is the thing people *actually write* when they need this —
``regexp_replace(page, '<[^>]*>', '', 'g')`` — and the point of the function is that the
idiom is wrong. So the tests come in two halves:

* **Agreement.** On markup with no script/style, no entities, and no adjacent block
  elements, `strip_html` must agree with DuckDB's regex strip (modulo the whitespace
  normalization it also performs). If it disagreed there, it would be doing something
  surprising rather than something better.
* **Divergence, pinned.** On script bodies, entities, and block boundaries, `strip_html`
  and the regex must differ, in the specific direction that makes the corpus correct.
  These assert *both* sides, so a future "simplification" back to a regex fails loudly.
"""

from __future__ import annotations

import pytest

import batcher as bt

pytestmark = pytest.mark.differential

_STRIP_REGEX = "regexp_replace(page, '<[^>]*>', '', 'g')"


def _batcher(pages: list[str | None]) -> list[str | None]:
    ds = bt.from_pydict({"page": pages})
    return ds.select(t=bt.col("page").str.strip_html()).to_pydict()["t"]


def _duck_regex(duck, pages: list[str | None]) -> list[str | None]:
    import pyarrow as pa

    duck.register("pages", pa.table({"page": pa.array(pages, type=pa.string())}))
    rows = duck.sql(f"SELECT {_STRIP_REGEX} AS t FROM pages").fetchall()
    return [r[0] for r in rows]


def test_it_agrees_with_the_regex_strip_on_plain_markup(duck):
    """No script, no entities, no adjacent blocks — the two must produce the same text."""
    pages = [
        "<b>hello world</b>",
        "the <i>quick</i> brown fox",
        "no markup at all",
        "<a href='http://x.com/?a=1&b=2'>link</a>",
    ]
    assert _batcher(pages) == _duck_regex(duck, pages)


def test_script_bodies_survive_the_regex_and_must_not_survive_strip_html(duck):
    """A regex strips the `<script>` *tags* and leaves the JavaScript as prose."""
    pages = ["real text<script>var x = 1; f();</script>"]
    assert _duck_regex(duck, pages) == ["real textvar x = 1; f();"]
    assert _batcher(pages) == ["real text"]


def test_style_bodies_likewise(duck):
    pages = ["copy<style>p { color: red }</style>"]
    assert _duck_regex(duck, pages) == ["copyp { color: red }"]
    assert _batcher(pages) == ["copy"]


def test_entities_stay_encoded_under_the_regex_and_are_decoded_here(duck):
    pages = ["Tom &amp; Jerry &lt;3 &#39;quotes&#39; &nbsp;end"]
    assert _duck_regex(duck, pages) == ["Tom &amp; Jerry &lt;3 &#39;quotes&#39; &nbsp;end"]
    assert _batcher(pages) == ["Tom & Jerry <3 'quotes' end"]


def test_adjacent_block_elements_are_welded_by_the_regex_and_separated_here(duck):
    """`<p>a</p><p>b</p>` is two words, not one. The regex makes it `ab`."""
    pages = ["<p>alpha</p><p>beta</p>"]
    assert _duck_regex(duck, pages) == ["alphabeta"]
    assert _batcher(pages) == ["alpha beta"]


def test_comments_are_dropped():
    assert _batcher(["a<!-- hidden <b>note</b> -->b"]) == ["ab"]


def test_nulls_propagate_and_empties_stay_empty():
    assert _batcher([None, "", "<div></div>", "   "]) == [None, "", "", ""]


def test_malformed_markup_never_raises():
    """A scrape of the open web contains every one of these; none may abort the scan.

    The last case pins the lenient rule rather than a guess: `<` opens a tag only when a
    `>` follows (so `a<b` and `2 < 3` are text), and the tag ends at the *first* `>` — so
    in `<<>>` the run `<<>` is a tag and the trailing `>` is literal text.
    """
    pages = ["a<b", "2 < 3", "<script>unterminated", "&notanentity;", "&#zz;", "<<>>"]
    assert _batcher(pages) == ["a<b", "2 < 3", "", "&notanentity;", "&#zz;", ">"]


def test_unicode_is_preserved_and_never_split():
    assert _batcher(["<p>héllo 🌍 café</p>"]) == ["héllo 🌍 café"]


def test_whitespace_collapses_and_edges_trim():
    assert _batcher(["  <p>\n  a   \t b \n </p>  "]) == ["a b"]


def test_it_composes_with_the_rest_of_the_ingest_chain():
    """strip_html -> chunk -> explode is the RAG ingest spine; it must stay one plan."""
    ds = bt.from_pydict({"page": ["<p>abcdef</p><p>ghi</p>"]})
    chunks = (
        ds.select(text=bt.col("page").str.strip_html())
        .select(chunk=bt.col("text").str.chunk(4))
        .explode("chunk")
        .to_pydict()["chunk"]
    )
    # The extracted text is "abcdef ghi" (10 chars) — the block boundary became a space,
    # which is the whole point: chunk 2 spans it rather than fusing `abcdefghi`.
    assert chunks == ["abcd", "ef g", "hi"]


def test_it_survives_multi_batch_input():
    import pyarrow as pa

    table = pa.table({"page": [f"<p>row {i}</p>" for i in range(100)]})
    ds = bt.from_arrow(table.to_batches(max_chunksize=7))
    got = ds.select(t=bt.col("page").str.strip_html()).to_pydict()["t"]
    assert got == [f"row {i}" for i in range(100)]
