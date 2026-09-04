# Text-metric audit: what the quality signals actually compute

**Status:** audit, 2026-09-02. Internal working document, excluded from the published site.

The `.str` namespace carries 132 zero-argument text-quality metrics -- `alpha_ratio`,
`avg_word_length`, `sentence_count`, `mean_line_length` -- the signals a pretraining or
retrieval pipeline filters documents on. Most have no DuckDB equivalent, so the differential
oracle the rest of the engine leans on does not reach them, and until 97db6545 they had no
automated check of any kind.

This records what a value-level audit of them found. The method is the part worth reusing:
**check each metric against an independent implementation, not against a property.** Property
tests -- null in, null out; a ratio inside `[0, 1]`; a count that is not negative -- pass for
a metric that computes the wrong number, and every one of these metrics passed them.

## The one defect: `mean_line_length` counts the newlines

`mean_line_length` is `len(s) / line_count(s)`, so the line *separators* are counted as
characters of the lines they separate. The mean is therefore biased upward by
`(line_count - 1) / line_count` characters, approaching one full character per line.

| Document | True mean | Reported | Error |
|---|---|---|---|
| 5 lines of one character | 1.00 | 1.80 | **+80%** |
| 20-line link dump, 4 characters each | 4.00 | 4.95 | **+24%** |
| `"ab\ncdef"` | 3.00 | 3.50 | +17% |
| a single 50-character line | 50.00 | 50.00 | 0% |

The bias is worst exactly where the metric is used. Its own docstring says "short means mark
navigation and link dumps", so it exists to find documents with short lines -- and the
shorter the lines, the more a fixed +1 per line distorts the answer. A threshold tuned to
drop documents under, say, 3 characters per line will keep a document of 2-character lines,
which reports 2.9.

**Its docstring's example enshrines the wrong value.** `"ab\ncdef"` is documented as `3.5`;
the lines are `ab` and `cdef`, of two and four characters, whose mean is `3.0`. So the
doctest passes and pins the defect, which is why nothing caught it: the example was written
from the implementation rather than from the definition.

The fix is one expression -- subtract the separators before dividing:

```python
# docs: skip
return (self.len() - self.regexp_count("\n")) / self.line_count()
```

Not applied here. `plan/expr_ir/namespaces/strings.py` had 86 uncommitted lines from another
session at the time, elsewhere in the file, and `git commit --only` takes a whole path.
Landing it means changing the docstring example in the same commit, since that example is
currently the specification.

## Four divergences that were not defects

Recorded because each cost time to run down, and each would look like a defect again to the
next person who checks these against a reasonable-seeming reference.

| Metric | Reference said | Batcher says | Why Batcher is right |
|---|---|---|---|
| `sentence_count` | `"Hello World"` is 1 sentence | 0 | It counts sentence-*ending punctuation*, which its docstring states. A string with no terminator has none. |
| `avg_word_length` | `"abc123"` averages 6.0 | 3.0 | It counts *letters* per word, not characters, which its docstring states. |
| `alnum_ratio` | `"café"` is 1.0 | 0.75 | ASCII-only, consistent with `alpha_ratio`, whose docstring says "ASCII letters". |
| `digit_to_word_ratio` | a ratio is in `[0, 1]` | 3.0 for `"123"` | It is digits *per word*, not a fraction. A Gopher-style threshold, correctly unbounded. |

`alnum_ratio` is the one to tidy: its docstring says "letters or digits" where the sibling it
matches says "ASCII letters", so the restriction is real but unstated.

## What agreed

Twenty metrics were checked value-for-value against independent Python implementations over
eighteen inputs -- unicode, empty, whitespace-only, URLs, emails, hashtags, tabs, newlines,
punctuation-only -- and all twenty agreed: `digit_count`, `space_count`, `tab_count`,
`newline_count`, `line_count`, `non_ascii_count`, `paren_count`, `quote_count`,
`hashtag_count`, `mention_count`, `url_count`, `email_count`, `uppercase_word_count`,
`has_digits`, `has_url`, `has_email`, `is_alpha`, `is_alnum`, `is_numeric`, `is_space`.

Eight more agreed on the second pass: `alpha_ratio`, `digit_ratio`, `whitespace_ratio`,
`word_count`, `remove_digits`, `remove_urls`, `remove_emails`, `remove_html_tags`.

So of 32 metrics checked against an independent implementation, 31 compute what they claim
and one does not.

## See also

- `tests/unit/test_text_metric_invariants.py` -- the property half, over all 132.
- `tests/unit/test_temporal_and_list_accessor_invariants.py` -- the same for `.dt` and `.list`.
- `docs/architecture/internals/competitive_architecture.md` -- the code-checked scorecard.
