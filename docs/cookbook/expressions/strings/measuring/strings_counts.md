# Counting text structure

Counts are the other half of a corpus filter. Length in characters says little. Length in words, sentences, or lines says whether a document is a fragment, a paragraph, or a scraped page, and the entity counts for URLs, hashtags, and mentions find rows that are mostly links rather than prose.

The script counts words, long words, lines, newlines, spaces, and sentences, measures average word length, and counts URLs, hashtags, and mentions. It ends with the filter these exist for: keep the prose and drop the link dump.

The whole script, executed on every test run:

```{literalinclude} ../../../../../examples/expressions/strings_counts.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/strings_counts.py
```

## See also

- {doc}`/cookbook/expressions/strings/shaping/strings_cleaning`: strip markup, URLs, emails, and stray punctuation.
- {doc}`/cookbook/expressions/strings/matching/strings_extraction`: pulling entities and leading fragments out of free text.
- {doc}`/user-guide/transform/columns/expressions`: what an expression is, and how it is evaluated.
- {doc}`/api/relational/expressions`: the complete {py:class}`Expr <batcher.plan.expr_ir.core.Expr>` reference.
