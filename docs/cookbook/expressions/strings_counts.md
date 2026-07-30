# Counting structure in text: words, lines, sentences, and entities

Counts are the other half of a corpus filter. Length in characters says little; length in words, sentences, or paragraphs says whether a document is a fragment, a paragraph, or a scraped page. The entity counts (urls, emails, hashtags, mentions) find rows that are mostly links rather than prose.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/expressions/strings_counts.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/strings_counts.py
```

## See also

- {doc}`strings_cleaning`: cleaning scraped text: strip markup, URLs, emails, and stray punctuation.
- {doc}`strings_extraction`: pulling entities and leading fragments out of free text.
- {doc}`../../user-guide/expressions`: how expressions are built, evaluated, and combined.
- {doc}`../../api/expressions`: the complete `Expr` reference.
