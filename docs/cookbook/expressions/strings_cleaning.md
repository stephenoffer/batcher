# Cleaning scraped text: strip markup, URLs, emails, and stray punctuation

This is the pre-processing pass in front of an embedding or LLM stage. Each call is one columnar operator, so a chain of ten of them still reads the column once per operator in Rust rather than materializing Python strings.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/expressions/strings_cleaning.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/strings_cleaning.py
```

## See also

- {doc}`strings_chunking`: splitting long documents into overlapping chunks for a RAG index.
- {doc}`strings_counts`: counting structure in text: words, lines, sentences, and entities.
- {doc}`../../user-guide/expressions`: how expressions are built, evaluated, and combined.
- {doc}`../../api/expressions`: the complete `Expr` reference.
