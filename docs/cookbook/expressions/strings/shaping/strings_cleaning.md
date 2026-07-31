# Cleaning scraped text

This is the pre-processing pass in front of an embedding or LLM stage. Each call is one columnar operator, so a chain of ten of them still reads the column once per operator in Rust rather than materializing Python strings.

The whole script, executed on every test run:

```{literalinclude} ../../../../../examples/expressions/strings_cleaning.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/strings_cleaning.py
```

## See also

- {doc}`/cookbook/expressions/strings/shaping/strings_chunking`: splitting long documents into overlapping chunks for a RAG index.
- {doc}`/cookbook/expressions/strings/measuring/strings_counts`: words, lines, sentences, and entities.
- {doc}`/user-guide/transform/expressions`: what an expression is, and how it is evaluated.
- {doc}`/api/relational/expressions`: the complete `Expr` reference.
