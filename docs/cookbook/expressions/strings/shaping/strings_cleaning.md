# Cleaning scraped text

Scraped text needs a cleaning pass before it goes to an embedding or LLM stage. Each call here is one columnar operator, so a chain of ten of them reads the column once per operator in Rust rather than materializing Python strings.

The script strips HTML tags, bullets, and digits, removes or masks URLs and emails, collapses repeated punctuation and runs of whitespace, and derives a URL-safe key with `slugify`. Masking rather than removing keeps a visible trace that something was redacted.

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
- {doc}`/user-guide/transform/columns/expressions`: what an expression is, and how it is evaluated.
- {doc}`/api/relational/expressions`: the complete {py:class}`Expr <batcher.plan.expr_ir.core.Expr>` reference.
