# Text chunking

Retrieval-augmented generation needs documents split into overlapping chunks, and `chunk` is the columnar version of the loop everyone writes by hand before indexing. Overlap matters: without it, a sentence spanning a boundary is retrievable from neither chunk, and that is exactly the passage the question was about.

The script produces fixed-size character chunks as a list column, shows that a short document yields one chunk and that more overlap yields more chunks, and chunks on word boundaries for prose. It ends with the shape a RAG index wants: one row per chunk, carrying its source id.

The whole script, executed on every test run:

```{literalinclude} ../../../../../examples/expressions/strings_chunking.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/strings_chunking.py
```

## See also

- {doc}`/cookbook/expressions/strings/shaping/strings_case`: normalizing capitalization before you compare or group.
- {doc}`/cookbook/expressions/strings/shaping/strings_cleaning`: strip markup, URLs, emails, and stray punctuation.
- {doc}`/user-guide/transform/columns/expressions`: what an expression is, and how it is evaluated.
- {doc}`/api/relational/expressions`: the complete {py:class}`Expr <batcher.plan.expr_ir.core.Expr>` reference.
