# Splitting long documents into overlapping chunks for a RAG index

``chunk`` is the columnar version of the loop everyone writes by hand before indexing. Overlap matters: without it, a sentence spanning a boundary is retrievable from neither chunk, and that is exactly the passage the question was about.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/expressions/strings_chunking.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/strings_chunking.py
```

## See also

- {doc}`strings_case`: string case: normalizing capitalization before you compare or group.
- {doc}`strings_cleaning`: cleaning scraped text: strip markup, URLs, emails, and stray punctuation.
- {doc}`../../user-guide/expressions`: how expressions are built, evaluated, and combined.
- {doc}`../../api/expressions`: the complete `Expr` reference.
