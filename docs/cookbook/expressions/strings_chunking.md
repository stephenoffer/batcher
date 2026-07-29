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
