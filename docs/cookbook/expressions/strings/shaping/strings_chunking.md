# Text chunking

``chunk`` is the columnar version of the loop everyone writes by hand before indexing. Overlap matters: without it, a sentence spanning a boundary is retrievable from neither chunk, and that is exactly the passage the question was about.

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
- {doc}`/user-guide/transform/expressions`: what an expression is, and how it is evaluated.
- {doc}`/api/relational/expressions`: the complete `Expr` reference.
