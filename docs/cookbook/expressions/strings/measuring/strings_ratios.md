# Character-class ratios

Character-class ratios are the cheap filters that keep junk out of a training set. A row that is mostly digits is probably a table dump. One that is mostly uppercase is probably a shouting header. One with a high non-ASCII ratio may be the wrong language or mojibake.

The script computes the alpha, alphanumeric, digit, uppercase, lowercase, whitespace, punctuation, and non-ASCII ratios, each a float between 0 and 1, over three rows chosen to trip them: prose, a number dump, and a shouted header. It then filters out the last two.

The whole script, executed on every test run:

```{literalinclude} ../../../../../examples/expressions/strings_ratios.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/strings_ratios.py
```

## See also

- {doc}`/cookbook/expressions/strings/matching/strings_predicates`: the screen in front of an expensive stage.
- {doc}`/cookbook/expressions/strings/matching/strings_regex`: extract, replace, and count.
- {doc}`/user-guide/transform/columns/expressions`: what an expression is, and how it is evaluated.
- {doc}`/api/relational/expressions`: the complete {py:class}`Expr <batcher.plan.expr_ir.core.Expr>` reference.
