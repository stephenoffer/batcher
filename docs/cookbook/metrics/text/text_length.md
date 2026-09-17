# Length and readability

Means hide the tail, which is where cost lives: a token budget is blown by the p99, not the average. `token_estimate_quantile` answers that directly, over a mergeable sketch.

Read the token figures as estimates. They divide a character count by an assumed characters-per-token ratio, which is enough to size a context window or compare two runs and not enough to reconcile an invoice. The script also measures character and word-count quantiles, the share of rows over a token budget, and readability.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/metrics/text_length.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/metrics/text_length.py
```

## See also

- {doc}`/cookbook/metrics/text/text_formatting`: did the model obey the output format you asked for?
- {doc}`/cookbook/metrics/text/text_overlap`: comparing a generated answer against a reference, without a model.
- {doc}`/ml/evaluation/evaluation`: scoring a model, per segment, in one pass.
- {doc}`/api/models/metrics`: the complete metric vocabulary.
