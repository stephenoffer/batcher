# Length and readability distribution over a text column

Means hide the tail, which is where cost lives: a token budget is blown by the p99, not the average. ``token_estimate_quantile`` answers that directly, and the token estimates turn a character count into the number you actually get billed for.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/metrics/text_length.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/metrics/text_length.py
```

## See also

- {doc}`text_formatting`: did the model obey the output format you asked for?
- {doc}`text_overlap`: comparing a generated answer against a reference, without a model.
- {doc}`../../ml/evaluation`: scoring a model, per segment, in one pass.
- {doc}`../../api/metrics`: the complete metric vocabulary.
