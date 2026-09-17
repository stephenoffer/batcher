# Output format checks

Format compliance is the cheapest eval there is. If you asked for JSON and `valid_json_rate` drops to 0.7, that is a production incident however good the prose is.

The script scores eight outputs for JSON, bullet and numbered lists, headings, code blocks, tables, links, and numeric, boxed, or tagged answers. It ends on the gate you would put in CI, and this sample fails it.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/metrics/text_formatting.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/metrics/text_formatting.py
```

## See also

- {doc}`/cookbook/metrics/text/text_diversity`: repetition, truncation, refusal, and empty output.
- {doc}`/cookbook/metrics/text/text_length`: length and readability distribution over a text column.
- {doc}`/ml/evaluation/evaluation`: scoring a model, per segment, in one pass.
- {doc}`/api/models/metrics`: the complete metric vocabulary.
