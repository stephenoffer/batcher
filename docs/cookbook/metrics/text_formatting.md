# Did the model obey the output format you asked for?

Format compliance is the cheapest eval there is and the one that catches the most regressions. If you asked for JSON and ``valid_json_rate`` drops to 0.7, that is a production incident regardless of how good the prose is.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/metrics/text_formatting.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/metrics/text_formatting.py
```

## See also

- {doc}`text_diversity`: degeneracy detection: repetition, truncation, refusal, and empty output.
- {doc}`text_length`: length and readability distribution over a text column.
- {doc}`../../ml/evaluation`: scoring a model, per segment, in one pass.
- {doc}`../../api/metrics`: the complete metric vocabulary.
