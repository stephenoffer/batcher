# Classification metrics

Batcher's classification metrics are aggregate expressions over a table of labels and predictions, so evaluation is a `select` rather than a pull into pandas.

The script builds a ten-row predictions table with a known confusion matrix, so every score can be checked by hand: accuracy 0.7, precision 0.6, recall 0.75. Put the same expressions under a `group_by` and you get the metrics per segment in the same pass. On a table too big for memory, that difference is the whole ballgame.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/metrics/classification.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/metrics/classification.py
```

## See also

- {doc}`/cookbook/metrics/model/agreement`: how well a prediction tracks the truth, not just how close.
- {doc}`/cookbook/metrics/model/diagnostic`: the epidemiology-style view of a binary classifier.
- {doc}`/ml/evaluation/evaluation`: scoring a model, per segment, in one pass.
- {doc}`/api/models/metrics`: the complete metric vocabulary.
