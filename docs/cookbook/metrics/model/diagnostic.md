# Diagnostic metrics

Accuracy hides everything on an imbalanced problem. Likelihood ratios, informedness, and markedness describe how much a prediction moves your belief, which is the number you want when positives are rare.

The script reuses the confusion matrix from {doc}`classification`, so the false discovery rate, false omission rate, Jaccard score, and Hamming loss it asserts can all be checked against those four counts.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/metrics/diagnostic.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/metrics/diagnostic.py
```

## See also

- {doc}`/cookbook/metrics/model/classification`: classification metrics computed as aggregates over a predictions table.
- {doc}`/cookbook/metrics/model/probabilistic_losses`: losses that score a probability or a margin rather than a hard label.
- {doc}`/ml/evaluation/evaluation`: scoring a model, per segment, in one pass.
- {doc}`/api/models/metrics`: the complete metric vocabulary.
