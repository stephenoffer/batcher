# Binning and clipping

Binning turns a continuous variable into a categorical one, which is how you let a linear model express a non-monotonic effect. Clipping and power transforms attack the other problem: a long tail that dominates the loss.

The script runs `KBinsDiscretizer` over a column with one extreme value, where equal-frequency bins behave and equal-width bins crowd every ordinary value into bin 0. Then it thresholds, clips, log-transforms, power-transforms, and rank-transforms the same column.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/ml/preprocessing_binning.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/ml/preprocessing_binning.py
```

## See also

- {doc}`/cookbook/ml/validation/outlier_detection`: per-column rules and a multivariate distance.
- {doc}`/cookbook/ml/preprocessing/preprocessing_chain`: chaining preprocessors into one fitted pipeline.
- {doc}`/ml/index`: the ML surface these recipes sit on.
- {doc}`/ml/preparing/preprocessors/index`: the fit and transform steps most pipelines start with.
