# Binning and clipping

Binning turns a continuous variable into a categorical one, which is how you let a linear model express a non-monotonic effect. Clipping and power transforms attack the other problem: a long tail that dominates the loss.

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
