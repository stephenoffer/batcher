# Scoring a supervised model

Every metric on these pages is an aggregate expression over a table of labels and predictions. Scoring a model is one `select`, and scoring it per customer segment is the same expression under a `group_by`, still in one pass.

Start with classification for a hard label, then the diagnostic view when positives are rare. Probabilistic losses score the probability rather than the label. The last two pages cover regression targets: how far off a prediction is, and whether it tracks the truth at all. Each page embeds a self-contained script that asserts on its own output, so the numbers you read are the numbers the engine returns.

| Recipe | What it shows |
|---|---|
| {doc}`/cookbook/metrics/model/classification` | Classification metrics as aggregates over a predictions table |
| {doc}`/cookbook/metrics/model/diagnostic` | The epidemiology-style view of a binary classifier |
| {doc}`/cookbook/metrics/model/probabilistic_losses` | Scoring a probability or a margin rather than a hard label |
| {doc}`/cookbook/metrics/model/regression_errors` | Absolute, squared, percentage, and robust error |
| {doc}`/cookbook/metrics/model/agreement` | How well a prediction tracks the truth, not just how close |
| {doc}`/cookbook/metrics/model/embeddings` | Corpus-level health checks for a vector column, in aggregate |

## See also

- {doc}`/cookbook/metrics/text/index`: scoring generated text rather than a label.
- {doc}`/cookbook/ml/validation/index`: cross-validation, so the score you compute here is an honest one.
- {doc}`/ml/evaluation/evaluation`: the guide to evaluating a model per segment.
- {doc}`/api/models/metrics`: the complete metric vocabulary.

```{toctree}
:hidden:

classification
diagnostic
probabilistic_losses
regression_errors
agreement
embeddings
```
