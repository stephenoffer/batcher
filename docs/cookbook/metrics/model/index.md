# Scoring a supervised model

Predictions against labels, for both the classification and the regression cases.

Each page embeds a complete, self-contained script that builds its own in-memory data and asserts on its own output, so a page that stops matching the engine fails the suite instead of drifting.

| Recipe | What it shows |
|---|---|
| {doc}`/cookbook/metrics/model/classification` | Classification metrics as aggregates over a predictions table |
| {doc}`/cookbook/metrics/model/diagnostic` | The epidemiology-style view of a binary classifier |
| {doc}`/cookbook/metrics/model/probabilistic_losses` | Scoring a probability or a margin rather than a hard label |
| {doc}`/cookbook/metrics/model/regression_errors` | Absolute, squared, percentage, and robust error |
| {doc}`/cookbook/metrics/model/agreement` | How well a prediction tracks the truth, not just how close |

```{toctree}
:hidden:

classification
diagnostic
probabilistic_losses
regression_errors
agreement
```
