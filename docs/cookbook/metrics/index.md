# Metrics cookbook

Model and text metrics computed as aggregate expressions, so evaluation is a `select` over the table rather than a pull into pandas.

Every page here embeds a complete, self-contained script from the
[`examples/metrics/`](https://github.com/batcher/batcher/tree/main/examples/metrics) directory.
The scripts build their own in-memory data and assert on their own output, and
`tests/docs/test_examples.py` runs all of them, so a page that stops matching the
engine fails the suite instead of drifting.

| Recipe | What it shows |
|---|---|
| {doc}`agreement` | Agreement metrics: how well a prediction tracks the truth, not just how close |
| {doc}`classification` | Classification metrics computed as aggregates over a predictions table |
| {doc}`diagnostic` | Diagnostic metrics: the epidemiology-style view of a binary classifier |
| {doc}`embeddings` | Corpus-level embedding metrics: monitoring a vector column in aggregate |
| {doc}`probabilistic_losses` | Losses that score a probability or a margin rather than a hard label |
| {doc}`regression_errors` | Regression error metrics: absolute, squared, percentage, and robust |
| {doc}`text_diversity` | Degeneracy detection: repetition, truncation, refusal, and empty output |
| {doc}`text_formatting` | Did the model obey the output format you asked for? |
| {doc}`text_length` | Length and readability distribution over a text column |
| {doc}`text_overlap` | Comparing a generated answer against a reference, without a model |
| {doc}`text_pii_safety` | PII leak rates over a text column |
| {doc}`text_quality` | Corpus hygiene rates: what fraction of a text column looks broken |
| {doc}`text_retrieval` | RAG groundedness: is the answer actually supported by the retrieved context? |
| {doc}`text_tone_and_script` | Tone and writing-system rates: style drift and language mix |

```{toctree}
:hidden:

agreement
classification
diagnostic
embeddings
probabilistic_losses
regression_errors
text_diversity
text_formatting
text_length
text_overlap
text_pii_safety
text_quality
text_retrieval
text_tone_and_script
```
