# Scoring generated text

Reference-free monitors first, since they need no labels, then the ones that compare against a reference.

Each page embeds a complete, self-contained script that builds its own in-memory data and asserts on its own output, so a page that stops matching the engine fails the suite instead of drifting.

| Recipe | What it shows |
|---|---|
| {doc}`/cookbook/metrics/text/text_formatting` | Did the model obey the output format you asked for? |
| {doc}`/cookbook/metrics/text/text_diversity` | Repetition, truncation, refusal, and empty output |
| {doc}`/cookbook/metrics/text/text_quality` | What fraction of a text column looks broken |
| {doc}`/cookbook/metrics/text/text_length` | Length and readability distribution over a text column |
| {doc}`/cookbook/metrics/text/text_tone_and_script` | Style drift and language mix |
| {doc}`/cookbook/metrics/text/text_pii_safety` | PII leak rates over a text column |
| {doc}`/cookbook/metrics/text/text_overlap` | Comparing an answer against a reference, without a model |
| {doc}`/cookbook/metrics/text/text_retrieval` | Whether the answer is supported by the retrieved context |

```{toctree}
:hidden:

text_formatting
text_diversity
text_quality
text_length
text_tone_and_script
text_pii_safety
text_overlap
text_retrieval
```
