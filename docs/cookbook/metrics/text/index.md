# Scoring generated text

These metrics turn a column of model output into a scorecard: format compliance, degenerate generations, broken text, length and token cost, tone, PII leaks, and grounding. Each one is an aggregate, so a whole generation run scores in one `select` with no judge model and no GPU.

The reference-free monitors come first, since they need no labels and can gate every batch. The last two compare an answer against a reference or against the context it was retrieved from. Each page embeds a self-contained script that asserts on its own output.

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

## See also

- {doc}`/ml/retrieval/llm-evaluation`: the same monitors applied to a generation pipeline.
- {doc}`/cookbook/ml/pipelines/text/llm-batch-scoring`: producing the output these pages score.
- {doc}`/cookbook/metrics/embeddings`: the aggregate checks for an embedding column.
- {doc}`/api/models/metrics`: the complete metric vocabulary.

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
