# Running a model over data

The two ways a fitted model meets a full table.

Each page embeds a complete, self-contained script that builds its own in-memory data and asserts on its own output, so a page that stops matching the engine fails the suite instead of drifting.

| Recipe | What it shows |
|---|---|
| {doc}`/cookbook/ml/inference/batch_inference` | A model over every row, without a Python loop |
| {doc}`/cookbook/ml/inference/vector_search` | Vector search over an embedding column, in the engine |

```{toctree}
:hidden:

batch_inference
vector_search
```
