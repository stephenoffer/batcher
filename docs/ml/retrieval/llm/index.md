# LLM inference

This section covers running a language model over a column: the call itself, the prompt and conversation handling around it, and the engine and throughput decisions underneath.

Offline text generation over millions of rows. The engine loads once per worker and
does its own continuous batching, so Batcher feeds it whole request lists and handles
the surrounding columnar work of building prompts from row columns and parsing structured
output.

## In this section

| Page | Covers |
|---|---|
| {doc}`/ml/retrieval/llm/calling` | The four shapes a generation call takes, from the one-liner to the class UDF. |
| {doc}`/ml/retrieval/llm/prompts` | Building the input, and reading a conversation column back out. |
| {doc}`/ml/retrieval/llm/engines` | Which backend runs the model, what it costs, and the two neighboring modalities. |

## See also

- {doc}`Inference </ml/inference/inference>`: the general batch-inference and embedding path.
- {doc}`Serving </ml/training/serving>`: expose a model behind an endpoint.
- {doc}`GPU scheduling </ml/inference/gpu>`: how `num_gpus` and `concurrency` map to actors.

```{toctree}
:hidden:

calling
prompts
engines
```
