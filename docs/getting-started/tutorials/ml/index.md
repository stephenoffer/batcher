# Models and features

These tutorials put a model inside a Batcher pipeline. Your function receives whole Arrow batches, a model class loads once per worker, and the data stays columnar from the scan to the tensor.

```python
import batcher as bt
import pyarrow.compute as pc

ds = bt.from_pydict({"feature": [0.5, 1.5, 2.5]})
scored = ds.map_batches(lambda b: b.append_column("score", pc.multiply(b.column("feature"), 2.0)))
print(scored.to_pydict())
# {'feature': [0.5, 1.5, 2.5], 'score': [1.0, 3.0, 5.0]}
```

Every tutorial runs on a laptop with `pip install batcher-engine`. A stub stands in wherever a GPU or a model download would be needed, with the real call shown beside it, so you learn the pipeline shape first and swap in the model later.

The following table lists the four tutorials:

| Tutorial | What you build |
|---|---|
| {doc}`Batch inference <batch-inference>` | A scoring function run over whole Arrow batches through the `.ml` accessor, then the class form that loads a model once per worker |
| {doc}`RAG from scratch <rag-from-scratch>` | Retrieval and generation as ordinary dataset work: chunk, embed, score with a top-N, and generate |
| {doc}`Feature engineering <feature-engineering>` | A model-ready feature matrix from a raw table, with preprocessors fitted on train and replayed on test |
| {doc}`A distributed training pipeline <distributed-training-pipeline>` | A loader that hands each data-parallel PyTorch rank a balanced, deterministic, resumable stream |

Start with batch inference if you're new to the `.ml` accessor. The other three can be taken in any order.

## See also

- {doc}`/ml/index`: the guide behind these tutorials, one capability per page.
- {doc}`/cookbook/ml/index`: shorter, focused ML recipes.
- {doc}`/benchmarks/results/ai-and-gpu`: the throughput these pipelines reach on real GPUs.
- {doc}`../pipelines/index`: the data-engineering tutorials that feed a model its inputs.

```{toctree}
:hidden:

batch-inference
rag-from-scratch
feature-engineering
distributed-training-pipeline
```
