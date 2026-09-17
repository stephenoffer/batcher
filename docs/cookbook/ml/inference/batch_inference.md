# Batch inference

`map_batches` hands your callable a whole pyarrow `RecordBatch`, never one row, so `batch["col"]` is an Arrow array. Call `.to_pylist()` once per batch rather than indexing it element by element. Pass a *class* rather than a function, and the model loads once per worker instead of once per batch.

The script runs a stand-in model both ways, streams its output with `iter_batches`, and filters before `map_batches` so the model never sees rows you would discard. It also scores the same rows as a plain expression. When the logic can be written as one, it runs in Rust and never crosses into Python.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/ml/batch_inference.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/ml/batch_inference.py
```

## See also

- {doc}`/cookbook/ml/inference/vector_search`: retrieval over an embedding column, as a projection plus a top-k.
- {doc}`/cookbook/ml/pipelines/multimodal/image-classification`: the same idiom with a real model on a GPU.
- {doc}`/cookbook/streaming/streaming-inference`: a resident model over micro-batches.
- {doc}`/ml/inference/batch-scoring`: filtering before the model, and sizing the batch to the device.
