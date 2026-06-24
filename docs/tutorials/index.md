# Tutorials

Worked, end-to-end walkthroughs. Each builds a small pipeline against the real API
and runs as written. Start with the first pipeline, then move to the workload that
matches what you are building.

- [Your first pipeline](first-pipeline.md): build, transform, aggregate, sort, and
  collect a dataset, then point the same code at files.
- [Batch inference](batch-inference.md): run a model over Arrow batches with the
  `.ml` accessor.
- [Synthetic data generation](synthetic-data-generation.md): build test datasets in
  memory with Python and {py:obj}`bt.from_pydict <batcher.from_pydict>`.

```{toctree}
:maxdepth: 1

first-pipeline
batch-inference
synthetic-data-generation
```
