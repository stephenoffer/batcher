# Multimodal data

This section covers turning media into columns a model can read: fetching the bytes, decoding them into tensors, curating what comes back, and moving the result through a pipeline without paying for it twice.

A multimodal pipeline turns references such as URLs and file paths into bytes, decodes
them into tensors, and feeds a model. Each step is a lazy operator that runs on whole
batches and parallelizes across the cluster.

## In this section

| Page | Covers |
|---|---|
| {doc}`/ml/preparing/multimodal/decoding` | Getting the bytes, and turning them into tensors. |
| {doc}`/ml/preparing/multimodal/video` | Sampling frames, pulling stills, and reading a clip without decoding it. |
| {doc}`/ml/preparing/multimodal/curating` | A scraped corpus is mostly rows that decode perfectly and teach a model nothing. |
| {doc}`/ml/preparing/multimodal/pipelines` | What changes once media is a column: what it costs to move, how it reaches a model, and how it is retrieved. |

## See also

- {doc}`Inference </ml/inference/inference>`: run a model over the decoded tensors.
- {doc}`Preprocessors </ml/preparing/preprocessors/index>`: assemble the decoded features into a training matrix.
- {doc}`Expressions API </api/relational/expressions>`: the {py:class}`.image <batcher.plan.expr_ir.image._ImageNamespace>`/{py:class}`.audio <batcher.plan.expr_ir.audio._AudioNamespace>`/{py:class}`.video <batcher.plan.expr_ir.video._VideoNamespace>` and vector
  {py:class}`.list <batcher.plan.expr_ir.namespaces.collections._ListNamespace>` method reference.

```{toctree}
:hidden:

decoding
video
curating
pipelines
```
