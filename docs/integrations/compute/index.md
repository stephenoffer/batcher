# ML and compute

Not data sources so much as the systems on either side of a model: where the work is
scheduled, where the tensors go, and where the weights come from.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`git-merge;1.1em` Ray
:link: /integrations/compute/ray
:link-type: doc
Scheduling only. Bulk data moves over Arrow Flight, not the object store.
:::

:::{grid-item-card} {octicon}`zap;1.1em` PyTorch
:link: /integrations/compute/pytorch
:link-type: doc
Streaming tensors into a training loop, and a shard per DDP rank.
:::

:::{grid-item-card} {octicon}`beaker;1.1em` Hugging Face
:link: /integrations/compute/huggingface
:link-type: doc
Datasets in with no copy; model ids that load once per worker.
:::

:::{grid-item-card} {octicon}`server;1.1em` Batch schedulers
:link: /integrations/compute/schedulers
:link-type: doc
Slurm, PBS, LSF, Kubernetes and the managed job services: sizing to the allocation, not the node.
:::

::::

```{toctree}
:hidden:

ray
schedulers
pytorch
huggingface
```
