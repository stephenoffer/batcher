# ML and compute

Batcher plugs into the systems on either side of a model: the cluster that schedules the work, the scheduler that granted the hardware, the training loop that consumes tensors, and the registries the weights come from. The data work stays in Batcher's Rust engine over Arrow, and each of these integrations hands off at the boundary where the other system is strongest.

On a Ray cluster, Ray schedules and Arrow Flight carries the shuffle, so bulk data never touches the object store. Under Slurm, PBS, LSF, Kubernetes, or a managed job service, Batcher sizes itself to the allocation you were granted, with no configuration. A PyTorch loop gets `{column: tensor}` batches already on the device, and a model from the Hugging Face Hub or an MLflow registry loads once per worker by its id or URI. A fitted scikit-learn, XGBoost, or LightGBM model needs no wrapper at all: `ds.ml.predict` calls the scikit-learn contract on whatever object you hand it.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`git-merge;1.1em` Ray
:link: /integrations/compute/ray
:link-type: doc
One argument takes a plan distributed. Ray schedules, and bulk data moves over Arrow Flight.
:::

:::{grid-item-card} {octicon}`server;1.1em` Batch schedulers
:link: /integrations/compute/schedulers
:link-type: doc
Slurm, PBS, LSF, Kubernetes, and managed job services. Sized to the allocation, not the node.
:::

:::{grid-item-card} {octicon}`zap;1.1em` PyTorch
:link: /integrations/compute/pytorch
:link-type: doc
Streaming tensors into a training loop, an equal shard per DDP rank, and load-once inference.
:::

:::{grid-item-card} {octicon}`beaker;1.1em` Hugging Face
:link: /integrations/compute/huggingface
:link-type: doc
Datasets in with zero copy, and Hub model ids that load once per worker.
:::

:::{grid-item-card} {octicon}`package;1.1em` MLflow
:link: /integrations/compute/mlflow
:link-type: doc
Score a `models:/` reference directly, resolved on each worker with its own credentials.
:::

:::{grid-item-card} {octicon}`beaker;1.1em` scikit-learn
:link: /integrations/compute/scikit-learn
:link-type: doc
Score a fitted estimator or `Pipeline` over a dataset, and the in-engine estimators for when training data doesn't fit either.
:::

:::{grid-item-card} {octicon}`git-branch;1.1em` Gradient boosting
:link: /integrations/compute/gradient-boosting
:link-type: doc
XGBoost, LightGBM, and CatBoost through the same contract, and the batch size that decides the throughput.
:::

::::

For the full batch-inference and training surface these pages build on, see {doc}`/ml/index`.

## See also

- {doc}`/ml/inference/index`: the inference pipeline these runtimes plug into.
- {doc}`/integrations/orchestration/index`: scheduling the job that calls them.
- {doc}`/user-guide/operate/running/index`: sizing a run, and what to watch while it goes.
- {doc}`/integrations/dataframes/index`: the in-process libraries on the other side of the model.
- {doc}`/examples/accelerators`: accelerator scripts, each run on every commit.

```{toctree}
:hidden:

ray
schedulers
pytorch
scikit-learn
gradient-boosting
huggingface
mlflow
```
