# Run an exported model

This page describes how to score a Batcher {py:class}`Dataset <batcher.Dataset>` with a model that was exported out of its training framework: an ONNX graph, a TorchScript archive, or an OpenVINO IR.

A trained model usually leaves the framework it was trained in before it ever runs over a hundred million rows. The exported form loads in a fraction of the time, holds a fraction of the memory, and runs on hardware the training stack was never installed on. Batcher runs all three exported forms **inside the worker**, so there is no serving fleet between the data and the model and the batch the engine already assembled is the batch the model sees.

## Which runtime

Each entry point is a load-once class UDF: the model is built once per worker and every batch goes through it as whole arrays.

| Entry point | Model | Reach for it when |
|---|---|---|
| {py:func}`bt.ml.onnx_predictor <batcher.ml.onnx_predictor>` | An `.onnx` graph | The model was exported. This is the portable default, and the way to reach TensorRT. |
| {py:func}`bt.ml.torch_predictor <batcher.ml.torch_predictor>` | TorchScript, a pickled module, or a factory | The model was never exported, or its forward has Python control flow in it. |
| {py:func}`bt.ml.openvino_predictor <batcher.ml.openvino_predictor>` | An OpenVINO IR, ONNX, or a saved model | The fleet is Intel CPUs and throughput per core is what you are paying for. |

All three are wired through {py:func}`bt.ml.serving_udf <batcher.ml.serving_udf>`, the same adapter the remote clients use. They inherit its behavior: a batch larger than the model's own window is split, results stay in input order, an output whose name the batch already carries replaces it rather than duplicating it, and the model is released when the worker finishes.

## Score an ONNX graph

Name the columns to feed and the columns to append:

```python
# docs: skip
import batcher as bt

udf = bt.ml.onnx_predictor(
    "resnet50.onnx",
    input_columns=["pixel_values"],
    output_columns=["logits"],
)

ds = bt.read.parquet("s3://images/features/")
scored = ds.ml.map_batches(udf, num_gpus=1, concurrency=8)
scored.write.parquet("s3://images/scored/")
```

`input_columns` are fed in the graph's input order. When the column names differ from the graph's own input names, pass `input_names=` to map them rather than renaming the columns in the plan:

```python
# docs: skip
udf = bt.ml.onnx_predictor(
    "bert.onnx",
    input_columns=["tokens", "mask"],
    input_names=["input_ids", "attention_mask"],
    output_columns=["embedding"],
)
```

A multi-output graph appends one column per fetched output. Use `output_names=` to fetch a subset, in the order you want them:

```python
# docs: skip
udf = bt.ml.onnx_predictor(
    "detector.onnx",
    input_columns=["image"],
    output_names=["boxes", "scores"],
    output_columns=["box", "confidence"],
)
```

### Execution providers

The provider list is what decides where the graph runs, and getting it wrong is the classic silent slowdown: a CPU session on a GPU worker still returns correct answers, ten to fifty times slower, and nothing in the result says so.

Leaving `providers` unset auto-selects every accelerated provider the installed build offers, but only when the worker can actually see an accelerator. Name them explicitly to be certain:

```python
# docs: skip
udf = bt.ml.onnx_predictor(
    "model.onnx",
    input_columns=["features"],
    providers=["cuda"],
)
```

Friendly names are accepted alongside ONNX Runtime's own spelling: `"cuda"`, `"tensorrt"`, `"rocm"`, `"openvino"`, `"coreml"`, `"directml"`, `"cpu"`. A name the installed build does not offer is dropped with a warning rather than raising, and `CPUExecutionProvider` is always appended as the terminal fallback so a graph with one unsupported operator still runs.

TensorRT is reached the same way. It compiles the supported subgraphs and leaves the rest on CUDA:

```python
# docs: skip
udf = bt.ml.onnx_predictor(
    "model.onnx",
    input_columns=["features"],
    providers=["tensorrt", "cuda"],
    provider_options={"tensorrt": {"trt_fp16_enable": True}},
)
```

That first batch pays for the compilation, which for a transformer can be minutes. Engine caching is enabled by default so each worker pays it once; point `trt_engine_cache_path` at shared storage to make the whole fleet pay it once.

```{warning}
`providers=["tensorrt"]` builds an engine specialized to the shapes it first sees. A dataset with varying sequence lengths triggers a rebuild per new shape unless the graph was exported with dynamic axes and TensorRT was given an optimization profile. Measure before adopting it.
```

### Fixed batch dimensions

Exporters bake the tracing batch size into the graph unless told not to. Batcher reads that static dimension at load and splits larger batches to match, so a model exported at 32 works against an engine batch of 4,096 without any configuration. Pass `max_batch_size=` to override.

## Score a PyTorch module

A TorchScript archive needs nothing but a path:

```python
# docs: skip
udf = bt.ml.torch_predictor(
    "scripted.pt",
    input_columns=["image"],
    output_columns=["logits"],
    channels_last=True,
)
```

A checkpoint is a `state_dict`, which carries weights but no architecture, so it needs a factory that builds the module and loads the file into it:

```python
# docs: skip
def build():
    import torch
    from my_project import Net

    model = Net()
    model.load_state_dict(torch.load("checkpoint.pt", map_location="cpu"))
    return model


udf = bt.ml.torch_predictor(build, input_columns=["x"], output_columns=["y"])
```

Two things are applied for you, because they are the pair that hand-written inference UDFs most often miss. The module is put in `eval()` mode, without which dropout randomizes predictions and batch normalization updates its running statistics from the data being scored. The forward runs under `torch.inference_mode()`, which frees the activation memory an unused backward graph would otherwise hold and is what lets the batch size go up.

Two more are opt-in, because each can regress a model it does not suit. `channels_last=True` is the layout convolution tensor cores want. `compile=True` runs `torch.compile`, measured as a win on fixed-shape vision models and a regression on dynamic-shape text models.

Precision is left alone unless you ask. `dtype="float16"` is a numerical change, so this path does not make one on your behalf:

```python
# docs: skip
udf = bt.ml.torch_predictor(
    "scripted.pt", input_columns=["image"], output_columns=["logits"], dtype="float16"
)
```

## Score on Intel CPUs

Most batch inference runs on whatever CPU the cluster already has. {py:func}`bt.ml.openvino_predictor <batcher.ml.openvino_predictor>` is the runtime for that case on Intel hardware, and the option that matters is `performance_hint`:

```python
# docs: skip
udf = bt.ml.openvino_predictor(
    "model.xml",
    input_columns=["features"],
    output_columns=["score"],
    performance_hint="THROUGHPUT",
    cache_dir="/mnt/shared/ov-cache",
)
ds.ml.map_batches(udf, concurrency=16).write.parquet("s3://out/")
```

Batcher defaults to `"THROUGHPUT"` where OpenVINO defaults to `"LATENCY"`. Latency mode spreads one inference across every core, which is right for a request-response server and wrong for scoring a table: the cores idle at each stage boundary. Throughput mode keeps several inferences in flight and saturates them.

`cache_dir` is worth setting on any fleet. Compilation takes seconds to minutes and every worker compiles the same model, so a shared cache turns a per-worker cost into a one-time one.

The runtime reads an OpenVINO IR `.xml`, an `.onnx` graph, or a saved TensorFlow model directly, so nothing needs converting first.

## Sizing the stage

The runtimes are ordinary UDFs, so everything on {py:meth}`ds.ml.map_batches <batcher.api.dataset.ml.DatasetML.map_batches>` applies: `num_gpus` and `concurrency` place and size the actor pool, `batch_size` sets the rows per call, and upstream preprocessing stays on CPU workers while the model stays on the accelerator. See {doc}`/ml/inference/gpu`.

Two options belong to the runtimes themselves:

`max_batch_size` splits an engine batch into model-sized calls. Leave it unset for ONNX, which reads the graph's own window.

`pipeline_depth` keeps several of those calls in flight, so the device is not idle while this worker converts the next sub-batch. Above 1 it costs that many sub-batches of memory; results stay in input order either way.

## Requirements and limitations

Each runtime needs its own package: `onnxruntime` (or `onnxruntime-gpu`) for ONNX, `torch` for TorchScript, and `openvino` for the IR path. A missing one raises with the install command rather than an `ImportError`.

Inputs must have a numeric array form. A string or nested column is rejected by name at the batch edge, so tokenize or cast before the model stage.

An ONNX graph does not coerce dtypes. Batcher casts each input to the type the graph declares, but a graph exported for `int64` token ids fed a float column is a cast that loses information, not a conversion.

`bfloat16` graph inputs are fed as float32 and cast by the runtime on ingest, because NumPy has no `bfloat16`. Half-precision *outputs* are widened to float32 on the way back into Arrow.

## See also

- {doc}`/ml/inference/inference` for the batch-first UDF model these build on.
- {doc}`/ml/inference/tabular-models` for scoring a fitted booster or scikit-learn model, including single-input ONNX graphs.
- {doc}`/ml/inference/gpu` for placement, batch sizing, and keeping a device busy.
- {doc}`/ml/training/serving` for calling a model that lives in another process instead.
