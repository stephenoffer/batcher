# Image classification

The GPU is not the bottleneck in an image-classification job. Decoding JPEGs is. Run the
decode and the forward pass in lockstep and the device sits idle through every decode,
which is how a ResNet-50 pipeline ends up at 942 img/s and ~30% utilization. Overlap them
so the CPU decodes morsel *k+1* while the GPU is still on morsel *k*, and the same hardware
does **2,504 img/s at 81%**. Batcher overlaps stages by default. The job of this
page is to not get in its way.

## Read, decode, classify

`bt.read.images(..., decode=True, size=(h, w))` lists the files, decodes and resizes in
the data plane (SIMD JPEG, SIMD resize, fanned out across every core), and hands you a
fixed-shape `(h, w, 3)` tensor column. Always pass a `size`: a batch of full-resolution
frames will exhaust host memory long before it reaches the model.

The same pipeline shape runs with a real network on GPUs, or with a stand-in on a laptop.
Only the model stage changes.

::::{tab-set}
:::{tab-item} ResNet-50 on GPUs

```python
# docs: skip
import batcher as bt
import pyarrow as pa


class ResNet:
    def __init__(self):
        import torch
        import torchvision

        self.torch = torch
        self.model = torchvision.models.resnet50(weights="DEFAULT").cuda().eval()

    def __call__(self, batch):
        # A tensor column arrives stacked: one (batch, 224, 224, 3) uint8 block.
        images = batch.column("image").to_numpy_ndarray()
        x = self.torch.from_numpy(images).cuda().permute(0, 3, 1, 2).float().div(255)
        with self.torch.no_grad():
            preds = self.model(x).argmax(dim=1).cpu().tolist()
        return batch.append_column("label", pa.array(preds))


images = bt.read.images("s3://bucket/photos/", decode=True, size=(224, 224))
scored = images.ml.infer(
    ResNet,  # the class: loads the weights once per actor
    output_columns=["path", "image", "label"],
    batch_size=256,
    num_gpus=1,
    concurrency=4,
    model_memory_gb=0.1,
)
scored.drop("image").write.parquet("s3://bucket/labels.parquet")
```

`ResNet` is passed as a class, not an instance and not a function. `map_batches` (and
`infer`, which lowers to it) constructs it once per worker; a function would rebuild the
model on every batch.
:::

:::{tab-item} No GPU, no weights

The decode half runs anywhere, with no GPU and no weights, because it is an engine
expression.
`.image.to_tensor(w, h)` decodes and resizes natively; the classifier below is a
brightness threshold standing in for a forward pass, with exactly the shape a real one
has.

```python
import io

import numpy as np
import pyarrow as pa
from PIL import Image

import batcher as bt
from batcher import col

# Two synthetic JPEGs: one dark, one bright.
def jpeg(value):
    buf = io.BytesIO()
    Image.fromarray(np.full((64, 64, 3), value, dtype="uint8")).save(buf, format="JPEG")
    return buf.getvalue()


photos = bt.from_pydict({"path": ["dark.jpg", "bright.jpg"], "bytes": [jpeg(20), jpeg(230)]})


class BrightnessModel:
    """Stands in for the forward pass: same call shape, no weights."""

    def __init__(self, cutoff=128):
        self.cutoff = cutoff

    def __call__(self, batch):
        pixels = np.array(batch.column("image").to_pylist())
        labels = ["day" if row.mean() >= self.cutoff else "night" for row in pixels]
        return batch.append_column("label", pa.array(labels))


scored = (
    photos.with_columns(image=col("bytes").image.to_tensor(32, 32))
    .ml.infer(BrightnessModel, output_columns=["path", "bytes", "image", "label"], batch_size=64)
    .select("path", "label")
)
print(scored.to_pydict())
# {'path': ['dark.jpg', 'bright.jpg'], 'label': ['night', 'day']}
```

Everything before `.ml.infer` is a lazy plan the engine runs on CPU workers while the
model stage holds the GPU. That separation is the whole trick: shape the input in the
engine, hand the actor a ready batch.
:::
::::

:::{warning}
A GPU stage handed a bare function raises a `PerformanceWarning`, and that warning is nearly
always a real 10× on the table. The weights belong in `__init__`, which runs once per worker,
not in the call the engine makes on every batch.
:::

:::{tip}
Leave `batch_size` unset if you do not have a number in mind. The pool starts from
`model_memory_gb` and the free VRAM, then hill-climbs the batch size against measured
throughput. Zero-config (`map_batches(Model, num_gpus=1)` with no `batch_size`) runs at
2,451 img/s and 82% utilization, within 2% of the hand-tuned batch size.
:::

## One corrupt JPEG should cost you one JPEG

:::{important}
Real corpora contain truncated files, HTML error pages saved with a `.jpg` extension, and
CMYK JPEGs that decode to four channels. A six-hour job that dies at hour five on one of
them is the default outcome in most engines. Batcher's error tolerance is per row rather
than per block, so with about 1% corrupt rows injected across 200k rows it keeps 99% of the
data and loses only the bad rows.
:::

Two knobs get you there. `on_error="null"` on the fetch turns a failed download into a
null instead of an exception. `max_errored_rows` on the model stage bisects a batch whose
`fn` raised, drops the offending rows (up to that budget, per worker), and carries on. A
corrupt image costs one row, while a genuine bug on clean data still fails fast once the
budget is spent.

:::{dropdown} The fetch → decode → score pipeline, error-tolerant end to end

```python
# docs: skip
import batcher as bt

catalog = bt.read.parquet("s3://bucket/catalog.parquet")  # a "url" column
scored = (
    catalog.ml.download("url", output_column="bytes", on_error="null", max_concurrency=32)
    .filter(bt.col("bytes").is_not_null())
    .with_columns(
        image=bt.col("bytes").image.to_tensor_f32(
            224,
            224,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            channels_first=True,
        )
    )
    .ml.infer(ResNet, batch_size=256, num_gpus=1, concurrency=4, max_errored_rows=1000)
)
```

`.image.to_tensor_f32(w, h, mean=, std=, channels_first=)` does the full torchvision
`ToTensor` + `Normalize` step natively: it decodes, resizes, scales to `[0, 1]`, applies
the per-channel ImageNet mean/std, and emits a channel-first `float32` tensor. The
whole preprocessing chain stays in the engine, and the model receives a ready tensor with
no per-batch Python (`/255`, `Normalize`, `permute`). Use the plain `.image.to_tensor(w, h)`
when the model wants raw `uint8` HWC pixels instead.

When the model was trained with the classic *resize-then-crop* recipe (`Resize(256)` then
`CenterCrop(224)`), `.image.center_crop(w, h)` is the crop half. It decodes and takes the
centered `(w, h)` window, zero-padding a too-small image the way torchvision `CenterCrop`
does, so you can chain it with the tensor step to match the model's exact eval
transform. For a model that takes a single-channel input, `.image.to_grayscale(w, h)` decodes,
resizes, and reduces to one Rec.601 luminance channel (`(h, w, 1)`) in the same native pass.
:::

Count what you dropped rather than trusting that you dropped nothing:

```python
import batcher as bt
from batcher import col

fetched = bt.from_pydict({"url": ["a.jpg", "b.jpg", "c.jpg"], "bytes": [b"x", None, b"z"]})
print(fetched.filter(col("bytes").is_null()).count())
# 1
```

If that number is not roughly what you expect, the problem is upstream and no amount of
error tolerance will fix it.

## Sizing the pool

`num_gpus` is how much of a device each actor holds; `concurrency` is how many actors
run. A ResNet-50 fills a T4, so `num_gpus=1, concurrency=4` across four devices. A small
model does not: `num_gpus=0.5, concurrency=4` packs two actors per GPU and roughly
doubles throughput on an EfficientNet-B0-sized network. Or state `model_memory_gb` and
let Kyber pick the fraction. Anything you set explicitly is honored, and only what you leave
unset is chosen for you.

| Argument | What it means | Leave it unset when |
| --- | --- | --- |
| `num_gpus` | the fraction of a device one actor holds | you would rather state `model_memory_gb` and let the engine pack |
| `concurrency` | how many actors run, or a `(min, max)` range to autoscale | you have no reason to pin the pool size |
| `batch_size` | rows per forward pass | you have no number in mind; the pool hill-climbs it |
| `model_memory_gb` | the model's footprint on the device | the model already fills a device |
| `accelerator_type` | pins the device model, e.g. `"NVIDIA_A100"` | the cluster is homogeneous |

```python
# docs: skip
# Two actors per GPU for a small model.
scored = images.ml.infer(SmallNet, num_gpus=0.5, concurrency=8, batch_size=512)

# Or: state the footprint, let the engine pack.
scored = images.ml.infer(SmallNet, model_memory_gb=0.6)
```

Pin a device model with `accelerator_type="NVIDIA_A100"` on a heterogeneous cluster.

## See also

- {doc}`Image captioning <image-captioning>`: the same pipeline with a vision-language model.
- {doc}`Audio transcription <audio-transcription>`: the same decode → model shape, for sound.
- {doc}`GPU scheduling <../../ml/gpu>`: fractional packing and autoscaling pools in full.
- {doc}`Inference <../../ml/inference>` and {doc}`batch scoring <../../ml/batch-scoring>`: the
  `map_batches` / `ml.infer` surface these calls lower to.
- {doc}`Multimodal <../../ml/multimodal>`: the `.image` decode expressions.
- {doc}`ML API reference <../../api/ml>`: every argument of `ds.ml.infer` and `ds.ml.download`.
- {doc}`AI and GPU benchmarks <../../benchmarks/ai-and-gpu>`: where the numbers above come from.
- {doc}`GPU execution <../../deep-dives/gpu-execution>`: how the CPU and GPU stages overlap.
- {doc}`PyTorch integration <../../integrations/pytorch>`: handing these tensors to a training
  loop.
