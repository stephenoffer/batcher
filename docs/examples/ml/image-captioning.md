# Image captioning

Captioning a product catalogue is a vision-language model over a URL column. The model is
the easy part. What sinks these jobs is everything on the way to it: a million HTTPS fetches
run one at a time, images decoded on the same thread that should be feeding the GPU, and a
single 404 that takes down a job five hours in.

## Fetch, decode, caption

`ds.ml.download` fetches each URL's bytes into a column (`s3://`, `gs://`, `az://`,
`http(s)://`, or a local path), fetching a batch's rows concurrently and parallelizing
across workers.

:::{warning}
`on_error="null"` turns a dead link into a null instead of an exception. On a catalogue of
any age some fraction of the links are dead, and that is a property of the data, not a bug to
crash on. Leave it at the default and a single 404 takes down a job that has been running for
five hours.
:::

::::{tab-set}
:::{tab-item} A vision model on GPUs

```python
# docs: skip
import batcher as bt
from batcher import col
from batcher.ml import vllm_engine

engine = vllm_engine("llava-hf/llava-1.5-7b-hf")  # a vision model: completion path

captions = (
    bt.read.parquet("s3://bucket/catalog.parquet")  # has "sku" and "url"
    .ml.download("url", output_column="photo", max_concurrency=32, on_error="null")
    .filter(col("photo").is_not_null())
    .ml.generate(
        engine,
        prompt_column="prompt",
        image_column="photo",
        output_column="caption",
        num_gpus=1,
        concurrency=4,
    )
)
captions.select("sku", "caption").write.parquet("s3://bucket/captions.parquet")
```
:::

:::{tab-item} A stub engine, no GPU

An engine is a zero-argument callable returning `requests -> list[str]`. With an
`image_column`, each request is a `{"prompt": str, "image": PIL.Image}` dict. So a stub
engine exercises the whole path (fetch, decode, request assembly, column append) with no GPU
and no weights.

```python
import io

import numpy as np
from PIL import Image

import batcher as bt


def png(value, size):
    buf = io.BytesIO()
    Image.fromarray(np.full((size, size, 3), value, dtype="uint8")).save(buf, format="PNG")
    return buf.getvalue()


catalog = bt.from_pydict({"sku": [1, 2], "photo": [png(30, 16), png(220, 32)]})


def stub_vlm():  # an EngineFactory: built once per worker
    def engine(requests):
        out = []
        for request in requests:
            image = request["image"]  # a PIL image, decoded by the engine boundary
            width, height = image.size
            shade = "dark" if np.array(image).mean() < 128 else "bright"
            out.append(f"{shade} {width}x{height} product photo")
        return out

    return engine


captioned = catalog.with_columns(prompt=bt.lit("Describe this product photo.")).ml.generate(
    stub_vlm, prompt_column="prompt", image_column="photo", output_column="caption"
)
print(captioned.select("sku", "caption").to_pydict())
# {'sku': [1, 2], 'caption': ['dark 16x16 product photo', 'bright 32x32 product photo']}
```

Swap `stub_vlm` for `vllm_engine("llava-hf/llava-1.5-7b-hf")` and add `num_gpus=1`, and this
is the production job. That is the whole point of the engine contract being one callable:
the pipeline you can test on a laptop is the pipeline that runs on eight GPUs.
:::
::::

`image_column` takes raw image bytes or a decoded `(H, W, 3)` tensor; the engine receives
one request per row carrying both the prompt and the image. A null image falls back to a
text-only request rather than failing the batch.

:::{note}
Vision models go through the **completion** path, so `chat=True` is rejected for an
`image_column`. That is a real constraint of how vLLM carries multimodal input, not a
Batcher preference.
:::

## Prompts that are worth sending

A caption model does what the prompt tells it. `template` builds the prompt from the row's
other columns, in the engine, with no Python loop of yours anywhere:

```python
# docs: skip
captions = catalog.ml.generate(
    engine,
    template="Describe this {category} for a shopping site in one sentence. Brand: {brand}.",
    prompt_column="prompt",
    image_column="photo",
    output_column="caption",
    num_gpus=1,
)
```

A shared prefix across rows costs almost nothing: `vllm_engine` enables prefix caching by
default, so the fixed part of the instruction is encoded once rather than a million times.

## Resize before the model, not inside it

A 4000×3000 product photo is 36 MB decoded, and a batch of them will exhaust host memory
before the GPU sees anything. `.image.resize(w, h)` decodes, resizes, and **re-encodes to
PNG bytes**, so the column stays a compact blob that is cheap to ship, spill, and shuffle.
`.image.to_tensor(w, h)` is the other half of the pair: it produces a tensor column, which
is what you want when the next stage is a model that takes pixels directly.

```python
from batcher import col

small = catalog.with_columns(thumb=col("photo").image.resize(8, 8)).collect()
print(small.schema.field("thumb").type, small.column("thumb")[0].as_py()[:4] == b"\x89PNG")
# binary True
```

| Expression | Emits | Reach for it when |
| --- | --- | --- |
| `.image.resize(w, h)` | PNG bytes: a compact blob | the column has to be shipped, spilled, shuffled, or written back out |
| `.image.to_tensor(w, h)` | a fixed-shape `(H, W, 3)` tensor column | the next stage is a model that takes pixels directly |

Both run natively in the data plane, SIMD-decoded and SIMD-resized across every core, which
is why image decode and resize runs at 5,693 img/s on a 96-core node, 2.4x Daft. Do it in a
Python UDF instead and that stage becomes the bottleneck the GPU waits on.

## Keep the GPU fed

The decode is CPU work and the forward pass is GPU work, and running them in lockstep idles
whichever one is not running. Batcher overlaps them, decoding morsel *k+1* on the CPU while
the GPU is still on morsel *k*, which took a two-stage ResNet-50 pipeline from **942 to
2,504 img/s** and utilization from ~30% to **81%**. You inherit that by expressing the decode as
an engine stage rather than doing it inside the model's `__call__`.

Two things to check when the GPU is idle anyway:

- The engine factory is passed as a **class or a factory**, never an instance or a plain
  function. A function rebuilds the engine, reloading the model, on every batch.
- `max_concurrency` on the download is high enough that the fetch is not the bottleneck. A
  million sequential HTTPS round trips at 50 ms each is fourteen hours of nothing.

## Writing media back out

:::{dropdown} Uploading the resized column back to object storage

`ds.ml.upload` is the counterpart to `download`: it writes a bytes column back to object
storage and appends the written paths, with concurrent writes and a content hash for a name
when you do not supply one.

```python
# docs: skip
written = (
    catalog.with_columns(thumb=col("photo").image.resize(256, 256))
    .ml.upload("thumb", "s3://bucket/thumbs/", name_column="sku", extension=".png")
)
```
:::

## See also

- {doc}`Image classification <image-classification>`: the discriminative version of this pipeline.
- {doc}`Audio transcription <audio-transcription>`: the same fetch → decode → model shape, for
  sound.
- {doc}`LLM inference <../../ml/llm>`: vision engines, guided decoding, token accounting.
- {doc}`Multimodal <../../ml/multimodal>`: decode expressions, tensor columns, blob offload.
- {doc}`GPU scheduling <../../ml/gpu>`: `num_gpus`, `concurrency`, and accelerator placement.
- {doc}`ML API reference <../../api/ml>`: `ds.ml.download`, `ds.ml.upload`, `ds.ml.generate`.
- {doc}`Multimodal-ingest benchmarks <../../benchmarks/multimodal-ingest>` and
  {doc}`AI and GPU benchmarks <../../benchmarks/ai-and-gpu>`: the decode and overlap numbers
  quoted here.
- {doc}`GPU execution <../../deep-dives/gpu-execution>`: how the decode stage and the model stage
  overlap.
- {doc}`HuggingFace integration <../../integrations/huggingface>`: where the vision model comes
  from.
