# GPU scheduling

GPU work in Batcher is requested per operation, not configured globally. The
`.ml` methods (`map_batches`, `infer`, `embed`) take two keywords that together
describe a pool of GPU actors:

- `num_gpus`: how much of a device each actor reserves.
- `concurrency`: how many actors run in parallel.

The engine places those actors on available GPUs, hands each one a stream of Arrow
batches, and collects the results. Your code requests GPUs and processes batches; it
never touches device placement.

## How the pool works

Each actor is a worker that holds `num_gpus` of a GPU for its lifetime. A class-based
function loads its model once when the actor starts, then processes many batches on that
reserved device. With `concurrency` actors, that many batches are in flight at once.

- `num_gpus=1, concurrency=4`: four actors, each owning a whole GPU. Use this when
  one model fills a device.
- `num_gpus=0.5, concurrency=4`: four actors packed two-per-GPU across two
  devices. Use this for small models so a single GPU is not underused.
- `num_gpus=0.0` (the default): CPU only, no GPU reserved.

Fractional packing is how you keep expensive GPUs busy: size `num_gpus` to the model's
memory footprint, then raise `concurrency` until the devices are saturated.

Leave `concurrency` unset and the engine sizes the pool automatically at one actor per
GPU the cluster reports. A multi-GPU cluster is never left idling a single engine, which
is a common scale-out mistake.

The call shape is the same with or without a device, so the pattern is runnable here on
CPU. Pass the *class*, not an instance: the engine constructs it once per worker, which
is what makes a multi-gigabyte model load once rather than once per batch.

```python
import pyarrow as pa

import batcher as bt

class Scorer:
    def __init__(self):
        self.weights = {"a": 1.5, "b": 2.0}   # a real model loads here, once per worker

    def __call__(self, batch):
        scores = [self.weights.get(k, 0.0) for k in batch.column("k").to_pylist()]
        return pa.table({"k": batch.column("k"), "score": pa.array(scores)})

ds = bt.from_pydict({"k": ["a", "b", "a"]})
print(ds.ml.map_batches(Scorer, num_gpus=0, concurrency=2).sort("k").to_pydict())
# {'k': ['a', 'a', 'b'], 'score': [1.5, 1.5, 2.0]}
```

Raise `num_gpus` to reserve a device (or a fraction of one) per actor, and the rest of
the call is unchanged.

## Autoscaling the pool

Pass `concurrency` as a `(min, max)` tuple to let the pool grow and shrink with the
backlog instead of holding a fixed actor count. The engine adds actors (up to `max`)
while batches queue and releases them (down to `min`) once the stage drains, so a
bursty workload does not pin every GPU for its whole duration.

```python
# docs: skip
import batcher as bt

ds = bt.read.parquet("data/features.parquet")

# Between 1 and 8 inference actors, scaled to the live backlog.
ds.ml.infer(Model, batch_size=512, num_gpus=1, concurrency=(1, 8))
```

## Pinning a GPU model

`accelerator_type` pins the actors to a specific device, a `ray.util.accelerators`
name such as `"NVIDIA_A100"` or `"NVIDIA_H100"`. Use it when a model needs a
particular GPU (enough VRAM, a required compute capability) on a heterogeneous
cluster.

```python
# docs: skip
ds.ml.infer(Model, num_gpus=1, concurrency=4, accelerator_type="NVIDIA_A100")
```

## Letting the engine pack by memory

`model_memory_gb` declares the model's footprint in gigabytes. State the size once and
Kyber sizes the stage for you. With `num_gpus` and `batch_size` left unset, it picks the
GPU fraction from the model's size against one GPU's memory (packing several copies of a
light model onto one device, or reserving whole GPUs for a model larger than one) and
seeds the initial `batch_size` from the VRAM left over. The online throughput controller
then refines that batch size from measured VRAM and throughput. Two other consumers read
the same number: the resource layer, to budget host RAM per worker (OOM protection), and
Kyber, to cost an inference stage by size.

Any value you set yourself is always honored; Kyber only fills what you leave unset.

```python
# docs: skip
# State only the model size — Kyber picks the GPU fraction and a starting batch size.
ds.ml.infer(Model, model_memory_gb=1.5)

# Or pin them yourself; the engine respects an explicit value.
ds.ml.infer(Model, num_gpus=0.25, concurrency=8, batch_size=256, model_memory_gb=1.5)
```

## The num_gpus request adapts across runs

GPU placement is also part of Batcher's adaptive loop. Each actor measures how busy
the device actually was; that utilization is recorded to the MetadataHub keyed by the
pipeline, and the next run's effective `num_gpus` adapts. It packs more tasks onto a
fraction of a device that sat idle, or asks for a whole GPU when one saturated. Your
declared `num_gpus` is the starting point; the measured load refines it. On a host with
no measurable utilization (Apple MPS, CPU, or no driver) the loop is a no-op and your
request stands unchanged.

## Requesting GPUs

The call shape is the same as any `.ml` operation; only `num_gpus` and
`concurrency` are added. Real GPU code needs a device and a model, so it is shown
but not run.

```python
# docs: skip
import batcher as bt
import pyarrow as pa


class Model:
    def __init__(self):
        import torch

        self.net = torch.load("model.pt").cuda().eval()

    def __call__(self, batch):
        import torch

        x = torch.tensor(batch.column("features").to_pylist()).cuda()
        with torch.no_grad():
            out = self.net(x).argmax(dim=1).cpu().tolist()
        return batch.append_column("prediction", pa.array(out))


ds = bt.read.parquet("data/features.parquet")

# One whole GPU per actor, four actors.
ds.ml.infer(Model(), batch_size=512, num_gpus=1, concurrency=4)

# Two actors share each GPU; good for a small model.
ds.ml.map_batches(Model(), batch_size=256, num_gpus=0.5, concurrency=4)
```

## Accelerators that are not GPUs (TPU, Trainium, Gaudi)

`num_gpus` covers everything Ray reports as the `GPU` resource, which means NVIDIA, AMD
(ROCm), Intel, and MetaX. Every other accelerator is a **named resource** instead, so
request it with `resources=`:

```python
# docs: skip
# Google TPU
ds.ml.map_batches(Model(), resources={"TPU": 4}, concurrency=2)

# AWS Trainium / Inferentia
ds.ml.map_batches(Model(), resources={"neuron_cores": 2}, concurrency=4)

# Intel Gaudi
ds.ml.map_batches(Model(), resources={"HPU": 8})
```

`resources` is a passthrough to Ray, not a fixed vendor list, so it equally requests a
resource you defined yourself on an on-prem cluster (`resources={"fpga_slot": 1}`).
`accelerator_type` works alongside it to pin a device generation
(`resources={"TPU": 4}, accelerator_type="TPU-V6E"`).

Do not pass `num_gpus` for these: a TPU or Trainium node advertises no `GPU` resource, so
the task would wait for a GPU that never appears rather than failing.

On the model side, `batcher.ml.gpu.detect_backend()` already resolves `cuda` / `rocm` /
`xpu` (Intel) / `mps` (Apple) / `tpu`, and `torch_device()` maps them to the right torch
device string (a TPU becomes `xla`). What `resources=` adds is the *placement* half. It
gets the task onto the node that has the device.

## Keeping GPUs fed

A GPU sits idle while it waits for data. To avoid that:

- Run input shaping (decode, filter, feature engineering) in the engine with
  expressions and CPU `map_batches`, so GPU actors receive ready batches.
- Stream rather than materialize, so batches arrive continuously; see
  [Streaming](streaming.md).
- Tune `batch_size` up to the largest batch that fits in device memory; larger
  batches amortize per-call overhead.
- Raise `concurrency` (and use fractional `num_gpus`) until the devices are fully
  utilized.

## Next steps

- [Inference](inference.md): the `infer` / `embed` workflow.
- [The ML accessor](../api/ml.md): the full argument reference.
- [Streaming](streaming.md): feed actors with a continuous batch stream.
