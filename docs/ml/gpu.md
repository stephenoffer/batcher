# GPU scheduling

GPU work in Batcher is requested per operation, not configured globally. The
`.ml` methods (`map_batches`, `infer`, `embed`) take two keywords that together
describe a pool of GPU actors:

- `num_gpus`: how much of a device each actor reserves.
- `concurrency`: how many actors run in parallel.

The engine places those actors on available GPUs, hands each one a stream of Arrow
batches, and collects the results. Your code never manages device placement
directly; it requests GPUs and processes batches.

## How the pool works

Each actor is a worker that holds `num_gpus` of a GPU for its lifetime. A
class-based function loads its model once when the actor starts, then processes
many batches on that reserved device. With `concurrency` actors, that many batches
are in flight at once.

- `num_gpus=1, concurrency=4`: four actors, each owning a whole GPU. Use this when
  one model fills a device.
- `num_gpus=0.5, concurrency=4`: four actors packed two-per-GPU across two
  devices. Use this for small models so a single GPU is not underused.
- `num_gpus=0.0` (the default): CPU only, no GPU reserved.

The fractional packing is how you keep expensive GPUs busy: size `num_gpus` to the
model's memory footprint and raise `concurrency` until the devices are saturated.

`concurrency` defaults to `"auto"` — one actor per GPU the cluster reports — so a
multi-GPU cluster is never left idling a single engine (a common scale-out foot-gun).

## Autoscaling the pool

Pass `concurrency` as a `(min, max)` tuple to let the pool grow and shrink with the
backlog instead of holding a fixed actor count. The engine adds actors (up to `max`)
while batches queue and releases them (down to `min`) when the stage drains — so a
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

`model_memory_gb` declares the model's footprint in gigabytes. The resource layer
uses it to budget host RAM per worker (OOM protection) and to VRAM-pack small models
onto a shared GPU — so you can state the size once and let the engine choose a safe
packing instead of hand-tuning `num_gpus`. Kyber also uses it to cost an inference
stage by model size.

```python
# docs: skip
# A 1.5 GB model: the engine budgets host RAM and packs replicas onto each GPU.
ds.ml.infer(Model, num_gpus=0.25, concurrency=8, model_memory_gb=1.5)
```

## The num_gpus request adapts across runs

GPU placement is also part of Batcher's adaptive loop. Each actor measures how busy
the device actually was; that utilization is recorded to the MetadataHub keyed by the
pipeline, and the next run's effective `num_gpus` adapts — packing more tasks onto a
fraction of a device that sat idle, or asking for a whole GPU when one saturated. The
declared `num_gpus` is the starting point; the measured load refines it. On a host
with no measurable utilization (Apple MPS, CPU, or no driver) the loop is a no-op and
your request stands unchanged.

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
