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
