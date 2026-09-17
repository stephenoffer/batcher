# GPU scheduling

This page describes how Batcher places model stages on GPUs and other accelerators, and how it sizes and refines the actor pool that runs them. GPU work is requested per operation rather than configured globally. The `.ml` methods `map_batches`, `infer`, and `embed` take two keywords that together describe a pool of GPU actors: `num_gpus`, how much of a device each actor reserves, and `concurrency`, how many actors run in parallel.

The engine places those actors, hands each a stream of Arrow batches, and collects the results. Your code requests devices and processes batches. It never touches placement.

## How the pool works

Each actor holds `num_gpus` of a GPU for its lifetime. A class-based function loads its model once when the actor starts, then processes many batches on that device, and `concurrency` actors means that many batches in flight at once. The common settings are the following:

- `num_gpus=1, concurrency=4`: four actors, each owning a whole GPU, for a model that fills a device.
- `num_gpus=0.5, concurrency=4`: four actors packed two per GPU across two devices, for a small model that would leave a whole GPU underused.
- `num_gpus=0.0`, the default: CPU only.

Size `num_gpus` to the model's memory footprint, then raise `concurrency` until the devices are saturated. Leave `concurrency` unset and the engine starts at one actor per GPU the cluster reports, so a multi-GPU cluster never idles behind a single actor.

The call shape is the same with or without a device, so the pattern runs here on CPU. Pass the *class*, not an instance. The engine constructs it once per worker, so a multi-gigabyte model loads once rather than once per batch.

```python
import pyarrow as pa

import batcher as bt


class Scorer:
    def __init__(self):
        self.weights = {"a": 1.5, "b": 2.0}  # a real model loads here, once per worker

    def __call__(self, batch):
        scores = [self.weights.get(k, 0.0) for k in batch.column("k").to_pylist()]
        return pa.table({"k": batch.column("k"), "score": pa.array(scores)})


ds = bt.from_pydict({"k": ["a", "b", "a"]})
print(ds.map_batches(Scorer, num_gpus=0, concurrency=2).sort("k").to_pydict())
# {'k': ['a', 'a', 'b'], 'score': [1.5, 1.5, 2.0]}
```

Against a real model and a real device, only `num_gpus` changes:

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
ds.ml.infer(Model, batch_size=512, num_gpus=1, concurrency=4)

# Two actors share each GPU; good for a small model.
ds.map_batches(Model, batch_size=256, num_gpus=0.5, concurrency=4)
```

Later examples on this page pass that same `Model`.

Past those two keywords, the sections below cover three more layers that size the pool. The following diagram stacks them in the order they act:

![Four tiers size a GPU actor pool. First, you declare any of num_gpus, concurrency, batch_size and model_memory_gb, or none, and a value you set always wins. Kyber fills only what is unset: before the run it picks the GPU fraction from model_memory_gb, several copies per device for a light model and whole GPUs for a model larger than one GPU, and seeds a starting batch_size from the VRAM left over. That starts the pool. During the run, concurrency=(min, max) adds actors while batches queue and drops back to min, and the throughput controller refines batch_size from measured VRAM and throughput. The run records utilization and peak device memory, and when concurrency is unset the next run packs toward 90% utilization, bounded by the measured peak memory and at most 8 actors per device, and holds the density of any device already at 80% or more.](/_static/diagrams/gpu_pool_sizing.svg)

## Autoscale the pool

Pass `concurrency` as a `(min, max)` tuple to let the pool follow the backlog. The engine adds actors, up to `max`, while batches queue, and releases them, down to `min`, once the stage drains. A bursty workload doesn't pin every GPU for its whole duration.

```python
# docs: skip
import batcher as bt

ds = bt.read.parquet("data/features.parquet")

# Between 1 and 8 inference actors, scaled to the live backlog.
ds.ml.infer(Model, batch_size=512, num_gpus=1, concurrency=(1, 8))
```

## Pin a GPU model

`accelerator_type` pins the actors to a device model, named as in `ray.util.accelerators`, such as `"NVIDIA_A100"` or `"NVIDIA_H100"`. Use it on a heterogeneous cluster when a model needs enough VRAM or a particular compute capability.

```python
# docs: skip
ds.ml.infer(Model, num_gpus=1, concurrency=4, accelerator_type="NVIDIA_A100")
```

## Let the engine pack by memory

`model_memory_gb` declares the model's footprint in gigabytes, and Kyber sizes the stage from it. With `num_gpus` and `batch_size` unset, it picks the GPU fraction from the model's size against one GPU's memory. A light model gets several copies per device, and a model larger than one GPU gets whole GPUs. It seeds the initial `batch_size` from the VRAM left over, and the online throughput controller refines that from measured VRAM and throughput.

The same number budgets host RAM per worker in the resource layer and costs the stage in Kyber. A value you set yourself always wins. Kyber fills only what you leave unset.

```python
# docs: skip
# State only the model size: Kyber picks the GPU fraction and a starting batch size.
ds.ml.infer(Model, model_memory_gb=1.5)

# Or pin them yourself; the engine respects an explicit value.
ds.ml.infer(Model, num_gpus=0.25, concurrency=8, batch_size=256, model_memory_gb=1.5)
```

## The num_gpus request adapts across runs

GPU placement is part of Batcher's adaptive loop. Each actor measures how busy its device was, the utilization is recorded to the `MetadataHub` keyed by the pipeline, and the next run's effective `num_gpus` adapts. A device that sat idle gets more actors, and a saturated one gets a whole GPU per actor. Your declared `num_gpus` is the starting point. On a host with no measurable utilization, such as Apple MPS, a CPU, or a missing driver, the loop does nothing and your request stands.

## Accelerators that are not GPUs

`num_gpus` covers everything Ray reports as the `GPU` resource: NVIDIA, AMD through ROCm, Intel, and MetaX. Every other accelerator is a *named resource*, requested with `resources=`:

```python
# docs: skip
# Google TPU
ds.map_batches(Model, resources={"TPU": 4}, concurrency=2)

# AWS Trainium / Inferentia
ds.map_batches(Model, resources={"neuron_cores": 2}, concurrency=4)

# Intel Gaudi
ds.map_batches(Model, resources={"HPU": 8})
```

`resources` is a passthrough to Ray rather than a vendor list, so it also requests a resource you defined on an on-prem cluster, such as `resources={"fpga_slot": 1}`. `accelerator_type` works alongside it to pin a device generation: `resources={"TPU": 4}, accelerator_type="TPU-V6E"`.

Don't pass `num_gpus` for these. A TPU or Trainium node advertises no `GPU` resource, so the task waits forever for a GPU instead of failing.

On the model side, `batcher.ml.gpu.detect_backend()` resolves `cuda`, `rocm`, `xpu` for Intel, `mps` for Apple, `tpu`, `neuron` for Trainium and Inferentia, `hpu` for Gaudi, and `npu` for Ascend. `torch_device()` maps each to its torch device string: a TPU or Trainium becomes `xla`, Gaudi `hpu`, and Ascend `npu`. `resources=` is the placement half that gets the task onto the node holding the device.

### What each accelerator reports

Placement gets a task onto a device, and the adaptive loops then need to *read* it. Both loops silently do nothing without a reading: with no utilization the packing target never applies, and with no memory reading the batch-size climb has no ceiling. The table shows what each backend reports:

| Backend | Utilization | Device memory | Fragmentation and per-process cap | Cache release |
|---|---|---|---|---|
| `cuda`, `rocm` | NVML / ROCm SMI | NVML, then torch | yes | yes |
| `xpu` (Intel) | torch counter | torch | yes | yes |
| `hpu` (Gaudi) | torch counter | torch | yes | yes |
| `npu` (Ascend) | torch counter | torch | yes | yes |
| `mps` (Apple) | none reported | torch (unified budget) | yes | yes |
| `tpu`, `neuron` | none reported | none (XLA has no caching allocator) | no | graph step |

Apple, Cloud TPU, and Trainium expose no stable per-process utilization counter, and Batcher reports none rather than inventing one that the packing loop would repack a fleet from. A stage on one of them keeps the `concurrency` you declared. `tpu` and `neuron` run through XLA, which has no caching allocator to read fragmentation from and no per-process cap to set, and it releases memory by stepping its execution graph.

## Keep GPUs fed

A GPU sits idle while it waits for data. To keep it busy, do the following:

- Run input shaping (decode, filter, feature engineering) in the engine with
  expressions and CPU `map_batches`, so GPU actors receive ready batches.
- Stream rather than materialize, so batches arrive continuously; see
  {doc}`Streaming </ml/inference/streaming>`.
- Tune `batch_size` up to the largest batch that fits in device memory; larger
  batches amortize per-call overhead.
- Raise `concurrency`, and use a fractional `num_gpus`, until the devices are saturated.

### The engine packs across runs

Leave `concurrency` unset and the packing is measured rather than guessed. Each run records the utilization its actors sustained and the peak device memory they used. The next run of the same pipeline packs actors toward 90% utilization, bounded by what that measured peak says memory allows.

The two bounds pull in opposite directions. Packing to utilization alone walks a fleet into an out-of-memory error, and packing to memory alone leaves a cheap model at one actor per device with most of the card idle. The loop takes the smaller of the two, so it converges to the densest packing that fits, capped at eight actors per device.

It settles rather than chases. A device at or above 80% counts as fed, and its density is held. Every change rebuilds the pool, which reloads the model on every device, and a measured step past a fed device came out slower and less even than the density it left: 2,602 img/s at 77/94/95/75% against 2,787 at 95/93/94/93%.

A first run has nothing measured. It reserves room for up to two actors per device and starts the second only once the loaded model's footprint shows it fits. The measured loop takes over from the second run.

### Adding devices adds throughput

The actor pool is sized from the cluster's devices, `total_devices / num_gpus` actors, so doubling the fleet doubles the pool. A named accelerator works the same way: a stage asking `resources={"TPU": 4}` opens `cluster_TPU / 4` actors.

A partition count is sized from *data*, though, and a 2.4 GB corpus takes the four-partition floor whatever the cluster looks like. Clamping the pool to partitions would let the data decide how many accelerators work. So both execution paths raise parallelism to match the devices. The batch path shards to one partition per actor, and the streaming path doesn't clamp its consumers at all, because the Flight hand-off feeds a consumer rather than a partition.

An explicit `concurrency` is always honored as written. This sizing applies only when you leave it to the engine.

### Reading the pool's own report

{py:class}`InferencePool <batcher.ml.InferencePool>`, which backs
{py:meth}`ds.ml.embed <batcher.api.dataset.ml.DatasetML.embed>` and the actor-pool paths,
publishes a per-batch event carrying the model's latency and `blocked_ms`, the time the
consumer spent waiting on the pool. A saturated pool and a starved one look identical in rows per second and want opposite fixes, and `blocked_ms` tells them apart:

| `blocked_ms` | What it means | What to change |
|---|---|---|
| Large, close to the batch latency | The pool is the bottleneck; the source keeps up. | More workers, a larger batch, or a smaller model. |
| Near zero | The pool is starved; it finishes before the next batch arrives. | Speed up the source, or widen the read and decode stages. |

The pool reads its source ahead on a background thread, so the read and the forward pass overlap instead of taking turns. The default depth of two batches bounds the extra resident memory to two batches. Raise `prefetch=` for a slow source, or set `prefetch=0` to pull inline.

## See also

- {doc}`Inference </ml/inference/inference>`: the `infer` / `embed` workflow.
- {doc}`The ML accessor </api/models/ml>`: the full argument reference.
- {doc}`Streaming </ml/inference/streaming>`: feed actors with a continuous batch stream.
- {doc}`/ml/inference/batch-scoring`: a full scoring job, with the pool-sizing table.
- {doc}`/ml/inference/runtimes`: ONNX Runtime providers, TensorRT, and OpenVINO on the same pool.
- {doc}`GPU execution </architecture/deep-dives/distribution/gpu-execution>`: how the actor pool works inside.
