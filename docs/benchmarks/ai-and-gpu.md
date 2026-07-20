# AI and GPU workloads

This is where Batcher's lead is widest. Measured on 8×T4 across a Ray cluster, against
Ray Data, with real models and a correctness gate on every run.

:::{important}
The gate on a model workload is **prediction agreement**, not a row count. Two engines that
return the same number of rows have not agreed on anything; two engines that return the same
predictions have. A run whose outputs disagree with the oracle reports `FAILED` and
contributes no timing, which is why an 11× or a 47× on this page is a claim about the same
work rather than a claim about less of it.
:::

## Ten workload families, all at least 2×

| Workload | Model | vs Ray Data |
|---|---|---:|
| Text embeddings | sentence-transformers MiniLM | **47×** |
| Audio feature extraction | torchaudio mel + ResNet-18 | **12.5×** |
| LLM batch inference | HF gpt2 | **11.1×** |
| Image generation (diffusion) | diffusers ddpm-cifar10 | **8.6×** |
| Video-clip inference | ResNet-18 per frame | **3.6×** |
| Training-data ingest | `iter_torch_batches` (DLPack) | **3.0×** |
| Batch inference | ResNet-50 | **2.05×** |
| Batch embeddings (image) | ResNet-50 features | **1.98×** |
| Fractional-GPU packing | EfficientNet-B0, 2 per GPU | **1.96×** |
| Multimodal (JPEG → GPU) | two-stage decode → model | **1.3x to 2x** |
| Zero-config GPU | `map_batches(Model, num_gpus=1)` | Ray Data hard-errors |

These come from general engine mechanisms, not per-workload tuning, which is why they carry
to workloads nobody benchmarked. RAG, for instance, is retrieval plus LLM, and both halves
are on this list.

### The honest exception

:::{warning}
A *single, maximally large, compute-bound* job reaches roughly parity. Both engines saturate
the same GPU at the same FLOPs, and no amount of scheduling cleverness changes arithmetic.
Beating that needs fewer FLOPs (FP16 or quantization), which is a model-side decision rather
than an engine one. If that is your workload, the table above does not describe it.
:::

## Why the GPU stays fed

The naive way to run decode → inference is to decode the whole partition, then run the
forward pass. The GPU idles through the decode. Batcher overlaps them: the CPU decode of
morsel *k+1* runs while the GPU forward of morsel *k* is still in flight.

| | img/s | GPU utilization |
|---|---:|---:|
| Sequential stages | 942 | ~30% |
| Stage-overlapped | **2,504** | **81%** |

Same result, same hardware. The device just stops waiting. This is an execution property
of the engine rather than a feature of the inference operator, so any CPU→GPU pipeline
inherits it.

A caveat worth stating, because it cuts the other way elsewhere: a *higher* GPU-utilization
percentage is not automatically better. A slower engine spreads the same GPU work over more
wall-clock and reads as higher utilization. Throughput is the number that matters;
utilization only explains it.

## Warm pools

A model that loads once per job pays its load cost once per job. Batcher's inference pools
are session-warm. The model loads once per *session* and is reused across calls, which is
worth about 2× on iterative or repeated inference and is the single biggest contributor to
the 11× on LLM batch inference.

## Zero-config

The simplest possible call, `ds.map_batches(Model, num_gpus=1)` with no `batch_size`, runs
at 2,451 img/s and 82% utilization. Ray Data rejects it outright. Adaptive batch sizing is
what makes the default a good default.

## Dirty data

Real corpora contain rows that fail to decode. With ~1% corrupt rows injected across 200k
rows, Batcher retains 99% of the data and Ray Data retains 0%. One bad image should cost you
one image, not the job.

## Multimodal ingest

:::{note}
The two tables below were measured on a single 96-core CPU node. Everything above them was
measured on an 8×T4 cluster. Read them as separate results: an img/s figure from one machine
and a clip/s figure from another are not on the same axis, however tempting the arithmetic
is. {doc}`methodology` has the hardware per family.
:::

Turning a corpus of media files into model-ready tensors, on one 96-core node, against both
competitors:

**Image decode and resize.** 2,000 JPEGs, 640x480 to 224x224:

| Engine | Time | Throughput | Batcher's lead |
|---|---:|---:|:---:|
| **Batcher** | 351 ms | 5,693 img/s | baseline |
| Daft | 838 ms | 2,388 img/s | **2.4×** |
| Ray Data | 2,136 ms | 936 img/s | **6.1×** |

**Point cloud and LiDAR to torch.** 20,000 frames of 4,096x3 points:

| Engine | Time | Throughput | Batcher's lead |
|---|---:|---:|:---:|
| **Batcher** | 932 ms | 21,467 frames/s | baseline |
| Ray Data | 2,198 ms | 9,099 frames/s | **2.4×** |

The point-cloud win needed no modality-specific work. It falls out of the same tensor-column
representation and concurrent file read the image path already used.

## Training ingest

`iter_torch_batches`, 10M rows × 32 float features, batch size 1024. Ray Data's own
documentation concedes it runs about 20% slower than a native PyTorch `DataLoader` here.

| Configuration | Batcher | Ray Data | vs Ray |
|---|---:|---:|---:|
| Plain | 1.76 Mrows/s | 0.58 | **3.02×** |
| Local shuffle buffer | 1.14 | 0.53 | **2.14×** |
| In-stream `map_batches` normalize | 1.33 | 0.56 | **2.38×** |
| DDP `streaming_split`, 4 ranks | 1.28 | 0.36 | **3.52×** |

## See also

- {doc}`vs-ray-data`: the same workloads, arranged as a head-to-head.
- {doc}`multimodal-ingest`: the image, point-cloud, audio and video
  pipelines in full, including the part where they lost.
- {doc}`../deep-dives/gpu-execution`: stage overlap and the warm pool, from the
  inside.
- {doc}`../deep-dives/tensor-columns`: the representation the point-cloud win
  falls out of.
- {doc}`../ml/index` and {doc}`../ml/gpu`: how to write these pipelines.
- {doc}`analytics`: the relational half.
- {doc}`methodology`: the machines, and the correctness gate.
