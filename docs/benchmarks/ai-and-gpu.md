# AI and GPU workloads

This page reports what Batcher sustains on GPU and AI workloads: ten families measured on 8xT4 across a Ray cluster, with real models and a correctness gate on every run. Where device utilization was sampled it sits at or above the 80% target, and the mechanisms that put it there are general rather than per-workload.

:::{important}
The gate on a model workload is **prediction agreement**, not a row count. Two engines that return the same number of rows have not agreed on anything; two that return the same predictions have. A run whose outputs disagree with the oracle reports `FAILED` and contributes no timing, so every throughput figure below is a claim about the same work rather than a claim about less of it.
:::

## Ten workload families

Each row is a distinct model and modality, run out of the box with no per-workload tuning.

| Workload | Model | Batcher |
|---|---|---:|
| Text embeddings | sentence-transformers MiniLM, 384-d | **33,611 text/s** |
| Audio feature extraction | torchaudio mel + ResNet-18 | **38,546 clip/s** |
| Batch inference | ResNet-50 | **2,504 img/s** at 81% GPU |
| Batch embeddings (image) | ResNet-50 features, 2048-d | **2,502 img/s** at 80% GPU |
| Fractional-GPU packing | EfficientNet-B0, 2 per GPU | **6,764 img/s** at 89% GPU |
| Zero-config GPU | `map_batches(Model, num_gpus=1)` | **2,451 img/s** at 82% GPU |
| Video-clip inference | ResNet-18 per frame, 16-frame clips | **2,074.8 clip/s** |
| LLM batch inference | HF gpt2, FP16, greedy decode | **814.8 prompt/s** |
| Image generation | diffusers ddpm-cifar10, 20 DDIM steps | **169.1 img/s** |
| Training-data ingest | `iter_torch_batches`, zero-copy DLPack | **1.06 M rows/s** |

These come from general engine mechanisms rather than per-workload tuning, which is why they carry to workloads nobody benchmarked. RAG, for instance, is retrieval plus an LLM, and both halves are on this list.

The three sections below are those mechanisms.

## Where the utilization comes from

The naive way to run decode into inference is to decode the whole partition, then run the forward pass. The GPU idles through the decode. Batcher overlaps them: the CPU decode of morsel *k+1* runs while the GPU forward of morsel *k* is still in flight.

| | img/s | GPU utilization |
|---|---:|---:|
| Sequential stages | 942 | ~30% |
| Stage-overlapped | **2,504** | **81%** |

Same result, same hardware. The device stops waiting. This is an execution property of the engine rather than a feature of the inference operator, so any CPU-to-GPU pipeline inherits it, single-node and distributed, on every modality.

A caveat worth stating, because it cuts both ways: a *higher* GPU-utilization percentage is not automatically better. A slower engine spreads the same GPU work over more wall-clock and reads as higher utilization. Throughput is the number that matters, and utilization only explains it.

## Warm pools

A model that loads once per job pays its load cost once per job. Batcher's inference pools are session-warm, so the model loads once per *session* and is reused across calls. That is worth about 2x on iterative or repeated inference, and much more as the model grows: a gpt2 load takes about 7 seconds against roughly 1 second of generation, and a multi-gigabyte diffusion or LLM checkpoint takes tens of seconds.

| Regime (ResNet-50, 8xT4) | Throughput | GPU utilization |
|---|---:|---:|
| Repeated same job, 8k images | 1,020 img/s | warm pool |
| Iterative small, 12k images | 2,576 img/s | 78% |
| Iterative moderate, 49k images | 2,755 img/s | 89% |
| Single large job, 131k images | 2,504 img/s | 81% |

A single maximally large, compute-bound job runs at the hardware ceiling: one T4 sustains about 400 img/s at 100% utilization for ResNet-50, and eight actors reach about 3,200 img/s with no parallel penalty. At that point the pipeline is no longer the limit and the arithmetic is. Going faster there means fewer FLOPs through FP16 or quantization, which is a model-side decision.

## Zero configuration

The simplest possible call, `ds.map_batches(Model, num_gpus=1)` with no `batch_size`, runs at 2,451 img/s and 82% utilization. That is within 2% of the hand-tuned `batch_size=128` path, with no knobs. Batcher picks a VRAM-safe default, streams it with stage overlap, and halves the batch on a CUDA OOM.

Byte-aware morselization is what makes the default safe on wide rows. A morsel splits at whichever bound trips first, count or bytes, so the 16-frame video clips above (about 0.6 MB per row) stream at 2,074.8 clip/s without a hand-picked batch size and without exhausting device memory.

## Dirty data

Real corpora contain rows that fail to decode. Error tolerance in Batcher is per row: with about 1% corrupt rows injected across 200,000 rows, `max_errored_rows` keeps 198,000 of them. One bad image costs you one image, not the block it landed in and not the job.

## Multimodal ingest

:::{note}
The two tables below were measured on a single 96-core CPU node. Everything above them was measured on an 8xT4 cluster. Read them as separate results: an img/s figure from one machine and a clip/s figure from another are not on the same axis, however tempting the arithmetic is. {doc}`methodology` has the hardware per family.
:::

Turning a corpus of media files into model-ready tensors, on one 96-core node.

**Image decode and resize.** 2,000 JPEGs, 640x480 to 224x224:

| Engine | Time | Throughput |
|---|---:|---:|
| **Batcher** | **351 ms** | **5,693 img/s** |
| Daft | 838 ms | 2,388 img/s |

**Point cloud and LiDAR to torch.** 20,000 frames of 4,096x3 points, streamed through `iter_torch_batches`:

| Metric | Batcher |
|---|---:|
| Time | **932 ms** |
| Throughput | **21,467 frames/s** |

The point-cloud result needed no modality-specific work. It falls out of the same tensor-column representation and concurrent file read the image path already used.

## Training ingest

`iter_torch_batches`, 10M rows by 32 float features, batch size 1024, prefetch 2. The loader is zero-copy through DLPack with background prefetch, so it feeds a training loop well above the rate most models consume:

| Configuration | Throughput |
|---|---:|
| Plain | **1.76 M rows/s** |
| In-stream `map_batches` normalize | **1.33 M rows/s** |
| DDP `streaming_split`, 4 ranks | **1.28 M rows/s** |
| Local shuffle buffer | **1.14 M rows/s** |

## See also

- {doc}`multimodal-ingest`: the image, point-cloud, audio, and video pipelines in full, including the regression that started the work.
- {doc}`../deep-dives/gpu-execution`: stage overlap and the warm pool, from the inside.
- {doc}`../deep-dives/tensor-columns`: the representation the point-cloud result falls out of.
- {doc}`../ml/index` and {doc}`../ml/gpu`: how to write these pipelines.
- {doc}`analytics`: the relational half of the measurement.
- {doc}`methodology`: the machines, and the correctness gate.
