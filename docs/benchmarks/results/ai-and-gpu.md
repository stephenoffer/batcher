# AI and GPU workloads

This page reports what Batcher sustains on GPU and model workloads: ten families measured on an 8xT4 Ray cluster with real models, and head-to-head runs against Ray Data and Daft.

Every run is gated on prediction agreement with the oracle, not on a row count. A run whose outputs disagree reports `FAILED` and contributes no timing.

## Ten workload families

Each row is a distinct model and modality, run out of the box on 8xT4 with no per-workload tuning and 100% agreement with the oracle. The scripts live in [`benchmarks/cluster/`](https://github.com/stephenoffer/batcher/tree/main/benchmarks/cluster), one per family, such as `gpu_text_embed.py`, `gpu_audio.py`, `gpu_llm.py` and `gpu_video.py`:

| Workload | Model | Batcher |
|---|---|---:|
| Text embeddings | sentence-transformers MiniLM, 384-d | **33,611 text/s** |
| Audio feature extraction | torchaudio mel plus ResNet-18 | **38,546 clip/s** |
| Fractional-GPU packing | EfficientNet-B0, two per GPU | **6,764 img/s** at 89% GPU |
| JPEG decode then inference | ResNet-50, two stages | **2,504 img/s** at 81% GPU |
| Batch embeddings (image) | ResNet-50 features, 2048-d | **2,502 img/s** at 80% GPU |
| Zero-config GPU | `map_batches(Model, num_gpus=1)` | **2,451 img/s** at 82% GPU |
| Video-clip inference | ResNet-18 per frame, 16-frame clips | **2,074.8 clip/s** |
| LLM batch inference | Hugging Face gpt2, greedy decode | **814.8 prompt/s** |
| Image generation | diffusers ddpm-cifar10, 20 DDIM steps | **169.1 img/s** |
| Training-data ingest | {py:meth}`iter_torch_batches <batcher.api.dataset.ml.DatasetML.iter_torch_batches>`, 200k rows of 1024-d features, no shuffle | **1.06M rows/s** |

On every family where device utilization was sampled, the GPU holds at or above an 80% target:

![Horizontal bar chart of sustained GPU utilization by workload family on 8xT4 with real models and 100 percent output agreement. Compute-bound ResNet-50 FP16 inference holds 100 percent at 4,707 images per second, a decode-heavy JPEG to ResNet pipeline 93.4 percent at 3,860, fractional GPU packing of EfficientNet-B0 89 percent at 6,764, zero-config inference with no batch size given 82 percent at 2,451, ResNet-50 batch inference 81 percent at 2,504, and image embeddings 80 percent at 2,502. A dashed line marks the 80 percent target.](/_static/diagrams/gpu_utilization.svg)

The throughput comes from general engine mechanisms rather than per-workload tuning: stage overlap, warm model pools and a VRAM-safe default batch. The pipeline shape is the same on a laptop and on 8xT4. A class passed to `map_batches` is constructed once per worker and reused, which is where a model load goes:

```python
import batcher as bt
import pyarrow as pa
import pyarrow.compute as pc

class Scale:
    def __init__(self):
        self.weight = 2.0  # stands in for a model loaded once per worker

    def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        return batch.append_column("score", pc.multiply(batch["x"], self.weight))

ds = bt.from_pydict({"x": [1.0, 2.0, 3.0]})
print(ds.map_batches(Scale).to_pydict())
# {'x': [1.0, 2.0, 3.0], 'score': [2.0, 4.0, 6.0]}
```

On a GPU cluster the same call takes `num_gpus=1`.

## Stage overlap

Batcher overlaps the CPU and GPU stages: the CPU decode of morsel *k+1* runs while the GPU forward of morsel *k* is still in flight. On the two-stage ResNet-50 pipeline:

| Execution | img/s | GPU utilization |
|---|---:|---:|
| Sequential stages | 942 | ~30% |
| Stage-overlapped | **2,504** | **81%** |

![Two panels comparing a two-stage ResNet-50 pipeline before and after stage overlap, with the same result and the same order. Throughput rises from 942 to 2,504 images per second. GPU utilization rises from 30 percent to 81 percent of the device kept busy.](/_static/diagrams/stage_overlap.svg)

Stage overlap is a property of the executor rather than of the inference operator, so any CPU-to-GPU pipeline inherits it, single-node or distributed.

## Warm pools

Batcher's inference pools are session-warm, controlled by `distributed.warm_inference_pools` and on by default: the model loads once per session and is reused across calls. A gpt2 load takes 7 to 10 seconds against roughly 1 second of generation, so the pool matters most on short and repeated jobs. ResNet-50 on 8xT4:

| Regime | Throughput | Against a cold pool |
|---|---:|---:|
| Repeated same job, 8k images | 1,020 img/s | 3.6x |
| Iterative small, 12k images | 2,576 img/s at 78% | 2.05x |
| Iterative moderate, 49k images | 2,755 img/s at 89% | 1.29x |
| Single large job, 131k images | 2,504 img/s at 81% | About parity, GPU-bound |

A single large job runs at the hardware ceiling: one T4 sustains about 400 img/s at 100% utilization on ResNet-50.

## Zero configuration

The simplest call, {py:meth}`ds.map_batches(Model, num_gpus=1) <batcher.Dataset.map_batches>` with no `batch_size`, runs at 2,451 img/s and 82% utilization, within 2% of the hand-tuned `batch_size=128` path. Batcher picks a VRAM-safe default, streams it with stage overlap, and halves the batch on a CUDA out-of-memory error.

Byte-aware morsels make that default safe on wide rows. A morsel splits at 16,384 rows or 1 MiB, whichever trips first, so the 16-frame video clips above, about 0.6 MB a row, stream at 2,074.8 clip/s with no batch size chosen and no device memory exhausted.

## Against Ray Data and Daft

Two cluster runs measure the same pipelines on other engines.

**Scoring an image corpus.** On six single-T4 nodes, every engine built the identical seeded network, scored the corpus through one actor per GPU, and returned a checksum that gated the timing ([`benchmarks/gpu_backend/vs_ray_daft_gpu_inference.py`](https://github.com/stephenoffer/batcher/blob/main/benchmarks/gpu_backend/vs_ray_daft_gpu_inference.py), 2026-09-06):

| Images | Batcher | Ray Data | Daft |
|---:|---:|---:|---:|
| 10,000 | **5.99 s** | 10.49 s | 13.24 s |
| 100,000 | **18.72 s** | 44.19 s | 101.10 s |

At 100,000 images that is **2.36x Ray Data and 5.40x Daft**.

**A compute-bound pipeline.** A corpus built so that neither the read nor the page cache can decide the answer ran the same `map(cpu) -> map(gpu) -> agg` pipeline on 17 nodes with 192 cores and 8 T4s ([`benchmarks/gpu_backend/compute_bound_inference.py`](https://github.com/stephenoffer/batcher/blob/main/benchmarks/gpu_backend/compute_bound_inference.py), 2026-09-11):

| Rows | Batcher | Ray Data | Batcher GPU | Ray Data GPU |
|---:|---:|---:|---:|---:|
| 125,000 | **1.2 s** | 10.5 s | 25% | 5% |
| 500,000 | **4.0 s** | 10.2 s | 40% | 18% |
| 2,000,000 | 14.9 s | **14.4 s** | 50% | 50% |
| 4,000,000 | 29.8 s | **22.6 s** | 48% | 65% |

Both engines return identical answers at every size. Warm pools give Batcher **8.6x at 125,000 rows** and **2.5x at 500,000**.

## Dirty data

Batcher's error tolerance is per row. With about 1% corrupt rows injected across 200,000 rows, `max_errored_rows` keeps 198,000 of them ([`benchmarks/cluster/robustness/gpu_dirty.py`](https://github.com/stephenoffer/batcher/blob/main/benchmarks/cluster/robustness/gpu_dirty.py)). One bad row costs one row, not its batch. Without the option the query raises:

```python
import batcher as bt
import pyarrow.compute as pc

def invert(batch):
    if 0.0 in batch["x"].to_pylist():
        raise ValueError("corrupt row")
    return batch.append_column("inv", pc.divide(1.0, batch["x"]))

ds = bt.from_pydict({"x": [1.0, 0.0, 2.0, 4.0]})
print(ds.map_batches(invert, max_errored_rows=1).to_pydict())
# {'x': [1.0, 2.0, 4.0], 'inv': [1.0, 0.5, 0.25]}
```

:::{dropdown} Scope of these numbers
- These results come from GPU clusters, named per table. Compare engines within a table.
- Utilization was sampled on some families, not all.
- On the compute-bound pipeline, quote a factor together with its row count.
:::

## See also

- {doc}`/benchmarks/results/multimodal-ingest`: the image, point-cloud, audio and video ingest paths on CPU.
- {doc}`/architecture/deep-dives/distribution/gpu-execution`: stage overlap and the warm pool, from the inside.
- {doc}`/architecture/deep-dives/memory/tensor-columns`: the representation tensor output rides on.
- {doc}`/ml/index` and {doc}`/ml/inference/gpu`: how to write these pipelines.
- {doc}`/benchmarks/results/analytics`: the relational half of the measurement.
- {doc}`/benchmarks/methodology`: the machines, and the correctness gate.
