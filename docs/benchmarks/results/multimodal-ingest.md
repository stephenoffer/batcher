# Multimodal ingest

This page reports how fast Batcher turns media files into model-ready tensors: camera frames, LiDAR sweeps, audio clips and video. For robotics and physical-AI work that step, not the model, is often the hot path, because the accelerator waits on it.

Every run is correctness-gated: frame counts, point counts and output shapes must match across engines before a time is recorded. Image, point-cloud and audio decode ran on a single 96-core CPU node. The audio-model and video results ran on an 8xT4 Ray cluster.

## Image decode and resize

The benchmark decodes 2,000 JPEG frames and resizes them from 640x480 to 224x224, the standard vision-model preprocessing step. On an idle 96-core node, best of three warm (2026-07-11):

| Engine | Time | Throughput |
|---|---:|---:|
| **Batcher** | **351 ms** | **5,693 img/s** |
| Daft | 838 ms | 2,388 img/s |

A later run on the same node under load (a load average of 13 to 17) added Ray Data:

| Engine | Time | Throughput | Batcher's lead |
|---|---:|---:|---:|
| **Batcher** | 418 to 430 ms | 4,649 to 4,788 img/s | |
| Daft 0.7.23 | 780 to 845 ms | 2,368 to 2,565 img/s | **1.87x to 1.96x** |
| Ray Data 2.56 | 2,731 to 2,762 ms | 724 to 732 img/s | **6.35x to 6.61x** |

## Curation and augmentation

Screening and augmenting a corpus needs measures such as entropy and perceptual hashes. Neither Daft nor Ray Data has these natively, so their users write a per-row Pillow UDF. On the same 2,000 frames, Batcher's native entropy, perceptual hash and horizontal flip ran in 1,298 ms against 7,361 ms for a per-row Pillow loop, **5.7x** faster, because the native path fans every row across the machine.

## Point clouds and LiDAR

On the idle 96-core node, 20,000 frames of 4,096 by 3 points streamed to torch through {py:meth}`iter_torch_batches <batcher.api.dataset.ml.DatasetML.iter_torch_batches>`:

| Metric | Batcher |
|---|---:|
| Time | **932 ms** |
| Throughput | **21,467 frames/s** |

That falls out of the tensor-column representation and the concurrent file read, with no modality-specific code.

## Audio

Audio decodes natively through `col(...).audio.decode()`, with per-row fan-out across every core, even on a corpus smaller than one morsel. Against a per-clip `soundfile` loop on the 96-core node, decode is **19.8x** faster, 20.4 ms against 405.8 ms for 2,000 clips, and resampling to 16 kHz matches `soxr`, the resampler `librosa` calls.

The end-to-end audio pipeline, a torchaudio mel spectrogram followed by ResNet-18 on the GPU, sustains **38,546 clip/s** over 8xT4 with 16,384 clips and 100% prediction agreement.

## Video

Each row is a 16-frame clip of about 0.6 MB, run through per-frame ResNet-18, mean-pooled and labeled. On 8xT4 with 4,096 clips, Batcher runs at **2,074.8 clip/s** with no configuration and identical predictions. No batch size is needed because a morsel splits at 16,384 rows or 1 MiB, whichever trips first:

```python
import batcher as bt

cfg = bt.active_config().execution
print(cfg.morsel_rows, cfg.morsel_bytes)
# 16384 1048576
```

## What makes the image path fast

Image ingest went from about 350 img/s to 5,693 img/s through five engine changes, each of which applies beyond images:

- Media-decode kernels decode per row in parallel, and a plan containing media decode uses every core.
- The image reader emits `arrow.fixed_shape_tensor` metadata directly, so the decode stays native end to end.
- SIMD resize through `fast_image_resize`.
- DCT-scaled JPEG decode at 1/2, 1/4 or 1/8 when the source is at least twice the target size.
- One concurrent read wave over all files.

:::{tip}
Keep media steps native. A per-batch Python UDF in the middle of a native pipeline costs about half the throughput even when it does nothing, so write the step as an expression wherever you can.
:::

## Reproduce

```bash
python benchmarks/scenarios/image_decode.py
python benchmarks/scenarios/image_decode.py --suite curate
python benchmarks/scenarios/point_cloud_load.py
python benchmarks/scenarios/audio_decode.py
python benchmarks/cluster/gpu_audio.py
python benchmarks/cluster/gpu_video.py
```

## See also

- {doc}`/benchmarks/results/ai-and-gpu`: the ten GPU workload families.
- {doc}`/benchmarks/comparisons/vs-daft`: the engine the image pipeline is measured against.
- {doc}`/ml/preparing/multimodal/index`: how to write these pipelines.
- {doc}`/architecture/deep-dives/memory/tensor-columns`: the `fixed_shape_tensor` representation the image reader emits.
- {doc}`/architecture/deep-dives/operators/morsel-parallelism`: how the executor sizes its thread pool.
- {doc}`/user-guide/transform/columns/udfs`: the cost of the Python boundary, and how to avoid paying it.
- {doc}`/benchmarks/methodology`: the machines behind each table.
