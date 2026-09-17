# Multimodal ingest

This page reports how fast Batcher turns media files into model-ready tensors: camera frames, LiDAR sweeps, audio clips and video. For robotics and physical-AI work that step, not the model, is often the hot path, because the accelerator waits on it.

:::{important}
Every run is correctness-gated. Frame counts, point counts and output shapes must be identical across engines before a time is recorded, so an engine that decoded fewer frames, or decoded them to a different shape, gets no timing.
:::

:::{note}
This page mixes two kinds of machine. Image, point-cloud and audio decode ran on a single 96-core CPU node. The audio pipeline and the video results ran on an 8xT4 Ray cluster, because they run a model. Compare within a table.
:::

## Image decode and resize

The benchmark decodes 2,000 JPEG frames and resizes them from 640x480 to 224x224, the standard vision-model preprocessing step. On an idle 96-core node, best of three warm (2026-07-11):

| Engine | Time | Throughput |
|---|---:|---:|
| **Batcher** | **351 ms** | **5,693 img/s** |
| Daft | 838 ms | 2,388 img/s |

A later run on the same node, with other sessions' test suites holding the load average at 13 to 17, measured the same benchmark against both Daft and Ray Data after two read-side fixes. Its absolute times are a floor rather than a peak:

| Engine | Time | Throughput | Batcher's lead |
|---|---:|---:|---:|
| **Batcher** | 418 to 430 ms | 4,649 to 4,788 img/s | |
| Daft 0.7.23 | 780 to 845 ms | 2,368 to 2,565 img/s | **1.87x to 1.96x** |
| Ray Data 2.56 | 2,731 to 2,762 ms | 724 to 732 img/s | **6.35x to 6.61x** |

The two read-side fixes were small. The per-file header parse ran whether or not the query asked for it, and local file reads were fanned across a thread pool, which suits an object store and is 2.5x slower than a serial loop on page cache: 2,000 local JPEGs read in 52 ms serially and 118 ms on eight threads.

## Curation and augmentation

Screening and augmenting a corpus needs measures such as entropy and perceptual hashes. Neither Daft nor Ray Data has these natively, so their users write a per-row Pillow UDF. On the same 2,000 frames, Batcher's native entropy, perceptual hash and horizontal flip ran in 1,298 ms against 7,361 ms for a per-row Pillow loop, **5.7x** faster, because the native path fans every row across the machine while the loop runs under the GIL on one core.

## Point clouds and LiDAR

On the idle 96-core node, 20,000 frames of 4,096 by 3 points streamed to torch through {py:meth}`iter_torch_batches <batcher.api.dataset.ml.DatasetML.iter_torch_batches>`:

| Metric | Batcher |
|---|---:|
| Time | **932 ms** |
| Throughput | **21,467 frames/s** |

No modality-specific work went into this. It falls out of the tensor-column representation and the concurrent file read the image path already needed.

## Audio

Audio decodes natively through `col(...).audio.decode()`, with per-row fan-out across every core. On a corpus smaller than one morsel of 16,384 rows the native path still uses the whole machine, because media-decode kernels parallelize per row rather than per morsel. There is no competing engine to measure against, since Daft has no native audio surface and Ray Data decodes through a Python UDF. Against a per-clip `soundfile` loop on the 96-core node, decode is **19.8x** faster, 20.4 ms against 405.8 ms for 2,000 clips, and resampling to 16 kHz matches `soxr`, the resampler `librosa` calls.

The end-to-end audio pipeline, a mel spectrogram on the CPU with torchaudio followed by ResNet-18 on the GPU, is a different measurement on different hardware. Distributed over 8xT4 with 16,384 clips, Batcher sustains **38,546 clip/s** with 100% prediction agreement.

## Video

Each row is a 16-frame clip of about 0.6 MB, run through per-frame ResNet-18, mean-pooled and labeled. That is the wide-row regime where a fixed `batch_size` either wastes memory or exhausts it. On 8xT4 with 4,096 clips, Batcher runs at **2,074.8 clip/s** with no configuration and identical predictions.

Byte-aware morsels are why no batch size is needed. A morsel splits at whichever bound trips first, rows or bytes, so a few very wide rows can't blow the memory budget.

## How the image path reached 5,693 img/s

Image ingest started this work at about 350 img/s, behind Daft. Five fixes took it to 5,693, about 16x, and each one generalized beyond images.

:::{dropdown} The five fixes, in order of how much they mattered
1. **The media-decode kernels ran on one core.** Per-row decode was serial, and the parallel executor capped its thread pool to the morsel count. A small-JPEG corpus is one morsel, so the whole decode ran single-threaded. The kernels now decode per row in parallel, and a plan containing media decode lifts the pool to every core. Decode alone got 17x to 22x faster.
1. **A re-typing UDF was halving throughput.** `read.images(decode=True)` appended a Python `map_batches` purely to re-type the flat list as a shaped tensor, and any downstream `map_batches`, even an identity one, roughly halves throughput and core use. The reader now emits `arrow.fixed_shape_tensor` field metadata directly, so pyarrow reconstructs the shaped column across the FFI and the decode stays native. That took the pipeline from 2,000 to 4,600 img/s.
1. **SIMD resize**, through `fast_image_resize` in place of a scalar filter.
1. **DCT-scaled JPEG decode** at 1/2, 1/4 or 1/8, used only when the source is at least twice the target size, which is the normal case for a large frame feeding a small model input.
1. **Bulk concurrent read.** `MediaSource.read()` read 64-file chunks serially with a fresh thread pool each time. One concurrent wave over all files took the read from 368 ms to 250 ms.
:::

:::{warning}
Fix 2 is the one to remember when you write pipelines. A per-batch Python UDF in the middle of a native pipeline costs about half your throughput even when it does nothing. Write the step as an expression wherever you can.
:::

## Requirements and limitations

The following limits apply to these results:

- **The busy-node table** measures a contended machine. It isn't comparable with the idle-node 5,693 img/s, and the ratio to reproduce is the one from your own hardware.
- **Audio operations** beyond decode and resample aren't separately benchmarked.

## Reproduce

The following scripts reproduce each result:

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
- {doc}`/architecture/deep-dives/memory/tensor-columns`: the `fixed_shape_tensor` representation fix 2 relies on.
- {doc}`/architecture/deep-dives/operators/morsel-parallelism`: the pool sizing fix 1 corrected.
- {doc}`/user-guide/transform/columns/udfs`: the cost of the Python boundary, and how to avoid paying it.
- {doc}`/benchmarks/methodology`: the machines behind each table.
