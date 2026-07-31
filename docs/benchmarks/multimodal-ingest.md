# Multimodal ingest

The robotics and physical-AI hot path is not a model. It is turning a corpus of media files (camera frames, LiDAR sweeps, audio clips) into model-ready tensors fast enough that the accelerator never waits. This page reports what Batcher sustains on each modality, and how the image path got there.

:::{important}
Every run is correctness-gated: frame and point counts and output shapes must be identical across engines before a time is recorded. An engine that decoded fewer frames, or decoded them to a different shape, gets no timing rather than a fast one.
:::

:::{note}
This page mixes two machines. The image, point-cloud, and audio-decode results are a single 96-core CPU node; the audio *pipeline* and the video results are an 8xT4 Ray cluster, because they run a model. The tables are comparable within themselves, not with each other. {doc}`methodology` lists the hardware per family.
:::

## Image decode and resize

One 96-core node, best-of-3 warm. 2,000 JPEG frames, 640x480 to 224x224, which is the vision-model preprocessing step.

| Engine | Time | Throughput |
|---|---:|---:|
| **Batcher** | **351 ms** | **5,693 img/s** |
| Daft | 838 ms | 2,388 img/s |

## Point cloud and LiDAR

Same node. 20,000 frames of 4,096x3 points, streamed to torch through `iter_torch_batches`.

| Metric | Batcher |
|---|---:|
| Time | **932 ms** |
| Throughput | **21,467 frames/s** |

No modality-specific work went into this. It falls out of the tensor-column representation and the concurrent file read that the image path already needed.

## Audio

Audio decode runs natively, through `col(bytes).audio.decode()` on symphonia, with per-row fan-out across every core. On a corpus smaller than a single 16,384-row morsel the native path still uses the whole machine, because the media-decode kernels parallelize per row rather than per morsel.

The end-to-end audio *pipeline* (waveform to mel-spectrogram on the CPU with torchaudio, then ResNet-18 on the GPU) is a different measurement on different hardware. Distributed over 8xT4 with 16,384 clips, Batcher sustains **38,546 clip/s** with 100% prediction agreement. That figure cannot be compared with the 96-core tables above.

## Video

Each row is a 16-frame clip of about 0.6 MB: per-frame ResNet-18, mean-pool, clip label. This is the large-row regime where a fixed `batch_size` either wastes memory or exhausts it. On 8xT4 with 4,096 clips, Batcher runs at **2,074.8 clip/s** with zero configuration and identical predictions.

Byte-aware morselization is why no batch size is needed. A morsel splits at whichever bound trips first, so a few very wide rows cannot blow the memory budget, and the zero-config batch shrinks by row width.

## How the image path reached 5,693 img/s

Image ingest began this work at about 350 img/s. Five fixes took it to 5,693, roughly 16x, and each one generalized beyond images.

:::{dropdown} The five fixes, in order of how much they mattered
1. **The media-decode kernels ran on one core.** Per-row decode was serial, *and* the parallel executor capped its rayon pool to the morsel count. A small-JPEG corpus is one morsel, so the entire decode of the entire corpus ran single-threaded. The kernels now do a rayon per-row `map_rows`, and `contains_media_decode()` on the plan lifts the pool to every core for a media query. Decode alone runs 17x to 22x ahead.
1. **A re-type UDF was halving throughput.** `read.images(decode=True)` appended a Python `map_batches` purely to re-type the flat list as a shaped tensor. *Any* downstream `map_batches`, even an identity one, roughly halves throughput and core use. The reader now emits `arrow.fixed_shape_tensor` field metadata directly, so pyarrow reconstructs the shaped column across the FFI and the decode stays on the fully-parallel native path. 2,000 to 4,600 img/s.
1. **SIMD resize** (`fast_image_resize`, replacing a scalar Triangle filter).
1. **DCT-scaled JPEG decode** at 1/2, 1/4, 1/8, used only when the source is at least 2x the target, which is the normal case for a large frame feeding a small model input.
1. **Bulk concurrent read.** `MediaSource.read()` read 64-file chunks serially with a fresh thread pool each time. One wide concurrent wave over all files took it from 368 ms to 250 ms.
:::

:::{warning}
Fix 2 is the one worth internalizing if you write pipelines: a per-batch Python UDF in the middle of a native pipeline costs about half your throughput **even when it does nothing**. An identity `map_batches` is not free, and it is not close to free. Say it as an expression if you possibly can.
:::

## Reproduce

```bash
python benchmarks/scenarios/image_decode.py
python benchmarks/scenarios/point_cloud_load.py
python benchmarks/scenarios/audio_decode.py
python benchmarks/cluster/gpu_video.py
```

## See also

- {doc}`ai-and-gpu`: the ten GPU workload families.
- {doc}`vs-daft`: the engine the image pipeline is measured against.
- {doc}`/ml/preparing/multimodal/index`: how to write these pipelines.
- {doc}`/deep-dives/memory/tensor-columns`: the `fixed_shape_tensor` representation fix 2 turned on.
- {doc}`/deep-dives/operators/morsel-parallelism`: the pool sizing that fix 1 corrected.
- {doc}`/user-guide/transform/udfs`: the cost of the Python boundary, and how to avoid paying it.
- {doc}`methodology`: the machines; the 96-core and 8xT4 tables above are not comparable with each other.
