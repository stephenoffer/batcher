"""Decode media columns into tensors — natively wherever the engine has the codec.

Multimodal sources read references + header metadata only (no pixels/samples at read
time). These helpers turn the raw ``bytes`` column into model-ready tensors at the
point a pipeline asks for it:

* **images** decode in the **Rust data plane** — the ``col.image.to_tensor`` kernel
  resizes and flattens to RGB8 in the engine, which tags the result as a fixed-shape
  ``(H, W, 3)`` tensor column so it crosses the FFI already shaped.
* **video** decodes in the data plane too (``col.video.frames``) on an engine built with
  the ``video`` cargo feature, which links the system FFmpeg. A build without it has no
  video codec to reach, so the stage falls back to a per-row PyAV loop.
* **audio** decodes natively for every mono case (``col.audio.to_waveform`` /
  ``.resample``); only multi-channel output falls back to `soundfile`.

Pixels stay ``uint8`` and samples ``float32`` all the way through: the float32 a model
wants is four times the bytes, and that conversion belongs at the GPU, not here.

Each returns a new lazy `Dataset`, so decode composes with the rest of a pipeline. A row
whose bytes are null or fail to decode yields **null** rather than failing the batch — the
multimodal convention, and the same answer whichever decoder ran. Zeros would be
indistinguishable from a legitimately black or silent input.

`transfer` moves bytes in and out (download/upload), `media` decodes images and audio,
`video` samples frames from clips, and `stage` holds the scaffolding all three share.
The private names re-exported below are the seams the decode tests drive directly.

`accelerated` is the alternative to all of it on a GPU node: decoding on the *device* means
the bus carries the compressed payload rather than the pixels, which for a photographic JPEG
is roughly twelve times less traffic, and for video means NVDEC does the work instead of the
SMs the model wanted. It reports an unavailable backend rather than failing wherever the
libraries are absent, which is most CPU images.
"""

from __future__ import annotations

from batcher.ml.decode.accelerated import (
    DecodeBackend,
    decode_jpeg_batch,
    hardware_decode_confirmed,
    image_decode_backend,
    transfer_saving_ratio,
    video_decode_backend,
)
from batcher.ml.decode.media import audio_dataset, image_tensor_dataset
from batcher.ml.decode.stage import (
    _bounded_map as _bounded_map,
)
from batcher.ml.decode.stage import (
    _shared_pool as _shared_pool,
)
from batcher.ml.decode.transfer import download_dataset, upload_dataset
from batcher.ml.decode.video import (
    _decode_video_bytes as _decode_video_bytes,
)
from batcher.ml.decode.video import (
    video_dataset,
)

__all__ = [
    "DecodeBackend",
    "audio_dataset",
    "decode_jpeg_batch",
    "download_dataset",
    "hardware_decode_confirmed",
    "image_decode_backend",
    "image_tensor_dataset",
    "transfer_saving_ratio",
    "upload_dataset",
    "video_dataset",
    "video_decode_backend",
]
