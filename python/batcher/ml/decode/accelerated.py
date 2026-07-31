"""Decoding on the device, so the bus carries compressed bytes instead of pixels.

The default image path decodes in the Rust data plane, on the host, and hands the model a
tensor. That is the right default and it pays a cost nobody sees, because the cost is not in
the decode — it is in what the decode *produces*.

A 500 KB JPEG is 6 MB of RGB8 at 1440x1440. Decoding on the host and copying the result to the
device moves the 6 MB. Copying the JPEG and decoding on the device moves the 500 KB. Same
pixels, same model, **twelve times less traffic across PCIe**, and on a node with eight devices
sharing a host link that ratio is frequently the difference between a stage that is
transfer-bound and one that is not. The decode itself also stops competing for the host cores
that are feeding seven other devices.

The same argument holds harder for video, where NVDEC is a *separate block* from the SMs: a
clip decoded there costs the model nothing at all, while a clip decoded on the SMs is taking
capacity directly from the thing it is being decoded for.

**Four backends, none of them required.** `torchvision` reaches nvJPEG through
`decode_jpeg(device="cuda")` and is present in most inference images; `nvidia.dali` is the
fastest and the least commonly installed; `torchcodec` and `PyNvVideoCodec` reach NVDEC for
video. When none is present this module reports so and the caller keeps the host path, which
works and is what runs today.

**Confirming it worked is a separate question from asking for it.** Requesting a device decode
and getting one are different events: a build of torchvision without nvJPEG silently falls back
to the CPU decoder, and the result is identical pixels arriving the slow way with nothing to
say so. `hardware_decode_confirmed` answers it from the device's own fixed-function counters,
which is the only source that can.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

__all__ = [
    "DecodeBackend",
    "decode_jpeg_batch",
    "hardware_decode_confirmed",
    "image_decode_backend",
    "reset_decode_backend_probe",
    "transfer_saving_ratio",
    "video_decode_backend",
]

#: Image-decode backends in preference order, as `(module path, backend name)`. One entry, and
#: deliberately so: `nvidia.dali` decodes faster still, and listing it here without a decode
#: path behind it would resolve a backend that `decode_jpeg_batch` then declines to use,
#: silently returning every caller to the host decoder while reporting a device backend. Add it
#: with its implementation or not at all.
_IMAGE_BACKENDS = (("torchvision.io", "torchvision"),)

#: Video-decode backends in preference order. Both reach NVDEC; `torchcodec` is the maintained
#: path and `PyNvVideoCodec` is what a deployment pinned to an older CUDA stack will have.
_VIDEO_BACKENDS = (
    ("torchcodec", "torchcodec"),
    ("PyNvVideoCodec", "pynvvideocodec"),
)

#: Bytes of RGB8 per pixel. Used only to size the saving in a report; nothing decides on it.
_RGB_BYTES_PER_PIXEL = 3


@dataclass(frozen=True, slots=True)
class DecodeBackend:
    """A resolved device-decode path, or the absence of one.

    Attributes:
        name: Backend identifier: `"dali"`, `"torchvision"`, `"torchcodec"`,
            `"pynvvideocodec"`, or `""` when none is available.
        device: The device string decoding would target, `""` when there is no device.
        batched: Whether the backend decodes a whole batch in one call. A per-image call into
            a device decoder is usually *slower* than the host decoder it replaced, because the
            launch overhead dominates a single small image — so this is a correctness-relevant
            property of the plan, not a performance footnote.
    """

    name: str = ""
    device: str = ""
    batched: bool = False

    @property
    def available(self) -> bool:
        """Whether device decode can be attempted at all."""
        return bool(self.name and self.device)

    def __bool__(self) -> bool:
        """A falsey backend is an unavailable one, so `if backend:` reads correctly."""
        return self.available


def _cuda_device() -> str:
    """`"cuda"` when a CUDA device is usable in this process, `""` otherwise.

    Asked through torch because every backend here needs torch's CUDA context anyway, and
    because `nvml_available` is a weaker condition: NVML answers on a host whose driver is
    present and whose container was never given a device.
    """
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else ""
    except Exception:
        return ""


def _importable(path: str) -> bool:
    """Whether a module imports, without keeping a reference to it."""
    import importlib

    try:
        importlib.import_module(path)
    except Exception:
        return False
    return True


@functools.lru_cache(maxsize=1)
def image_decode_backend() -> DecodeBackend:
    """The best available on-device image decoder, or an unavailable backend.

    Memoized: the answer is an import probe and a CUDA handshake, neither of which changes
    within a process, and the question is asked once per decode stage.

    Returns:
        A `DecodeBackend`. Falsey when there is no device or no installed decoder, in which
        case the caller keeps the host path rather than failing.
    """
    device = _cuda_device()
    if not device:
        return DecodeBackend()
    for path, name in _IMAGE_BACKENDS:
        if _importable(path):
            return DecodeBackend(name=name, device=device, batched=True)
    return DecodeBackend()


@functools.lru_cache(maxsize=1)
def video_decode_backend() -> DecodeBackend:
    """The best available NVDEC-backed video decoder, or an unavailable backend.

    Returns:
        A `DecodeBackend`. Falsey when there is no device or no installed decoder, in which
        case `video_dataset`'s PyAV path continues to serve, on the host.
    """
    device = _cuda_device()
    if not device:
        return DecodeBackend()
    for path, name in _VIDEO_BACKENDS:
        if _importable(path):
            # Per-clip rather than batched: a video decoder is stateful across frames, so
            # there is no batch call to make, and the win comes from NVDEC being a separate
            # block rather than from amortizing a launch.
            return DecodeBackend(name=name, device=device, batched=False)
    return DecodeBackend()


def reset_decode_backend_probe() -> None:
    """Forget the resolved decode backends so the next call re-probes.

    The hook a test faking an installed library needs; nothing in a running process can change
    the answer otherwise.
    """
    image_decode_backend.cache_clear()
    video_decode_backend.cache_clear()


def decode_jpeg_batch(payloads, backend: DecodeBackend | None = None):
    """Decode a batch of JPEG payloads straight into device memory.

    Whole-batch by construction. A per-image call into a device decoder loses to the host
    decoder it replaced — the kernel launch dominates one small image — so a caller with one
    image should not reach for this, and a caller with a batch should not loop.

    Args:
        payloads: A sequence of `bytes`, one JPEG each. A null or empty entry is skipped
            rather than failing the batch, matching the multimodal convention the rest of
            `decode` follows.
        backend: The backend to use, or `None` to resolve it. Passed explicitly by a stage
            that resolved it once rather than per batch.

    Returns:
        A list of decoded `uint8` tensors on the device, in input order with skipped entries
        omitted, or `None` when no device decoder is available *or* the decode failed. `None`
        is the signal to use the host path; it is never a partial result, because a batch half
        on the device and half on the host is worse than either.
    """
    resolved = image_decode_backend() if backend is None else backend
    if not resolved:
        return None
    usable = [p for p in payloads if p]
    if not usable:
        return []
    try:
        import torch

        buffers = [torch.frombuffer(bytearray(p), dtype=torch.uint8) for p in usable]
        if resolved.name == "torchvision":
            from torchvision.io import decode_jpeg

            # torchvision takes the whole list and dispatches one batched nvJPEG call; passing
            # them one at a time is the mistake this signature exists to prevent.
            return decode_jpeg(buffers, device=resolved.device)
    except Exception:
        # A build of torchvision without nvJPEG raises here rather than falling back, which is
        # the outcome to prefer: a silent fallback would leave the caller believing the bus
        # saving happened.
        return None
    return None


def hardware_decode_confirmed() -> bool | None:
    """Whether the device's fixed-function decoders are actually doing work.

    The check to run once after a decode stage has started, because *asking* for a device
    decode and *getting* one are different events. A torchvision build without nvJPEG, a codec
    NVDEC does not implement, and a driver too old for the path all produce identical pixels
    arriving the slow way, with nothing in the pipeline's own timings to say which happened.

    Returns:
        True when NVDEC or NVJPG reported activity, False when the counters were readable and
        idle, and `None` when the part publishes no engine counters at all — which is not
        evidence either way and must not be reported as a failure.
    """
    try:
        from batcher._internal.hardware.telemetry.engines import device_engines

        readings = device_engines()
    except Exception:
        return None
    if not any(r.readable for r in readings):
        return None
    return any(r.decoder > 0.0 or r.jpeg > 0.0 for r in readings)


def transfer_saving_ratio(width: int, height: int, compressed_bytes: int) -> float:
    """How much less the bus carries when the decode happens on the device.

    The number that justifies the whole module, computed from the actual image geometry rather
    than asserted. A caller sizing a multimodal stage should look at this before deciding
    whether the extra dependency is worth taking on: below about 3x it rarely is, and above
    10x it usually decides whether the stage is transfer-bound.

    Args:
        width: Decoded image width in pixels.
        height: Decoded image height in pixels.
        compressed_bytes: Size of the encoded payload.

    Returns:
        Decoded bytes divided by compressed bytes, so `12.0` means the host path moves twelve
        times as much across PCIe. `0.0` when either figure is missing, rather than a guess.
    """
    if width <= 0 or height <= 0 or compressed_bytes <= 0:
        return 0.0
    return (width * height * _RGB_BYTES_PER_PIXEL) / compressed_bytes
