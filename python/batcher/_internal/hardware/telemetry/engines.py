"""The fixed-function engines beside the SMs — NVDEC, NVENC, NVJPG, OFA — and whether they run.

A datacenter GPU is not one processor. Beside the SMs sit dedicated video decode, video encode,
JPEG decode, and optical-flow blocks, each with its own clock and its own utilization counter,
and none of them contributes to the `sm_utilization` figure every scheduler reads. That gap is
the reason a multimodal decode stage is so often mis-diagnosed: a pipeline decoding H.264 on
the SMs shows 95% GPU utilization and looks saturated, while the NVDEC block that would have
done the same work at a fraction of the cost sits at zero and the SMs that should have been
running the model are busy doing a video codec's job.

What each counter decides:

* **Decoder utilization** — whether `bt.read.video` is reaching NVDEC. Zero here on a stage
  that is demonstrably decoding video means the decode is happening on the CPU or on the SMs,
  and the fix is the decode backend, not the batch size.
* **Encoder utilization and session stats** — the same question for a write path, plus the
  session count, which is capped in hardware on some parts and is a hard admission limit rather
  than a performance one.
* **JPEG utilization** — the equivalent for image corpora, which is the far more common shape
  in a training-data pipeline than video is.
* **Optical-flow utilization** — the block a frame-interpolation or tracking stage uses; zero
  on such a stage means it is running on the SMs.

**These counters are duty cycles over the driver's own sampling window, exactly like
`sm_utilization`.** A decoder at 100% is saturated for the window; it is not necessarily the
bottleneck, because a single stream can saturate one decoder engine while other engines idle.
The count of engines is not published, which is why saturation here is reported as a fact about
the counter rather than as a verdict about the device.

Every field degrades to `0` when the driver is absent, the query is refused, or the part has no
such block — which is the case for OFA and JPG on everything before Ampere, and for NVENC on
the datacenter parts that ship without an encoder at all.
"""

from __future__ import annotations

from dataclasses import dataclass

from batcher._internal.hardware.nvml import _device_count, _nvml, _read

__all__ = [
    "EngineUtilization",
    "device_engines",
    "engine_idle_devices",
    "hardware_decode_active",
]

#: NVML utilization getters for the fixed-function blocks, as
#: `(getter name, record field, human label)`. Read by name because the JPG and OFA getters
#: arrived in the 12.x line and are simply absent on an older binding, where their absence must
#: read as "no such block" rather than raise.
_ENGINE_GETTERS = (
    ("nvmlDeviceGetDecoderUtilization", "decoder", "NVDEC"),
    ("nvmlDeviceGetEncoderUtilization", "encoder", "NVENC"),
    ("nvmlDeviceGetJpgUtilization", "jpeg", "NVJPG"),
    ("nvmlDeviceGetOfaUtilization", "optical_flow", "OFA"),
)


@dataclass(frozen=True, slots=True)
class EngineUtilization:
    """One device's fixed-function engine duty cycles and encoder session state.

    Attributes:
        index: NVML device index on this host.
        decoder: Fraction of the sample window NVDEC was busy, in [0, 1].
        encoder: Fraction of the sample window NVENC was busy, in [0, 1].
        jpeg: Fraction of the sample window NVJPG was busy, in [0, 1].
        optical_flow: Fraction of the sample window the optical-flow block was busy, in [0, 1].
        encoder_sessions: Live encode sessions on the device. Capped in hardware on some parts,
            so this is an admission limit as well as a load figure.
        encoder_fps: Mean frames per second across those sessions, `0` when none are running.
        encoder_latency_us: Mean encode latency in microseconds across those sessions.
        supported: Labels of the blocks that answered at all, from `_ENGINE_GETTERS`. A block
            absent here has no counter on this part; a block present here reading `0.0` is
            genuinely idle. The distinction is the whole diagnosis.
        readable: Whether NVML answered any query.
    """

    index: int
    decoder: float = 0.0
    encoder: float = 0.0
    jpeg: float = 0.0
    optical_flow: float = 0.0
    encoder_sessions: int = 0
    encoder_fps: int = 0
    encoder_latency_us: int = 0
    supported: tuple[str, ...] = ()
    readable: bool = False

    @property
    def any_active(self) -> bool:
        """Whether any fixed-function block is doing work.

        The one-line answer to "is this device using its hardware codecs at all". False on a
        device demonstrably processing media means the media path is on the SMs or the host.
        """
        return max(self.decoder, self.encoder, self.jpeg, self.optical_flow) > 0.0

    @property
    def decode_saturated(self) -> bool:
        """Whether the decode block was busy for effectively the whole sample window.

        A 90% band rather than 100% because the counter is quantized to whole percent and a
        fully-loaded engine routinely reports 97-99. A saturated decoder caps a media stage at
        the engine's rate regardless of how much SM capacity is left.
        """
        return self.decoder >= 0.9

    @property
    def encoder_saturated(self) -> bool:
        """Whether the encode block was busy for effectively the whole sample window."""
        return self.encoder >= 0.9


def _utilization(nv, handle, getter: str) -> float | None:
    """One engine's duty cycle in [0, 1], or `None` when the part has no such counter.

    NVML returns these as `(percent, sampling period)` pairs on every binding that has them.
    `None` and `0.0` are held apart on purpose: the first means there is no such block to use,
    and the second means there is one and nothing is using it.
    """
    fn = getattr(nv, getter, None)
    if fn is None:
        return None
    sentinel = object()
    value = _read(lambda: fn(handle), sentinel)
    if value is sentinel:
        return None
    if isinstance(value, (tuple, list)):
        value = value[0] if value else 0
    return min(1.0, max(0.0, float(value or 0) / 100.0))


def _encoder_stats(nv, handle) -> tuple[int, int, int]:
    """`(sessions, mean fps, mean latency microseconds)`, all `0` when unreported."""
    fn = getattr(nv, "nvmlDeviceGetEncoderStats", None)
    if fn is None:
        return (0, 0, 0)
    stats = _read(lambda: fn(handle), None)
    if not isinstance(stats, (tuple, list)) or len(stats) < 3:
        return (0, 0, 0)
    return (int(stats[0] or 0), int(stats[1] or 0), int(stats[2] or 0))


def device_engines() -> tuple[EngineUtilization, ...]:
    """Fixed-function engine utilization for every local device, in NVML index order.

    Not memoized: every field is a live duty cycle. Costs four to five NVML calls per device.

    Returns:
        One record per device, empty when NVML is unavailable. A device whose every engine query
        was refused still reports a record with `readable=False` and an empty `supported`, which
        is what separates "this part has no codec blocks" from "we could not ask".
    """
    nv = _nvml()
    if nv is None:
        return ()
    out: list[EngineUtilization] = []
    for index in range(_device_count(nv)):
        handle = _read(lambda i=index: nv.nvmlDeviceGetHandleByIndex(i), None)
        if handle is None:
            continue
        values: dict[str, float] = {}
        supported: list[str] = []
        for getter, field, label in _ENGINE_GETTERS:
            reading = _utilization(nv, handle, getter)
            if reading is None:
                continue
            values[field] = reading
            supported.append(label)
        sessions, fps, latency = _encoder_stats(nv, handle)
        out.append(
            EngineUtilization(
                index=index,
                decoder=values.get("decoder", 0.0),
                encoder=values.get("encoder", 0.0),
                jpeg=values.get("jpeg", 0.0),
                optical_flow=values.get("optical_flow", 0.0),
                encoder_sessions=sessions,
                encoder_fps=fps,
                encoder_latency_us=latency,
                supported=tuple(supported),
                readable=bool(supported),
            )
        )
    return tuple(out)


def hardware_decode_active(readings: tuple[EngineUtilization, ...] | None = None) -> bool:
    """Whether any local device is decoding media on its hardware blocks.

    The check a multimodal pipeline should make once, at the point it has decided its decode
    backend, to confirm the decision took effect. It answers a question no timing can: a
    software decode path is not slower per frame in a way that is obviously wrong, it is slower
    while consuming the SM capacity the model was supposed to get.

    Args:
        readings: Records to inspect, or `None` to read them live.

    Returns:
        True when any device shows non-zero decode or JPEG utilization. False when none do
        *and* when nothing was readable, so this is evidence for a problem only on a host where
        `device_engines` reports `readable=True`.
    """
    records = device_engines() if readings is None else readings
    return any(r.decoder > 0.0 or r.jpeg > 0.0 for r in records)


def engine_idle_devices(
    readings: tuple[EngineUtilization, ...] | None = None,
) -> tuple[EngineUtilization, ...]:
    """Devices with codec blocks that are supported and entirely unused, in index order.

    Not a fault. It is the capacity report for a media pipeline: these devices have decode
    hardware sitting idle, and a stage doing media work on their SMs is choosing to leave it
    that way.

    Args:
        readings: Records to inspect, or `None` to read them live.

    Returns:
        The subset that reported at least one engine counter and zero activity on all of them.
    """
    records = device_engines() if readings is None else readings
    return tuple(r for r in records if r.readable and not r.any_active)
