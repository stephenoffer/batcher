"""Which wire codec a shuffle should use, decided against the link it will actually cross.

Compression on a shuffle is a trade between a core and a wire, and the exchange rate is the
link. On a 10 Gb/s VM the wire moves 1.25 GB/s and a compressor running at several GB/s per
core is free money: every byte it removes is a byte the NIC does not have to carry. On a
400 Gb/s InfiniBand port the wire moves 50 GB/s, one core cannot feed it even at LZ4's rate,
and compressing turns a NIC-bound stage into a CPU-bound one at a fraction of the throughput.

One default cannot be right on both, and the failure is silent in the expensive direction: the
fast-fabric node is exactly the node someone paid for, and it runs the shuffle at compressor
speed while every counter reads healthy.

This is the decision, as a pure function of the measured fabric rate and the cores available to
feed it. It is Carbonite's because it is a resource trade rather than a plan rewrite, and it is
consulted only when the operator asked for `auto`: an explicit `lz4`, `zstd`, or `none` is a
decision somebody made about their own deployment, and a measurement must not overrule it.
"""

from __future__ import annotations

__all__ = ["CODEC_CODES", "codec_for_fabric", "resolve_codec"]

#: The engine's codec codes, by name. One mapping, because the call site had it inline and a
#: second spelling of "zstd is 2" is a silent mis-selection rather than an error.
CODEC_CODES = {"none": 0, "lz4": 1, "zstd": 2}

#: What one core sustains, in gigabytes per second, for each codec. LZ4-frame is the widely
#: reproduced ~1 GB/s/core figure and zstd at its low levels is several times slower. These are
#: order-of-magnitude figures used only to compare against a link rate, and the comparison is
#: made with a wide margin, so a factor-of-two error in either does not move the answer.
_CODEC_GBPS = {"lz4": 1.0, "zstd": 0.3}

#: Compression ratios real shuffle data reaches: sorted runs, repeated group keys, dictionary
#: strings, and nulls. Conservative, because over-stating the ratio is what makes a compressor
#: look able to keep up with a wire it cannot.
_CODEC_RATIO = {"lz4": 2.0, "zstd": 3.0}

#: Cores a worker is assumed to be able to give the compressor when the caller does not say.
#: One, deliberately, and it is the conservative direction rather than the realistic one: a
#: single core cannot keep up with any modern fabric, so an uninformed caller gets "do not
#: compress" wherever the wire is fast and the shipped default everywhere else. The real figure
#: is the worker's usable core count, and the call site passes it.
_DEFAULT_CORES = 1.0

#: The share of a worker's cores a shuffle's compression may be assumed to have. The rest are
#: doing the work the shuffle exists to feed — decoding, computing, filling a device — and a
#: model that hands the compressor the whole machine concludes that compression wins on every
#: fabric, which is how a NIC-bound stage becomes a CPU-bound one on the fastest hardware in
#: the fleet.
_CORE_SHARE = 0.25


def codec_for_fabric(fabric_gbps: float, cores: float = _DEFAULT_CORES) -> str:
    """The codec whose *effective* throughput is highest on a link of this rate.

    A codec's effective rate is the smaller of two things: what the cores can compress, and
    what the wire can carry once the payload has shrunk. Whichever codec maximizes that is the
    right one, and on a fast enough fabric the answer is `none` — no codec keeps up, so every
    one of them is a ceiling below the wire.

    Args:
        fabric_gbps: The node's measured off-node rate in gigabits per second, `0.0` when
            unknown.
        cores: The worker's usable cores. A share of them is assumed to be available to the
            compressor; the rest are feeding the stage.

    Returns:
        `"none"`, `"lz4"`, or `"zstd"`. An unknown fabric returns `"lz4"`, which is the shipped
        default and the answer that is never badly wrong: it costs a fast node some throughput
        and saves a slow one a great deal.
    """
    if fabric_gbps <= 0.0 or cores <= 0.0:
        return "lz4"
    wire_gbytes = fabric_gbps / 8.0
    usable = max(1.0, cores * _CORE_SHARE)
    best, best_rate = "none", wire_gbytes
    for name, per_core in _CODEC_GBPS.items():
        compress_rate = per_core * usable
        # The payload shrinks, so the wire carries the compressed bytes: the stage's rate in
        # *uncompressed* bytes is the wire's rate times the ratio, capped by what the cores
        # can compress in the first place.
        effective = min(compress_rate, wire_gbytes * _CODEC_RATIO[name])
        if effective > best_rate:
            best, best_rate = name, effective
    return best


def resolve_codec(configured: str, fabric_gbps: float = 0.0, cores: float = _DEFAULT_CORES) -> int:
    """The codec code to hand the engine, honoring an explicit setting over any measurement.

    The rate is passed in rather than probed here. The one implementation of "what is this
    node's off-node bandwidth" lives where the cost model reads it, and Carbonite cannot import
    that subsystem — so the caller, which can, hands over the figure instead of a second
    implementation being written next to this one.

    Args:
        configured: The `distributed.flight_compression` value: `none`, `lz4`, `zstd`, or
            `auto`.
        fabric_gbps: The measured off-node rate in gigabits per second. Only read for `auto`,
            so a deployment that named its codec pays nothing for this.
        cores: Cores available to the compressor.

    Returns:
        The engine's codec code. An unrecognized name resolves to LZ4, which is what the call
        site's inline mapping already did with its default.
    """
    name = (configured or "").strip().lower()
    if name != "auto":
        return CODEC_CODES.get(name, CODEC_CODES["lz4"])
    return CODEC_CODES[codec_for_fabric(fabric_gbps, cores)]
