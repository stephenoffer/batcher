"""Accelerator and energy tunables — the facts about a GPU fleet only its operator knows.

Everything Batcher can discover about a device it discovers: model, memory, count, live draw.
What it cannot discover is the *deployment* around that device — what a rack's busway is
allowed to draw, what a kilowatt-hour costs here, how carbon-intense this grid is, how hot an
inlet is allowed to get before a device should be taken out of rotation. Those are facts about
a datacenter, and this section is where its operator states them.

Every default is deliberately inert. A power budget of zero means unbounded, a carbon intensity
of zero means unconfigured (and reports no emissions rather than an invented figure), and the
placement switches default to the behavior the scheduler already had. A fleet that configures
nothing here behaves exactly as it did before energy became a plannable resource, which is what
makes the section safe to ship enabled.

Kept beside `config.py` rather than inside it: that module is already at its size limit, and
these tunables carry their own range checks, which would otherwise push
`validation/sections.py` over its own.
"""

from __future__ import annotations

from dataclasses import dataclass

from batcher._internal.errors import ConfigError

__all__ = [
    "AcceleratorConfig",
    "DeviceHealthConfig",
    "DeviceMemoryConfig",
    "EnergyConfig",
    "validate_accelerator",
]

#: Cache element types an inference stage may be sized for. Mirrors what the KV-cache math in
#: `carbonite.accel.kv_cache` knows an element width for, so a config that validates is a
#: config that can be sized.
_CACHE_DTYPES = frozenset({"fp32", "float32", "fp16", "float16", "bf16", "fp8", "int8"})


@dataclass(frozen=True, slots=True)
class EnergyConfig:
    """The site's power envelope and what its energy costs.

    Examples:
        .. doctest::

            >>> from batcher.config import EnergyConfig
            >>> EnergyConfig().power_budget_watts
            0.0
    """

    #: Watts the whole job may draw, `0.0` for unbounded (the default). A fleet under a
    #: rack- or hall-level cap sets this and gets a device count bounded by the breaker
    #: rather than by the slot count, which is the tighter bound above about eight
    #: 700 W devices per rack.
    power_budget_watts: float = 0.0
    #: Fraction of the budget deliberately left unused, absorbing the gap between the linear
    #: power model and a real board's curve.
    power_headroom: float = 0.1
    #: Grid carbon intensity in grams CO2e per kilowatt-hour. `0.0` means unconfigured, and
    #: every emissions figure then reports `0.0` — deliberately, because a shipped default
    #: would be authoritative-looking and wrong for any specific site.
    carbon_intensity: float = 0.0
    #: Energy price per kilowatt-hour, in whatever currency the operator reports in. `0.0`
    #: means unpriced.
    price_per_kwh: float = 0.0
    #: Facility power usage effectiveness: total facility draw divided by IT load. `1.0`
    #: accounts for the servers only; a site that knows its PUE sets it and gets cost and
    #: carbon figures that include cooling.
    pue: float = 1.0
    #: Free-form site or region identifier, carried into energy reports for attribution.
    region: str = ""
    #: Fraction of supply from carbon-free generation, in [0, 1]. Reporting only — the
    #: carbon intensity above already accounts for the mix.
    renewable_fraction: float = 0.0
    #: Record per-stage energy into the run's ledger. Cheap (one arithmetic roll-up per
    #: stage) and on by default, because an efficiency figure nobody collected is the one
    #: nobody can act on.
    accounting: bool = True
    #: Seconds between device telemetry samples during a stage. Each sample is a handful of
    #: NVML calls per device; below about a second the sampling itself starts to show up.
    telemetry_interval_s: float = 5.0


@dataclass(frozen=True, slots=True)
class DeviceHealthConfig:
    """When a device stops being worth scheduling on.

    Examples:
        .. doctest::

            >>> from batcher.config import DeviceHealthConfig
            >>> DeviceHealthConfig().quarantine_on_ecc
            True
    """

    #: Take a device out of rotation when it reports an uncorrectable ECC error. On by
    #: default: an uncorrectable error means data already read back wrong, and no throughput
    #: argument outweighs a silently corrupted tensor.
    quarantine_on_ecc: bool = True
    #: Device temperature above which a device is treated as degraded even before the driver
    #: clamps it, because it is about to be clamped.
    max_temperature_c: float = 87.0
    #: Derate at or below which a degraded device stops being scheduled entirely.
    quarantine_below_derate: float = 0.3
    #: Resident device memory fraction above which a device is treated as full, so a stage is
    #: not admitted onto a device another tenant has already filled.
    max_memory_fraction: float = 0.95
    #: Consult live telemetry before placing accelerator work. Off by default because it
    #: requires `pynvml` on every worker; with it off, every device is assumed healthy, which
    #: is the behavior the scheduler already had.
    enabled: bool = False


#: Device allocator strategies. `default` is one driver allocation per request (CUDA's own,
#: and what RAPIDS uses unconfigured); `pool` suballocates from one large reservation; `async`
#: uses the driver's stream-ordered pool; `managed` backs the pool with unified memory so a
#: working set larger than the device migrates over the bus instead of failing.
_ALLOCATORS = frozenset({"default", "pool", "async", "managed"})


@dataclass(frozen=True, slots=True)
class DeviceMemoryConfig:
    """How a GPU worker's device memory is allocated and what it does when it runs out.

    Examples:
        .. doctest::

            >>> from batcher.config import DeviceMemoryConfig
            >>> DeviceMemoryConfig().allocator
            'default'
    """

    #: Allocator strategy: `default`, `pool`, `async`, or `managed`. `default` is the
    #: unconfigured driver allocator, where every intermediate column costs a `cudaMalloc`
    #: that synchronizes the device. A pool pays that once and suballocates, which is the
    #: single largest constant-factor lever on a chain of many small operators.
    allocator: str = "default"
    #: Fraction of a device's *reservable* memory the pool takes at startup. Reserved up
    #: front, so a large value trades a longer first allocation for no growth pauses later.
    pool_initial_fraction: float = 0.5
    #: Fraction of a device's reservable memory the pool may grow to. `1.0` means all of
    #: what the VRAM headroom leaves; the remainder stays available to a co-tenant.
    pool_max_fraction: float = 1.0
    #: Let cuDF move columns to host memory rather than fail when the device fills. Turns a
    #: class of hard OOM into a slowdown, which is what makes a shard that misjudged its size
    #: survivable. Off by default because it makes an over-large query slow rather than loud.
    spill_to_host: bool = False
    #: Track allocation counts and the device high-water mark, so a stage reports the device
    #: memory it actually peaked at rather than the footprint it declared. Costs an atomic
    #: per allocation.
    statistics: bool = False


@dataclass(frozen=True, slots=True)
class AcceleratorConfig:
    """How accelerator work is placed, partitioned, and budgeted.

    Examples:
        .. doctest::

            >>> from batcher.config import AcceleratorConfig
            >>> AcceleratorConfig().vram_headroom
            0.15
    """

    energy: EnergyConfig = EnergyConfig()
    health: DeviceHealthConfig = DeviceHealthConfig()
    memory: DeviceMemoryConfig = DeviceMemoryConfig()
    #: Fraction of each device's memory held back from reservation: CUDA context, allocator
    #: fragmentation, and activation peaks no declared model footprint includes.
    vram_headroom: float = 0.15
    #: Keep a multi-device collective inside one NVLink domain when the fleet has one wide
    #: enough, and report it when it does not. On by default: the alternative is an
    #: all-reduce that silently leaves the fast path.
    fabric_aware_placement: bool = True
    #: Prefer a MIG partition over a whole device when a model fits one. On by default — a
    #: partition gives memory and fault isolation that fractional scheduling does not.
    prefer_mig: bool = True
    #: On a heterogeneous fleet, place work on the most throughput-per-watt device that fits
    #: rather than the first that fits. Off by default because it constrains placement, which
    #: costs queue time on a busy fleet.
    efficiency_first_placement: bool = False
    #: KV-cache element type for inference sizing (`fp16`, `bf16`, `fp8`, `int8`, `fp32`).
    #: The single largest lever on concurrency: FP8 halves the cache against FP16.
    kv_cache_dtype: str = "fp16"
    #: Fraction of a device left free when sizing an inference stage's KV cache.
    kv_cache_headroom: float = 0.1
    #: Maximum context length to size the KV cache for. `0` means "the model's own maximum",
    #: which is often far longer than a workload's actual prompts and costs concurrency
    #: proportionally.
    max_context_tokens: int = 0


def validate_accelerator(cfg: AcceleratorConfig) -> None:
    """Raise `ConfigError` when an accelerator tunable is out of range.

    Args:
        cfg: The accelerator section to check.

    Raises:
        ConfigError: On the first out-of-range value, naming the field and its bound.
    """
    energy, health, memory = cfg.energy, cfg.health, cfg.memory
    checks: tuple[tuple[bool, str], ...] = (
        (
            memory.allocator in _ALLOCATORS,
            f"accelerator.memory.allocator {memory.allocator!r} must be one of "
            f"{sorted(_ALLOCATORS)}",
        ),
        (
            0.0 < memory.pool_initial_fraction <= 1.0,
            "accelerator.memory.pool_initial_fraction must be in (0, 1]",
        ),
        (
            0.0 < memory.pool_max_fraction <= 1.0,
            "accelerator.memory.pool_max_fraction must be in (0, 1]",
        ),
        (
            memory.pool_initial_fraction <= memory.pool_max_fraction,
            "accelerator.memory.pool_initial_fraction must not exceed pool_max_fraction",
        ),
        (energy.power_budget_watts >= 0, "accelerator.energy.power_budget_watts must be >= 0"),
        (0.0 <= energy.power_headroom < 1.0, "accelerator.energy.power_headroom must be in [0, 1)"),
        (energy.carbon_intensity >= 0, "accelerator.energy.carbon_intensity must be >= 0"),
        (energy.price_per_kwh >= 0, "accelerator.energy.price_per_kwh must be >= 0"),
        (
            energy.pue >= 1.0,
            "accelerator.energy.pue must be >= 1.0 (a facility cannot beat its IT load)",
        ),
        (
            0.0 <= energy.renewable_fraction <= 1.0,
            "accelerator.energy.renewable_fraction must be in [0, 1]",
        ),
        (energy.telemetry_interval_s > 0, "accelerator.energy.telemetry_interval_s must be > 0"),
        (health.max_temperature_c > 0, "accelerator.health.max_temperature_c must be > 0"),
        (
            0.0 <= health.quarantine_below_derate <= 1.0,
            "accelerator.health.quarantine_below_derate must be in [0, 1]",
        ),
        (
            0.0 < health.max_memory_fraction <= 1.0,
            "accelerator.health.max_memory_fraction must be in (0, 1]",
        ),
        (0.0 <= cfg.vram_headroom < 1.0, "accelerator.vram_headroom must be in [0, 1)"),
        (0.0 <= cfg.kv_cache_headroom < 1.0, "accelerator.kv_cache_headroom must be in [0, 1)"),
        (cfg.max_context_tokens >= 0, "accelerator.max_context_tokens must be >= 0"),
        (
            cfg.kv_cache_dtype.lower() in _CACHE_DTYPES,
            f"accelerator.kv_cache_dtype {cfg.kv_cache_dtype!r} is not a supported cache dtype",
        ),
    )
    for ok, message in checks:
        if not ok:
            raise ConfigError(message)
