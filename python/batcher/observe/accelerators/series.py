"""Sampling devices over a run, so a stage gets a distribution instead of a snapshot.

Every reading in `hardware.telemetry` is instantaneous, and a single instantaneous reading is
the wrong shape for the question anyone asks. "Was the GPU busy during this query" sampled once,
at the end, systematically samples the moment the work drained. Sampled once at the start, it
samples the moment before it began. The honest answer needs the device watched *while* the work
runs, and nothing in the engine's own timings can substitute for it.

This module is the watcher: one daemon thread per process, folding readings into the bounded
accumulator in `hardware.telemetry.sampler` on the configured interval.

**One thread, started explicitly, never at import.** A `_internal` probe that started a sampler
on import would start one in every Ray worker on the cluster, including the short-lived ones
where the sampling costs more than the stage it measures. So the lifecycle is owned here, in the
observability layer, and it is opt-in: `bt.start_ui()` and the accelerator report turn it on for
their own duration, and `accelerator.telemetry_sampling` turns it on for a whole process.

**Bounded regardless of run length.** The accumulator keeps running aggregates rather than
samples, so a sampler left running for a day costs exactly what one running for a second costs.
That is what makes it safe to leave on.

**A failing sample must never take the run with it.** The loop catches everything, and a source
that raises repeatedly is simply absent from the window rather than fatal — a device that
disappeared mid-run is a real condition, and it is reported by the samples stopping, not by an
exception on a thread nobody is waiting on.
"""

from __future__ import annotations

import threading

from batcher._internal.hardware.telemetry.sampler import TelemetrySampler
from batcher._internal.logging import note_suppressed

__all__ = [
    "device_window",
    "reset_device_series",
    "sampling_active",
    "start_device_series",
    "stop_device_series",
]

#: The one sampler and the one thread controlling it. A list rather than a module global so the
#: start/stop pair can swap both atomically under `_LOCK` without a global statement in four
#: functions.
_STATE: list[tuple[TelemetrySampler, threading.Thread, threading.Event]] = []
_LOCK = threading.Lock()

#: Floor on the sampling interval. NVML costs tens of microseconds per field per device and the
#: full sweep here is dozens of fields; below this the sampler is competing with the workload it
#: is measuring, which is the one thing a measurement must not do.
_MIN_INTERVAL_S = 0.1


def _interval() -> float:
    """The configured sampling interval in seconds, floored at `_MIN_INTERVAL_S`."""
    try:
        from batcher.config import active_config

        configured = float(active_config().accelerator.energy.telemetry_interval_s)
    except Exception:
        configured = 1.0
    return max(_MIN_INTERVAL_S, configured)


def _sample_once(sampler: TelemetrySampler) -> None:
    """Fold one round of readings from every source into the accumulator.

    Each source is guarded separately rather than as a group: on a host where NVML answers and
    DCGM does not, grouping them would drop the NVML readings too, and the NVML half is the one
    that produces a verdict on almost every host.
    """
    try:
        from batcher._internal.hardware.nvml import device_telemetry

        sampler.observe_telemetry(device_telemetry())
    except Exception as exc:
        note_suppressed("observe", "sample device telemetry", exc)
    try:
        from batcher._internal.hardware.telemetry.throughput import device_throughput

        sampler.observe_throughput(device_throughput())
    except Exception as exc:
        note_suppressed("observe", "sample device link throughput", exc)
    try:
        from batcher._internal.hardware.telemetry.dcgm import device_profiles

        for profile in device_profiles():
            sampler.observe(profile.index, "sm_active", profile.sm_active)
            sampler.observe(profile.index, "sm_occupancy", profile.sm_occupancy)
            sampler.observe(profile.index, "tensor_active", profile.tensor_active)
            sampler.observe(profile.index, "dram_active", profile.dram_active)
    except Exception as exc:
        note_suppressed("observe", "sample device performance counters", exc)
    try:
        from batcher._internal.hardware.telemetry.engines import device_engines

        for engine in device_engines():
            sampler.observe(engine.index, "codec", max(engine.decoder, engine.encoder, engine.jpeg))
    except Exception as exc:
        note_suppressed("observe", "sample device codec engines", exc)


def _loop(sampler: TelemetrySampler, stop: threading.Event) -> None:
    """Sample until asked to stop, on the configured interval.

    The interval is re-read each pass rather than captured once, so a `config_context` that
    tightens it takes effect on a running sampler instead of at the next restart.
    """
    while not stop.is_set():
        _sample_once(sampler)
        # `wait` rather than `sleep`: a stop during the interval returns immediately instead of
        # leaving the caller of `stop_device_series` blocked for up to a full period.
        stop.wait(_interval())


def start_device_series() -> bool:
    """Begin sampling this host's devices into a rolling window.

    Idempotent: a second call while sampling is active is a no-op and reports False, so nesting
    a report inside a running dashboard does not start a second thread or reset the first one's
    window.

    Returns:
        True when this call started the sampler, False when it was already running.
    """
    with _LOCK:
        if _STATE:
            return False
        sampler = TelemetrySampler()
        stop = threading.Event()
        thread = threading.Thread(
            target=_loop,
            args=(sampler, stop),
            name="batcher-device-telemetry",
            daemon=True,
        )
        _STATE.append((sampler, thread, stop))
        thread.start()
        return True


def stop_device_series(timeout: float = 2.0) -> None:
    """Stop sampling and let the thread finish its current pass.

    The accumulated window survives the stop, so a caller reads `device_window` after stopping
    rather than racing the last sample. `reset_device_series` is what discards it.

    Args:
        timeout: Seconds to wait for the thread. It is a daemon thread, so a timeout leaves it
            to be reaped at interpreter exit rather than hanging the process; the bound exists
            so a wedged NVML call cannot stall a shutdown.
    """
    with _LOCK:
        if not _STATE:
            return
        _thread, stop = _STATE[0][1], _STATE[0][2]
    # Joined outside the lock: the loop takes no lock of its own, but a caller blocking here
    # while holding `_LOCK` would stall `device_window` for the length of the timeout.
    stop.set()
    _thread.join(timeout=timeout)


def sampling_active() -> bool:
    """Whether the sampling thread is running right now."""
    with _LOCK:
        return bool(_STATE) and _STATE[0][1].is_alive()


def device_window() -> TelemetrySampler | None:
    """The accumulator holding this run's device window, or `None` when never started.

    Returned rather than copied: the accumulator's summaries are computed on read from running
    aggregates, so a caller taking a summary while the thread is still sampling gets a
    consistent-enough answer without a lock — every field it reads is a single float, and a
    window that gained one more sample between two of them is not a meaningful inconsistency.

    Returns:
        The sampler, or `None` when sampling has not been started in this process.
    """
    with _LOCK:
        return _STATE[0][0] if _STATE else None


def reset_device_series() -> None:
    """Stop sampling and discard the accumulated window.

    The hook a test needs, and what a caller uses to start a fresh window around a specific
    query rather than reading one that has been accumulating since the process started.
    """
    stop_device_series()
    with _LOCK:
        _STATE.clear()
