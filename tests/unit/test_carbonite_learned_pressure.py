"""Learned pressure hysteresis + morsel working-set sizing.

Two measured→tuned loops here, both purely about *when/how* the engine batches and
throttles — never about the result:

* Pressure hysteresis: a channel measured to flap SPILL↔NORMAL gets a stickier
  de-escalation weight so its shuffle credit window stops oscillating.
* Morsel working-set: a workload whose rows proved far wider than the assumed
  `optimizer.row_bytes` (embeddings, blobs) gets a smaller morsel *row* count so the
  morsel's true byte working set stays within the byte budget — before RAM is pressured.

Both fall back to the exact current behavior on a cold store, and a real query is
byte-identical whether the tuned morsel fires or not.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher import Config, col, config_context
from batcher.carbonite import ResourceManager
from batcher.carbonite.memory.pressure import (
    PressureLevel,
    PressureMonitor,
    hysteresis_alpha_from_flap,
    load_flap_rate,
    record_flap_rate,
)
from batcher.config import active_config
from batcher.config.config import ExecutionConfig
from batcher.metadata import MetadataHub
from batcher.metadata.backends import InProcessBackend
from batcher.plan.feedback import OperatorFeedback
from batcher.plan.ids import OpId

_ROW_BYTES = active_config().optimizer.row_bytes


def _hub() -> MetadataHub:
    return MetadataHub(InProcessBackend())


# --- pressure-flap hysteresis ------------------------------------------------


def test_flap_rate_recorded_and_loaded():
    hub = _hub()
    assert load_flap_rate(hub) is None
    record_flap_rate(hub, 0.6)
    assert abs(load_flap_rate(hub) - 0.6) < 1e-9


def test_hysteresis_alpha_stiffens_with_flap():
    assert hysteresis_alpha_from_flap(None) is None  # cold → the monitor's static default
    assert hysteresis_alpha_from_flap(0.0) == 0.5  # quiet history → default weight
    # A fully-flapping history stiffens the de-escalation weight far below the default.
    assert hysteresis_alpha_from_flap(1.0) < 0.2


def test_monitor_uses_learned_alpha():
    hub = _hub()
    record_flap_rate(hub, 1.0)
    rm = ResourceManager(hub=hub)
    assert rm._pressure._alpha < 0.2  # stiffened
    assert ResourceManager()._pressure._alpha == PressureMonitor._EWMA_ALPHA  # cold default


def test_stickier_hysteresis_holds_level_longer_on_falling_edge(monkeypatch):
    # Feed both monitors the SAME raw sequence — a CRITICAL spike then relief — and show
    # the stiffer (learned-flap) monitor de-escalates more slowly: it is still CRITICAL a
    # sample after the looser one has relaxed to SPILL. Escalation is instant for both
    # (protective spill is never delayed); only the falling edge is damped, so a flapping
    # channel stops oscillating its shuffle credit window.
    holder = {"raw": 0.0}
    monkeypatch.setattr(
        PressureMonitor, "_engine_used_fraction", staticmethod(lambda: holder["raw"])
    )
    stiff = PressureMonitor(hysteresis_alpha=0.1)
    loose = PressureMonitor(hysteresis_alpha=0.5)
    levels_stiff, levels_loose = [], []
    for raw in (0.95, 0.80, 0.80):  # spike (>hard 0.90), then relief into the ELEVATED band
        holder["raw"] = raw
        levels_stiff.append(stiff.level())
        levels_loose.append(loose.level())
    # By the third sample the loose monitor has de-escalated below the stiff one.
    assert levels_stiff[-1] == PressureLevel.CRITICAL
    assert levels_loose[-1] < levels_stiff[-1]


# --- morsel working-set from learned per-row bytes ---------------------------


def _seed_width(hub: MetadataHub, kind: str, bytes_per_row: float, *, n: int = 30) -> None:
    for _ in range(n):
        hub.record(
            OperatorFeedback(
                op_id=OpId(1),
                kind=kind,
                n_actual=100,
                t_op_ms=1.0,
                m_peak_bytes=int(bytes_per_row * 1000),
                selectivity=0.1,
                batch_size=16384,
                n_input=1000,
            )
        )


def test_morsel_rows_capped_by_learned_width():
    hub = _hub()
    # Rows measured 64x wider than assumed → cap rows so rows*width stays within morsel_bytes.
    width = _ROW_BYTES * 64
    _seed_width(hub, "aggregate", bytes_per_row=width)
    rm = ResourceManager(hub=hub)
    target = rm.recommend_morsel_target()
    assert target is not None
    rows, nbytes = target
    exec_cfg = active_config().execution
    expected = int(exec_cfg.morsel_bytes / width)
    assert rows == expected
    assert rows < exec_cfg.morsel_rows  # genuinely tightened
    assert nbytes == exec_cfg.morsel_bytes  # byte budget unchanged (unpressured)
    # The property this test exists to prove, which its own comment stated and its
    # assertion did not: the resulting morsel is inside the byte budget. It previously
    # asserted `max(1024, expected)`, so at this width it accepted 1,024 rows x 4 KiB =
    # 4 MiB against a 1 MiB budget — the flat row floor overriding the bound it accompanies.
    assert rows * width <= exec_cfg.morsel_bytes


def test_morsel_unchanged_on_cold_store():
    # No learned width and no pressure → keep the configured target (the fast path).
    assert ResourceManager(hub=_hub()).recommend_morsel_target() is None


def test_morsel_narrow_rows_no_change():
    hub = _hub()
    _seed_width(hub, "aggregate", bytes_per_row=8.0)  # narrower than assumed
    assert ResourceManager(hub=hub).recommend_morsel_target() is None


# --- result-invariance -------------------------------------------------------


def _rows(tbl: pa.Table) -> list[tuple]:
    return sorted(tuple(r.values()) for r in tbl.to_pylist())


def test_learned_morsel_is_result_invariant():
    # A learned-tightened morsel produces an identical aggregate to the default morsel.
    t = pa.table({"k": [i % 7 for i in range(5000)], "v": list(range(5000))})

    def q() -> pa.Table:
        return bt.from_arrow(t).group_by("k").agg(s=col("v").sum()).collect()

    baseline = q()
    # Force the learned-tightened morsel target for the execution scope.
    tightened = Config().replace(
        execution=ExecutionConfig(morsel_rows=1024, morsel_bytes=64 * 1024)
    )
    with config_context(tightened):
        adapted = q()
    assert _rows(adapted) == _rows(baseline)


def test_the_monitor_measures_the_flap_rate_its_hysteresis_consumes():
    """`record_flap_rate` had no producer, so the anti-oscillation mechanism never engaged.

    `ResourceManager.__init__` reads a past run's flap rate back
    (`load_flap_rate` -> `hysteresis_alpha_from_flap`) to stiffen de-escalation for a
    workload observed to oscillate SPILL<->NORMAL. Nothing ever wrote one, so the read was
    permanently cold, `hysteresis_alpha_from_flap(None)` returned `None`, and the monitor
    always used the static `_EWMA_ALPHA`. The whole `_FLAP_NS`/`_FLAP_STIFFEN` path was dead.

    The monitor is the natural producer — it is the component that sees every level — and it
    only *measures*, which is all `PressureMonitor` is allowed to do. Hysteresis damps *when*
    the engine spills or throttles, never what it computes, so this is result-invariant.
    """
    from batcher.carbonite.memory.pressure import (
        PressureLevel,
        PressureMonitor,
        hysteresis_alpha_from_flap,
        load_flap_rate,
        record_flap_rate,
    )
    from batcher.metadata import MetadataHub
    from batcher.metadata.backends import InProcessBackend

    monitor = PressureMonitor()
    assert monitor.flap_rate() is None  # fewer than two samples: nothing to conclude

    # An oscillating run: the level keeps reversing direction.
    for level in (
        PressureLevel.NORMAL,
        PressureLevel.SPILL,
        PressureLevel.NORMAL,
        PressureLevel.SPILL,
        PressureLevel.NORMAL,
    ):
        monitor._observe_flap(level)
    flappy = monitor.flap_rate()
    assert flappy is not None and flappy > 0.0

    # It round-trips through the hub and actually stiffens the hysteresis.
    hub = MetadataHub(InProcessBackend())
    record_flap_rate(hub, flappy)
    assert load_flap_rate(hub) == flappy
    stiffened = hysteresis_alpha_from_flap(load_flap_rate(hub))
    assert stiffened is not None and stiffened < PressureMonitor._EWMA_ALPHA

    # A monotonic climb is not a flap, however many steps it takes.
    steady = PressureMonitor()
    for level in (PressureLevel.NORMAL, PressureLevel.ELEVATED, PressureLevel.SPILL):
        steady._observe_flap(level)
    assert steady.flap_rate() == 0.0
    # ...and a quiet history keeps the static default.
    assert hysteresis_alpha_from_flap(steady.flap_rate()) == PressureMonitor._EWMA_ALPHA
