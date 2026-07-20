# Credit-based flow control

*Credit-based flow control* bounds how much data a producer can put in flight toward a
consumer: the producer may only send what the consumer has said it has room for. This page
describes the credit protocol on the shuffle wire, how credits are replenished, how
Carbonite sizes the window, and how the AIMD controller moves that window at run time.

A Parquet scanner produces batches at 10,000/s. A GPU embedding model consumes them at
100/s. Connect them and, with no intervention, 99% of the scanned data sits in RAM waiting
its turn, until the worker's memory is gone. This isn't a pathological case. It's what
every pipeline with a slow stage does by default.

The shuffle has the same shape. A mapper can serialize its bucket far faster than a reducer
can fold it, and a reducer fetching from sixteen mappers at once is sixteen producers
against one consumer. The fix is the one TCP uses.

## One credit is one batch slot

:::{important}
A channel's in-flight memory is at most `credits × batch_bytes`. Not "usually bounded", not
"bounded under normal load". Bounded, because the producer physically can't send batch `n+1`
until a permit for it exists. That's what makes the memory bound arithmetic rather than
aspirational.
:::

```text
   PRODUCER  (mapper: FlightHandler::do_exchange)        CONSUMER  (reducer)
   ─────────────────────────────────────────────          ──────────────────

                     credits: tokio::sync::Semaphore
                                   │                       seeds the window in the
                                   │  ◄──── seed(n) ────── first DoExchange message
   for batch in bucket:            │                       (a little-endian u32 in
       credits.acquire().await ────┤                        app_metadata)
                                   │
              ── blocks here at 0 ─┤
                                   │
       gauge.on_send()             │
       yield batch  ───────────────┼──────────────────────► receive; pending += 1
                                   │
                                   │                        if pending >= credits/2:
                                   │  ◄──── grant(n) ─────      send one grant
                                   │
       (a spawned pump task drains the grants and calls credits.add_permits)
                                   │
                                   ▼
```

The producer blocks in one place. `FlightHandler::do_exchange`, in
`crates/bc-transport/src/handler.rs`:

```rust
let gated = async_stream::stream! {
    let _pump = pump;
    for batch in batch_vec {
        match credits.acquire().await {   // blocks at zero
            Ok(permit) => permit.forget(),
            Err(_) => break,
        }
        gauge.on_send();
        yield Ok(batch);
    }
};
```

`credits` is a `tokio::sync::Semaphore`. The consumer seeds it in the first `DoExchange`
message as a little-endian `u32` in `app_metadata`. A decoded zero, from a missing or
malformed seed, falls back to a compiled-in default rather than stalling forever.

## Replenishment

Granting one credit per batch would double the message count for no benefit. The consumer
does a batched low-watermark refill in `credit_exchange_inner` in `exchange.rs`:

```rust
let refill_at = (credits / 2).max(1);
let mut pending: u32 = 0;
loop {
    // ... receive a batch, with an idle timeout so a dead peer doesn't hang the fetch
    pending += 1;
    if pending >= refill_at {
        grant_tx.send(/* encode_credits(pending) */).await;
        pending = 0;
    }
}
```

The server runs a spawned pump task that drains inbound grant messages and calls
`credits.add_permits(granted)`. Half the window is the refill point, so the producer never
runs dry while a grant is in flight.

An `InflightGauge` runs alongside, tracking `current` and a high-water `max`. It isn't
decoration. It's how the credit bound gets *tested*, surfaced as
`ShuffleSession.max_inflight` so a test can assert that no channel ever held more batches
than its window allowed.

## The window, and who sets it

Carbonite is the authority. `ResourceManager.grant_credits(requested, signature=...)` is
the single entry point, and it clamps every request:

```python
# docs: skip
# python/batcher/carbonite/policies.py
def credit_ceiling(config, effective_morsel_bytes=None) -> int:
    count_ceiling = fc.default_credits * fc.credit_ceiling_factor
    morsel_bytes = max(1, effective_morsel_bytes or config.execution.morsel_bytes)
    byte_ceiling = max(1, _channel_byte_budget(config) // morsel_bytes)
    return max(1, min(count_ceiling, byte_ceiling))
```

Two ceilings, and the tighter wins. The count ceiling is the obvious one, `default_credits
× credit_ceiling_factor`. The byte ceiling exists because a credit is *one batch*, and a
batch of 768-dimensional embeddings isn't the same object as a batch of int64 keys.
Sixty-four credits of wide rows can be gigabytes. `_learned_channel_morsel_bytes` widens
the assumed batch size from the learned row width, so a wide-row workload gets *fewer*
credits automatically.

The per-channel byte budget is itself memory-aware. `_channel_byte_budget` caps the
configured `credit_byte_budget` at `_SHUFFLE_BUFFER_FRACTION` of the machine's total RAM,
which is 10%, divided by `shuffle_fetch_fan_in`, with a floor of one morsel. That matters
because 256 MiB per channel across 32 concurrent fetches is 8 GiB in flight, which is
unremarkable on a 512 GiB node and more than half the RAM of a 16 GiB one. The cap only
ever lowers the configured value, so tuning `credit_byte_budget` down keeps your number.

These are the defaults under `config.flow_control`:

| Knob | Default | Meaning |
|---|---|---|
| `default_credits` | 16 | starting window when there's no learned estimate |
| `credit_ceiling_factor` | 4 | count ceiling is `default_credits` times this |
| `credit_byte_budget` | 256 MiB | configured byte ceiling per channel, before the RAM cap |
| `shuffle_fetch_fan_in` | 32 | channels fetching at once, which divides the RAM share |
| `aimd_alpha` | 1 | additive increase, +1 credit per round |
| `aimd_beta` | 0.5 | multiplicative decrease on congestion |

Which ceiling binds depends on the host, so read it rather than assuming:

```python
from batcher.config import Config
from batcher.carbonite.policies import credit_ceiling

cfg = Config()
count_ceiling = cfg.flow_control.default_credits * cfg.flow_control.credit_ceiling_factor
print("count ceiling:", count_ceiling)
print("effective ceiling:", credit_ceiling(cfg))
print("with 8 MiB batches:", credit_ceiling(cfg, 8 << 20))
```

On a machine with enough RAM the count ceiling of 64 binds. Raise `morsel_bytes`, run on a
smaller node, or let the learned row width widen the assumed batch for an embedding column,
and the byte ceiling drops below the count ceiling and starts binding instead. That
crossover is the whole point of having both.

The starting window of 16 matters more than it sounds. A cross-node fetch's throughput is
`window × batch / RTT`, so a small window throttles the opening rounds before any
adaptation can help. The rationale recorded alongside the default in
`config/config.py` measures a single 18 MiB partition over a 50 ms-RTT link at 2.4 MiB/s
with 4 credits against 7.7 MiB/s with 16.

## AIMD

A static window is a compromise. Too small and you leave bandwidth on the floor. Too large
and you buffer memory you didn't need.

`distributed.adaptive_credits` selects between the two, and it defaults to `True`.

Set it to `False` and the window is whatever `grant_credits()` returned, clamped to
`credit_ceiling`. There's no slow start and no reaction to pressure. That's simple, and
wrong in one direction or the other on most links.

Leave it at `True` and a TCP-like controller runs. Slow start doubles the window each
uncongested round until the first congestion signal. After that it's classic
additive-increase, multiplicative-decrease: `+aimd_alpha` per uncongested round, and
`×aimd_beta` on congestion. The window is always clamped to `[1, credit_ceiling]`, so
AIMD moves the window *inside* the envelope Carbonite set and never moves the envelope
itself.

```python
# docs: skip
# python/batcher/carbonite/policies.py, AIMDFlowControl
def observe(self, *, congested: bool) -> int:
    if congested:
        self._window = max(self._floor, self._window * self._beta)
        self._slow_start = False
    elif self._slow_start:
        self._window = min(self._ceiling, self._window * _SLOW_START_FACTOR)
    else:
        self._window = min(self._ceiling, self._window + self._alpha)
    return self.window
```

The congestion signal is memory pressure, not queue occupancy.
`ShuffleSession._observe_backpressure` reads it:

```python
# docs: skip
congested = self._pressure.level() >= PressureLevel.SPILL
self._flow_control.observe(congested=congested)
```

That's worth being precise about, because it's easy to assume otherwise. The channel
doesn't measure its own buffer depth. It asks the process-wide `PressureMonitor` whether
the node is past `memory.soft_limit`, and shrinks if so. The `PressureMonitor` escalates
instantly and de-escalates only as an EWMA relaxes, and that asymmetric hysteresis is what
keeps the window from oscillating.

:::{note}
`flow_control.backpressure_high` (0.70) and `backpressure_low` (0.40) are declared and
validated in the config, and are documented elsewhere as buffer-occupancy thresholds. They
have no runtime consumer. The live thresholds are the `PressureLevel` ladder in
`carbonite/memory/pressure.py`.
:::

## Warm-starting a recurring channel

Slow start costs a few rounds every time, and a shuffle of the same shape converges to the
same window every time. So the converged window is persisted per shuffle signature under
the `carbonite.shuffle_window` namespace, exponentially smoothed, and
`grant_credits(signature=…)` warm-starts the next run from it, skipping slow start
entirely.

Learning only moves the *starting point*. AIMD still governs the window actually used from
live pressure, and the ceiling still clamps it. A credit window bounds in-flight batches and
nothing else, so none of this can change a result. That's exactly why it's safe to learn
aggressively.

## Costs and limits

Credits cost a semaphore acquire per batch on the producer and one grant message per half
window on the consumer. At 1 MiB batches that's negligible. At very small batches the
per-batch permit is a real fraction of the work, which is one more reason morsels are
16,384 rows and not 100.

:::{warning}
Striping is the sharp edge. `ClientPool::fetch_secured_striped` gives **each shard its own full
window**, so a peer fetched over 4 connections can have `4 × credits` batches in flight. The
byte bound is per-channel, not per-peer, and `distributed.flight_connections_per_peer`, which
defaults to 4, is the only thing bounding the multiplier.
:::

The bound also covers *transport* memory only. Credits stop a mapper from flooding a
reducer's socket buffers. They don't stop the reducer's own hash table from growing. That's
the buffer pool's job, and the two meet in the pressure signal that AIMD reads.

## See also

:::{seealso}
- [Architecture](../architecture/index.md): Carbonite as the authority over every backpressure knob
- [Carbonite](../internals/carbonite.md): the flow-control knob reference
- `docs/internals/mathematical_foundations.md` (in the repo, not a site page): the stability argument for AIMD under a clamp
- [Configuration options](../configuration/options.md): `flow_control.*` and `distributed.adaptive_credits`
- [Streaming](../user-guide/streaming.md): the other place a fast producer meets a slow consumer
- [Scaling benchmarks](../benchmarks/scaling.md): what the credited shuffle sustains as nodes are added
- [The shuffle over Arrow Flight](shuffle-flight.md): what the credits are gating
- [The buffer pool](buffer-pool.md): where `PressureLevel` comes from
- [Distributed scheduling](distributed-scheduling.md): how many channels there are
:::
