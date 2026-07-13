# Credit-based flow control

A Parquet scanner produces batches at 10,000/s. A GPU embedding model consumes them at
100/s. Connect them and, with no intervention, 99% of the scanned data sits in RAM waiting
its turn, until the worker's memory is gone. This is not a pathological case; it is what
every pipeline with a slow stage does by default.

The shuffle has the same shape. A mapper can serialize its bucket far faster than a reducer
can fold it, and a reducer fetching from sixteen mappers at once is sixteen producers
against one consumer.

The fix is the one TCP uses: the producer may only send what the consumer has said it has
room for.

## One credit is one batch slot

:::{important}
A channel's in-flight memory is at most `credits × batch_bytes`. Not "usually bounded", not
"bounded under normal load". Bounded, because the producer physically cannot send batch `n+1`
until a permit for it exists. That is what makes the memory bound arithmetic rather than
aspirational.
:::

```text
   PRODUCER  (mapper — FlightHandler::do_exchange)        CONSUMER  (reducer)
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
message, a little-endian `u32` in `app_metadata`, and a decoded zero (missing or
malformed seed) falls back to a compiled-in default rather than stalling forever.

## Replenishment

Granting one credit per batch would double the message count for no benefit. The consumer
does batched low-watermark refill (`credit_exchange_inner` in `exchange.rs`):

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

There is an `InflightGauge` alongside, tracking `current` and a high-water `max`. It is not
decoration: it is how the credit bound is *tested*, surfaced as
`ShuffleSession.max_inflight` so a test can assert that no channel ever held more batches
than its window allowed.

## The window, and who sets it

Carbonite is the authority. `ResourceManager.grant_credits(requested, signature=...)` is
the single entry point, and it clamps every request:

```python
# docs: skip
# python/batcher/carbonite/policies.py
def credit_ceiling(config, effective_morsel_bytes=None) -> int:
    count_ceiling = fc.default_credits * fc.credit_ceiling_factor   # 16 * 4 = 64
    morsel_bytes = max(1, effective_morsel_bytes or config.execution.morsel_bytes)
    byte_ceiling = max(1, fc.credit_byte_budget // morsel_bytes)    # 256 MiB // 1 MiB = 256
    return max(1, min(count_ceiling, byte_ceiling))
```

Two ceilings, and the tighter wins. The count ceiling is the obvious one. The byte ceiling
exists because a credit is *one batch*, and a batch of 768-dimensional embeddings is not
the same object as a batch of int64 keys. Sixty-four credits of wide rows can be gigabytes.
`_learned_channel_morsel_bytes` widens the assumed batch size from the learned row width,
so a wide-row workload gets *fewer* credits automatically.

Defaults (`config.flow_control`):

| Knob | Default | Meaning |
|---|---|---|
| `default_credits` | 16 | starting window when there is no learned estimate |
| `credit_ceiling_factor` | 4 | count ceiling = 16 × 4 = 64 |
| `credit_byte_budget` | 256 MiB | byte ceiling per channel |
| `aimd_alpha` | 1 | additive increase, +1 credit per round |
| `aimd_beta` | 0.5 | multiplicative decrease on congestion |

With the shipped defaults the count ceiling binds and the byte ceiling is slack:

```python
from batcher.config import Config

cfg = Config()
count_ceiling = cfg.flow_control.default_credits * cfg.flow_control.credit_ceiling_factor
byte_ceiling = cfg.flow_control.credit_byte_budget // cfg.execution.morsel_bytes
print(count_ceiling, byte_ceiling, "->", min(count_ceiling, byte_ceiling))
```

```text
64 256 -> 64
```

Raise `morsel_bytes`, or let the learned row width widen it for an embedding column, and
the byte ceiling drops below 64 and starts binding instead. That crossover is the whole
point of having both.

The starting window was 4 and is now 16, which matters more than it sounds. A cross-node
fetch's throughput is `window × batch / RTT`, so a small window throttles the opening
rounds before any adaptation can help. On a 50 ms-RTT link, a single 18 MiB partition
transferred at 2.4 MiB/s at 4 credits and 7.7 MiB/s at 16.

## AIMD

A static window is a compromise: too small and you leave bandwidth on the floor, too large
and you buffer memory you did not need.

::::{tab-set}
:::{tab-item} Static window
```text
distributed.adaptive_credits = False

  the window is whatever grant_credits() returned, clamped to credit_ceiling
  no slow start, no reaction to pressure
  simple, and wrong in one direction or the other on most links
```
:::

:::{tab-item} AIMD (the default)
```text
distributed.adaptive_credits = True

  slow start:  window × 2 each uncongested round, until the first congestion signal
  then:        +aimd_alpha per uncongested round;  ×aimd_beta on congestion
  always clamped to [1, credit_ceiling]
```
AIMD moves the window *inside* the envelope Carbonite set. It does not move the envelope.
:::
::::

`distributed.adaptive_credits` turns on the TCP-like controller.

```python
# docs: skip
# python/batcher/carbonite/policies.py — AIMDFlowControl
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

Slow-start doubles the window each uncongested round until the first congestion signal,
then it is classic additive-increase/multiplicative-decrease. The window is always clamped
to `[1, credit_ceiling]`, so adaptation can never escape the memory bound Carbonite set.

The congestion signal is memory pressure, not queue occupancy.
`ShuffleSession._observe_backpressure`:

```python
# docs: skip
congested = self._pressure.level() >= PressureLevel.SPILL
self._flow_control.observe(congested=congested)
```

That is worth being precise about, because it is easy to assume otherwise. The channel does
not measure its own buffer depth. It asks the process-wide `PressureMonitor` whether the
node is past `memory.soft_limit`, and shrinks if so. The `PressureMonitor` has asymmetric
hysteresis (escalate instantly, de-escalate on an EWMA), which is what keeps the window
from oscillating.

:::{note}
`flow_control.backpressure_high` (0.70) and `backpressure_low` (0.40) are declared and
validated in the config, and are documented elsewhere as buffer-occupancy thresholds. They
have no runtime consumer. The live thresholds are the `PressureLevel` ladder in
`carbonite/memory/pressure.py`.
:::

## Warm-starting a recurring channel

Slow-start costs a few rounds every time, and a shuffle of the same shape converges to the
same window every time. So the converged window is persisted per shuffle signature
(namespace `carbonite.shuffle_window`, exponentially smoothed) and `grant_credits(signature=…)`
warm-starts the next run from it, skipping slow-start entirely.

Learning only moves the *starting point*. AIMD still governs the window actually used from
live pressure, and the ceiling still clamps it. A credit window bounds in-flight batches and
nothing else, so none of this can change a result. Which is exactly why it is safe to learn
aggressively.

## Costs and limits

Credits cost a semaphore acquire per batch on the producer and one grant message per half
window on the consumer. At 1 MiB batches that is negligible; at very small batches the
per-batch permit is a real fraction of the work, which is one more reason morsels are
16,384 rows and not 100.

:::{warning}
Striping is the sharp edge. `ClientPool::fetch_secured_striped` gives **each shard its own full
window**, so a peer fetched over 4 connections can have `4 × credits` batches in flight. The
byte bound is per-channel, not per-peer, and `distributed.flight_connections_per_peer` (4) is
the only thing bounding the multiplier.
:::

And the bound is on *transport* memory only. Credits stop a mapper from flooding a reducer's
socket buffers; they do not stop the reducer's own hash table from growing. That is the
buffer pool's job, and the two meet in the pressure signal that AIMD reads.

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
