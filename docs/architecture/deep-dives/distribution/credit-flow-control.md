# Credit-based flow control

*Credit-based flow control* bounds how much data a producer can put in flight toward a consumer: the producer may only send what the consumer has said it has room for. This page describes the credit protocol on the shuffle wire, how Carbonite sizes the window, and how the controller moves it at run time.

A scanner producing 10,000 batches a second feeding a model consuming 100 a second would, unchecked, park 99% of the scanned data in RAM. A shuffle has the same shape: a reducer fetching from sixteen mappers is sixteen producers against one consumer. The fix is the one TCP uses.

## One credit is one batch slot

:::{important}
A channel's in-flight memory is at most `credits x batch_bytes`. The producer can't send batch `n+1` until a permit for it exists, which makes the memory bound arithmetic rather than aspirational.
:::

```text
   PRODUCER  (mapper: FlightHandler::do_exchange)        CONSUMER  (reducer)
   ----------------------------------------------        -------------------

                     credits: tokio::sync::Semaphore
                                   |                       seeds the window in the
                                   |  <---- seed(n) ------ first DoExchange message
   for batch in bucket:            |                       (a little-endian u32 in
       credits.acquire().await ----+                        app_metadata)
                                   |
              -- blocks here at 0 -+
                                   |
       gauge.on_send()             |
       yield batch  ---------------+----------------------> receive; pending += 1
                                   |
                                   |                        if pending >= credits/2:
                                   |  <---- grant(n) -----      send one grant
                                   |
       (a spawned pump task drains the grants and calls credits.add_permits)
```

The producer blocks in one place, `FlightHandler::do_exchange` in [`crates/bc-transport/src/handler.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-transport/src/handler.rs):

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

The consumer refills at half the window (`credit_exchange_inner` in `exchange.rs`), sending one grant per `credits / 2` batches, so the producer never runs dry while a grant is in flight. The pump clamps every top-up so outstanding permits never exceed the seeded window, which makes the bound the *producer's* invariant: an over-granting consumer can't make a mapper buffer its whole partition. A missing or zero seed falls back to a compiled-in default rather than stalling.

![The credit protocol on one shuffle channel. The consumer opens the exchange, names the ticket and seeds the window at 16 credit slots, which becomes the producer's semaphore. The producer acquires one permit per batch before the Flight encoder ever sees it, so one batch spends one permit and the consumer's pending count rises by one. When the permits are exhausted the producer is blocked at zero and the next batch is never encoded. Once the consumer's pending count reaches 8, half the window, it sends a single grant covering all eight rather than one grant per batch, and the producer resumes. The top-up is clamped to the seeded window, so an over-granting consumer cannot make the producer buffer the whole partition, and batching the grants cuts control traffic without loosening the bound, because a grant is deferred and never anticipated. One credit is one in-flight batch slot, so the window is the channel's memory bound, about 16 MiB at a 1 MiB morsel, and Carbonite's AIMD controller sizes it per channel.](/_static/diagrams/credit_backpressure.svg)

An `InflightGauge` tracks the high-water mark and surfaces as `ShuffleSession.max_inflight`, so a test can assert that no channel ever held more batches than its window.

## The window, and who sets it

Carbonite is the authority. `ResourceManager.grant_credits(requested, signature=..., channels=...)` clamps every request through `credit_ceiling`, which takes the tighter of two ceilings:

- **Count.** `default_credits x credit_ceiling_factor`, 64 by default.
- **Bytes.** The per-channel byte budget divided by the batch size, because a credit is one batch, and 64 batches of 768-dimensional embeddings can be gigabytes. `learned_channel_morsel_bytes` widens the assumed batch from the learned row width, so wide rows get fewer credits automatically.

```python
from batcher.config import Config
from batcher.carbonite.policies import credit_ceiling

cfg = Config()
fc = cfg.flow_control
print(fc.default_credits * fc.credit_ceiling_factor)  # 64
print(credit_ceiling(cfg, 8 << 20) < credit_ceiling(cfg))  # True: wide batches, fewer credits
```

The byte budget is memory-aware. `_channel_byte_budget` caps `credit_byte_budget` at 10% of total RAM (`_SHUFFLE_BUFFER_FRACTION`) divided by the number of channels fetching at once, the caller's `channels` or else `shuffle_fetch_fan_in`. The cap only ever lowers the configured value.

| Knob | Default | Meaning |
|---|---|---|
| `default_credits` | 16 | starting window when there's no learned or measured estimate |
| `credit_ceiling_factor` | 4 | the count ceiling is `default_credits` times this |
| `credit_byte_budget` | 256 MiB | configured byte ceiling per channel, before the RAM cap |
| `shuffle_fetch_fan_in` | 32 | channels assumed to fetch at once when the caller doesn't say |
| `aimd_alpha` | 1 | additive increase, +1 credit per round |
| `aimd_beta` | 0.5 | multiplicative decrease on congestion |
| `backpressure_high` | 0.70 | occupancy above which a channel reads as saturated |
| `backpressure_low` | 0.40 | occupancy below which a channel reads as starved |

The starting window of 16 is the cold-start value. A cross-node fetch's throughput is `window x batch / RTT`; the rationale beside the default in `config/config.py` measured an 18 MiB partition over a 50 ms-RTT link at 2.4 MiB/s with 4 credits against 7.7 MiB/s with 16. Once the process has completed a fetch, the starting window comes from the {ref}`measured bandwidth-delay product <bdp-window>` instead.

## AIMD

`distributed.adaptive_credits`, on by default, runs a TCP-like controller inside Carbonite's envelope. Turned off, the window is whatever `grant_credits()` returned.

```python
import batcher as bt

print(bt.Config().distributed.adaptive_credits)  # True
```

Slow start doubles the window each starved round until the first congestion signal. After that, a congested round cuts it by `aimd_beta`, and a starved round grows it by the larger of `+aimd_alpha` and the CUBIC curve back toward the last congestion point. The window is always clamped to `[1, credit_ceiling]`.

`carbonite.policies.congestion` fuses two independent measurements into one verdict per round: the process-wide `PressureMonitor` (is the node in trouble?) and the transport's measured *occupancy*, the share of the round the consumer had data ready (is the window covering the bandwidth-delay product?). Memory outranks occupancy unconditionally.

| Verdict | What was observed | What the window does |
|---|---|---|
| `CONGESTED` | The node is past its spill threshold. | Cut by `aimd_beta`, and leave slow start. |
| `SATURATED` | The consumer never waited on the wire. | Hold. More credits buy buffering, not throughput. |
| `STARVED` | The consumer waited, and memory is fine. | Grow: double in slow start, else follow the CUBIC curve. |

`backpressure_high` and `backpressure_low` form a Schmitt trigger around occupancy, so a noisy ratio doesn't flip the window on alternate rounds. `ShuffleSession.stats()` reports `credit_hold_rate`: a window at its ceiling reading `SATURATED` found its bandwidth-delay product.

:::{dropdown} The control law
```python
# docs: skip
# python/batcher/carbonite/policies/flow_control.py, AIMDFlowControl
def observe_signal(self, signal: CongestionSignal) -> int:
    self._rounds += 1
    if signal is CongestionSignal.SATURATED:
        self._holds += 1
        return self.window
    if signal is CongestionSignal.CONGESTED:
        self._window = max(self._floor, self._window * self._beta)
        self._slow_start = False
    elif self._slow_start:
        self._window = min(self._ceiling, self._window * _SLOW_START_FACTOR)
    else:
        self._window = min(self._ceiling, self._grown_window())
    return self.window
```

A held round doesn't advance the CUBIC recovery clock, because a converged channel isn't recovering. An unmeasured channel reports `STARVED`, the permissive verdict. `carbonite.policies.congestion.StarvationMeter` differences the transport's running totals per round, so the controller reacts to the current round rather than a lifetime average.
:::

(bdp-window)=
## Measuring the window instead of probing for it

TCP has to search for its window because a sender knows almost nothing about the path. A shuffle's peers are the same query's workers, and the path is measured on every fetch. So the transport keeps two filters per peer, BBR's construction:

| Estimate | Filter | Why that filter |
|---|---|---|
| `RTprop`, the propagation delay | running **minimum** of observed round trips | every error in a round-trip sample is non-negative, so the truth is the smallest sample |
| `BtlBw`, the bottleneck bandwidth | running **maximum** of observed delivery rates | a fetch can finish slower than the bottleneck allows but never faster |

Their product is the bandwidth-delay product, and `carbonite.policies.bdp` converts it into credits. The window is `2 x BDP`, and the factor is forced by the half-window refill: up to `w / 2` batches sit taken but unacknowledged on top of those in flight.

```text
w  >=  L x R  +  w / 2   =>   w / 2  >=  BDP_batches   =>   w  >=  2 x BDP_batches
```

Only the *starting* window moves; the control law and the ceiling are unchanged, so a result can't move. Precedence is a learned window for this shuffle's signature, then the measured product, then the configured default.

## Warm-starting a recurring channel

The converged window is persisted per shuffle signature under the `carbonite.shuffle_window` namespace, exponentially smoothed, and `grant_credits(signature=...)` starts the next run from it. AIMD still governs the window from live pressure, and since a credit window bounds in-flight batches and nothing else, learning can't change a result.

:::{dropdown} Stability and bounded influence
Slow start is skipped only for a window past runs agreed on. `metadata.smoothed` tracks an exponentially weighted variance alongside the mean, and `shuffle_window_is_stable` asks whether the coefficient of variation is inside the band; an unstable window still supplies the starting point but keeps its ramp.

The same machinery bounds every learned scalar in the engine. The deviation is clamped into `+/-3 sigma` before blending, so one throttled GPU or one swapping node can't drag the estimate: a settled estimate of 100 moves to 101 when handed 100,000, where plain smoothing would move it to 10,090. The variance sees the *unclamped* deviation, so a workload that genuinely moved to a new regime widens its own band and is tracked within a couple of runs.
:::

## Practical limits

- **Per-batch cost.** One semaphore acquire per batch and one grant per half window, negligible at 1 MiB batches, which is one reason a morsel targets 16,384 rows or 1 MiB.
- **Striping.** Each shard of a striped bucket gets its own full window. One reducer runs `flow_control.gather_streams` streams (48) across its peers, so it holds about `max(gather_streams, peers) x credits` batches in flight, and `_channel_byte_budget` divides the 10%-of-RAM transit share by that stream count. `distributed.flight_connections_per_peer` (4) caps TCP connections, over which shards multiplex as HTTP/2 streams.
- **Transport memory only.** Credits bound socket-side buffering, not a reducer's hash table, which is the {doc}`buffer pool's </architecture/deep-dives/memory/buffer-pool>` job. Published buckets are bounded by the shuffle store's cap.
- **Skew.** The gather divides its byte budget evenly across channels. The optimal split is proportional to each channel's bytes, so a skewed shuffle pays the skew factor `s_max / mean(s)` in completion time.

## See also

- {doc}`The shuffle over Arrow Flight </architecture/deep-dives/distribution/shuffle-flight>`: what the credits are gating.
- {doc}`Distributed scheduling </architecture/deep-dives/distribution/distributed-scheduling>`: how many channels there are.
- {doc}`The buffer pool </architecture/deep-dives/memory/buffer-pool>`: where `PressureLevel` comes from.
- {doc}`Carbonite </architecture/internals/carbonite>`: the flow-control knob reference.
- {doc}`Configuration options </configuration/options>`: `flow_control.*` and `distributed.adaptive_credits`.
- {doc}`Streaming </user-guide/moving-data/streaming/index>`: the other place a fast producer meets a slow consumer.
- {doc}`Scaling benchmarks </benchmarks/results/scaling>`: what the credited shuffle sustains as nodes are added.
- `docs/architecture/internals/mathematical_foundations.md`, an in-repo v1-era paper rather than a site page, with an errata list at its top. It covers the boundedness argument for AIMD under a clamp, which replaces the stability claim it once made.
