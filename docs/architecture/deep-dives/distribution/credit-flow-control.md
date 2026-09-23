# Credit-based flow control

*Credit-based flow control* bounds how much data a producer can put in flight toward a consumer: the producer may only send what the consumer has said it has room for. This page describes the credit protocol on the shuffle wire, how credits are replenished, how Carbonite sizes the window, and how the controller moves that window at run time.

A Parquet scanner produces batches at 10,000 a second. A GPU embedding model consumes them at 100 a second. Connect them with no intervention and 99% of the scanned data sits in RAM waiting its turn, until the worker's memory is gone. That isn't a pathological case. It's what every pipeline with a slow stage does by default.

The shuffle has the same shape. A mapper can serialize its bucket far faster than a reducer can fold it, and a reducer fetching from sixteen mappers at once is sixteen producers against one consumer. The fix is the one TCP uses.

## One credit is one batch slot

:::{important}
A channel's in-flight memory is at most `credits x batch_bytes`. Not usually bounded, and not bounded under normal load. Bounded, because the producer can't send batch `n+1` until a permit for it exists. That's what makes the memory bound arithmetic rather than aspirational.
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
                                   |
                                   v
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

`credits` is a `tokio::sync::Semaphore`. The consumer seeds it in the first `DoExchange` message as a little-endian `u32` in `app_metadata`. A decoded zero, from a missing or malformed seed, falls back to a compiled-in default rather than stalling forever.

The bound is the *producer's* invariant, not only the consumer's arithmetic. The pump clamps every top-up so outstanding permits never exceed the seeded window. Without that clamp a consumer that over-granted, through a bug or by choice, would make a healthy mapper encode and hold its whole partition.

## Replenishment

Granting one credit per batch would double the message count for no benefit. The consumer does a batched low-watermark refill in `credit_exchange_inner` in `exchange.rs`:

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

The server's pump task drains inbound grant messages and calls `credits.add_permits(granted)`. Half the window is the refill point, so the producer never runs dry while a grant is in flight.

Seed, spend, block and refill form one loop per channel, and the blocked state in the middle is the one worth seeing drawn.

![The credit protocol on one shuffle channel. The consumer opens the exchange, names the ticket and seeds the window at 16 credit slots, which becomes the producer's semaphore. The producer acquires one permit per batch before the Flight encoder ever sees it, so one batch spends one permit and the consumer's pending count rises by one. When the permits are exhausted the producer is blocked at zero and the next batch is never encoded. Once the consumer's pending count reaches 8, half the window, it sends a single grant covering all eight rather than one grant per batch, and the producer resumes. The top-up is clamped to the seeded window, so an over-granting consumer cannot make the producer buffer the whole partition, and batching the grants cuts control traffic without loosening the bound, because a grant is deferred and never anticipated. One credit is one in-flight batch slot, so the window is the channel's memory bound, about 16 MiB at a 1 MiB morsel, and Carbonite's AIMD controller sizes it per channel.](/_static/diagrams/credit_backpressure.svg)

An `InflightGauge` runs alongside, tracking `current` and a high-water `max`. It's how the credit bound gets *tested*: it surfaces as `ShuffleSession.max_inflight`, so a test can assert that no channel ever held more batches than its window allowed.

## The window, and who sets it

Carbonite is the authority. `ResourceManager.grant_credits(requested, signature=..., channels=...)` is the single entry point, and it clamps every request through the flow-control policy:

```python
# docs: skip
# python/batcher/carbonite/policies/flow_control.py
def credit_ceiling(config, effective_morsel_bytes=None, *, channels=None) -> int:
    fc = config.flow_control
    count_ceiling = fc.default_credits * fc.credit_ceiling_factor
    morsel_bytes = max(1, effective_morsel_bytes or config.execution.morsel_bytes)
    byte_ceiling = max(1, _channel_byte_budget(config, channels) // morsel_bytes)
    return max(1, min(count_ceiling, byte_ceiling))
```

There are two ceilings, and the tighter wins. The count ceiling is the obvious one, `default_credits x credit_ceiling_factor`. The byte ceiling exists because a credit is *one batch*, and a batch of 768-dimensional embeddings isn't the same object as a batch of int64 keys. Sixty-four credits of wide rows can be gigabytes. `learned_channel_morsel_bytes` widens the assumed batch size from the learned row width, so a wide-row workload gets *fewer* credits automatically.

The per-channel byte budget is itself memory-aware. `_channel_byte_budget` caps the configured `credit_byte_budget` at `_SHUFFLE_BUFFER_FRACTION`, 10% of the machine's total RAM, divided by the number of channels fetching at once. That's the `channels` the caller passes when it knows its width, and `shuffle_fetch_fan_in` when it doesn't. Dividing by the real width matters: a reducer with three upstreams handed a budget sized for eight channels could buffer nearly three times its intended share.

The RAM cap matters because 256 MiB per channel across 32 concurrent fetches is 8 GiB in flight. That's unremarkable on a 512 GiB node and more than half the RAM of a 16 GiB one. The cap only ever lowers the configured value, so tuning `credit_byte_budget` down keeps your number.

The following table lists the defaults under `config.flow_control`:

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

On a machine with enough RAM the count ceiling of 64 binds. Raise `morsel_bytes`, run on a smaller node, or let the learned row width widen the assumed batch for an embedding column, and the byte ceiling drops below the count ceiling and starts binding instead. That crossover is the reason to have both.

The starting window of 16 matters more than it sounds. A cross-node fetch's throughput is `window x batch / RTT`, so a small window throttles the opening rounds before any adaptation can help. The rationale recorded beside the default in `config/config.py` measured a single 18 MiB partition over a 50 ms-RTT link at 2.4 MiB/s with 4 credits, against 7.7 MiB/s with 16.

It's the cold-start value and only that. Once this process has completed a fetch, the starting window comes from the measured bandwidth-delay product instead, as {ref}`measuring the window instead of probing for it <bdp-window>` describes.

## AIMD

A static window is a compromise. Too small leaves bandwidth on the floor, and too large buffers memory nobody needed.

`distributed.adaptive_credits` selects between the two, and it defaults to `True`.

Set it to `False` and the window is whatever `grant_credits()` returned, clamped to `credit_ceiling`. There's no slow start and no reaction to pressure. That's simple, and wrong in one direction or the other on most links.

Leave it at `True` and a TCP-like controller runs. Slow start doubles the window each starved round until the first congestion signal. After that, a congested round cuts the window by `aimd_beta`, and a starved round grows it by the larger of `+aimd_alpha` and the CUBIC curve back toward the window where congestion was last found. A round that is neither holds the window where it is.

The window is always clamped to `[1, credit_ceiling]`, so AIMD moves the window *inside* the envelope Carbonite set and never moves the envelope itself.

## The two facts behind a window

Deciding a window needs two independent measurements. They answer different questions and have different remedies.

*Is the node in trouble?* The process-wide `PressureMonitor` answers this. It escalates instantly and de-escalates only as an EWMA relaxes, and that asymmetric hysteresis keeps a shrinking window from oscillating. The signal is a property of the *machine*, so it cuts every channel on the node, including the ones that were behaving. On a healthy node it never fires.

*Is the window doing anything?* A credit window's only job is to cover the channel's bandwidth-delay product: enough batches in flight that the consumer never waits on the wire, and not one more, because every credit past that point is buffered memory bought for no throughput. The transport measures this directly. `credit_exchange_inner` times how long the consumer sits blocked awaiting the next batch and folds it into the per-peer registry alongside the bytes and the elapsed time. So `bytes / seconds` reads as a per-stream rate, and `starved / seconds` reads as how much of the fetch was spent waiting.

Occupancy is the complement of that ratio, and `backpressure_high` and `backpressure_low` bound it. A channel above the high threshold had data available almost the whole time, so its window already covers the BDP. One below the low threshold spent most of the round waiting, so a wider window would fill the link. Between them the previous verdict stands. The band is a Schmitt trigger, because a single threshold on a noisy ratio produces a window that grows and cuts on alternate rounds.

`carbonite.policies.congestion` fuses the two into one of three verdicts. Memory outranks occupancy unconditionally, because a slow shuffle is recoverable and an OOM-killed worker is not.

| Verdict | What was observed | What the window does |
|---|---|---|
| `CONGESTED` | The node is past its spill threshold. | Cut by `aimd_beta`, and leave slow start. |
| `SATURATED` | The consumer never waited on the wire. | Hold. More credits buy buffering, not throughput. |
| `STARVED` | The consumer waited, and memory is fine. | Grow: double in slow start, else follow the CUBIC curve. |

An unmeasured channel reports `STARVED`. That's deliberately the permissive verdict: a window that has never been tested hasn't earned the right to stop growing.

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

A held round doesn't advance the CUBIC recovery clock. That clock counts rounds spent *recovering* toward a known-good window, and a converged channel isn't recovering. If held rounds advanced it, a channel that sat saturated for a minute and then starved once would evaluate `w_max + C(t - K)^3` far out on the curve and leap straight to its ceiling.

:::{note}
The occupancy measurement is what makes this a control loop rather than a ramp. With memory as the only signal, every round on a healthy node read as "grow", so a channel climbed to its ceiling whether or not the extra credits moved a byte. That reserved up to a full `credit_byte_budget` per channel of transit buffering, and manufactured the pressure it would then back off on. `ShuffleSession.stats()` reports `credit_hold_rate` for this reason: a window pinned at its ceiling reading `STARVED` is being throttled by the ceiling, and the same window reading `SATURATED` found its bandwidth-delay product.
:::

### Why the totals are differenced

The transport exposes running totals rather than a ratio, and `carbonite.policies.congestion.StarvationMeter` differences them against the previous reading.

A controller acts once per round, so it needs the ratio over *that round*. A few seconds into a long shuffle the lifetime denominator is large enough that a round of pure starvation barely moves it. A controller reading the lifetime figure converges on a number and then stops responding to the link. A round that moved too little data to divide, such as one served entirely from locality, reports no opinion and leaves the hysteresis band holding the last real verdict.

(bdp-window)=
## Measuring the window instead of probing for it

Everything above is a *search*. Slow start doubles until something breaks, CUBIC recovers toward where it broke last time, and the memory signal decides when to back off. That's the right design for TCP, because a TCP sender knows almost nothing: not the receiver's capacity, not the path's, not how much it will send, and not who else is on the wire. Every constant in the law is a fairness heuristic for anonymous flows on the public internet.

A shuffle has none of those unknowns. The peers are the same query's workers. The objective is makespan, not fairness among strangers. The receiver's memory envelope is Carbonite's own number, and the path is measured on every fetch the engine already performs.

So the transport keeps two filters per peer, which is BBR's construction:

| Estimate | Filter | Why that filter |
|---|---|---|
| `RTprop`, the propagation delay | running **minimum** of observed round trips | every error in a round-trip sample is non-negative, whether from queueing, a busy worker or a lost scheduler slice, so the truth is the smallest sample and an average is biased upward |
| `BtlBw`, the bottleneck bandwidth | running **maximum** of observed delivery rates | a fetch can finish slower than the bottleneck allows but never faster, so the ceiling is the largest rate seen; an average is dragged down by every application-limited fetch, which on a wide shuffle is most of them |

Their product is the bandwidth-delay product: the bytes a path holds when it's exactly busy. Below it the producer idles waiting for permission. Above it the surplus doesn't move a byte sooner, it sits in a buffer. That's what a credit window is for, and `carbonite.policies.bdp` converts the product into credits.

### The factor of two is forced, not tuned

The window is `2 x BDP`, and the two is arithmetic rather than a constant someone picked.

The consumer doesn't return a credit per batch. It sends one grant once half the window has drained. So on top of the `L x R` batches genuinely in flight sit up to `w / 2` more that the consumer has taken and not yet acknowledged. The producer stalls unless its window covers both:

```text
w  >=  L x R  +  w / 2   =>   w / 2  >=  BDP_batches   =>   w  >=  2 x BDP_batches
```

A window set to exactly the product would idle the producer for about half of every round trip. BBR arrives at the same factor of two from the analogous cause, delayed and aggregated acknowledgements.

### What it changes

Only the *starting* window moves. The control law and the ceiling are unchanged, so a result can't move, but the number of round trips spent finding the operating point can. Doubling from 16 credits to 64 costs two round trips, and on a 100 ms link that's 200 ms during which the transfer runs below the window it needs. A short bucket finishes inside that window and spends its entire life under-provisioned.

Precedence is history, then measurement, then the configured default. A learned window for this shuffle's own signature beats a product measured over this process's paths, which this shuffle may not be typical of, and both beat a constant. A process that has completed no fetch has no estimate, and the probing ramp runs as before.

## Warm-starting a recurring channel

Slow start costs a few rounds every time, and a shuffle of the same shape converges to the same window every time. So the converged window is persisted per shuffle signature under the `carbonite.shuffle_window` namespace, exponentially smoothed, and `grant_credits(signature=...)` starts the next run from it.

Learning only moves the *starting point*. AIMD still governs the window actually used from live pressure, and the ceiling still clamps it. A credit window bounds in-flight batches and nothing else, so none of this can change a result, which is why it's safe to learn aggressively.

### Only as far as the runs agree

Skipping slow start is the aggressive part, and it's only worth doing when the learned value has earned it. A shuffle whose window has scattered across an order of magnitude hasn't learned a window. It has averaged a bimodal population, and starting there *and* switching off the search that would find the answer is worse than never having learned.

So the learned scalar carries its own dispersion. `metadata.smoothed` tracks an exponentially weighted variance alongside the mean, on the same decay, and `ScalarEstimate.stable` asks whether there are enough observations and whether their coefficient of variation is inside the band. The two questions have separate callers, so they have separate functions: `load_shuffle_window` returns the number and `shuffle_window_is_stable` answers whether to trust it. Slow start is skipped only for a window past runs agreed on. An unstable one still supplies the starting point, the best guess available, but keeps its ramp.

That dispersion buys a second thing, for every learned scalar in the engine. Plain exponential smoothing moves an estimate by `step x (value - prior)`, which is unbounded in the observation, so one GPU that thermally throttled or one shuffle measured while the node was swapping drags the learned value by however wrong it was. Clamping the deviation into `+/-3 sigma` before blending bounds any single run's influence, the same bounded-influence property `ml.HuberRegressor` provides against outliers in a fit. A settled estimate of 100 moves to 101 when handed 100,000, where plain smoothing would move it to 10,090.

The variance is fed the **unclamped** deviation, and that asymmetry is load-bearing. A workload that genuinely moved to a new regime must be able to widen its own acceptance band and follow. One real excursion widens it far past the floor, so the second observation of a new regime is barely clamped and the estimate tracks it within a couple of runs. Protect the mean, and let the variance see what really happened.

## Requirements and limitations

Credits cost a semaphore acquire per batch on the producer and one grant message per half window on the consumer. At 1 MiB batches that's negligible. At very small batches the per-batch permit is a real fraction of the work, which is one more reason a morsel targets 16,384 rows or 1 MiB rather than 100 rows.

:::{warning}
Striping is the sharp edge. `ClientPool::fetch_secured_striped` gives **each shard its own full window**, so a peer fetched over 4 connections can have `4 x credits` batches in flight. Each shard is an independent TCP flow, and on a high bandwidth-delay-product link splitting one window across the flows starves each to about one credit. The byte bound is per channel, not per peer, and `distributed.flight_connections_per_peer`, which defaults to 4, bounds the multiplier.
:::

The bound covers *transport* memory only. Credits stop a mapper from flooding a reducer's socket buffers. They don't stop the reducer's own hash table from growing. That's the buffer pool's job, and the two meet in the pressure signal AIMD reads. Published buckets waiting to be fetched are bounded separately, by the shuffle store's cap described in {doc}`the shuffle over Arrow Flight </architecture/deep-dives/distribution/shuffle-flight>`.

Even division of the byte budget across a reducer's channels is right when nothing is known about the buckets, and wrong on skewed data. Channel `i` carrying `s_i` bytes with a window of `w_i` batches of `b` bytes over round-trip time `R` finishes at `t_i = s_i R / (w_i b)`, and the reducer is done when its slowest channel is. Minimizing that maximum under a fixed total `W` sets every `t_i` equal, which gives `w_i = W x s_i / sum(s)`. Even division is worse by `s_max / mean(s)`, the skew factor exactly. `carbonite.policies.proportional_windows` computes that allocation, with a floor of one credit per channel, and is unit-tested. The gather doesn't call it yet, so a skewed shuffle still divides its budget evenly.

## See also

- {doc}`The shuffle over Arrow Flight </architecture/deep-dives/distribution/shuffle-flight>`: what the credits are gating.
- {doc}`Distributed scheduling </architecture/deep-dives/distribution/distributed-scheduling>`: how many channels there are.
- {doc}`The buffer pool </architecture/deep-dives/memory/buffer-pool>`: where `PressureLevel` comes from.
- {doc}`Carbonite </architecture/internals/carbonite>`: the flow-control knob reference.
- {doc}`Configuration options </configuration/options>`: `flow_control.*` and `distributed.adaptive_credits`.
- {doc}`Streaming </user-guide/moving-data/streaming/index>`: the other place a fast producer meets a slow consumer.
- {doc}`Scaling benchmarks </benchmarks/results/scaling>`: what the credited shuffle sustains as nodes are added.
- `docs/architecture/internals/mathematical_foundations.md`, an in-repo paper rather than a site page: the stability argument for AIMD under a clamp.
