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

It is the cold-start value, and only that. Once this process has completed a fetch, the
starting window comes from the measured bandwidth-delay product instead — see
{ref}`measuring the window instead of probing for it <bdp-window>`.

## AIMD

A static window is a compromise. Too small and you leave bandwidth on the floor. Too large
and you buffer memory you didn't need.

`distributed.adaptive_credits` selects between the two, and it defaults to `True`.

Set it to `False` and the window is whatever `grant_credits()` returned, clamped to
`credit_ceiling`. There's no slow start and no reaction to pressure. That's simple, and
wrong in one direction or the other on most links.

Leave it at `True` and a TCP-like controller runs. Slow start doubles the window each
starved round until the first congestion signal. After that it's multiplicative decrease
(`×aimd_beta`) on congestion, and on a starved round the larger of additive increase
(`+aimd_alpha`) and the CUBIC curve back toward the window congestion was last found at. A
round that is neither holds the window where it is. The next section covers how a round is
classified.

The window is always clamped to `[1, credit_ceiling]`, so AIMD moves the window *inside* the
envelope Carbonite set and never moves the envelope itself.

## The two facts behind a window

Deciding a window needs two independent measurements, and it is worth separating them
because they answer different questions and have different remedies.

*Is the node in trouble?* The process-wide `PressureMonitor` answers this. It escalates
instantly and de-escalates only as an EWMA relaxes, and that asymmetric hysteresis is what
keeps a shrinking window from oscillating. The signal is a property of the *machine*, so it
cuts every channel on the node, including the ones that were behaving. On a healthy node it
never fires at all.

*Is the window doing anything?* A credit window's only job is to cover the channel's
bandwidth-delay product: enough batches in flight that the consumer never waits on the wire,
and not one more, because every credit past that point is buffered memory bought for no
throughput. The transport measures this directly. `credit_exchange_inner` times how long the
consumer sits blocked awaiting the next batch and folds it into the per-peer registry
alongside the bytes and the elapsed time, so `bytes / seconds` still reads as a per-stream
rate while `starved / seconds` reads as how much of the fetch was spent waiting.

Occupancy is the complement of that ratio, and it is what
`flow_control.backpressure_high` (0.70) and `backpressure_low` (0.40) bound. A channel above
the high threshold had data available almost the whole time, so its window already covers the
BDP. One below the low threshold spent most of the round waiting, so a wider window would
fill the link. Between them the previous verdict stands: the band is a Schmitt trigger, and
a single threshold on a noisy ratio produces a window that grows and cuts on alternate
rounds.

`carbonite.policies.congestion` fuses the two into one of three verdicts, with memory
outranking occupancy unconditionally — a slow shuffle is recoverable and an OOM-killed worker
is not.

| Verdict | What was observed | What the window does |
|---|---|---|
| `CONGESTED` | The node is past its spill threshold. | Cut by `aimd_beta`, leave slow start. |
| `SATURATED` | The consumer never waited on the wire. | Hold. More credits buy buffering, not throughput. |
| `STARVED` | The consumer waited, and memory is fine. | Grow: double in slow start, else follow the CUBIC curve. |

An unmeasured channel reports `STARVED`, which is deliberately the permissive verdict: a
window that has never been tested has not earned the right to stop growing.

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

A held round deliberately does not advance the CUBIC recovery clock. That clock counts rounds
spent *recovering* toward a known-good window, and a converged channel is not recovering. If
held rounds advanced it, a channel that sat saturated for a minute and then starved once would
evaluate `w_max + C(t - K)³` far out on the curve and leap straight to its ceiling.

:::{note}
The occupancy measurement is what makes this a control loop rather than a ramp. With memory
as the only signal, every round on a healthy node read as "grow", so a channel climbed to its
ceiling whether or not the extra credits moved a byte — reserving up to a full
`credit_byte_budget` per channel of transit buffering, and manufacturing the very pressure it
would then back off on. `ShuffleSession.stats()` reports `credit_hold_rate` for exactly this:
a window pinned at its ceiling reading `STARVED` is being throttled by the ceiling, and the
same window reading `SATURATED` found its bandwidth-delay product.
:::

(bdp-window)=
## Measuring the window instead of probing for it

Everything above is a *search*. Slow start doubles until something breaks, CUBIC recovers
toward where it broke last time, and the memory signal decides when to back off. That is the
right design for TCP, and it is the right design because a TCP sender knows almost nothing: not
the receiver's capacity, not the path's, not how much data it is going to send, not who else is
on the wire. Every constant in the law is a fairness heuristic for anonymous flows on the
public internet.

A shuffle has none of those unknowns. The peers are the same query's workers. The objective is
makespan, not fairness among strangers. The receiver's memory envelope is Carbonite's own
number. And the path is measured on every fetch the engine already performs.

So the transport keeps two filters per peer, which is BBR's construction:

| Estimate | Filter | Why that filter |
|---|---|---|
| `RTprop`, the propagation delay | running **minimum** of observed round trips | every error in a round-trip sample is non-negative — queueing, a busy worker, a lost scheduler slice — so the truth is the smallest sample, and an average is biased upward by all of them |
| `BtlBw`, the bottleneck bandwidth | running **maximum** of observed delivery rates | a fetch can finish slower than the bottleneck allows but never faster, so the ceiling is the largest rate seen; averaging would be dragged down by every application-limited fetch, which on a wide shuffle is most of them |

Their product is the bandwidth-delay product: the bytes a path holds when it is exactly busy.
Below it the producer idles waiting for permission. Above it the surplus does not move a byte
sooner, it sits in a buffer. That is precisely what a credit window is for, and
`carbonite.policies.bdp` converts it into credits.

### The factor of two is forced, not tuned

The window is `2 x BDP`, and the two is arithmetic rather than a constant someone picked.

The consumer does not return a credit per batch. It accumulates and sends one grant once half
the window has drained, which cuts control-message traffic by that factor. So on top of the
`L x R` batches genuinely in flight sit up to `w / 2` more that the consumer has taken and not
yet acknowledged. The producer stalls unless its window covers both:

```text
w  >=  L x R  +  w / 2   =>   w / 2  >=  BDP_batches   =>   w  >=  2 x BDP_batches
```

A window set to exactly the product would idle the producer for about half of every round trip.
BBR arrives at the same factor of two from the analogous cause, delayed and aggregated
acknowledgements.

### What it changes

Only the *starting* window. The control law and the ceiling are unchanged, so a result cannot
move — but the number of round trips spent finding the operating point can. Doubling from 16
credits to 64 costs two round trips, and on a 100 ms link that is 200 ms during which the
transfer runs below the window it needs. A short bucket finishes inside that window and spends
its entire life under-provisioned.

Precedence is history, then measurement, then the configured default. A learned window for this
shuffle's own signature beats a product measured over this process's paths, which this shuffle
may not be typical of; both beat a constant. A process that has completed no fetch has no
estimate, and the probing ramp runs exactly as it did before.

## Splitting one budget across skewed channels

A reducer fetches several buckets at once out of one byte budget, and `_channel_byte_budget`
divides it by the number of channels. Even division is the correct answer when nothing is known
about the buckets. On skewed data it is badly wrong, and Batcher is not in the dark: the
sketches estimate each bucket before the shuffle and the mappers measure them while publishing.

Channel `i` carrying `s_i` bytes with a window of `w_i` batches of `b` bytes over a path of
round-trip time `R` is window-limited to `w_i b / R` bytes per second, so it finishes at
`t_i = s_i R / (w_i b)`. The reducer is done when its slowest channel is, so the objective is
`T = max(t_i)` subject to a fixed `sum(w_i) = W`.

At an optimum every `t_i` is equal. If some channel finished early, moving a credit from it to
the slowest one strictly lowers the maximum, so an unequal allocation is never optimal. Setting
them equal and summing gives

```text
w_i  =  W x s_i / sum(s)          T*  =  R x sum(s) / (W b)
```

Even division instead yields `T = R k s_max / (W b)`, worse by a factor of

```text
k x s_max / sum(s)  =  s_max / mean(s)
```

which is the skew factor exactly. A shuffle with one bucket ten times the average takes ten
times longer than it needs to, and every credit of the difference was already paid for.
`proportional_windows` is that allocation, with a floor of one credit per channel — a zero
window is not a small share, it is a channel that never completes.

### Why the totals are differenced

The transport exposes running totals rather than a ratio, and
`carbonite.policies.congestion.StarvationMeter` differences them against the previous reading.

That is not incidental. A controller acts once per round, so what it needs is the ratio over
*that round*. A few seconds into a long shuffle the lifetime denominator is large enough that
a round of pure starvation barely moves it, and a controller reading the lifetime figure
converges on a number and then stops responding to the link entirely. A round that moved too
little data to divide — one served entirely from locality, say — reports no opinion and leaves
the hysteresis band holding the last real verdict.

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

### Only as far as the runs agree

Skipping slow start is the aggressive part, and it is worth only doing when the learned value
has earned it. A shuffle whose window has scattered across an order of magnitude hasn't learned
a window — it has averaged a bimodal population, and starting there *and* switching off the
search that would find the answer independently is worse than never having learned.

So the learned scalar carries its own dispersion. `metadata.smoothed` tracks an
exponentially-weighted variance alongside the mean, on the same decay, and
`ScalarEstimate.stable` asks whether there are enough observations and whether their
coefficient of variation is inside the band. `load_shuffle_window` returns both, and slow start
is skipped only for a window past runs actually agreed on. An unstable one still supplies the
starting point — it is the best guess available — but keeps its ramp.

That dispersion buys a second thing, for every learned scalar in the engine rather than just
this one. Plain exponential smoothing moves an estimate by `step × (value - prior)`, which is
unbounded in the observation, so one GPU that thermally throttled or one shuffle measured while
the node was swapping drags the learned value by however wrong it was. Clamping the deviation
into `±3σ` before blending bounds any single run's influence — the same bounded-influence
property `ml.HuberRegressor` provides against outliers in a fit. A settled estimate of 100 moves
to 101 when handed 100,000, where before it moved to 10,090.

The variance is fed the **unclamped** deviation, and that asymmetry is load-bearing: a workload
that genuinely moved to a new regime must be able to widen its own acceptance band and follow.
One real excursion widens it far past the floor, so the second observation of a new regime is
barely clamped and the estimate tracks it within a couple of runs. Protect the mean; let the
variance see what really happened.

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

- {doc}`Architecture </architecture/index>`: Carbonite as the authority over every backpressure knob.
- {doc}`Carbonite </architecture/internals/carbonite>`: the flow-control knob reference.
- `docs/architecture/internals/mathematical_foundations.md` (in the repo, not a site page): the stability argument for AIMD under a clamp.
- {doc}`Configuration options </configuration/options>`: `flow_control.*` and `distributed.adaptive_credits`.
- {doc}`Streaming </user-guide/moving-data/streaming>`: the other place a fast producer meets a slow consumer.
- {doc}`Scaling benchmarks </benchmarks/results/scaling>`: what the credited shuffle sustains as nodes are added.
- {doc}`The shuffle over Arrow Flight </architecture/deep-dives/distribution/shuffle-flight>`: what the credits are gating.
- {doc}`The buffer pool </architecture/deep-dives/memory/buffer-pool>`: where `PressureLevel` comes from.
- {doc}`Distributed scheduling </architecture/deep-dives/distribution/distributed-scheduling>`: how many channels there are.
