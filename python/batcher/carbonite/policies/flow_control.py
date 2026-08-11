"""Credit-window flow control: how many in-flight batch slots a shuffle channel may hold.

One credit is one in-flight `RecordBatch` slot, so the window *is* the channel's buffered
memory bound. Two policies implement the seam: a static clamp into a memory-safe band, and
an AIMD controller that adapts the window from observed backpressure the way TCP adapts a
congestion window. Both share one ceiling, which bounds the window by count *and* by bytes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.mathx import clamp
from batcher.carbonite.memory.pressure import total_memory_bytes
from batcher.carbonite.policies.congestion import CongestionSignal
from batcher.config import Config, active_config
from batcher.metadata.smoothed import load_scalar_estimate, record_smoothed_scalar

if TYPE_CHECKING:
    from batcher.carbonite.base import ResourceContext
    from batcher.metadata import MetadataHub

__all__ = [
    "AIMDFlowControl",
    "StaticCreditFlowControl",
    "credit_ceiling",
    "learned_channel_morsel_bytes",
    "load_shuffle_window",
    "record_shuffle_window",
    "shuffle_store_cap",
    "shuffle_window_is_stable",
]

# Learned-parameter namespace for the converged AIMD credit window, keyed by a shuffle
# channel's stable signature. One smoothed integer window per signature.
_SHUFFLE_WINDOW_NS = "carbonite.shuffle_window"


def credit_ceiling(
    config: Config, effective_morsel_bytes: int | None = None, *, channels: int | None = None
) -> int:
    """The upper bound on a shuffle channel's credit window (count *and* bytes).

    The count ceiling (`default_credits x credit_ceiling_factor`) is further capped
    so the window's *bytes* (`credits x morsel_bytes`) never exceed
    `credit_byte_budget` — bounding a channel's buffered memory regardless of row
    width. Always >= 1.

    Args:
        config: The active config.
        effective_morsel_bytes: Overrides the config `morsel_bytes` when a channel's real
            per-batch size is known to be wider than the configured target — the
            learned-row-width case (embeddings/blobs), where the assumed `morsel_bytes`
            under-counts the buffered bytes and a fast producer would run the window well
            past `credit_byte_budget`.
        channels: How many channels are actually fetching at once. The whole-shuffle byte
            budget is divided by this rather than by the configured `shuffle_fetch_fan_in`,
            which is a *cap* on concurrency and not a measurement of it. A reducer with
            three upstreams was being handed a budget sized for eight, so its three
            channels could together buffer nearly three times the intended share — while a
            reducer whose fan-out exceeds the configured cap got a share that was too
            generous in the other direction. `None` keeps the configured fan-in.

    Returns:
        The maximum credits this channel may hold, at least 1.
    """
    fc = config.flow_control
    count_ceiling = fc.default_credits * fc.credit_ceiling_factor
    morsel_bytes = max(1, effective_morsel_bytes or config.execution.morsel_bytes)
    byte_ceiling = max(1, _channel_byte_budget(config, channels) // morsel_bytes)
    return max(1, min(count_ceiling, byte_ceiling))


#: Share of the machine's memory a worker's *published* shuffle output may hold resident.
#:
#: The in-flight fraction below bounds bytes on the wire; this bounds the bytes a mapper
#: has already produced and is holding for reducers to collect. They are different pools
#: and the second is the larger one: with `workers` mappers each producing `workers`
#: buckets, a node holds its whole share of the shuffle in anonymous memory that no
#: reservation covers and the kernel cannot reclaim.
#:
#: A quarter rather than the transit fraction's tenth, because this memory is doing the
#: query's actual work rather than moving it, and because exceeding it costs a disk
#: round-trip rather than a stall. Above the cap the store spills its largest buckets to
#: local disk and reads them back on fetch — result-preserving, so the only cost is the
#: re-read.
_SHUFFLE_STORE_FRACTION = 0.25


def shuffle_store_cap(config: Config, envelope_bytes: int | None = None) -> int:
    """Bytes a worker's published shuffle output may hold in memory before it spills.

    Carbonite's buffer pool cannot see this memory: a published bucket is never *reserved*,
    it is simply held until a reducer fetches it. `PressureMonitor` names the store as the
    reason it must fall back to reading process RSS. Giving it an explicit cap is what turns
    the largest un-governed footprint on a shuffle-heavy worker into a bounded one.

    Args:
        config: The active config, for the memory envelope's limits.
        envelope_bytes: The query's sampled memory envelope; `None` reads the machine's.

    Returns:
        The cap in bytes, or `0` when no envelope can be determined — which the engine
        reads as unbounded, preserving the historical behaviour rather than guessing.
    """
    envelope = envelope_bytes if envelope_bytes is not None else total_memory_bytes()
    if envelope <= 0:
        return 0
    # Held under the hard limit as well as the fraction: the shuffle store is not entitled
    # to memory the query as a whole may not use.
    return max(
        config.execution.morsel_bytes,
        int(min(envelope * _SHUFFLE_STORE_FRACTION, envelope * config.memory.hard_limit)),
    )


#: Share of the machine's memory the *whole* shuffle may hold in flight across all its
#: concurrent channels. Deliberately small: this buffer is pure transit, competing with the
#: build tables and aggregate state that are the query's actual working set, and a shuffle
#: that stalls for credit is slow while one that OOMs the node is fatal.
_SHUFFLE_BUFFER_FRACTION = 0.10


def _channel_byte_budget(config: Config, channels: int | None = None) -> int:
    """The per-channel buffered-byte budget: the configured cap, held under a share of the
    memory this machine actually has.

    `credit_byte_budget` is a fixed 256 MiB per channel, and `shuffle_fetch_fan_in` channels
    fetch at once — up to 8 GiB in flight on the defaults. That is unremarkable on a 512 GiB
    node and more than half the RAM of a 16 GiB one, and nothing in the credit path had ever
    read how much memory exists: the admission path is memory-aware (`BudgetingAdmission`),
    the backpressure path was not, so a node could be admitted for a query and then OOM'd by
    the transit buffers carrying it.

    **Caps only.** It never raises the configured budget. The floor that keeps a tiny
    container able to move a batch is therefore `min(configured, morsel_bytes)` and not
    `morsel_bytes`: the latter silently *raised* a deliberately tiny `credit_byte_budget`
    up to a morsel, so an operator who had tuned the transit buffer down below one morsel
    got more buffering than they asked for from the one function whose contract is that it
    only ever gives less.

    Args:
        config: The active config.
        channels: Channels actually fetching at once; `None` uses the configured fan-in cap.

    Returns:
        The per-channel byte budget.
    """
    fc = config.flow_control
    configured = fc.credit_byte_budget
    total = total_memory_bytes()
    if total <= 0:
        return configured
    fan_in = max(1, channels if channels and channels > 0 else fc.shuffle_fetch_fan_in)
    headroom_per_channel = int(total * _SHUFFLE_BUFFER_FRACTION) // fan_in
    floor = min(configured, config.execution.morsel_bytes)
    return max(floor, min(configured, headroom_per_channel))


def learned_channel_morsel_bytes(ctx: ResourceContext) -> int | None:
    """A channel's effective per-batch bytes from the learned row width, or `None`.

    The credit→bytes conversion assumes a `morsel_bytes`-sized batch; a workload whose
    rows proved *wide* anywhere (the learned `max_bytes_per_row`) fills a `morsel_rows`
    batch to far more than that, so its real buffered footprint per credit is larger.
    Returning `max(morsel_bytes, width x morsel_rows)` lets `credit_ceiling` hand out
    fewer credits for wide-row shuffles, keeping buffered memory within budget. `None`
    (cold model / narrow rows) leaves the conversion at the configured `morsel_bytes`.
    """
    model = ctx.memory_model
    if model is None:
        return None
    width = model.max_bytes_per_row()
    if width is None or width <= 0:
        return None
    ex = ctx.config.execution
    return max(ex.morsel_bytes, int(width * max(1, ex.morsel_rows)))


class StaticCreditFlowControl:
    """Credit-window flow control: clamp the requested window to a memory-safe band.

    This is the Carbonite authority that replaces the engine's hardcoded
    `DEFAULT_CREDITS`: one credit = one in-flight `RecordBatch` slot, so the window
    directly bounds a shuffle channel's buffered memory. The window comes from
    `FlowControlConfig`: a non-positive request (operator with no `c_max_credits`
    estimate) gets `default_credits`; a positive request is clamped into
    `[1, credit_ceiling(config)]` (a count *and* byte bound) so neither a stale zero
    stalls the channel nor an over-large estimate (or a wide-row morsel) lets a fast
    producer run unbounded.
    """

    def grant(self, requested: int, ctx: ResourceContext) -> int:
        fc = ctx.config.flow_control
        ceiling = credit_ceiling(
            ctx.config, learned_channel_morsel_bytes(ctx), channels=ctx.shuffle_channels
        )
        if requested <= 0:
            return min(fc.default_credits, ceiling)
        return min(max(requested, 1), ceiling)


# AIMD's multiplicative-decrease factor must lie strictly inside (0, 1): at 1.0 the
# congested branch stops decreasing (the window never backs off), and above 1.0 it grows
# on congestion. The floor keeps a decrease from collapsing the window to the floor in one
# round, which would serialize the shuffle.
_MIN_AIMD_BETA = 0.1
_MAX_AIMD_BETA = 0.95
# CUBIC's window-growth aggressiveness, in credits per round-cubed. The congestion-avoidance
# law is `W(t) = W_max + C·(t - K)³`, where `t` is rounds since the last backoff and `K` is
# where the cubic passes back through `W_max`.
#
# Additive increase (`+alpha` per round, TCP Reno's law) is the wrong shape for a credit
# window that has already found its operating point. After a multiplicative decrease from a
# converged window of, say, 64 credits, `+1` per round needs 32 rounds to climb back to a
# value the channel *already measured as safe* — on a cross-node shuffle each round is a
# network round trip, so the transfer spends most of its life below the bandwidth-delay
# product it had already discovered. The cubic recovers to `W_max` quickly (its slope is
# steepest far below it), flattens as it approaches — spending most of its rounds near the
# known-good window, which is exactly where a controller should sit — and only then probes
# above it, gently at first and faster the longer no congestion appears.
#
# `C = 0.4` is TCP CUBIC's own constant and carries the same meaning here: rounds, not
# seconds, since a credit round *is* the control interval.
_CUBIC_C = 0.4
# Slow-start multiplier: while a channel has never hit congestion, the window *doubles*
# each headroom-to-spare round (TCP slow-start) instead of crawling up `+alpha`/RTT. A
# cross-node shuffle's throughput ceiling is `window x batch / RTT`, so on a high-RTT link
# a window that starts at `default_credits` (4) and grows `+1`/RTT reaches a BDP-filling
# ~64 credits only after ~60 round trips — long after a short shuffle has finished, so it
# runs the whole transfer memory-safe but bandwidth-starved. Doubling reaches the same
# ceiling in ~log2 rounds (4->8->16->32->64 = 4 RTTs), then the first congestion switches
# to additive-increase / multiplicative-decrease (congestion avoidance). Purely the
# window's *ramp*: same ceiling, same backoff, so the memory bound and results are
# unchanged.
_SLOW_START_FACTOR = 2
# How far past a backoff the recovery clock keeps counting. The cubic is monotone in `t`
# beyond `K`, so once `t` is this far out the curve is above any reachable ceiling and the
# window is pinned there regardless — advancing the clock further changes nothing except
# the size of the number being cubed. A long-lived streaming channel observes millions of
# rounds, and `(t - k)**3` on an unbounded `t` is both wasted work and, at the far
# extreme, a float overflow.
_MAX_RECOVERY_ROUNDS = 1 << 20


def _opening_window(
    default_credits: int,
    initial_window: int | None,
    initial_window_stable: bool,
    bdp_window: int | None,
) -> tuple[int, bool]:
    """Where a channel's window starts, and whether that start was actually *informed*.

    Three sources, in increasing order of authority, and the second half of the answer matters
    as much as the first. Slow start is a *search*: it doubles the window until the first
    congestion signal, costing `log2(W)` round trips — two on a 100 ms link between 16 credits
    and 64, by which time a short bucket has finished and spent its whole life below the window
    it needed. Skipping that search is worth it only when something firm already knows the
    answer, because an informed start with no ramp cannot correct itself upward.

    **The configured default** is the floor, and it is a measured one rather than a guess: an
    18 MiB partition over a 50 ms link moves at 2.4 MiB/s on 4 credits against 7.7 on 16.

    **A measured bandwidth-delay product** may only ever raise that floor. This is the one place
    the measurement can do harm taken literally — a loopback path really does have a product of
    a credit or two, correct for that path and a disastrous opening window for the cross-node
    fetch the same process makes next. Held one-sided, the failure stays one-sided with it: a
    window may start wider than a short path needs, which the ceiling bounds and the control law
    trims, but never narrower than the value chosen for the case where narrowness costs most.

    **A learned window** for this shuffle's own signature wins outright, because it is where a
    full control loop of this exact shape converged — it already accounts for the paths and for
    everything the product cannot see: the reducer's fold rate, the memory the query holds
    elsewhere, the skew across buckets. But it only counts as *informed* when past runs agreed
    on it. A window averaged over runs that scattered is still the best guess available and is
    used as the start; it has not earned the right to switch off the search.

    Args:
        default_credits: The configured cold-start window.
        initial_window: A learned window for this shuffle signature, or `None`.
        initial_window_stable: Whether past runs agreed on that learned window.
        bdp_window: The measured bandwidth-delay product in credits, or `None`.

    Returns:
        The starting window before clamping, and whether slow start should be skipped.
    """
    if initial_window is not None:
        return initial_window, initial_window_stable
    if bdp_window is not None and bdp_window > default_credits:
        return bdp_window, True
    return default_credits, False


class AIMDFlowControl:
    """Adaptive credit window via AIMD with TCP-style slow-start.

    The static policy fixes the window; this one *adapts* it from observed
    backpressure, the TCP-style control law the architecture specifies. It starts at
    the config default window and, per round, `observe`s whether the channel was
    congested. While it has never congested it is in **slow-start** — a headroom round
    *doubles* the window (`_SLOW_START_FACTOR`) so it fills the bandwidth-delay product
    in ~log2 rounds rather than crawling up `+alpha`/RTT. The first congestion exits
    slow-start into classic AIMD: a congested round cuts by `aimd_beta` (relieve memory
    pressure fast), a headroom round grows by `aimd_alpha` (pipeline deeper while memory
    is plentiful). The window is always clamped to the same memory-safe band
    `[1, default_credits x credit_ceiling_factor]` the static policy uses, so slow-start
    can never exceed the byte-bounded ceiling.

    Stateful — hold one per adaptive channel. `grant` ignores its `requested`
    argument because the controller, not the caller, owns the evolving window.
    """

    def __init__(
        self,
        config: Config | None = None,
        *,
        initial_window: int | None = None,
        effective_morsel_bytes: int | None = None,
        channels: int | None = None,
        ceiling: int | None = None,
        bdp_window: int | None = None,
        initial_window_stable: bool = True,
    ) -> None:
        cfg = config or active_config()
        fc = cfg.flow_control
        self._alpha = max(1, fc.aimd_alpha)
        # A multiplicative *decrease* requires 0 < beta < 1. A misconfigured `beta >= 1`
        # would make the congested branch *grow* the window — the opposite of AIMD, and
        # an unstable control law (the window would only ever increase, congested or not).
        # Clamped rather than raised: flow control must never fail a query on a tunable.
        self._beta = min(max(fc.aimd_beta, _MIN_AIMD_BETA), _MAX_AIMD_BETA)
        self._floor = 1
        # `effective_morsel_bytes` carries the learned wide-row width (embeddings/blobs) so the
        # ceiling's *byte* bound (`credit_byte_budget`) is honored on this adaptive path exactly
        # as it is on the static grant — otherwise AIMD would grow the window to the un-corrected
        # count ceiling and a fast producer would buffer far past the byte budget.
        # `channels` for the same reason `effective_morsel_bytes` is here: the ceiling must
        # be the *real* one on the adaptive path too. AIMD grows the window toward this
        # ceiling, so a ceiling divided by the configured fan-in rather than the measured
        # channel count is not a cosmetic difference — it is the value the controller
        # converges to, and the memory it buffers there.
        #
        # An explicit `ceiling` wins, for the case a controller runs somewhere that cannot
        # derive the right one. A shuffle worker is exactly that: it is a Ray actor, so it
        # sees neither the driver's `config_context` nor the metadata hub the learned
        # row-width comes from, and every input to `credit_ceiling` is therefore wrong or
        # missing there. The driver knows all of them, so it computes the ceiling once and
        # ships the integer.
        self._ceiling = (
            max(1, ceiling)
            if ceiling is not None and ceiling > 0
            else credit_ceiling(cfg, effective_morsel_bytes, channels=channels)
        )
        # A recurring shuffle warm-starts at the window its past runs converged to
        # (`initial_window`, learned per shuffle signature) instead of re-climbing from
        # `default_credits` every time — the AIMD control law still governs from there,
        # so the window a channel actually uses is unchanged, only its starting point. A
        # warm-started channel skips slow-start: its window already reflects a prior run's
        # congestion, so exponential ramp from there would overshoot the learned value.
        # Precedence: a learned window beats a measured bandwidth-delay product, and both
        # beat the configured default.
        #
        # Not because the learned value is more recent — it is usually older — but because it
        # is a strictly more informed *kind* of number. The BDP is what the network alone
        # allows, and so it is a lower bound on the useful window. The learned value is where
        # a full control loop of this exact shuffle shape actually converged, which already
        # accounts for the paths *and* for everything the BDP cannot see: the reducer's fold
        # rate, the memory the query holds elsewhere, the skew across buckets. Preferring the
        # network-only figure would discard all of that.
        #
        # All three are only a starting point — the control law governs from there — so this
        # decides how many rounds are spent finding the operating point, never what it is.
        start, informed = _opening_window(
            fc.default_credits, initial_window, initial_window_stable, bdp_window
        )
        self._window: float = float(min(max(start, self._floor), self._ceiling))
        self._slow_start = not informed
        # CUBIC state: the window the last backoff started from, and how many rounds ago that
        # was. `None` means no backoff has happened yet, so there is no known-good window to
        # recover toward and growth is slow-start or plain additive.
        self._w_max: float | None = None
        self._rounds_since_backoff = 0
        # `K`, the round at which the cubic returns to `_w_max`. It is a function of
        # `_w_max` and `_beta` alone, both of which change only at a backoff — so it is
        # computed there rather than re-deriving a cube root on every uncongested round of
        # every channel of every shuffle.
        self._k = 0.0
        # Lifetime counters, so a channel can say what its window actually did. A shuffle
        # that backs off on most rounds is a shuffle whose reducer is memory-bound, and
        # that reads as "the query is slow" with nothing pointing at the cause.
        self._backoffs = 0
        self._rounds = 0
        # Rounds the window was held because the channel measured as saturated. The figure
        # that says the controller is *converged* rather than merely uncongested: a window
        # that holds most rounds has found the channel's bandwidth-delay product, and one
        # that never holds is climbing to its ceiling on no evidence.
        self._holds = 0
        self._peak_window = int(self._window)
        # The network's own answer, kept so `stats` can say which constraint is actually
        # binding. See `network_limited`.
        self._bdp_window = bdp_window

    @property
    def window(self) -> int:
        """The current credit window (clamped to the band)."""
        return self._clamp(self._window)

    def grant(self, requested: int, ctx: ResourceContext) -> int:  # noqa: ARG002
        return self.window

    def rewindow(self, credits: int) -> int:
        """Restart the window at a new grant — a reused channel now serving a *different* query.

        A warm shuffle fleet outlives the query that spawned it, and `set_grant` re-grants each
        worker for the query about to borrow it. Under adaptive credits that re-grant had
        nowhere to land: the controller owns the window, so the new grant was dropped and the
        fleet kept the window it had converged to for the *previous* query — the very
        stale-grant regression (`0.6 s -> 3.2 s` on TPC-H sf10) that `set_grant` exists to
        prevent, reintroduced on the default path.

        Leaves slow-start off, exactly as a warm `initial_window` does: the grant already
        reflects a real estimate, so an exponential ramp from it would overshoot.

        Clearing the CUBIC recovery state is what makes the re-grant *hold*. `_w_max` is the
        window congestion was measured at, and `_rounds_since_backoff` is how long recovery
        toward it has been running — both describe the **previous** query's channel. Left in
        place, the first uncongested round after a re-grant evaluates that stale curve at a
        large `t`, and `(t - k)³` returns the old window immediately: a channel re-granted 4
        went to 64 (the ceiling) in one round, where a fresh channel granted 4 goes to 5. The
        re-grant survived exactly one round, which is the stale-grant regression this method
        exists to prevent, one round later. With the state cleared, growth is the plain
        additive law until this query finds its own congestion point.

        Args:
            credits: The new grant (1 credit = 1 in-flight batch), clamped to the band.

        Returns:
            The new current window.
        """
        self._window = float(min(max(credits, self._floor), self._ceiling))
        self._slow_start = False
        self._w_max = None
        self._k = 0.0
        self._rounds_since_backoff = 0
        return self.window

    @property
    def backoffs(self) -> int:
        """How many rounds this channel has backed off on over its life."""
        return self._backoffs

    def stats(self) -> dict[str, int | float]:
        """What this channel's window actually did — the shuffle's backpressure story.

        Returns:
            The live and peak window, the ceiling it was clamped to, how many rounds were
            observed, how many backed off, and the resulting backoff *rate*. A high rate
            means the reducer is memory-bound rather than network-bound, which is the
            distinction an operator otherwise has no way to make.
        """
        return {
            "window": self.window,
            "peak_window": self._peak_window,
            "ceiling": self._ceiling,
            "rounds": self._rounds,
            "backoffs": self._backoffs,
            "backoff_rate": (self._backoffs / self._rounds) if self._rounds else 0.0,
            "holds": self._holds,
            # A window that holds most of its rounds has converged on the channel's
            # bandwidth-delay product. One that never holds is climbing on no evidence, which
            # is what this controller did before it could see the channel.
            "hold_rate": (self._holds / self._rounds) if self._rounds else 0.0,
            "slow_start": int(self._slow_start),
            # Which constraint the window is actually up against. A shuffle capped by memory
            # and one capped by the wire look identical in every other figure here, and they
            # have opposite remedies: the first wants a bigger node or a narrower fan-in, the
            # second wants a faster link or more of them. Nothing else in the engine can tell
            # them apart, because nothing else knows what the path could carry.
            "network_limited": int(self.network_limited),
        }

    @property
    def network_limited(self) -> bool:
        """Whether the wire, rather than the memory ceiling, is what bounds this window.

        `True` when the path's bandwidth-delay product fits inside the ceiling Carbonite set,
        so the window is free to sit where the network wants it. `False` when the ceiling is
        the tighter of the two and the channel is being held below what the link could carry
        — the memory-limited shuffle, which is a node-sizing problem rather than a network one.

        `True` when no product has been measured, which reads as "no evidence of a memory
        constraint" rather than as a diagnosis.
        """
        return self._bdp_window is None or self._bdp_window <= self._ceiling

    def observe(self, *, congested: bool) -> int:
        """Update the window from one round's *memory* signal; return the new window.

        The two-state shorthand for `observe_signal`, kept for a caller that can see whether
        the node is in trouble but not whether the channel is doing anything: `True` cuts,
        `False` grows. Growing on every uncongested round is the permissive reading, and it is
        what this controller did when memory was its only evidence.

        Prefer `observe_signal` where the channel's occupancy is measurable — it is the
        difference between a control loop and a ramp.

        Args:
            congested: Whether this round hit memory backpressure.

        Returns:
            The new credit window.
        """
        return self.observe_signal(
            CongestionSignal.CONGESTED if congested else CongestionSignal.STARVED
        )

    def observe_signal(self, signal: CongestionSignal) -> int:
        """Update the window from one round's three-state verdict; return the new window.

        - `CONGESTED` — cut multiplicatively and leave slow-start, relieving memory fast.
        - `SATURATED` — hold. The consumer never waited on the wire, so the window already
          covers the channel's bandwidth-delay product and further credits buy buffered
          memory rather than throughput. This is the state the two-state law could not
          express, and its absence is why an uncongested channel climbed to its ceiling
          whether or not the extra credits moved a byte.
        - `STARVED` — grow: double the window in slow-start, otherwise follow the CUBIC curve
          back toward the window the last backoff happened at, never below what additive
          increase would give.

        A held round still counts as a round, so `hold_rate` reads as convergence rather than
        as an idle controller.

        Args:
            signal: What the channel and the node measured this round.

        Returns:
            The new credit window.
        """
        self._rounds += 1
        if signal is CongestionSignal.SATURATED:
            # Held, not grown — and the recovery clock is *not* advanced. Advancing it would
            # walk the cubic forward through the rounds a converged window spent doing
            # nothing, so the first later round that did starve would evaluate `(t - k)**3`
            # far out on the curve and jump straight to the ceiling. The clock measures
            # rounds spent recovering, and a held round is not one of them.
            self._holds += 1
            return self.window
        if signal is CongestionSignal.CONGESTED:
            # Remember the window congestion was found at: that is the channel's measured
            # capacity, and the point growth should return to rather than crawl toward.
            self._w_max = self._window
            self._rounds_since_backoff = 0
            # K depends only on `w_max` and `beta`, so this is the one place it can change.
            # Deriving it here instead of per growth round takes a cube root off the
            # per-round path of every channel of every shuffle.
            self._k = (self._w_max * (1.0 - self._beta) / _CUBIC_C) ** (1.0 / 3.0)
            self._window = max(self._floor, self._window * self._beta)
            self._slow_start = False
            self._backoffs += 1
        elif self._slow_start:
            self._window = min(self._ceiling, self._window * _SLOW_START_FACTOR)
        else:
            # The clock only matters until the cubic has passed the ceiling; past that the
            # window is pinned there anyway. Holding it bounds `(t - k)**3` to a small
            # number instead of letting a long-lived channel cube a growing integer every
            # round forever (and, at the extreme, overflow the float).
            if self._rounds_since_backoff < _MAX_RECOVERY_ROUNDS:
                self._rounds_since_backoff += 1
            self._window = min(self._ceiling, self._grown_window())
        self._peak_window = max(self._peak_window, self.window)
        return self.window

    def _grown_window(self) -> float:
        """The window after one uncongested round of congestion avoidance.

        The larger of the CUBIC curve and the additive-increase (Reno) window. Taking the
        maximum is CUBIC's own "TCP-friendly region" rule, and it is what guarantees this can
        never be *slower* to recover than the additive law it replaces: on a small window,
        where the cubic is flat, `+alpha` wins and the behaviour is exactly as before; on a
        large window recovering toward a known-good value, the cubic wins by a wide margin.
        """
        reno = self._window + self._alpha
        if self._w_max is None:
            return reno
        # `W(t) = w_max + C·(t - K)³`, with K (computed at the backoff) the round the curve
        # returns to `w_max`. K comes from requiring the curve to pass through the
        # *post-backoff* window at t = 0, i.e. `C·K³ = w_max·(1 - beta)`.
        t = float(self._rounds_since_backoff)
        cubic = self._w_max + _CUBIC_C * (t - self._k) ** 3
        return max(reno, min(cubic, self._ceiling))

    def _clamp(self, w: float) -> int:
        return int(clamp(w, self._floor, self._ceiling))


def load_shuffle_window(hub: MetadataHub | None, signature: str) -> int | None:
    """The learned converged credit window for a shuffle `signature`, or `None` if unseen.

    Best-effort: any read failure (or a cold store) yields `None`, so the channel starts
    at the configured default. Only the *starting* window is affected — flow control still
    governs the window it actually uses — so this is purely a warm-start, never a result
    or a correctness change.

    The stored value is a smoothed float; a window is a count of batch slots, so it is
    rounded here rather than truncated.

    Whether the runs behind that average actually *agreed* is a separate question, asked
    separately: see `shuffle_window_is_stable`. It is deliberately not folded into the return
    value here, because the two have different consumers — the credit grant wants only the
    number, and the adaptive controller wants both.
    """
    estimate = load_scalar_estimate(hub, _SHUFFLE_WINDOW_NS, signature)
    return None if estimate is None else round(estimate.value)


def shuffle_window_is_stable(hub: MetadataHub | None, signature: str) -> bool:
    """Whether past runs of this shuffle agreed closely enough to act on the learned window.

    A shuffle whose window has scattered across an order of magnitude has not learned a
    window, it has averaged a bimodal population. Warm-starting from that is still the best
    guess available, but *skipping slow start* as well leaves a badly-sized channel with no
    ramp to escape on — strictly worse than never having learned. See
    `metadata.smoothed.ScalarEstimate.stable`.

    Args:
        hub: The metadata hub, or `None` when learning is off.
        signature: The shuffle channel's stable signature.

    Returns:
        `False` for a cold store, an unseen signature, or a value written before dispersion
        was tracked — the conservative reading in every case, since an unknown spread is not
        a small one.
    """
    estimate = load_scalar_estimate(hub, _SHUFFLE_WINDOW_NS, signature)
    return estimate is not None and estimate.stable


def record_shuffle_window(
    hub: MetadataHub | None, signature: str, window: int, config: Config | None = None
) -> None:
    """Persist a shuffle channel's converged credit `window`, exp-smoothed across runs.

    Best-effort and non-raising (mirrors `ml.gpu.record_gpu_utilization`): a recurring
    shuffle's window is smoothed toward each run's converged value so the next run
    warm-starts near it. Records nothing for a non-positive window."""
    if window <= 0:
        return
    record_smoothed_scalar(hub, _SHUFFLE_WINDOW_NS, signature, float(window), config)
