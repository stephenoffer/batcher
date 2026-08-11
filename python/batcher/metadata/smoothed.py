"""Best-effort read/write of a single learned scalar, exponentially smoothed across runs.

Several subsystems learn one number per key and want the same three properties: a cold
store yields `None` rather than an error, a write blends the new observation into the
prior instead of overwriting it, and neither direction can ever raise into the query path.
Carbonite learns a converged shuffle credit window and a memory-pressure flap rate; `io`
learns a per-source read throughput; `dist` learns a partition skew factor.

Those had grown three byte-identical copies of the same twelve lines, one of them across
the `carbonite`/`metadata` boundary. The subsystems cannot import each other, so a shared
helper has to live in a neutral layer, and `metadata` is where the Hub already is.

The smoothing weight is `max(floor, 1/(n+1))` for `floor =
optimizer.learned_scalar_alpha_floor`: a **running mean** while evidence is thin, decaying
into an exponential average with a `~1/floor`-observation memory once enough runs have
accrued. A fixed weight is the wrong step at both ends — at a static 0.5 the *first*
observation keeps a quarter of the estimate after three runs and an eighth after four,
forever, so one anomalous cold run anchors the value it was supposed to be smoothing; and a
pure running mean never forgets a regime the workload has left.

The floor is `learned_scalar_alpha_floor` and deliberately not `learning_smoothing_alpha`.
The latter is a *static blend weight* used elsewhere, and at its value of 0.5 it would
dominate `1/(n+1)` from the second observation onward — which is to say the floor would
never bind and the running-mean phase would not exist. `kyber.learning._smooth` makes the
same distinction for the same reason; this is the neutral-layer twin of it.

# Dispersion, and the two things it buys

A stored scalar used to be a point with no sense of how firmly it was held, and a point
estimate cannot answer either of the questions its consumers actually have.

**How much should one observation move it?** Under plain exponential smoothing, a single wild
run moves the estimate by `step x (value - prior)`, which is unbounded in the observation. A
GPU that thermally throttled once, a shuffle that ran while the node was swapping, a read
throughput measured against a cold page cache: each drags the learned value by an amount
proportional to how wrong it was. The estimate is a least-squares mean, and least squares has
no bounded-influence property.

The fix is the one this repository already reaches for in `ml.HuberRegressor`: clamp the
observation into `mean ± k sigma` before blending it. Any single run can then move the estimate
by at most `k sigma x step` no matter how extreme it was, which is exactly Huber's bounded
influence, and under normality a genuine observation is clamped about 0.3% of the time at
`k = 3` — a negligible bias against a large protection.

The variance, though, is updated from the **true** deviation rather than the clamped one. That
asymmetry is load-bearing and it is worth stating why: a workload that genuinely moved to a new
regime must be able to widen its own acceptance band and follow, and an estimator that clamped
*and* learned from the clamped value would narrow its band toward zero and lock up. Protect the
mean; let the variance see what really happened.

**Is this number worth acting on?** A learned window whose runs scatter over an order of
magnitude is not an estimate, it is an average of a bimodal population, and a consumer that
warm-starts from it and then switches its own search off is worse than one that never learned
anything. `ScalarEstimate.stable` is that check, and it is a coefficient of variation rather
than an absolute band because these scalars span bytes, ratios and counts.

Both are backward compatible. A record written before dispersion existed reports no variance,
which reads as "unknown" everywhere and reproduces the previous behaviour exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from batcher._internal.logging import note_suppressed
from batcher.config import active_config

if TYPE_CHECKING:
    from batcher.config import Config
    from batcher.metadata.hub import MetadataHub

__all__ = ["ScalarEstimate", "load_scalar", "load_scalar_estimate", "record_smoothed_scalar"]

#: How many standard deviations an observation may sit from the mean before it is clamped.
#:
#: Three is the conventional robust-statistics cut and it is conservative on purpose: under
#: normality a genuine observation lands outside it about 0.3% of the time, so the bias this
#: introduces is negligible while the protection against a single pathological run is total.
_WINSOR_SIGMAS = 3.0

#: Observations required before the dispersion is trusted enough to clamp against.
#:
#: The failure this prevents is subtle and terminal. Two nearly-equal early observations give a
#: variance near zero, so the acceptance band collapses, so every later observation is clamped
#: to the mean, so the variance stays near zero. The estimate freezes at whatever the first two
#: runs happened to say and no amount of later evidence can move it. Five is enough that a
#: near-zero variance means the quantity really is steady.
_MIN_OBSERVATIONS_FOR_ROBUSTNESS = 5

#: Narrowest acceptance band, as a fraction of the estimate's own magnitude.
#:
#: A variance of exactly zero is an artifact of finite sampling, not a claim of infinite
#: precision, and without this floor it reads as one: ten identical observations collapse the
#: band to nothing, and the eleventh — however absurd — is admitted at full weight because the
#: guard has no width to reject it with. That is the *most* common shape a learned scalar has
#: and the case where an outlier is most obviously an outlier.
#:
#: Ten percent is wide enough that ordinary run-to-run drift passes unclamped and narrow enough
#: that a thousand-fold excursion cannot move the estimate by more than a per cent. It does not
#: prevent the engine from following a genuine regime change, because the variance is fed the
#: *true* deviation: one real excursion widens the band far past this floor, and the next
#: observation is barely clamped at all.
_MIN_RELATIVE_BAND = 0.1

#: Coefficient of variation at or below which an estimate is called `stable`.
#:
#: A ratio rather than an absolute band, because these scalars are bytes, rates, counts and
#: fractions and no single width means the same thing to all of them. A quarter is generous:
#: a quantity whose runs scatter by more than that is not one a consumer should switch its own
#: search off for.
_STABLE_COEFFICIENT_OF_VARIATION = 0.25


@dataclass(frozen=True, slots=True)
class ScalarEstimate:
    """A learned scalar together with how firmly it is held.

    Attributes:
        value: The smoothed estimate.
        observations: Runs folded into it.
        variance: The exponentially-weighted variance of the observations, or `None` for a
            record written before dispersion was tracked. `None` means *unknown*, never zero —
            a consumer that reads an unmeasured spread as "perfectly steady" would act with
            more confidence than any evidence supports.
    """

    value: float
    observations: float
    variance: float | None = None

    @property
    def stddev(self) -> float | None:
        """The spread in the scalar's own units, or `None` when it was never tracked."""
        if self.variance is None or self.variance < 0:
            return None
        return math.sqrt(self.variance)

    @property
    def coefficient_of_variation(self) -> float | None:
        """Spread relative to magnitude, or `None` when unmeasurable.

        The scale-free form, because these scalars are bytes, rates, counts and fractions and
        an absolute band means something different to each. `None` for an estimate centred on
        zero, where the ratio is undefined rather than infinite.
        """
        sigma = self.stddev
        if sigma is None or self.value == 0:
            return None
        return sigma / abs(self.value)

    @property
    def stable(self) -> bool:
        """Whether this estimate is firm enough for a consumer to act on it confidently.

        The question a warm start actually has. A value averaged over runs that scattered by an
        order of magnitude is not an estimate of anything — it is the mean of a bimodal
        population — and a consumer that starts from it *and switches its own search off* ends
        up worse than one that never learned. Requires both enough observations and a
        coefficient of variation inside the band.

        `False` for a record with no tracked variance, which is the conservative reading: an
        unknown spread is not a small one.
        """
        if self.observations < _MIN_OBSERVATIONS_FOR_ROBUSTNESS:
            return False
        cv = self.coefficient_of_variation
        return cv is not None and cv <= _STABLE_COEFFICIENT_OF_VARIATION


def load_scalar_estimate(
    hub: MetadataHub | None, namespace: str, key: str
) -> ScalarEstimate | None:
    """Read one learned scalar with its dispersion, or `None` when it was never recorded.

    The richer sibling of `load_scalar`, for a consumer that needs to know how firmly the value
    is held and not only what it is.

    Args:
        hub: The metadata hub, or `None` when learning is off.
        namespace: The learned-parameter namespace.
        key: The identity the value is learned per.

    Returns:
        The estimate, or `None` for a cold store, an absent key, or any read failure.
    """
    if hub is None:
        return None
    try:
        stored = hub.get_keyed_param(namespace, key)
    except Exception as exc:  # pragma: no cover - a learned read must never break a query
        note_suppressed("metadata", "load a smoothed scalar estimate", exc)
        return None
    value = _value_of(stored)
    if value is None:
        return None
    return ScalarEstimate(value, _count_of(stored), _variance_of(stored))


def _bounded(delta: float, variance: float, count: float, prior: float) -> float:
    """Clamp one observation's deviation into `±k sigma`, so no single run can dominate.

    Plain exponential smoothing moves the estimate by `step x delta`, which is unbounded in the
    observation: a GPU that thermally throttled once, or a shuffle measured while the node was
    swapping, drags the learned value by however wrong it was. Bounding the deviation bounds
    that influence at `k sigma x step` regardless — the same bounded-influence property
    `ml.HuberRegressor` provides against outliers in a fit.

    Inert until there is enough evidence to say what "far" means. Two nearly-equal early
    observations give a variance near zero, and clamping against that would pin every later
    observation to the mean and freeze the estimate permanently.

    Args:
        delta: The observation's deviation from the current estimate.
        variance: The tracked variance before this observation.
        count: Observations already folded in.
        prior: The current estimate, for the relative floor on the band.

    Returns:
        The deviation to smooth with, equal to `delta` whenever the guard is inactive.
    """
    if count < _MIN_OBSERVATIONS_FOR_ROBUSTNESS:
        return delta
    limit = max(
        _WINSOR_SIGMAS * math.sqrt(max(0.0, variance)),
        _MIN_RELATIVE_BAND * abs(prior),
    )
    return delta if limit <= 0 else max(-limit, min(limit, delta))


def _variance_of(stored: object) -> float | None:
    """The tracked variance in a stored record, or `None` when it does not carry one.

    `None` rather than `0.0` for the missing case, and the distinction matters: a record
    written before dispersion existed has an *unknown* spread, and reading that as zero would
    make every pre-existing learned value look perfectly steady to `stable`.
    """
    if not isinstance(stored, dict):
        return None
    raw = stored.get("var")
    if raw is None or isinstance(raw, bool):
        return None
    try:
        variance = float(raw)
    except (TypeError, ValueError):
        return None
    return variance if math.isfinite(variance) and variance >= 0 else None


def load_scalar(hub: MetadataHub | None, namespace: str, key: str) -> float | None:
    """Read one learned scalar, or `None` when it was never recorded.

    Reads both the current shape (a `{"value", "n"}` record carrying the observation count)
    and the bare float an older store may hold, so a hub written by a previous build keeps
    answering.

    Args:
        hub: The metadata hub, or `None` when learning is off.
        namespace: The learned-parameter namespace, e.g. `"carbonite.shuffle_window"`.
        key: The identity the value is learned per, e.g. a shuffle signature.

    Returns:
        The stored value, or `None` for a cold store, an absent key, or any read failure.
    """
    if hub is None:
        return None
    try:
        stored = hub.get_keyed_param(namespace, key)
    except Exception as exc:  # pragma: no cover - a learned read must never break a query
        note_suppressed("metadata", "load a smoothed scalar", exc)
        return None
    return _value_of(stored)


def _value_of(stored: object) -> float | None:
    """The scalar held by a stored record, in either shape, or `None` if it is not usable.

    A non-finite stored value reads as "never recorded". It should not be reachable — the
    writer refuses to fold one in — but a store outlives the build that wrote it, and every
    consumer of these scalars divides by them or compares them against a threshold, where a
    NaN silently fails every comparison and an infinity produces a zero-sized budget. Reading
    it as absent costs one cold estimate; adopting it costs a wrong decision on every query
    until the entry is manually deleted.
    """
    if stored is None:
        return None
    raw = stored.get("value") if isinstance(stored, dict) else stored
    if raw is None or isinstance(raw, bool):
        return None
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):  # a foreign blob
        return None
    return value if math.isfinite(value) else None


def _count_of(stored: object) -> float:
    """Observations already folded into a stored record; 1.0 when it does not say."""
    if not isinstance(stored, dict):
        return 1.0
    try:
        count = float(stored.get("n", 1.0))
    except (TypeError, ValueError):
        return 1.0
    return count if math.isfinite(count) else 1.0


def record_smoothed_scalar(
    hub: MetadataHub | None,
    namespace: str,
    key: str,
    value: float,
    config: Config | None = None,
) -> None:
    """Blend one observation into the learned scalar at `namespace`/`key`.

    The first observation is stored as-is; every later one moves the stored value a step
    `max(floor, 1/(n+1))` of the way toward it, where `n` counts the observations already
    folded in. That step is a plain running mean while `n` is small and settles into a fixed
    exponential average once `n` passes `1/floor`, which is what makes an early anomalous run
    wash out instead of anchoring the estimate.

    Best-effort in both directions, so a metadata backend that is unreachable, read-only, or
    mid-migration degrades to "learned nothing this run".

    Args:
        hub: The metadata hub, or `None` when learning is off.
        namespace: The learned-parameter namespace.
        key: The identity the value is learned per.
        value: The new observation.
        config: Config to read the smoothing weight from; defaults to the active one.
    """
    if hub is None:
        return
    # A non-finite observation is dropped rather than folded in. Exponential smoothing is
    # `prior + step * (value - prior)`, which propagates a NaN or an infinity into the stored
    # value and, from there, into *every* subsequent update — the entry is poisoned for the
    # life of the store, and nothing raises. The producers are ratios (bytes over elapsed,
    # observed over predicted, used over capacity), so a zero denominator or an empty
    # measurement window is the ordinary way one arises, not an exotic one.
    if not math.isfinite(value):
        return
    try:
        floor = (config or active_config()).optimizer.learned_scalar_alpha_floor
        stored = hub.get_keyed_param(namespace, key)
        prior = _value_of(stored)
        if prior is None:
            smoothed, count, variance = value, 1.0, 0.0
        else:
            # `count` decides the step, so a nonsensical stored count is a nonsensical blend
            # weight: a negative one at -1 divides by zero, and below -1 it flips the step's
            # sign and moves the estimate *away* from every observation. Clamping to at least
            # one observation makes the worst case "smooths as if this were the second run".
            count = max(1.0, _count_of(stored))
            step = min(1.0, max(floor, 1.0 / (count + 1.0)))
            prior_var = _variance_of(stored) or 0.0
            # The true deviation drives the variance; a clamped one drives the mean.
            delta = value - prior
            smoothed = prior + step * _bounded(delta, prior_var, count, prior)
            # Exponentially-weighted variance, the companion of the mean update above: it
            # decays on the same `step`, so the two describe the same window of history rather
            # than one tracking a regime the other has already forgotten. Fed the *unclamped*
            # deviation on purpose — a workload that genuinely moved must be able to widen its
            # own acceptance band and follow, and an estimator that both clamped and learned
            # from the clamped value would narrow toward zero and lock up.
            variance = (1.0 - step) * (prior_var + step * delta * delta)
            count += 1.0
        hub.put_keyed_param(namespace, key, {"value": smoothed, "n": count, "var": variance})
    except Exception as exc:  # pragma: no cover - a learned write must never break a query
        note_suppressed("metadata", f"record learned scalar {namespace}/{key}", exc)
