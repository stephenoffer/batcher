"""Learned join-skew: persist the hot join-key values measured by the detection
pre-pass, keyed by join shape, so a later run of the same shape engages salting from
learned skew without re-running the pre-pass (and without the user re-opting in).

The loop is result-preserving — salting only moves a hot key's work between reducers,
never the joined relation — so a learned hot set is always safe to act on. Stored as
neutral learned params in the process-wide `MetadataHub`; `dist` reads/writes them
directly (it is outside the kyber/carbonite/core independence set).
"""

from __future__ import annotations

import hashlib
import json
import math

from batcher._internal.logging import note_suppressed
from batcher.plan.logical import Join

__all__ = [
    "DEFAULT_LEARNED_SALT",
    "hot_keys_from_column_stats",
    "join_skew_key",
    "load_learned_hot_keys",
    "persist_hot_keys",
    "resolve_hot_keys",
    "salt_factor",
    "salting_preserves_result",
]

# Learned-skew namespace + the salt fan-out used when learned hot keys engage salting
# on a run that did not explicitly request it.
_SKEW_NAMESPACE = "dist.skew"
DEFAULT_LEARNED_SALT = 4
# Upper bound on the salt fan-out. Each sub-partition of a hot key costs a replicated copy
# of the matching build-side rows, so an unbounded fan-out trades one overloaded reducer for
# a network-wide broadcast. 64 covers a key holding a third of the data across a 200-way
# shuffle, which is already an extreme skew.
_MAX_SALT = 64


def salt_factor(hot_fraction: float, partitions: int) -> int:
    """How many sub-partitions one hot key should be fanned across.

    A fixed fan-out is the wrong shape for this problem, because the imbalance it has to
    repair depends on the shuffle's width. With `P` reducers the average reducer receives
    `N/P` rows, while the reducer owning a value of frequency `f` receives `f·N` — an
    overload factor of `f·P`, which grows with the cluster. Splitting that key across `s`
    sub-partitions divides its load by `s`, so levelling it needs

    ``s >= f · P``

    and nothing more: past that point the hot key's reducers are already below average and
    every extra sub-partition only replicates more of the build side. A constant 4 is
    therefore simultaneously too much on a 16-way shuffle (where a 10% key overloads by 1.6x)
    and far too little on a 200-way one (where the same key overloads by 20x and stays 5x
    over average after salting) — and it is the wide shuffle, the one that only appears at
    scale, that the constant serves worst.

    Args:
        hot_fraction: The hot value's share of the side's rows.
        partitions: The number of reduce partitions the shuffle fans into.

    Returns:
        The salt fan-out, at least 2 (salting at all means splitting in two) and capped.
    """
    if hot_fraction <= 0.0 or partitions <= 1:
        return 2
    needed = math.ceil(hot_fraction * partitions)
    return max(2, min(_MAX_SALT, needed))


def join_skew_key(left_ir: str, right_ir: str, join: Join) -> str:
    """A stable key identifying this join's shape (both sides + keys + type), so the
    hot values learned on one run are reused on the next run of the same shape."""
    payload = json.dumps(
        [left_ir, right_ir, list(join.left_keys), list(join.right_keys), join.join_type],
        sort_keys=True,
    )
    # `usedforsecurity=False` because this is a cache key, not a security claim — and on a
    # FIPS-enforcing host a bare `sha1()` *raises*, so learned skew would not merely be
    # unavailable there, the join would fail. Saying what the digest is for is what makes
    # OpenSSL allow it, and it changes no byte of the key.
    return hashlib.sha1(payload.encode(), usedforsecurity=False).hexdigest()[:16]


def load_learned_skew(shape_key: str) -> tuple[list[str], float] | None:
    """The hot values learned for this shape and the largest one's measured share.

    `None` when never measured. A learned empty list means "measured, not skewed", which is
    distinct from never-measured, so a shape known to be uniform never re-runs the pre-pass.

    The share is what sizes the salt fan-out, and it has to be *measured* rather than assumed
    — see `salt_factor`. A record written before the share was stored reads back as `0.0`,
    which the caller treats as "unknown" and falls back to the detection threshold, i.e. the
    previous behavior for an old record rather than a wrong number.

    Args:
        shape_key: This join shape's key (`join_skew_key`).

    Returns:
        `(hot_values, max_share)`, or `None` if this shape has never been measured.
    """
    raw = _load_raw(shape_key)
    if raw is None:
        return None
    if isinstance(raw, dict):
        return [str(v) for v in raw.get("hot", ())], float(raw.get("share", 0.0))
    return [str(v) for v in raw], 0.0  # legacy record: values only


def load_learned_hot_keys(shape_key: str) -> list[str] | None:
    """The hot join-key values learned for this shape, or `None` if never measured.

    A learned empty list means "measured, not skewed" — distinct from never-measured,
    so a non-skewed shape never re-runs the detection pre-pass. Best-effort; the hub
    being unavailable simply means no learned skew (fall back to the config behavior).
    """
    learned = load_learned_skew(shape_key)
    return None if learned is None else learned[0]


def _load_raw(shape_key: str):
    """The stored learned-skew record for this shape, or `None`. Best-effort: a hub that
    cannot be reached is *noted*, not swallowed — a silent failure here disables learned
    skew forever with nothing in the log to say why the pre-pass keeps re-running."""
    try:
        from batcher.core import default_hub

        return default_hub().get_keyed_param(_SKEW_NAMESPACE, shape_key)
    except Exception as exc:
        note_suppressed("dist", "load learned hot keys", exc)
        return None


def persist_hot_keys(shape_key: str, hot: list[str], share: float = 0.0) -> None:
    """Record what the detection pre-pass measured, so a later run of the same join shape
    engages salting from learned skew without re-running the pre-pass.

    `share` — the hottest value's measured share of its side — is stored with the values
    because it is what sizes the fan-out (`salt_factor`), and a later run has no other way
    to recover it. Best-effort; never breaks the join.

    Args:
        shape_key: This join shape's key (`join_skew_key`).
        hot: The hot values, as strings. Empty means "measured, not skewed".
        share: The hottest value's share of its side; `0.0` when unknown.
    """
    try:
        from batcher.core import default_hub

        default_hub().put_keyed_param(
            _SKEW_NAMESPACE, shape_key, {"hot": list(hot), "share": float(share)}
        )
    except Exception as exc:
        note_suppressed("dist", "persist learned hot keys", exc)


def salting_preserves_result(join: Join, reducer_finalizes: bool) -> bool:
    """Whether salting this join's shuffle leaves the answer unchanged.

    Salting spreads a hot key's probe rows over several reducers and replicates its build
    rows to each, so the key's work fans across the cluster instead of overloading one node.
    Three conditions make that a pure scheduling change:

    * a **single** equi-key on each side, because the salt is appended to one key value;
    * a **left-driven** join type, because replicating build rows would otherwise duplicate
      output rows a RIGHT or FULL join must emit exactly once;
    * reducers that **concatenate** rather than finalize. A reducer may finalize a fused
      aggregate only because co-partitioning puts every group in exactly one bucket, and
      salting deliberately breaks that: each salted reducer would finalize a *partial*
      group and the union would carry several half-summed rows for the hot key. No error is
      raised — the query simply returns a wrong answer, which is why this is a guard rather
      than a preference.

    Args:
        join: The join being scheduled.
        reducer_finalizes: Whether a reducer closes a fused aggregate on its own bucket.

    Returns:
        True when salting may engage.
    """
    from batcher.plan.distribution import BROADCAST_SAFE_JOINS

    return (
        not reducer_finalizes
        and len(join.left_keys) == 1
        and len(join.right_keys) == 1
        and join.join_type in BROADCAST_SAFE_JOINS
    )


def hot_keys_from_column_stats(
    join: Join, sources, fraction: float, partitions: int
) -> tuple[list[str], float]:
    """The join key's hot values as Kyber already measured them — no pass over the data.

    Kyber owns the column statistics (Core measures, Kyber decides), so *which values are
    hot* is asked of it rather than re-derived here. Skew is a property of the column, not
    of the query: if one value is 40% of the rows that is true of every join on it,
    including this one's first ever run, so this costs nothing and needs no opt-in.

    The hottest value's **measured share** comes back with the values, because it is what
    sizes the fan-out and Kyber has it in hand. Returning the values alone made this path
    substitute `fraction` — the threshold at which a value starts counting as hot — and
    `salt_factor(0.10, 8)` floors to a fan-out of 2 whatever the real skew is. That is the
    same defect `resolve_hot_keys` records having fixed on the *detection* path, left
    standing on the path that fires first and needs no opt-in.

    Args:
        join: The join whose key is being examined.
        sources: The plan's bound inputs.
        fraction: The share of rows a value must hold to count as hot.
        partitions: The shuffle's reducer count, which sets how badly a hot key overloads.

    Returns:
        `(hot values as strings, the hottest one's measured share)`; `([], 0.0)` when
        nothing is known.
    """
    try:
        from batcher import kyber
        from batcher.core import default_hub

        return kyber.hot_join_value_shares(join, sources, default_hub(), fraction, partitions)
    except Exception as exc:  # pragma: no cover - statistics must never break a join
        note_suppressed("dist", "read hot keys from column stats", exc)
        return [], 0.0


def resolve_hot_keys(
    join: Join,
    sources,
    shape_key: str,
    fraction: float,
    partitions: int,
    salt: int,
    detect,
) -> tuple[list[str], int]:
    """The hot values of this join's key and the salt fan-out to spread them over.

    Cheapest source first, because salting is result-preserving and an approximate hot set
    therefore costs at most a little fan-out, never an answer:

    1. the set **learned** for this exact join shape on a previous run — free and exact, but
       silent about a shape that has not run before. An empty learned list means "measured,
       not skewed", which is distinct from never-measured, so a shape known to be uniform
       never re-runs the pre-pass;
    2. the column's **measured most-common values**, which Kyber already holds;
    3. the **detection pre-pass** (`detect`), a distributed Misra-Gries scan of both sides.
       Correct, and the only option when nothing has been measured. It costs an extra pass,
       so it runs when the config asks for it (`salt > 0`) **or** the join is large enough
       that `_detect_is_worth_it` says the insurance is cheaper than the exposure — and its
       result is persisted either way, so it is paid at most once per shape.

    Shared by both transports so the disk and Flight joins cannot drift on which keys they
    consider hot; only how they *publish* the salted buckets differs.

    Args:
        join: The join being scheduled.
        sources: The plan's bound inputs.
        shape_key: This join shape's learned-skew key (`join_skew_key`).
        fraction: The share of rows a value must hold to count as hot.
        partitions: The shuffle's reducer count.
        salt: The configured salt fan-out. A positive value forces the pre-pass and pins the
            fan-out; `0` means "not requested", which still allows a *measured* hot key to
            engage salting at a sized fan-out; a negative value means never, and is the only
            way to pin the plain co-partition shuffle.
        detect: A zero-argument callable running the pre-pass, returning
            `(hot_values, max_share)`.

    Returns:
        `(hot_values, salt_count)`. `salt_count` is 0 when nothing is hot, which is the
        plain co-partition shuffle.
    """
    if salt < 0:
        # The explicit off switch. Nothing above this line runs, so a query that pins it
        # pays neither the pre-pass nor a hub lookup — "never salt" has to be free, or it
        # is only a different kind of cost.
        return [], 0
    learned = load_learned_skew(shape_key)
    if learned is not None:
        hot, share = learned
    else:
        hot, share = hot_keys_from_column_stats(join, sources, fraction, partitions)
        if not hot and (salt > 0 or _detect_is_worth_it(join, sources)):
            hot, share = detect()
            persist_hot_keys(shape_key, hot, share)
    if not hot:
        return [], 0
    # A known-skewed key engages salting even when the config left it off: the skew is
    # measured, and salting cannot regress a plain shuffle.
    #
    # Size the fan-out from the key's **measured** share, not from `fraction`. `fraction` is
    # the threshold at which a value *starts* counting as hot (0.10 by default), so feeding
    # it to `salt_factor` answers "how far must I fan a key that is barely hot" — and the
    # answer, `ceil(0.10 x 8) = 1`, floors to a fan-out of 2 no matter how skewed the key
    # really is. A value holding 40% of a side across 8 reducers needs `ceil(0.40 x 8) = 4`;
    # it was getting 2, which is why the default path still ran ~12.5 s where an explicit
    # `skew_join_salt=8` ran ~1.9 s. The formula was right and its input was the threshold
    # rather than the measurement. `share` is 0.0 only when genuinely unknown — a learned
    # record written before the share was stored, or a hot set with no frequency behind it —
    # and then `fraction` is the only figure available and stands as the conservative floor it
    # always was. The column-statistics path used to land there too, on a measurement it was
    # already holding; it now carries it.
    return hot, salt if salt > 0 else salt_factor(max(share, fraction), partitions)


#: Total input rows above which one Misra-Gries pre-pass is cheaper than risking a skewed
#: shuffle. The trade is asymmetric and both halves are measured (8 workers, 40M ⋈ 10M on the
#: Flight transport, `benchmarks/BENCHMARK_RESULTS.md`): detection costs ~4% on a join that
#: turns out uniform (1,657 ms -> 1,731 ms), while an undetected 40% hot key costs 5.8x
#: (12,801 ms against 2,206 ms salted) — and, worse, *anti*-scales, going from 6,138 ms at 2
#: workers to 12,801 ms at 8, because the cluster grows and the overloaded reducer does not.
#: Below this the exposure is small in absolute terms and the pass is not worth its scan.
_DETECT_MIN_INPUT_ROWS = 1 << 23  # ~8.4M rows across both sides


def _detect_is_worth_it(join: Join, sources) -> bool:
    """Whether this join is large enough to pay for one hot-key detection pass.

    Asked only when nothing cheaper knew anything, and the answer is persisted, so a shape
    pays this at most once. Estimated rather than measured on purpose — it is deciding
    whether to *take* a measurement, so requiring one first would be circular. Best-effort:
    an estimator that cannot answer reads as "not worth it", which is the previous
    opt-in-only behavior.
    """
    try:
        from batcher.config import active_config
        from batcher.kyber.cardinality import CardinalityEstimator

        est = CardinalityEstimator(sources, {}, active_config().optimizer.cardinality)
        rows = est.estimate(join.left).rows + est.estimate(join.right).rows
    except Exception as exc:  # pragma: no cover - estimation must never break a join
        note_suppressed("dist", "size the skew detection pre-pass", exc)
        return False
    return rows >= _DETECT_MIN_INPUT_ROWS
