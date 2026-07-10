"""Fuzzy deduplication — MinHash signatures + LSH banding, as relational algebra.

Exact deduplication is `distinct()`. It is also nearly useless on a web-scale training
corpus, where the duplicates are the same article behind three different headers, or the
same page with a changed timestamp. Removing those is the single highest-leverage
preprocessing step for an LLM pretraining set, and it is a similarity *join*: every pair
of documents whose Jaccard similarity clears a threshold.

Comparing every pair is quadratic and impossible. The standard escape is two-stage:

1. **MinHash** replaces each document with a fixed-length signature (`str.minhash`) whose
   positional agreement rate estimates the documents' Jaccard similarity.
2. **LSH banding** splits each signature into `bands` bands and hashes each band. Two
   documents that share *any* band hash are candidates. A pair with similarity `s`
   survives with probability `1 - (1 - s^rows_per_band)^bands` — an S-curve whose knee
   sits near `(1/bands)^(1/rows_per_band)`, so band count *is* the recall/cost dial.

Everything below is that recipe expressed in operators Batcher already has: a projection
(the signature), an `array` + `explode` (the bands), a self-join on the band hash (the
candidates), and a filter on the exact signature agreement (`list.jaccard`) to discard
the false positives banding admits. Nothing is materialized on the driver, so it runs
wherever a join runs — including distributed.

Candidate generation is a *recall* filter, never a precision one: every returned pair is
verified against `threshold` by its signature agreement. Recall is not total — a pair
above the threshold can miss every band — which is inherent to LSH and is what `bands`
trades against cost.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir import Col, array, col, hash_rows

if TYPE_CHECKING:
    from batcher.api.dataset.frame import Dataset
    from batcher.plan.expr_ir import Expr

__all__ = ["build_drop_near_duplicates", "build_near_duplicates"]

_KEY = "__bc_dedup_key"
_SIG = "__bc_dedup_sig"
_BAND = "__bc_dedup_band"


def _row_key(column: str, key: str | None) -> Expr:
    """The value identifying a row: an explicit `key` column, or the text's own digest.

    Hashing the text means two byte-identical documents share a key and collapse to one
    row before any similarity work — exact duplicates handled by construction.
    """
    return Col(key) if key is not None else hash_rows(Col(column))


def _validate(
    ds: Dataset, column: str, threshold: float, num_perm: int, bands: int, key: str | None
) -> int:
    if column not in ds.columns:
        raise PlanError(f"near_duplicates(): unknown column {column!r}")
    if key is not None and key not in ds.columns:
        raise PlanError(f"near_duplicates(): unknown key column {key!r}")
    if not 0.0 < threshold <= 1.0:
        raise PlanError(f"near_duplicates(): threshold must be in (0, 1], got {threshold}")
    if num_perm < 1 or bands < 1:
        raise PlanError(
            f"near_duplicates(): num_perm and bands must be >= 1, got {num_perm}/{bands}"
        )
    if num_perm % bands:
        raise PlanError(
            f"near_duplicates(): bands must divide num_perm, got {num_perm} % {bands} != 0"
        )
    return num_perm // bands


def build_near_duplicates(
    ds: Dataset,
    column: str,
    *,
    threshold: float = 0.8,
    num_perm: int = 128,
    ngram: int = 5,
    bands: int = 16,
    key: str | None = None,
) -> Dataset:
    """Candidate near-duplicate pairs, verified against `threshold` (see `Dataset.ml`)."""
    rows_per_band = _validate(ds, column, threshold, num_perm, bands, key)
    signatures = _signatures(ds, column, key, num_perm, ngram)

    # One hash per band. The band index seeds its hash, so a value appearing in band 0
    # and band 3 cannot make two documents look like band-mates.
    band_hashes = [
        hash_rows(
            *(col(_SIG).list.get(b * rows_per_band + r) for r in range(rows_per_band)), seed=b
        )
        for b in range(bands)
    ]
    # Band on the key alone. Carrying the 128-value signature into the self-join would
    # copy it once per band and once per candidate — the join is over the *keys*, and the
    # signatures are attached afterwards, to the (far smaller) candidate set.
    banded = signatures.select(_KEY, **{_BAND: array(*band_hashes)}).explode(_BAND)
    left = banded.rename({_KEY: "key_a"})
    right = banded.rename({_KEY: "key_b"})
    # `key_a < key_b` drops the self-pair and one of each mirrored pair in one predicate;
    # `distinct` collapses a pair that collided in several bands.
    candidates = (
        left.join(right, on=_BAND)
        .filter(col("key_a") < col("key_b"))
        .select("key_a", "key_b")
        .distinct()
    )

    scored = candidates.join(
        signatures.rename({_KEY: "key_a", _SIG: "sig_a"}), on="key_a"
    ).join(signatures.rename({_KEY: "key_b", _SIG: "sig_b"}), on="key_b")
    verified = scored.select("key_a", "key_b", jaccard=col("sig_a").list.jaccard(col("sig_b")))
    return verified.filter(col("jaccard") >= threshold)


def _signatures(
    ds: Dataset, column: str, key: str | None, num_perm: int, ngram: int
) -> Dataset:
    """One `(key, signature)` row per distinct key.

    Collapsing duplicate keys here is what makes the joins below safe: a key appearing
    twice would multiply every pair it takes part in, and (with the default key, a digest
    of the text) two byte-identical documents *do* share one.
    """
    from batcher.plan.expr_ir.nodes import row_number

    base = ds.select(
        **{_KEY: _row_key(column, key), _SIG: Col(column).str.minhash(num_perm, ngram)}
    )
    # Rows sharing a key hold the same document, so which one survives is immaterial.
    return base.filter(row_number().over(partition_by=[Col(_KEY)], order_by=[Col(_KEY)]) == 1)


def build_drop_near_duplicates(
    ds: Dataset,
    column: str,
    *,
    threshold: float = 0.8,
    num_perm: int = 128,
    ngram: int = 5,
    bands: int = 16,
    key: str | None = None,
) -> Dataset:
    """Drop every row that has a near-duplicate with a smaller key (see `Dataset.ml`)."""
    from batcher.plan.expr_ir.nodes import row_number

    pairs = build_near_duplicates(
        ds, column, threshold=threshold, num_perm=num_perm, ngram=ngram, bands=bands, key=key
    )
    keyed = ds.with_columns(**{_KEY: _row_key(column, key)})
    # Rows sharing a key are the *same* document (the default key is the text's digest),
    # and a key is what the pair table names — so they must collapse to one row first, or
    # a winning key would carry all of its exact copies through.
    # Every row in a key-partition holds the same document, so the ordering key is
    # immaterial; when a row's *other* columns differ, the representative is arbitrary.
    first_of_key = row_number().over(partition_by=[Col(_KEY)], order_by=[Col(_KEY)])
    keyed = keyed.filter(first_of_key == 1)
    # Survivors are then the keys minimal among their near-duplicates. For a duplicate
    # cluster where every member matches every other — the usual shape — that is one row.
    losers = pairs.select("key_b").distinct()
    return keyed.join(losers, left_on=_KEY, right_on="key_b", how="anti").drop(_KEY)
