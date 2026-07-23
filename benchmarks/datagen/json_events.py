"""A deterministic semistructured (JSON) event log — the one input every engine sees.

The tabular benchmarks read canonical public parquet; there is no public parquet corpus
of *nested JSON documents* to point at, so this builds one deterministically and — the
property the correctness gate actually relies on — hands the **same** Arrow table to every
engine. A fixed-seed generator over a shared table is exactly as parity-safe as reading one
parquet file: all engines parse byte-identical JSON, so a disagreement is a real engine bug,
not a data-skew artifact.

The shape is a realistic web/mobile analytics event: a nested ``user``/``event``/``device``
object plus a ``tags`` array, serialized one JSON document per row into a single Utf8
``payload`` column (how semistructured data actually lands — a text/JSON column, not a
pre-shredded struct). The benchmark then makes every engine do the real work: parse the
document and pull typed fields out of it by path.

Row count scales with ``--scale`` (1 -> 1,000,000 documents, ~250 MiB of JSON text), enough
to keep 96 cores busy on the parse-bound path the suite is built to measure.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa

ROWS_PER_SCALE = 1_000_000
_SEED = 0x5EED

# Field domains — small, realistic cardinalities so group-by outputs stay comparable.
_COUNTRIES = ("US", "DE", "GB", "FR", "JP", "BR", "IN", "CA", "AU", "NL", "SE", "ES")
_TIERS = ("bronze", "silver", "gold", "platinum")
_EVENTS = ("view", "click", "search", "add_to_cart", "purchase", "logout")
_OS = ("iOS", "Android", "Windows", "macOS", "Linux")
_TAGS = ("mobile", "web", "promo", "organic", "returning", "new")


def _pick(rng: np.random.Generator, choices: tuple[str, ...], n: int) -> np.ndarray:
    """``n`` random members of ``choices`` (as a NumPy object array of str)."""
    return np.asarray(choices, dtype=object)[rng.integers(0, len(choices), size=n)]


def build_events(scale: float) -> dict[str, pa.Table]:
    """Build the shared ``events`` table (one Utf8 ``payload`` JSON column + an ``id``).

    Deterministic for a given ``scale``: the same seed yields byte-identical documents, so
    every engine and every repeat sees the same input.
    """
    n = int(ROWS_PER_SCALE * scale)
    rng = np.random.default_rng(_SEED)

    uid = rng.integers(1, 1_000_000, size=n)
    country = _pick(rng, _COUNTRIES, n)
    tier = _pick(rng, _TIERS, n)
    etype = _pick(rng, _EVENTS, n)
    # Monetary value: 0 for non-purchase events, a positive amount for purchases, so a
    # WHERE type = 'purchase' filter selects a meaningful, checkable subset.
    is_purchase = etype == "purchase"
    value = np.where(is_purchase, np.round(rng.uniform(1.0, 500.0, size=n), 2), 0.0)
    items = rng.integers(1, 6, size=n)
    os = _pick(rng, _OS, n)
    ver_major = rng.integers(10, 18, size=n)
    ver_minor = rng.integers(0, 9, size=n)
    ts = 1_700_000_000 + rng.integers(0, 30 * 86_400, size=n)
    tag0 = _pick(rng, _TAGS, n)
    tag1 = _pick(rng, _TAGS, n)

    # Assemble one JSON document per row. A single f-string per row keeps the (untimed)
    # generation straightforward; the documents are what every engine then parses.
    payload = [
        f'{{"user":{{"id":{uid[i]},"country":"{country[i]}","tier":"{tier[i]}"}},'
        f'"event":{{"type":"{etype[i]}","value":{value[i]},"items":{items[i]}}},'
        f'"device":{{"os":"{os[i]}","version":"{ver_major[i]}.{ver_minor[i]}"}},'
        f'"tags":["{tag0[i]}","{tag1[i]}"],"ts":{ts[i]}}}'
        for i in range(n)
    ]
    table = pa.table(
        {
            "id": pa.array(np.arange(n, dtype=np.int64)),
            "payload": pa.array(payload, type=pa.string()),
        }
    )
    return {"events": table}
