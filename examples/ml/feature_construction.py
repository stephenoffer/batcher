"""Building new features: interactions, ratios, calendar parts, lags, and rolling windows.

These are the featurizers that turn a raw table into a model-ready one. The time-series
ones (``LagFeaturizer``, ``RollingFeaturizer``) need an ``order_by`` and usually a
``partition_by``: forgetting the partition silently leaks one entity's history into
another's features.

    python examples/ml/feature_construction.py
"""

from __future__ import annotations

from datetime import datetime

import batcher as bt
from batcher import ml


def main() -> None:
    numeric = bt.from_pydict({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})

    # Every pairwise product, plus squares.
    poly = ml.PolynomialFeatures(["a", "b"], degree=2).fit(numeric).transform(numeric).to_pydict()
    print(sorted(poly))
    assert len(poly) > 2

    # Products only, without the squared terms.
    inter = ml.InteractionFeatures(["a", "b"]).fit(numeric).transform(numeric).to_pydict()
    assert len(inter) > 2

    # Explicit ratios, which often carry more signal than either column alone.
    ratio = ml.RatioFeatures([("a", "b")]).fit(numeric).transform(numeric).to_pydict()
    print(sorted(ratio))
    assert len(ratio) > 2

    # Calendar features from a timestamp.
    events = bt.from_pydict(
        {
            "ts": [datetime(2024, 3, 1, 9), datetime(2024, 3, 2, 18), datetime(2024, 3, 4, 12)],
            "user": ["u1", "u1", "u2"],
            "amount": [10.0, 20.0, 30.0],
        }
    )
    dtf = ml.DateTimeFeaturizer("ts", parts=("year", "month", "hour", "is_weekend"))
    feats = dtf.fit(events).transform(events).to_pydict()
    print(sorted(feats))
    assert any("year" in c for c in feats)
    assert any("is_weekend" in c for c in feats)

    # Cyclical encoding: hour 23 and hour 0 should be neighbours, not opposites.
    cyc = ml.CyclicalEncoder("ts", parts=("hour",)).fit(events).transform(events).to_pydict()
    print(sorted(cyc))
    assert len(cyc) > 3  # sin/cos pair added

    # Lags, partitioned per user so one user's history never leaks into another's.
    lag = ml.LagFeaturizer("amount", order_by="ts", lags=(1,), partition_by="user")
    lagged = lag.fit(events).transform(events).sort("ts").to_pydict()
    print(sorted(lagged))
    lag_col = next(c for c in lagged if "lag" in c)
    # The first row of each user has no prior row.
    assert lagged[lag_col][0] is None

    # Rolling aggregates over the same ordering.
    roll = ml.RollingFeaturizer(
        "amount", order_by="ts", window=2, aggregates=("mean",), partition_by="user"
    )
    rolled = roll.fit(events).transform(events).to_pydict()
    assert any("mean" in c for c in rolled)

    # Group statistics as a feature: how does this row compare to its group?
    gse = ml.GroupStatEncoder("amount", by="user", statistics=("mean",))
    grouped = gse.fit(events).transform(events).to_pydict()
    assert len(grouped) > 3

    # Cheap text features, without a model.
    text = bt.from_pydict({"body": ["Hello world 123", "Short"]})
    tsf = ml.TextStatFeaturizer("body", features=("char_count", "word_count", "digit_ratio"))
    tf = tsf.fit(text).transform(text).to_pydict()
    print(sorted(tf))
    assert any("word_count" in c for c in tf)

    # Drop columns that never vary -- they cannot help any model.
    flat = bt.from_pydict({"varies": [1.0, 2.0, 3.0], "constant": [7.0, 7.0, 7.0]})
    vt = ml.VarianceThreshold(["varies", "constant"], threshold=0.0)
    kept = vt.fit(flat).transform(flat).to_pydict()
    print(sorted(kept))
    assert "varies" in kept


if __name__ == "__main__":
    main()
