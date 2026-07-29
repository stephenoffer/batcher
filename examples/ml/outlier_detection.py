"""Finding outliers: per-column rules and a multivariate distance.

A per-column rule misses the point that a row can be unremarkable on every axis and still
be absurd as a combination. Mahalanobis distance catches that, which is why it is the one
to reach for on correlated features.

    python examples/ml/outlier_detection.py
"""

from __future__ import annotations

import batcher as bt
from batcher import ml


def main() -> None:
    # `height` and `weight` move together, plus one row that breaks the relationship.
    people = bt.from_pydict(
        {
            "height": [150.0, 160.0, 170.0, 180.0, 190.0, 155.0],
            "weight": [50.0, 60.0, 70.0, 80.0, 90.0, 200.0],
        }
    )

    # The IQR rule, per column.
    bounds = ml.outlier_bounds(people, "weight", method="iqr", threshold=1.5)
    print("weight bounds:", bounds)
    assert bounds[0] < bounds[1]

    counts = ml.count_outliers(people, ["height", "weight"], method="iqr")
    print("outlier counts:", counts)
    assert counts["weight"] >= 1
    # Height alone looks perfectly ordinary.
    assert counts["height"] == 0

    flagged = ml.flag_outliers(people, ["height", "weight"], method="iqr").to_pydict()
    print(sorted(flagged))
    assert "weight_outlier" in flagged
    assert flagged["weight_outlier"][-1] is True

    # A z-score rule instead of IQR.
    zbounds = ml.outlier_bounds(people, "weight", method="zscore", threshold=2.0)
    assert zbounds[0] < zbounds[1]

    # Multivariate: distance from the joint centre, accounting for correlation.
    scored = ml.mahalanobis_distance(people, ["height", "weight"]).to_pydict()
    print("mahalanobis:", [round(v, 3) for v in scored["mahalanobis"]])
    assert "mahalanobis" in scored
    # The row that breaks the height/weight relationship is the furthest out.
    assert scored["mahalanobis"][-1] == max(scored["mahalanobis"])

    # The screen this exists for.
    ranked = (
        ml.mahalanobis_distance(people, ["height", "weight"])
        .sort("mahalanobis", descending=True)
        .limit(1)
        .to_pydict()
    )
    assert ranked["weight"] == [200.0]


if __name__ == "__main__":
    main()
