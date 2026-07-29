"""Discretizing, clipping, and reshaping the distribution of a numeric column.

Binning turns a continuous variable into a categorical one, which is how you let a linear
model express a non-monotonic effect. Clipping and power transforms attack the other
problem: a long tail that dominates the loss.

    python examples/ml/preprocessing_binning.py
"""

from __future__ import annotations

import batcher as bt
from batcher import ml


def main() -> None:
    data = bt.from_pydict({"v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 500.0]})

    # Equal-frequency bins: each bin holds about the same number of rows.
    quantile_binner = ml.KBinsDiscretizer("v", n_bins=5, strategy="quantile")
    q = quantile_binner.fit(data).transform(data).to_pydict()
    print("quantile bins:", q["v"])
    assert set(q["v"]) <= set(range(5))

    # Equal-width bins: the outlier drags every real value into bin 0.
    u = ml.KBinsDiscretizer("v", n_bins=5, strategy="uniform").fit(data).transform(data).to_pydict()
    print("uniform bins:", u["v"])
    assert u["v"][:9].count(0) >= 8

    # A threshold, for a straight yes/no feature.
    b = ml.Binarizer("v", threshold=5.0).fit(data).transform(data).to_pydict()
    assert b["v"][:5] == [0, 0, 0, 0, 0]
    assert b["v"][-1] == 1

    # Clip to fixed quantiles, so the tail stops dominating.
    c = ml.Clipper("v", lower=0.05, upper=0.95).fit(data).transform(data).to_pydict()
    print("clipped:", c["v"])
    assert max(c["v"]) < 500.0

    # Clip by the IQR rule instead of fixed quantiles.
    oc = ml.OutlierClipper("v", method="iqr", threshold=1.5).fit(data).transform(data).to_pydict()
    assert max(oc["v"]) < 500.0

    # Compress the tail rather than cutting it off.
    lg = ml.LogTransformer("v", offset=1.0).fit(data).transform(data).to_pydict()
    print("log:", [round(x, 3) for x in lg["v"]])
    assert lg["v"][-1] < 10.0

    # Power transforms make a skewed column more symmetric.
    pt = ml.PowerTransformer("v").fit(data).transform(data).to_pydict()
    assert len(pt["v"]) == 10
    bc = ml.BoxCoxTransformer("v").fit(data).transform(data).to_pydict()
    assert len(bc["v"]) == 10

    # Map to a uniform distribution by rank, which is scale-free by construction.
    qt = ml.QuantileTransformer("v", n_quantiles=10).fit(data).transform(data).to_pydict()
    assert all(0.0 <= x <= 1.0 for x in qt["v"])
    rt = ml.RankTransformer("v").fit(data).transform(data).to_pydict()
    print("ranks:", rt["v"])
    assert sorted(rt["v"]) == rt["v"]


if __name__ == "__main__":
    main()
