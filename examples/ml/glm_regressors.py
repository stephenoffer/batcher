"""Generalized linear models for counts, costs, and mixed zero-and-positive targets.

Ordinary least squares assumes a symmetric, unbounded target. Counts are neither, and
insurance-style cost data is a spike at zero plus a long positive tail. Poisson, gamma,
and Tweedie regressions carry the right assumption for each.

    python examples/ml/glm_regressors.py
"""

from __future__ import annotations

import batcher as bt
from batcher import ml


def main() -> None:
    # Count target: visits rising with exposure. Poisson is the natural fit.
    counts = bt.from_pydict(
        {
            "exposure": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "visits": [1.0, 2.0, 4.0, 5.0, 8.0, 11.0, 14.0, 20.0],
        }
    )

    poisson = ml.PoissonRegressor(["exposure"], "visits", alpha=1.0, max_iter=200).fit(counts)
    scored = poisson.predict(counts)
    quality = scored.select(
        deviance=bt.poisson_deviance("visits", "prediction"),
        mae=bt.mae("visits", "prediction"),
    ).to_pydict()
    print("poisson:", poisson.coef_, quality)

    # Predictions stay positive, which is the point of the log link.
    preds = scored.to_pydict()["prediction"]
    assert all(p > 0 for p in preds)
    assert quality["deviance"][0] >= 0.0

    # Strictly positive, right-skewed target: gamma.
    costs = bt.from_pydict(
        {
            "size": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "cost": [10.0, 22.0, 35.0, 60.0, 90.0, 150.0],
        }
    )
    # Keep `alpha` at 1.0 or above here: the gamma solver needs the regularizer to stay
    # numerically stable on a small sample, and returns NaN coefficients without it.
    gamma = ml.GammaRegressor(["size"], "cost", alpha=1.0, max_iter=200).fit(costs)
    gpred = gamma.predict(costs).to_pydict()["prediction"]
    print("gamma:", gamma.coef_, gpred[:3])
    assert all(p > 0 for p in gpred)

    # Zero-inflated positive target: Tweedie, with `power` between 1 and 2.
    claims = bt.from_pydict(
        {
            "risk": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "claim": [0.0, 0.0, 5.0, 0.0, 12.0, 20.0, 0.0, 40.0],
        }
    )
    tweedie = ml.TweedieRegressor(["risk"], "claim", power=1.5, alpha=1.0, max_iter=200).fit(claims)
    tpred = tweedie.predict(claims).to_pydict()["prediction"]
    print("tweedie:", tweedie.coef_, tpred[:3])
    assert all(p >= 0 for p in tpred)

    dev = (
        tweedie.predict(claims)
        .select(d=bt.tweedie_deviance("claim", "prediction", power=1.5))
        .to_pydict()
    )
    assert dev["d"][0] >= 0.0

    # The baseline to beat: predict the mean for everyone.
    baseline = ml.DummyRegressor("claim", strategy="mean").fit(claims)
    b = baseline.predict(claims).select(mae=bt.mae("claim", "prediction")).to_pydict()
    print("dummy mae:", b)
    assert b["mae"][0] > 0.0


if __name__ == "__main__":
    main()
