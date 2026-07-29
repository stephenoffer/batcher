"""A/B test statistics computed in the engine: effect size, t-statistic, and intervals.

The whole test is aggregate expressions over the assignment table, so it runs where the
data is instead of pulling a sample into SciPy. ``welch_*`` does not assume equal
variances, which is the right default for a real experiment.

    python examples/statistics/ab_test_inference.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col


def main() -> None:
    # Two arms. B is genuinely better on `value`, and converts more often.
    trial = bt.from_pydict(
        {
            "arm": ["a"] * 8 + ["b"] * 8,
            # First eight rows are arm A, last eight are arm B.
            "value": [
                10.0,
                12.0,
                11.0,
                9.0,
                10.5,
                11.5,
                10.0,
                12.0,
                14.0,
                15.0,
                13.5,
                16.0,
                14.5,
                15.5,
                14.0,
                15.0,
            ],
            "converted": [0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1],
        }
    )

    is_b = col("arm") == "b"

    result = trial.select(
        # Effect size: how big is the difference, in standard deviations?
        cohens_d=bt.cohens_d("value", is_b),
        hedges_g=bt.hedges_g("value", is_b),
        # Welch's t-test pieces (unequal variances allowed).
        t_stat=bt.welch_t_statistic("value", is_b),
        df=bt.welch_df("value", is_b),
        # Group means, and the confidence half-width around the pooled mean.
        group_mean=bt.group_mean("value", is_b),
        mean_ci=bt.mean_ci_half_width("value", confidence=0.95),
        # The proportion test, for the binary outcome.
        prop_z=bt.proportion_z_statistic(col("converted") == 1, is_b),
        prop_ci=bt.proportion_ci_half_width(col("converted") == 1, confidence=0.95),
    ).to_pydict()

    print(result)

    # B is better, so the effect size and t-statistic are large and same-signed.
    assert abs(result["cohens_d"][0]) > 2.0
    # Hedges' g is the small-sample-corrected d, so it is slightly shrunk toward zero.
    assert abs(result["hedges_g"][0]) < abs(result["cohens_d"][0])
    assert abs(result["t_stat"][0]) > 2.0
    assert result["df"][0] > 0.0
    # The half-widths are positive spans you add and subtract from the point estimate.
    assert result["mean_ci"][0] > 0.0
    assert result["prop_ci"][0] > 0.0
    # More conversions in B than A, so the z-statistic is non-zero.
    assert result["prop_z"][0] != 0.0

    # The same statistics per segment, in one pass -- the reason they are aggregates.
    per_arm = (
        trial.group_by("arm")
        .agg(mean=bt.mean("value"), n=bt.count(), ci=bt.mean_ci_half_width("value"))
        .sort("arm")
        .to_pydict()
    )
    print(per_arm)
    assert per_arm["arm"] == ["a", "b"]
    assert per_arm["mean"][1] > per_arm["mean"][0]


if __name__ == "__main__":
    main()
