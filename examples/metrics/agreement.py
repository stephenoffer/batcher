"""Agreement metrics: how well a prediction tracks the truth, not just how close.

Correlation says the shapes match; these say the *values* match. A forecast that is
perfectly correlated but biased high scores well on correlation and badly here, which is
usually the honest answer.

    python examples/metrics/agreement.py
"""

from __future__ import annotations

import batcher as bt


def main() -> None:
    series = bt.from_pydict(
        {
            "observed": [1.0, 2.0, 3.0, 4.0, 5.0],
            # A near-perfect tracker.
            "close": [1.1, 2.0, 2.9, 4.1, 5.0],
            # Perfectly correlated but shifted up by 10: correlation is blind to this.
            "biased": [11.0, 12.0, 13.0, 14.0, 15.0],
        }
    )

    scores = series.select(
        ccc_close=bt.concordance_correlation("observed", "close"),
        ccc_biased=bt.concordance_correlation("observed", "biased"),
        nse_close=bt.nash_sutcliffe_efficiency("observed", "close"),
        nse_biased=bt.nash_sutcliffe_efficiency("observed", "biased"),
        kge_close=bt.kling_gupta_efficiency("observed", "close"),
        kge_biased=bt.kling_gupta_efficiency("observed", "biased"),
        # Plain correlation, for contrast.
        corr_biased=bt.corr("observed", "biased"),
    ).to_pydict()

    print(scores)

    # The biased series is perfectly correlated ...
    assert abs(scores["corr_biased"][0] - 1.0) < 1e-9
    # ... and every agreement metric still marks it down against the close tracker.
    assert scores["ccc_close"][0] > scores["ccc_biased"][0]
    assert scores["nse_close"][0] > scores["nse_biased"][0]
    assert scores["kge_close"][0] > scores["kge_biased"][0]
    # Nash-Sutcliffe is 1.0 for a perfect fit and can go negative when a forecast is
    # worse than simply predicting the mean.
    assert scores["nse_close"][0] > 0.9
    assert scores["nse_biased"][0] < 0.0


if __name__ == "__main__":
    main()
