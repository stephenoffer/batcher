"""Losses that score a probability or a margin rather than a hard label.

A classifier that says "0.51" and one that says "0.99" both predict the positive class,
but they are not equally right. These losses read the score column, which is what you need
to tell a confident model from a lucky one.

    python examples/metrics/probabilistic_losses.py
"""

from __future__ import annotations

import batcher as bt


def main() -> None:
    scored = bt.from_pydict(
        {
            "y_true": [1, 1, 0, 0],
            # A confident, mostly-correct model.
            "good": [0.95, 0.80, 0.10, 0.05],
            # A hedging model: right side of 0.5, but barely.
            "timid": [0.55, 0.52, 0.48, 0.45],
        }
    )

    losses = scored.select(
        log_loss_good=bt.log_loss("y_true", "good"),
        log_loss_timid=bt.log_loss("y_true", "timid"),
        brier_good=bt.brier_score("y_true", "good"),
        brier_timid=bt.brier_score("y_true", "timid"),
        hinge=bt.hinge_loss("y_true", "good"),
        squared_hinge=bt.squared_hinge_loss("y_true", "good"),
    ).to_pydict()

    print(losses)

    # Confidence is rewarded: the same hard labels, a much lower loss.
    assert losses["log_loss_good"][0] < losses["log_loss_timid"][0]
    assert losses["brier_good"][0] < losses["brier_timid"][0]
    # Brier is a mean squared error on probabilities, so it stays in [0, 1].
    assert 0.0 <= losses["brier_good"][0] <= 1.0
    assert losses["hinge"][0] >= 0.0
    assert losses["squared_hinge"][0] >= 0.0

    # Count-target deviances, for Poisson/gamma/Tweedie-style regressions.
    counts = bt.from_pydict(
        {
            "y_true": [1.0, 2.0, 3.0, 10.0],
            "y_pred": [1.2, 1.8, 3.3, 9.0],
        }
    )
    dev = counts.select(
        poisson=bt.poisson_deviance("y_true", "y_pred"),
        gamma=bt.gamma_deviance("y_true", "y_pred"),
        tweedie=bt.tweedie_deviance("y_true", "y_pred", power=1.5),
    ).to_pydict()
    print(dev)
    assert all(v[0] >= 0.0 for v in dev.values())


if __name__ == "__main__":
    main()
