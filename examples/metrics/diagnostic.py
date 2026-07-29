"""Diagnostic metrics: the epidemiology-style view of a binary classifier.

Accuracy hides everything on an imbalanced problem. Likelihood ratios, informedness, and
markedness describe how much a prediction actually moves your belief, which is the number
you want when positives are rare.

    python examples/metrics/diagnostic.py
"""

from __future__ import annotations

import batcher as bt


def main() -> None:
    # TP=3, FP=2, FN=1, TN=4 again, so these line up with the classification example.
    preds = bt.from_pydict(
        {
            "y_true": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            "y_pred": [1, 1, 1, 0, 1, 1, 0, 0, 0, 0],
        }
    )

    scores = preds.select(
        # Error rates among the *predicted* classes.
        fdr=bt.false_discovery_rate("y_true", "y_pred"),
        fomr=bt.false_omission_rate("y_true", "y_pred"),
        # How much a positive / negative call shifts the odds.
        plr=bt.positive_likelihood_ratio("y_true", "y_pred"),
        nlr=bt.negative_likelihood_ratio("y_true", "y_pred"),
        dor=bt.diagnostic_odds_ratio("y_true", "y_pred"),
        # Informedness combines the two recall rates; markedness the two precisions.
        informedness=bt.informedness("y_true", "y_pred"),
        markedness=bt.markedness("y_true", "y_pred"),
        # Balanced summaries that survive class imbalance.
        gmean=bt.geometric_mean_score("y_true", "y_pred"),
        fowlkes_mallows=bt.fowlkes_mallows_index("y_true", "y_pred"),
        jaccard=bt.jaccard_score("y_true", "y_pred"),
        prevalence_threshold=bt.prevalence_threshold("y_true", "y_pred"),
        hamming=bt.hamming_loss("y_true", "y_pred"),
    ).to_pydict()

    print(scores)

    # FDR = FP / (TP + FP) = 2/5; FOR = FN / (FN + TN) = 1/5.
    assert abs(scores["fdr"][0] - 0.4) < 1e-12
    assert abs(scores["fomr"][0] - 0.2) < 1e-12
    # A positive call is more likely under a true positive than a false one.
    assert scores["plr"][0] > 1.0
    assert scores["nlr"][0] < 1.0
    assert scores["dor"][0] > 1.0
    # Both stay in [-1, 1]; positive means better than chance.
    assert 0.0 < scores["informedness"][0] <= 1.0
    assert 0.0 < scores["markedness"][0] <= 1.0
    # Jaccard = TP / (TP + FP + FN) = 3/6.
    assert abs(scores["jaccard"][0] - 0.5) < 1e-12
    # Hamming loss is the misclassification rate: 3 of 10 wrong.
    assert abs(scores["hamming"][0] - 0.3) < 1e-12


if __name__ == "__main__":
    main()
