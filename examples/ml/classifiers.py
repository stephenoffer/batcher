"""Classifiers that fit in the engine: naive Bayes, discriminant analysis, and baselines.

Reach for a dummy baseline first. A model that cannot beat "always predict the most
frequent class" is not a model, and on an imbalanced problem that baseline can look
deceptively strong on accuracy alone.

    python examples/ml/classifiers.py
"""

from __future__ import annotations

import batcher as bt
from batcher import ml


def main() -> None:
    # Two well-separated clusters: low values are class 0, high values are class 1.
    train = bt.from_pydict(
        {
            "f1": [1.0, 1.5, 2.0, 1.2, 8.0, 8.5, 9.0, 8.2],
            "f2": [1.0, 1.2, 0.8, 1.1, 9.0, 8.8, 9.2, 9.1],
            "label": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    features = ["f1", "f2"]

    def accuracy_of(model) -> float:
        scored = model.predict(train)
        return scored.select(a=bt.accuracy("label", "prediction")).to_pydict()["a"][0]

    # A baseline that always predicts the majority class.
    dummy = ml.DummyClassifier("label", strategy="most_frequent").fit(train)
    base = accuracy_of(dummy)
    print("dummy accuracy:", base)
    assert base == 0.5  # the classes are balanced here

    # Gaussian naive Bayes: continuous features, assumed independent per class.
    gnb = ml.GaussianNB(features, "label").fit(train)
    print("GaussianNB accuracy:", accuracy_of(gnb))
    assert accuracy_of(gnb) > base

    # Discriminant analysis: linear and quadratic decision boundaries.
    lda = ml.LinearDiscriminantAnalysis(features, "label").fit(train)
    qda = ml.QuadraticDiscriminantAnalysis(features, "label").fit(train)
    assert accuracy_of(lda) > base
    assert accuracy_of(qda) > base

    # Nearest centroid: assign the class whose mean is closest.
    nc = ml.NearestCentroid(features, "label").fit(train)
    assert accuracy_of(nc) > base

    # A regularized linear classifier, and logistic regression.
    rc = ml.RidgeClassifier(features, "label", alpha=1.0).fit(train)
    lr = ml.LogisticRegression(features, "label", max_iter=200).fit(train)
    assert accuracy_of(rc) > base
    assert accuracy_of(lr) > base

    # Count-valued features call for a different naive Bayes.
    counts = bt.from_pydict(
        {
            "w1": [5, 6, 4, 0, 1, 0],
            "w2": [0, 1, 0, 7, 6, 8],
            "topic": [0, 0, 0, 1, 1, 1],
        }
    )
    mnb = ml.MultinomialNB(["w1", "w2"], "topic", alpha=1.0).fit(counts)
    bnb = ml.BernoulliNB(["w1", "w2"], "topic", alpha=1.0, threshold=0.5).fit(counts)
    for model in (mnb, bnb):
        acc = model.predict(counts).select(a=bt.accuracy("topic", "prediction")).to_pydict()
        print(type(model).__name__, acc)
        assert acc["a"][0] >= 0.5


if __name__ == "__main__":
    main()
