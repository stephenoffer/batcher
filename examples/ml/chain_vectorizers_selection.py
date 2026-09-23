"""One fitted Chain that selects numeric features, scales them, and vectorizes text.

A tabular-plus-text model needs three kinds of preprocessing: drop the columns that carry no
signal, put the survivors on one scale, and turn the free text into a vector. `Chain` fits
them in order on the training split, then replays the fitted steps over any split, so the
held-out rows are transformed with the training set's vocabulary and statistics.

The chain is built with ``cache=False``: with the default ``cache=True`` it collects the
whole training split to the driver once, which is the right trade for a small split and the
wrong one for a large distributed dataset.

    python examples/ml/chain_vectorizers_selection.py
"""

from __future__ import annotations

import math

import batcher as bt
from batcher import ml


def main() -> None:
    train = bt.from_pydict(
        {
            "text": ["the cat sat", "the dog sat", "a cat and a dog", "dog dog dog"],
            "label": [1, 0, 1, 0],
            "length": [3.0, 3.0, 5.0, 3.0],
            "has_cat": [1.0, 0.0, 0.8, 0.1],
            "constant": [7.0, 7.0, 7.0, 7.0],
        }
    )
    test = bt.from_pydict(
        {
            "text": ["a cat", "an unseen word"],
            "label": [1, 0],
            "length": [2.0, 3.0],
            "has_cat": [1.0, 0.0],
            "constant": [7.0, 7.0],
        }
    )

    # Univariate selection on its own: which numeric column best separates the label?
    best = ml.SelectKBest("label", k=1, features=["length", "has_cat"]).fit(train)
    print("SelectKBest keeps", best.selected_)
    assert best.selected_ == ["has_cat"]  # it tracks the label; length barely moves

    # A bag of words learns its vocabulary in fit, sorted, from the training split only.
    counts = ml.CountVectorizer("text", dense=True).fit(train)
    print("vocabulary", counts.vocabulary_)
    assert counts.vocabulary_ == ["and", "cat", "dog", "sat", "the"]
    bag = counts.transform(train).to_pydict()["features"]
    assert bag[3] == [0.0, 0.0, 3.0, 0.0, 0.0]  # "dog dog dog"

    chain = ml.Chain(
        ml.VarianceThreshold(["length", "has_cat", "constant"]),
        ml.StandardScaler(["length", "has_cat"]),
        ml.TfidfVectorizer("text", dense=True),
        cache=False,
    ).fit(train)

    # The zero-variance column is gone, so nothing downstream sees it.
    train_out = chain.transform(train).to_pydict()
    test_out = chain.transform(test).to_pydict()
    print("columns", sorted(train_out))
    assert "constant" not in train_out and "constant" not in test_out

    # The scaler's statistics are the training split's: population std, like scikit-learn.
    scaler = chain[1]
    assert scaler.mean_["length"] == 3.5
    assert math.isclose(scaler.scale_["length"], math.sqrt(0.75))

    # Every TF-IDF row is L2-normalized; a test document with no training words is all zero.
    for row in train_out["features"]:
        assert math.isclose(sum(v * v for v in row), 1.0)
    assert test_out["features"][1] == [0.0] * 5
    assert len(test_out["features"][0]) == len(chain[2].vocabulary_)

    # The whole fitted chain saves as one JSON document and reloads identically.
    import os
    import tempfile

    path = os.path.join(tempfile.mkdtemp(), "chain.json")
    chain.save(path)
    reloaded = ml.Preprocessor.load(path)
    assert reloaded.transform(test).to_pydict() == test_out
    print("reloaded chain reproduces the test features")


if __name__ == "__main__":
    main()
