"""Classifying order priority from order features.

The label is a string with five values, so this is multi-class. Everything that matters
happens before the model: the split is done first, the encoder is fitted on the training
half only, and the same fitted encoder is applied to the test half.

    python examples/ml/classification_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher import col, ml


def main() -> None:
    orders = tpch("orders").select("o_totalprice", "o_orderstatus", "o_orderpriority").head(20_000)

    train, test = orders.ml.train_test_split(test_size=0.25, seed=3)

    # Fit the encoders on the training half only. Fitting on everything leaks the test
    # set's category distribution into the model.
    status = ml.OrdinalEncoder("o_orderstatus").fit(train)
    label = ml.LabelEncoder("o_orderpriority").fit(train)

    encoded_train = label.transform(status.transform(train))
    encoded_test = label.transform(status.transform(test))
    print(encoded_train.columns)

    # `LogisticRegression` fits one weight vector, so it separates two classes and no more.
    # This label has five. `OneVsRestClassifier` fits one binary model per class and takes
    # the highest score, which is what makes a linear model answer a multi-class question.
    model = ml.OneVsRestClassifier(
        ml.LogisticRegression,
        ["o_totalprice", "o_orderstatus"],
        "o_orderpriority",
        params={"max_iter": 50},
    ).fit(encoded_train)

    scored = model.predict(encoded_test)
    assert "prediction" in scored.columns
    assert scored.count() == encoded_test.count()

    accuracy = (
        scored.select(hit=(col("prediction") == col("o_orderpriority")).cast("int64"))
        .agg(rate=col("hit").mean())
        .to_pydict()["rate"][0]
    )
    print(f"accuracy {accuracy:.4f}")

    # Priority is close to independent of price in TPC-H, so a real model scores near the
    # majority-class rate. Asserting it beats chance would be asserting a fiction; what
    # must hold is that it is a valid probability and predicts only seen classes.
    assert 0.0 <= accuracy <= 1.0
    seen = set(encoded_train.to_pydict()["o_orderpriority"])
    assert set(scored.to_pydict()["prediction"]) <= seen

    # The assertions above are all satisfied by a model that answers one class for every
    # row, which is what this script used to build: a binary model fitted on a five-class
    # label converges and predicts nearly a constant. Pinning one sub-model per class is
    # what distinguishes a multi-class model from a broken one.
    assert len(model.estimators_) == len(seen) == len(model.classes_)


if __name__ == "__main__":
    main()
