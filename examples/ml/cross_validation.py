"""K-fold cross-validation on real data.

One train/test split gives one number with no error bar. K-fold gives k numbers, and the
spread between them tells you whether the model is stable or whether you got lucky with
the split — which is usually the more actionable fact.

    python examples/ml/cross_validation.py
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher import col, ml


def main() -> None:
    lineitem = tpch("lineitem").select("l_quantity", "l_extendedprice", "l_discount").head(20_000)

    folds = list(lineitem.ml.kfold(k=5, seed=5))
    print("folds:", len(folds))
    assert len(folds) == 5

    scores: list[float] = []
    for index, (train, validate) in enumerate(folds):
        # Each fold's validation set is disjoint from its training set.
        assert train.count() + validate.count() == lineitem.count()

        model = ml.Ridge(["l_quantity", "l_discount"], "l_extendedprice", alpha=1.0).fit(train)
        scored = model.predict(validate)
        mae = (
            scored.select(e=(col("prediction") - col("l_extendedprice")).abs())
            .agg(m=col("e").mean())
            .to_pydict()["m"][0]
        )
        scores.append(mae)
        print(f"  fold {index}: MAE {mae:,.1f}")

    spread = max(scores) - min(scores)
    print(f"mean MAE {statistics.mean(scores):,.1f}, spread {spread:,.1f}")

    # Five folds of the same distribution should agree closely. A large spread here is
    # the signal that a single split would have been misleading.
    assert all(value > 0 for value in scores)
    assert spread < statistics.mean(scores) * 0.5


if __name__ == "__main__":
    main()
