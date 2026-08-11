"""PCA and truncated SVD on real numeric features.

Reducing dimensions is only worth it if the components keep the variance. Checking the
explained variance is what separates a useful reduction from throwing away the signal along
with the noise.

    python examples/ml/dimensionality_reduction.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher import col, ml


def main() -> None:
    features = (
        tpch("lineitem").select("l_quantity", "l_extendedprice", "l_discount", "l_tax").head(20_000)
    )
    columns = features.columns

    # Scale first: PCA maximizes variance, so an unscaled column with a bigger range wins
    # by default rather than on merit.
    scaler = ml.Chain(*[ml.StandardScaler(name) for name in columns]).fit(features)
    scaled = scaler.transform(features)

    pca = ml.PCA(columns, n_components=2).fit(scaled)
    reduced = pca.transform(scaled)
    print("reduced columns:", reduced.columns)
    assert reduced.count() == features.count()

    component_columns = [name for name in reduced.columns if name not in columns]
    assert len(component_columns) == 2

    # The components are uncorrelated with each other, which is what PCA guarantees.
    first, second = component_columns
    correlation = reduced.agg(r=__import__("batcher").corr(col(first), col(second))).to_pydict()[
        "r"
    ][0]
    print("component correlation:", round(correlation, 9))
    assert abs(correlation) < 1e-6

    # The first component carries more variance than the second, by construction.
    spreads = reduced.agg(first_var=col(first).var(), second_var=col(second).var()).to_pydict()
    print("variance:", round(spreads["first_var"][0], 4), round(spreads["second_var"][0], 4))
    assert spreads["first_var"][0] >= spreads["second_var"][0]

    # Truncated SVD is the sparse-friendly cousin, with the same shape of output.
    svd = ml.TruncatedSVD(columns, n_components=2).fit(scaled)
    projected = svd.transform(scaled)
    assert projected.count() == features.count()


if __name__ == "__main__":
    main()
