"""K-means over real customer features.

Clustering is unsupervised, so the only honest checks are structural: every row gets a
label, the labels are in range, and the within-cluster spread is smaller than the overall
spread. Anything about what the clusters *mean* is interpretation, not a result.

    python examples/ml/clustering_customers.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col, ml


def main() -> None:
    customer = tpch("customer").select("c_custkey", "c_acctbal", "c_nationkey")

    # Scale first: k-means measures Euclidean distance, so an unscaled feature with a
    # bigger range silently dominates the clustering.
    scaler = ml.Chain(ml.StandardScaler("c_acctbal"), ml.StandardScaler("c_nationkey")).fit(
        customer
    )
    scaled = scaler.transform(customer)

    model = ml.KMeans(["c_acctbal", "c_nationkey"], n_clusters=4, seed=7).fit(scaled)
    labelled = model.predict(scaled)
    print(labelled.columns)

    # The label lands in `cluster`, not `prediction`: a clusterer assigns a group
    # rather than predicting a target, and the column name says so.
    sizes = labelled.value_counts("cluster").sort("cluster").to_pydict()
    print(sizes)

    # Every row is labelled, exactly once, with a label in range.
    assert sum(sizes["count"]) == customer.count()
    assert set(sizes["cluster"]) <= {0, 1, 2, 3}

    # Within-cluster spread is tighter than the overall spread — the point of clustering.
    overall = scaled.agg(sd=col("c_acctbal").std()).to_pydict()["sd"][0]
    per_cluster = (
        labelled.group_by("cluster").agg(sd=col("c_acctbal").std(), n=bt.count()).to_pydict()
    )
    print(
        "overall sd:", round(overall, 3), "cluster sds:", [round(v, 3) for v in per_cluster["sd"]]
    )
    assert min(per_cluster["sd"]) < overall


if __name__ == "__main__":
    main()
