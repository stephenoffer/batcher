"""Saving a fitted model and loading it back.

A fitted preprocessor carries statistics, and those statistics are the thing that must reach
serving unchanged. Round-tripping the model and re-checking its output on the same rows is
the test that catches a serialization gap.

    python examples/ml/model_persistence.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher import col, ml


def main() -> None:
    customer = tpch("customer").select("c_custkey", "c_acctbal", "c_nationkey")
    train, holdout = customer.ml.train_test_split(test_size=0.2, seed=9)

    scaler = ml.StandardScaler("c_acctbal").fit(train)
    before = scaler.transform(holdout).sort("c_custkey").to_pydict()["c_acctbal"]

    # A fitted estimator round-trips through a dict, which is the portable form.
    as_dict = ml.to_dict(scaler)
    print("serialized keys:", sorted(as_dict)[:5])
    assert isinstance(as_dict, dict)

    restored = ml.from_dict(as_dict)
    after = restored.transform(holdout).sort("c_custkey").to_pydict()["c_acctbal"]

    # The restored model produces identical output on identical input.
    assert len(before) == len(after)
    assert all(abs(a - b) < 1e-12 for a, b in zip(before, after, strict=True))

    # And through a file.
    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "scaler.json")
        ml.save(scaler, path)
        assert Path(path).exists()

        loaded = ml.load(path)
        from_file = loaded.transform(holdout).sort("c_custkey").to_pydict()["c_acctbal"]
        assert all(abs(a - b) < 1e-12 for a, b in zip(before, from_file, strict=True))

    # The statistics really came from the training half: the holdout does not centre.
    holdout_mean = sum(before) / len(before)
    print("holdout mean after scaling:", round(holdout_mean, 6))
    assert abs(holdout_mean) < 0.2
    assert holdout_mean != 0.0
    assert col is not None


if __name__ == "__main__":
    main()
