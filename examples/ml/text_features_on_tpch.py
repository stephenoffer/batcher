"""Turning a text column into numeric features without a model.

Hashing and count features are what you reach for when a transformer is too expensive and a
bag of words is enough. They are also deterministic and need no vocabulary file, which makes
them safe to compute at serving time.

    python examples/ml/text_features_on_tpch.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher import col, ml


def main() -> None:
    parts = tpch("part").select("p_partkey", "p_name").head(5_000)

    # Statistical text features: length, word counts, character classes.
    featurizer = ml.TextStatFeaturizer("p_name").fit(parts)
    statistical = featurizer.transform(parts)
    added = [name for name in statistical.columns if name not in parts.columns]
    print("statistical features:", added)
    assert added
    assert statistical.count() == parts.count()

    # No nulls, which is what a model needs.
    nulls = statistical.select(*added).null_count().to_pydict()
    assert all(value == 0 for column in nulls.values() for value in column)

    # Hashing: a fixed-width representation with no vocabulary to ship.
    hasher = ml.HashingEncoder("p_name", n_buckets=8).fit(parts)
    hashed = hasher.transform(parts)
    hash_columns = [name for name in hashed.columns if name not in parts.columns]
    print("hashed features:", len(hash_columns))
    assert hashed.count() == parts.count()

    # Deterministic: the same text hashes the same way twice.
    again = ml.HashingEncoder("p_name", n_buckets=8).fit(parts).transform(parts)
    assert again.sort("p_partkey").to_pydict() == hashed.sort("p_partkey").to_pydict()

    # And the expression-level features, for when a preprocessor is more than you need.
    direct = parts.select(
        "p_partkey",
        words=col("p_name").str.word_count(),
        length=col("p_name").str.len_chars(),
        vowels=col("p_name").str.count_char("a") + col("p_name").str.count_char("e"),
    )
    values = direct.to_pydict()
    assert all(value > 0 for value in values["words"])
    assert all(
        vowels <= length for vowels, length in zip(values["vowels"], values["length"], strict=True)
    )


if __name__ == "__main__":
    main()
