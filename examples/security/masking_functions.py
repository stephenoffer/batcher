"""The masking functions, and what each preserves.

A mask is a trade between utility and disclosure. Showing the last four keeps a value
recognizable to the person it belongs to; hashing keeps it joinable and nothing else;
redaction keeps only the fact that there was a value.

    python examples/security/masking_functions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    customer = tpch("customer").select("c_custkey", "c_phone").head(500)

    masked = customer.select(
        "c_phone",
        last_four=bt.mask(col("c_phone"), show_last=4),
        first_two=bt.mask(col("c_phone"), show_first=2),
        fully=bt.mask(col("c_phone")),
        hashed=col("c_phone").str.sha256(),
    )
    result = masked.head(3).to_pydict()
    for key in ("c_phone", "last_four", "first_two", "fully", "hashed"):
        print(f"  {key:<10} {result[key][0][:40]}")

    full = masked.to_pydict()

    # Length is preserved, so a downstream format check still passes.
    assert all(
        len(hidden) == len(original)
        for original, hidden in zip(full["c_phone"], full["last_four"], strict=True)
    )

    # The visible tail really is the original tail.
    assert all(
        hidden[-4:] == original[-4:]
        for original, hidden in zip(full["c_phone"], full["last_four"], strict=True)
    )
    assert all(
        hidden[:2] == original[:2]
        for original, hidden in zip(full["c_phone"], full["first_two"], strict=True)
    )

    # Full masking discloses nothing but the length.
    assert all(set(value) == {"X"} for value in full["fully"])

    # A hash is not reversible and is not length-preserving, but it is joinable: two equal
    # phone numbers hash the same and different ones do not.
    assert all(len(value) == 64 for value in full["hashed"])
    distinct_phones = len(set(full["c_phone"]))
    assert len(set(full["hashed"])) == distinct_phones

    # And the masked value is not joinable at that granularity, which is the trade.
    assert len(set(full["fully"])) < distinct_phones


if __name__ == "__main__":
    main()
