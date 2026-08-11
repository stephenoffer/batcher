"""Converting between dates, timestamps and epoch integers.

An epoch integer is what a wire format carries, and its unit is the thing that gets lost.
Milliseconds read as seconds puts you in 1970; seconds read as milliseconds puts you fifty
thousand years out. Both are silent, so convert explicitly.

    python examples/expr_temporal/epochs_and_timestamps.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders").select("o_orderdate").head(200)

    converted = orders.select(
        "o_orderdate",
        stamp=col("o_orderdate").cast("timestamp"),
    ).with_columns(
        millis=col("stamp").dt.epoch_ms(),
        micros=col("stamp").dt.epoch_us(),
        nanos=col("stamp").dt.epoch_ns(),
    )

    result = converted.head(2).to_pydict()
    print(result)

    full = converted.to_pydict()
    # The three units are exact multiples of each other.
    assert all(
        micro == milli * 1_000 for milli, micro in zip(full["millis"], full["micros"], strict=True)
    )
    assert all(
        nano == micro * 1_000 for micro, nano in zip(full["micros"], full["nanos"], strict=True)
    )

    # These dates are all after the epoch, so the counts are positive.
    assert all(value > 0 for value in full["millis"])

    # Round trip: a date cast to a timestamp and back is the same date.
    round_trip = converted.select("o_orderdate", back=col("stamp").dt.date()).to_pydict()
    assert round_trip["back"] == round_trip["o_orderdate"]

    # The timestamp sits at midnight, so its time parts are zero.
    parts = converted.select(
        hour=col("stamp").dt.hour(), minute=col("stamp").dt.minute()
    ).to_pydict()
    assert set(parts["hour"]) == {0}
    assert set(parts["minute"]) == {0}


if __name__ == "__main__":
    main()
