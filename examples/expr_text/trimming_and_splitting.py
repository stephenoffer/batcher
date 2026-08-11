"""Trimming whitespace and splitting on a delimiter.

`strip` removes **spaces**, not all whitespace — a leading tab or a trailing newline
survives it, unlike Python's `str.strip()`. Pass the character set to `strip_chars` when
you mean all of it, which is almost always.

Splitting produces a list column, which keeps the row intact. `split_part` is the shortcut
when you only want one field, and it avoids materializing the whole list.

    python examples/expr_text/trimming_and_splitting.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from batcher import col


def main() -> None:
    records = bt.from_pydict(
        {
            "raw": [
                "  alice|engineering|london  ",
                "bob|sales|new york",
                "\tcarol|research|tokyo\n",
            ]
        }
    )

    cleaned = records.select(
        trimmed=col("raw").str.strip(),
        left_only=col("raw").str.lstrip(),
        right_only=col("raw").str.rstrip(),
    )
    result = cleaned.to_pydict()
    print([repr(value) for value in result["trimmed"]])

    # Spaces go; the tab and the newline on the third record do not.
    assert result["trimmed"][0] == "alice|engineering|london"
    assert result["trimmed"][2].startswith("\t")
    assert result["left_only"][0].endswith("  ")
    assert result["right_only"][0].startswith("  ")

    # `strip_chars` with an explicit set removes all of it.
    thorough = records.select(x=col("raw").str.strip_chars(" \t\n")).to_pydict()
    assert all(value == value.strip() for value in thorough["x"])

    # Splitting into a list keeps one row per record.
    fields = records.select(parts=col("raw").str.strip_chars(" \t\n").str.split("|"))
    assert fields.count() == records.count()
    assert fields.select(n=col("parts").list.len()).to_pydict()["n"] == [3, 3, 3]

    # `split_part` takes one field directly, 1-based.
    picked = records.select(
        name=col("raw").str.strip_chars(" \t\n").str.split_part("|", 1),
        team=col("raw").str.strip_chars(" \t\n").str.split_part("|", 2),
        city=col("raw").str.strip_chars(" \t\n").str.split_part("|", 3),
    ).to_pydict()
    print(picked)
    assert picked["name"] == ["alice", "bob", "carol"]
    assert picked["city"] == ["london", "new york", "tokyo"]

    # And it agrees with indexing the list.
    via_list = fields.select(name=col("parts").list.get(0)).to_pydict()
    assert via_list["name"] == picked["name"]


if __name__ == "__main__":
    main()
