"""String search: substring tests, multi-pattern tests, and match counting.

Every predicate here is a columnar expression, so the whole column is tested in Rust
rather than one Python call per row. ``contains_any``/``contains_all`` take an iterable
of patterns and fold to a single boolean column, which is what you want for a keyword
screen: one pass, not one pass per keyword.

    python examples/expressions/strings_search.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col


def main() -> None:
    tickets = bt.from_pydict(
        {
            "subject": [
                "Cannot login to account",
                "Refund request for order 8812",
                "login page is slow",
                "Where is my refund?",
            ],
        }
    )

    screened = tickets.with_columns(
        # Plain substring test, case-sensitive.
        mentions_login=col("subject").str.contains("login"),
        starts_upper=col("subject").str.starts_with("C"),
        ends_question=col("subject").str.ends_with("?"),
        # Any of several keywords -> one boolean column.
        is_billing=col("subject").str.contains_any(["refund", "Refund", "invoice"]),
        # All of several keywords must appear.
        login_and_slow=col("subject").str.contains_all(["login", "slow"]),
        # How many times a pattern occurs.
        vowel_runs=col("subject").str.count_matches("[aeiou]+"),
        # How many times one character occurs.
        spaces=col("subject").str.count_char(" "),
    )

    result = screened.to_pydict()
    print(result)

    assert result["mentions_login"] == [True, False, True, False]
    assert result["starts_upper"] == [True, False, False, False]
    assert result["ends_question"] == [False, False, False, True]
    assert result["is_billing"] == [False, True, False, True]
    assert result["login_and_slow"] == [False, False, True, False]
    # "login page is slow" -> o, i, a, e, i, o  (six vowel runs)
    assert result["vowel_runs"][2] == 6
    assert result["spaces"] == [3, 4, 3, 3]


if __name__ == "__main__":
    main()
