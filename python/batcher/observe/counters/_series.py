"""The two primitives every fold needs to turn an event field into a metric series.

Both existed three times over before this module: each fold re-implemented "read a number
off an untrusted event payload" and "escape a label value for the exposition format". Three
copies of an escaper is how one of them silently stops escaping a character the others
handle — and an unescaped label value produces a line no scraper can parse, which drops the
*whole* exposition rather than the one series that was wrong.
"""

from __future__ import annotations

__all__ = ["as_number", "escape_label"]


def as_number(value: object) -> float:
    """`value` as a float, or `0.0` when it is missing or not a number.

    Events cross a JSON boundary and are published by many subsystems, so a field may be
    absent, `None`, or a string. A malformed payload must degrade to an unmeasured zero
    rather than raise inside a bus sink, which by contract cannot fail a query.

    Booleans are *not* numbers here. Python says `True == 1`, so without this a `spilled`
    flag folded into a byte counter would read as one byte.

    Args:
        value: Whatever the event carried.

    Returns:
        The value as a float, or `0.0`.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def escape_label(value: str) -> str:
    """Escape `value` for use as a Prometheus label value.

    The three characters the text format reserves in a label: a backslash, a double quote,
    and a newline. Names reaching here are not always tame — a regex data-quality constraint
    embeds its pattern, and a streaming query is named by its author.

    Args:
        value: The raw label value.

    Returns:
        The escaped value, safe to place inside double quotes.
    """
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
