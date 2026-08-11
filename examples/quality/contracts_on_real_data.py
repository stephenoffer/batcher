"""Asserting a data contract against a real table.

The three endings are the whole API: `validate` reports, `drop` removes bad rows,
`quarantine` splits them out, and `fail` raises. Which one you want depends on whether a
violation is a data problem to route or a promise that must hold.

    python examples/quality/contracts_on_real_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher._internal.errors import DataQualityError


def main() -> None:
    lineitem = tpch("lineitem").select(
        "l_orderkey", "l_quantity", "l_discount", "l_returnflag", "l_shipdate"
    )

    # A contract the real data satisfies.
    report = (
        lineitem.dq.not_null("l_orderkey")
        .in_range("l_discount", 0.0, 0.1)
        .in_range("l_quantity", 1, 50)
        .accepted_values("l_returnflag", ["A", "N", "R"])
        .validate()
    )
    print(report)
    assert report.ok
    assert report.total_violations == 0

    # A contract it does not: discounts do not all exceed 5%.
    strict = lineitem.dq.in_range("l_discount", 0.05, 0.1).validate()
    print("violations:", strict.total_violations)
    assert not strict.ok
    assert strict.total_violations > 0

    # `drop` keeps the conforming rows.
    kept = lineitem.dq.in_range("l_discount", 0.05, 0.1).drop()
    assert kept.count() == lineitem.count() - strict.total_violations

    # `quarantine` keeps both sides, which is what a dead-letter sink needs.
    clean, rejected = lineitem.dq.in_range("l_discount", 0.05, 0.1).quarantine()
    assert clean.count() + rejected.count() == lineitem.count()
    assert rejected.count() == strict.total_violations

    # `fail` is the gate: it raises rather than returning bad data.
    try:
        lineitem.dq.in_range("l_discount", 0.05, 0.1).fail()
    except DataQualityError as error:
        print("gate raised:", str(error)[:80])
    else:
        raise AssertionError("expected DataQualityError")


if __name__ == "__main__":
    main()
