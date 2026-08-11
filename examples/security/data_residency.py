"""Data residency: refusing to process a dataset in the wrong region.

Residency resolves against a catalog of rules keyed by dataset prefix, longest match wins.
The verdict carries the obligation behind the refusal — a regulation, a contract — so a
blocked job explains itself instead of just failing.

The mode defaults to `off`, which makes every check pass. That is deliberate: a fleet
turns residency on in `advisory` to measure what it would block before moving to `strict`.
A catalog left at the default enforces nothing, so setting the mode is the whole control.

An unregistered dataset is *not* the same as one permitted everywhere, and an empty
permitted set is a deliberate quarantine rather than a mistake.

    python examples/security/data_residency.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batcher.governance import DataResidency, ResidencyCatalog


def main() -> None:
    # `off` is the default and passes everything; `strict` is what enforces.
    permissive = ResidencyCatalog()
    permissive.register(
        DataResidency("s3://eu-customers/", frozenset({"eu-north-1"}), "GDPR Art. 44")
    )
    assert permissive.check("s3://eu-customers/orders", "us-east-1").allowed

    catalog = (
        ResidencyCatalog(mode="strict")
        .register(DataResidency("s3://eu-customers/", frozenset({"eu-north-1"}), "GDPR Art. 44"))
        # A narrower prefix carries an exception; longest match wins.
        .register(
            DataResidency(
                "s3://eu-customers/public/",
                frozenset({"eu-north-1", "us-east-1"}),
                "published aggregates",
            )
        )
    )

    rule = catalog.rule_for("s3://eu-customers/orders")
    print("rule:", rule)
    assert rule is not None
    assert rule.allowed_regions == frozenset({"eu-north-1"})
    assert "GDPR" in rule.obligation

    # The narrower prefix wins for the paths it covers.
    exception = catalog.rule_for("s3://eu-customers/public/summary")
    assert exception is not None
    assert "us-east-1" in exception.allowed_regions

    # An unregistered dataset has no rule, which is distinct from an unrestricted one.
    assert catalog.rule_for("s3://public-data/reference") is None

    # The verdicts.
    permitted = catalog.check("s3://eu-customers/orders", "eu-north-1")
    refused = catalog.check("s3://eu-customers/orders", "us-east-1")
    print("in region :", permitted.allowed, permitted.message())
    print("out of region:", refused.allowed, refused.message())

    assert permitted.allowed
    assert not refused.allowed
    # The refusal names the region and the obligation, which is what makes it actionable.
    assert "us-east-1" in refused.message()
    assert "GDPR" in refused.message()

    # And the exception path is permitted in the wider set.
    assert catalog.check("s3://eu-customers/public/summary", "us-east-1").allowed


if __name__ == "__main__":
    main()
