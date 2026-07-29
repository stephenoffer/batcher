"""Masking, hashing, and encrypting a sensitive column.

These are ordinary expressions, so they run in Rust at full speed and compose with
everything else. Pick by what you need back: masking is one-way and readable, hashing is
one-way and joinable, encryption is reversible with the key.

    python examples/governance/pii_transforms.py
"""

from __future__ import annotations

import os

import batcher as bt
from batcher import col


def main() -> None:
    # A `env:` reference is resolved at execution time. Set one here so the script is
    # self-contained; in production this comes from the deployment, not the code.
    os.environ.setdefault("DEMO_HMAC_KEY", "supersecretvalue")

    people = bt.from_pydict(
        {
            "email": ["ada@example.com", "grace@example.com"],
            "card": ["4111111111111111", "4222222222222222"],
        }
    )

    protected = people.with_columns(
        # Readable but redacted: keep a tail so a human can still recognize a record.
        masked=bt.mask(col("email"), show_last=6),
        # Fully opaque, but stable -- equal inputs give equal outputs, so it still joins.
        hashed=col("email").str.sha256(),
        # Keyed, so the same value hashes differently in another system.
        keyed=bt.hmac_sha256(col("email"), key="env:DEMO_HMAC_KEY"),
        # Card numbers: keep the last four, the industry convention.
        last4=col("card").str.tail(4),
    ).to_pydict()

    print({k: v[0] for k, v in protected.items()})

    # Masking keeps the tail and hides the rest.
    assert protected["masked"][0] != "ada@example.com"
    assert protected["masked"][0].endswith("le.com")

    # Hashing is deterministic, which is what keeps a hashed key joinable.
    assert len(protected["hashed"][0]) == 64
    again = people.select(h=col("email").str.sha256()).to_pydict()
    assert again["h"] == protected["hashed"]
    # Different inputs, different digests.
    assert protected["hashed"][0] != protected["hashed"][1]

    assert protected["last4"] == ["1111", "2222"]

    # A hashed column still supports the join it was protecting.
    left = people.select(k=col("email").str.sha256(), email=col("email"))
    right = bt.from_pydict({"email": ["ada@example.com"]}).select(
        k=col("email").str.sha256(), flag=bt.lit(True)
    )
    joined = left.join(right, on="k").to_pydict()
    print("joined on the hash:", joined["email"])
    assert joined["email"] == ["ada@example.com"]

    # Secrets are referenced, never inlined: `env:`/`file:`/`cmd:` are resolved at
    # execution time so the key never enters the plan, the logs, or an explain output.
    keyed_plan = people.select(k=bt.hmac_sha256(col("email"), key="env:DEMO_HMAC_KEY")).explain()
    print(keyed_plan)
    # The literal secret value never reaches the plan, the logs, or an explain output --
    # only the reference does, and it is resolved at execution time.
    assert "supersecretvalue" not in keyed_plan


if __name__ == "__main__":
    main()
