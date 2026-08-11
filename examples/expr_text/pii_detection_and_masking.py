"""Finding and masking personal data in a text column.

Masking keeps the shape of the value while destroying its content, which is what makes a
masked dataset still useful for debugging. Detection and masking are separate steps on
purpose: you usually want to count what you found before you destroy it.

    python examples/expr_text/pii_detection_and_masking.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from batcher import col


def main() -> None:
    tickets = bt.from_pydict(
        {
            "id": [1, 2, 3, 4],
            "note": [
                "Customer ada@example.com asked about the invoice.",
                "Call back on (555) 010-0199 tomorrow morning.",
                "See https://internal.example.com/tickets/91 for history.",
                "No contact details in this one.",
            ],
        }
    )

    # Detect first, so the counts are recorded before anything is destroyed.
    flagged = tickets.select(
        "id",
        has_email=col("note").str.has_email(),
        has_phone=col("note").str.has_phone(),
        has_url=col("note").str.has_url(),
        emails=col("note").str.email_count(),
        phones=col("note").str.phone_count(),
    )
    found = flagged.to_pydict()
    print(found)

    assert found["has_email"] == [True, False, False, False]
    assert found["has_phone"] == [False, True, False, False]
    assert found["has_url"] == [False, False, True, False]
    assert found["emails"] == [1, 0, 0, 0]

    # Then mask. The sentence still reads; the identifier does not.
    masked = tickets.select(
        "id",
        note=col("note").str.mask_emails().str.mask_urls().str.remove_phones(),
    ).to_pydict()
    for value in masked["note"]:
        print(repr(value))

    assert "ada@example.com" not in masked["note"][0]
    assert "010-0199" not in masked["note"][1]
    assert "internal.example.com" not in masked["note"][2]
    # The row with nothing to mask is untouched.
    assert masked["note"][3] == tickets.to_pydict()["note"][3]

    # A stable pseudonym, for when rows must still be joinable after masking.
    pseudonymous = tickets.select("id", token=col("note").str.sha256()).to_pydict()
    assert len(set(pseudonymous["token"])) == 4
    assert all(len(value) == 64 for value in pseudonymous["token"])


if __name__ == "__main__":
    main()
