"""An Ibis bridge pilot: compile an Ibis expression with Ibis's public SQL compiler, run it here.

Not yet verified against a live Ibis installation; see tests/PENDING_VERIFICATION.md.
"""

from __future__ import annotations

from batcher.integrations.ibis.bridge import table, to_dataset

__all__ = ["table", "to_dataset"]
