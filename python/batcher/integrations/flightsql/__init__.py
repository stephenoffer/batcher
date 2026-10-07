"""A Flight SQL service pilot over a Batcher `Session`, built on pyarrow Flight.

Not yet verified against a live Flight SQL client such as the ADBC or JDBC driver; see
tests/PENDING_VERIFICATION.md.
"""

from __future__ import annotations

from batcher.integrations.flightsql.server import serve

__all__ = ["serve"]
