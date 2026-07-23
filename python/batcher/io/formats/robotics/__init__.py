"""Robotics / ADAS log formats — the containers a vehicle or robot records into.

One file multiplexes every sensor as timestamped messages on named topics, so a row is a
*message* and the natural queries are per-topic and per-time-window. These containers are
indexed, which is what makes topic and time filters a seek rather than a scan.
"""

from __future__ import annotations

from batcher.io.formats.robotics.mcap import MCAP_SCHEMA, MCAPSource
from batcher.io.formats.robotics.mdf import MDF_SCHEMA, MDFSource

__all__ = ["MCAP_SCHEMA", "MDF_SCHEMA", "MCAPSource", "MDFSource"]
