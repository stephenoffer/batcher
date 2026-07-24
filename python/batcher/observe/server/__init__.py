"""The Batcher UI — a local web dashboard for queries, plans, metrics, and logs.

Serves the single-page app in `../assets/` plus a read-only JSON API over an
`ActivityStore`, on its own port so it never interferes with the process it is observing.
The Spark UI analog: open it while a job runs and watch the plan, the per-operator
throughput, the spill and memory picture, and the log stream side by side.

`core` owns the lifecycle, `handler` the HTTP mechanics, and `routes` the API surface as a
table — so adding an endpoint is one entry rather than an edit in three places.
"""

from __future__ import annotations

from batcher.observe.server.core import UIServer
from batcher.observe.server.routes import ROUTES

__all__ = ["ROUTES", "UIServer"]
