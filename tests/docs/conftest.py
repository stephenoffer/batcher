"""Doc examples run the way a reader runs them: one process, no attached cluster.

`resolve_distributed("auto", ...)` consults the *live* Ray session. In a test process that
makes the docs suite order-dependent: run it on its own and every example executes locally,
exactly as a reader would see it; run it after a suite that happened to start Ray (the io
and distributed tests do) and the same examples suddenly route to an 8-node cluster.

That is not a hypothetical. A doc example that defines its own `Source` — the custom
connector guide does, and it is the whole point of the page — cannot report a row count, so
"auto" takes the *unknown size means assume large* branch and distributes it. The source
class is defined in the doc block, so it does not exist on any worker, and the page fails
for a reason that has nothing to do with the page.

Pinning the docs suite to a reader's environment fixes it at the root and keeps the suite
deterministic regardless of collection order. A page that genuinely wants a cluster marks
its block `# docs: skip`; none of them execute one.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_inherited_cluster(monkeypatch):
    """Make routing see no Ray session, however earlier tests left the process."""
    try:
        import ray
    except ImportError:
        return
    monkeypatch.setattr(ray, "is_initialized", lambda: False)
