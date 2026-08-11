"""Threading primitives the whole tree may use — layer 0, no engine, no plan.

Today: carrying a caller's `contextvars` context onto a worker thread, which is what keeps
a `config_context` applying to work that is handed off rather than run inline.
"""

from __future__ import annotations

from batcher._internal.concurrency.context import bound_to_context, start_context_thread

__all__ = ["bound_to_context", "start_context_thread"]
