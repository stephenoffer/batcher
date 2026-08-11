"""A streaming query runs under the config it was started with, on its own thread.

Two independent holes, both silent.

`threading.Thread` does not copy context variables, and the control plane keeps everything
that answers "what does this query think the machine looks like" in one: the active
`Config`, the cancellation scope, the machine-scoping key learned statistics are filed
under. So a `config_context` wrapped around `write_stream(...)` governed the setup and then
stopped applying the moment the loop thread started -- a pinned `max_memory_bytes` reverted
to the static fallback, an adjusted morsel size reverted to 16,384 rows, with no error
anywhere.

And `start_streaming_query` carried no `with_auto_config`, unlike every batch terminal, so
`max_memory_bytes` was never sensed at all. `spill_budget_bytes()` then fell back to the
static 8 GiB `default_total_bytes` -- the figure the data plane's spill backstop and every
streaming operator's state cap derive from. A batch query on the same box got the real
envelope.
"""

from __future__ import annotations

import dataclasses
import threading

import pytest

from batcher.config import Config, active_config, config_context

pytestmark = pytest.mark.unit

_MIB = 1 << 20


# --- the mechanism ------------------------------------------------------------


def test_a_bare_thread_does_not_inherit_the_active_config() -> None:
    """The negative control, and the reason the fix is needed at all.

    If this ever starts passing, `contextvars` has changed its threading semantics and the
    snapshot in `StreamingQueryEngine.start` is no longer load-bearing.
    """
    base = Config()
    pinned = base.replace(memory=dataclasses.replace(base.memory, max_memory_bytes=123_456_789))
    seen: dict[str, object] = {}
    with config_context(pinned):
        thread = threading.Thread(target=lambda: seen.update(v=active_config().memory))
        thread.start()
        thread.join()
    assert seen["v"].max_memory_bytes != 123_456_789


def test_a_context_snapshot_carries_the_config_across_the_thread() -> None:
    """What `StreamingQueryEngine.start` does, on the smallest possible subject."""
    import contextvars

    base = Config()
    pinned = base.replace(memory=dataclasses.replace(base.memory, max_memory_bytes=123_456_789))
    seen: dict[str, object] = {}
    with config_context(pinned):
        thread = threading.Thread(
            target=contextvars.copy_context().run,
            args=(lambda: seen.update(v=active_config().memory),),
        )
        thread.start()
        thread.join()
    assert seen["v"].max_memory_bytes == 123_456_789


# --- the engine ---------------------------------------------------------------


def test_the_streaming_loop_thread_sees_the_launching_config() -> None:
    """End to end through the real engine: the loop thread reads the pinned envelope.

    The processor is a stub that records what the *loop thread* saw, because that is the
    thread every micro-batch executes on and the one that was reading defaults.
    """
    from batcher.core.streaming_query.engine import StreamingQueryEngine
    from batcher.plan.streaming import OutputMode, Trigger

    observed: list[int | None] = []
    done = threading.Event()

    class _RecordingProcessor:
        def process(self, batch):
            return []

        def finalize(self):
            observed.append(active_config().memory.max_memory_bytes)
            done.set()
            return []

    class _EmptySource:
        """One empty micro-batch, then end of stream, so the loop finalizes at once."""

        def iter_batches(self, *args, **kwargs):
            return iter(())

    base = Config()
    pinned = base.replace(memory=dataclasses.replace(base.memory, max_memory_bytes=321 * _MIB))
    with config_context(pinned):
        engine = StreamingQueryEngine(
            name="ctx-probe",
            source=_EmptySource(),
            sink=None,
            processor=_RecordingProcessor(),
            trigger=Trigger.once(),
            output_mode=OutputMode.APPEND,
        )
        engine.start()
    assert done.wait(timeout=30), "the streaming loop never reached finalize"
    engine.stop()
    assert observed == [321 * _MIB], (
        "the streaming loop thread ran under the default config, not the one the query "
        "was started with"
    )


# --- the envelope is sensed ----------------------------------------------------


def test_the_launcher_senses_the_memory_envelope() -> None:
    """`start_streaming_query` must resolve `max_memory_bytes` the way batch terminals do.

    Checked at the decorator, because invoking the launcher needs a real unbounded source
    and a sink; what matters is that the entry point is wrapped at all, since an unwrapped
    one leaves the whole query on the static 8 GiB fallback.
    """
    from batcher.api.streaming._launch import start_streaming_query

    assert getattr(start_streaming_query, "__wrapped__", None) is not None, (
        "start_streaming_query is not decorated with @with_auto_config, so a streaming "
        "query never senses its memory envelope"
    )


def test_the_sensed_envelope_beats_the_static_fallback() -> None:
    """The figure the fix is about: an unset cap falls back to a fixed 8 GiB.

    That number is neither the container's nor the host's, and it is what the spill
    backstop and every streaming state cap are derived from.
    """
    from batcher.api.orchestration import resolve_auto_config

    unset = Config()
    assert unset.memory.max_memory_bytes is None
    assert unset.spill_budget_bytes() == int(
        unset.memory.default_total_bytes * unset.memory.hard_limit
    )

    resolved = resolve_auto_config(unset)
    assert resolved.memory.max_memory_bytes is not None, "the envelope could not be sensed"
    assert resolved.spill_budget_bytes() != unset.spill_budget_bytes(), (
        "sensing produced exactly the static fallback, so the test proves nothing about "
        "this machine -- pick a box whose RAM is not 8 GiB"
    )
