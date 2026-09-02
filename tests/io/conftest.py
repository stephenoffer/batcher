"""Keep the IO suite's routing independent of whether an earlier test started Ray.

`resolve_distributed("auto", ...)` consults the **live** Ray session, so a test's execution
path depends on whether some *other* test happened to call `ray.init` first. `test_interop`
does, which makes the rest of this directory order-dependent: run alone it executes locally,
run after `test_interop` it routes to whatever cluster is up.

That is not a theoretical ordering nit. Measured on this suite: fifteen tests across
`test_text_encoding`, `test_sql_dbapi_sink`, `test_sql_read_round_trips`,
`test_sql_uri_and_dbapi`, `test_lakehouse_merge` and `test_public_reader_writer_surface` pass
in isolation and fail in the full run, and the failure is a `RayTaskError` raised inside a
worker running a *stale package snapshot* -- a long-lived shared cluster keeps the copy of
Batcher it started with. So a text-encoding test failed for a reason with nothing to do with
text, encoding, or any code in the change under review.

`tests/conftest.py` already carries this fixture and scopes it to `tests/docs/`, with a
docstring naming this directory as one of the suites that starts Ray. This is that same
fixture, applied where it was already known to be needed. It lives here rather than being
widened there because a subdirectory `conftest` is the narrowest place that can hold it, and
because widening the shared one would silently change suites nobody has measured.

**What this must not do is hide a distributed test.** It patches only the *auto-routing*
probe, so a test that asks for `distributed=True` explicitly still gets it. Exactly one file
here does (`test_streaming_write_file_size.py`), and it is unaffected.

`test_interop.py` is exempt, and finding out why is what makes the exemption trustworthy
rather than a convenience: it is the file that is *about* Ray, and
`test_from_ray_dataset_streams_blocks` reads a live `ray.data.Dataset`, so lying to it about
whether Ray is initialized breaks the one suite that legitimately asks. It is also the file
whose `ray.init` causes the ordering problem for everything else -- which is the whole shape
of the thing: one suite genuinely needs the session, and the other twenty must not inherit it.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _route_io_tests_locally(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Make `"auto"` resolve as it does on a machine with no cluster attached.

    No-op for `test_interop.py`, which is the suite about Ray itself, and where Ray is not
    installed, which is how CI runs.
    """
    if request.path.name == "test_interop.py":
        return
    try:
        import ray
    except ImportError:
        return
    monkeypatch.setattr(ray, "is_initialized", lambda: False)
