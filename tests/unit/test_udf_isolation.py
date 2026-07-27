"""A UDF must not be able to read the engine's credentials.

The exploit these guard against is two lines long. `map_batches` runs a UDF in a
`forkserver` child, children inherit the parent's environment, and the engine's
environment is exactly where credentials live — `bc-secrets` resolves `env:NAME`
references so a plan never carries a secret inline, and `BATCHER_SECRET_COMMAND` names
the operator's helper for fetching arbitrary secrets from Vault, AWS, or GCP.

So the first test here is not a test of a feature. It is the exploit, asserted to fail.

What is deliberately **not** claimed: this is not a sandbox. A UDF is arbitrary Python and
can reach any syscall through `ctypes`. These tests pin the accidental exposure being
closed and the resource ceilings working — not a containment property the engine does not
have and does not claim.
"""

from __future__ import annotations

import os
import sys

import pytest

from batcher.core.udf.isolation import (
    DEFAULT_ENV_ALLOWLIST,
    ResourceLimits,
    _apply_limits,
    child_initializer,
    resolve_isolation,
    shard_directory,
)

pytestmark = pytest.mark.unit

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions/rlimits")


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch):
    """An environment carrying a credential, a secret helper, and a legitimate variable."""
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "super-secret")
    monkeypatch.setenv("BATCHER_SECRET_COMMAND", "/usr/local/bin/fetch-any-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-a-real-key")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    return None


class TestEnvironmentScrub:
    def test_the_exploit_no_longer_works(self, clean_env: None) -> None:
        """A UDF child cannot read the credential the parent holds.

        This is the whole point of the module. `child_initializer` runs in the child
        before it takes any work; after it, the credential is simply not there.
        """
        child_initializer(DEFAULT_ENV_ALLOWLIST, ResourceLimits())
        assert os.environ.get("AWS_SECRET_ACCESS_KEY") is None
        assert os.environ.get("OPENAI_API_KEY") is None

    def test_the_secret_helper_is_dropped_too(self, clean_env: None) -> None:
        """`BATCHER_SECRET_COMMAND` is sharper than any single credential.

        It names a program that hands out secrets on request, so a child keeping it does
        not need to have inherited any *particular* secret to obtain one. It is dropped by
        prefix rather than by name, so a future `BATCHER_*` variable is covered without
        anyone remembering to add it.
        """
        child_initializer((*DEFAULT_ENV_ALLOWLIST, "BATCHER_SECRET_COMMAND"), ResourceLimits())
        assert os.environ.get("BATCHER_SECRET_COMMAND") is None

    def test_what_a_udf_legitimately_needs_survives(self, clean_env: None) -> None:
        # Scrubbing must not break the UDF. `PATH` finds binaries; the thread-pinning
        # variables are what keep N workers from each spinning a full BLAS pool, which is
        # the oversubscription `forkserver` was chosen to avoid.
        child_initializer(DEFAULT_ENV_ALLOWLIST, ResourceLimits())
        assert os.environ.get("PATH") == "/usr/bin:/bin"
        assert os.environ.get("OMP_NUM_THREADS") == "1"

    def test_an_operator_can_name_extra_variables(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # An allowlist that cannot be extended would force operators to turn isolation off
        # entirely to pass one application variable through.
        monkeypatch.setenv("ACME_REGION", "eu-west-1")
        child_initializer((*DEFAULT_ENV_ALLOWLIST, "ACME_REGION"), ResourceLimits())
        assert os.environ.get("ACME_REGION") == "eu-west-1"
        assert os.environ.get("AWS_SECRET_ACCESS_KEY") is None

    @posix_only
    def test_the_child_umask_is_private(self, clean_env: None) -> None:
        child_initializer(DEFAULT_ENV_ALLOWLIST, ResourceLimits())
        current = os.umask(0o022)  # read it back, then restore
        os.umask(current)
        assert current == 0o077


class TestResourceLimits:
    """The clamping and ordering logic, not the kernel's `setrlimit`.

    An earlier draft spawned a child per assertion to observe a real limit taking effect.
    That tested the standard library — slowly, and flakily, because a child with a lowered
    address space cannot always allocate enough to report back. What is actually worth
    pinning is the logic this module adds on top: never raise a hard limit, keep going when
    one limit is unsupported, and scrub the environment *before* any of it.
    """

    @posix_only
    def test_a_limit_is_clamped_to_the_inherited_hard_ceiling(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # An unprivileged process cannot raise its hard limit, so asking would raise where
        # lowering succeeds — and a raised exception here would skip every later limit.
        import resource

        applied: dict[int, tuple[int, int]] = {}
        monkeypatch.setattr(resource, "getrlimit", lambda _which: (512, 1024))
        monkeypatch.setattr(
            resource, "setrlimit", lambda which, pair: applied.update({which: pair})
        )
        _apply_limits(ResourceLimits(memory_bytes=1 << 40))
        assert applied[resource.RLIMIT_AS] == (1024, 1024), "must clamp to the hard limit"

    @posix_only
    def test_an_infinite_hard_limit_accepts_the_requested_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import resource

        applied: dict[int, tuple[int, int]] = {}
        monkeypatch.setattr(
            resource, "getrlimit", lambda _w: (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
        )
        monkeypatch.setattr(resource, "setrlimit", lambda w, pair: applied.update({w: pair}))
        _apply_limits(ResourceLimits(memory_bytes=2 * 1024**3))
        assert applied[resource.RLIMIT_AS][0] == 2 * 1024**3

    @posix_only
    def test_zero_means_leave_the_inherited_limit_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The default configuration must change nothing about how a UDF runs. Only
        # `RLIMIT_CORE` is touched unconditionally, because a core dump of a UDF child
        # writes the whole address space to disk.
        import resource

        applied: dict[int, tuple[int, int]] = {}
        monkeypatch.setattr(resource, "getrlimit", lambda _w: (0, resource.RLIM_INFINITY))
        monkeypatch.setattr(resource, "setrlimit", lambda w, pair: applied.update({w: pair}))
        _apply_limits(ResourceLimits())
        assert set(applied) == {resource.RLIMIT_CORE}
        assert applied[resource.RLIMIT_CORE] == (0, 0)

    @posix_only
    def test_one_unsupported_limit_does_not_stop_the_others(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A platform rejecting one rlimit must not silently drop the rest."""
        import resource

        applied: dict[int, tuple[int, int]] = {}

        def flaky(which, pair):
            if which == resource.RLIMIT_AS:
                raise OSError("unsupported on this platform")
            applied[which] = pair

        monkeypatch.setattr(resource, "getrlimit", lambda _w: (0, resource.RLIM_INFINITY))
        monkeypatch.setattr(resource, "setrlimit", flaky)
        _apply_limits(ResourceLimits(memory_bytes=1 << 30, cpu_seconds=60))
        assert resource.RLIMIT_CPU in applied, "a later limit was skipped after an earlier failure"

    def test_the_scrub_happens_before_any_limit(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The security half must not depend on the resource half succeeding.

        If `_apply_limits` raises outright, the credentials must already be gone — which is
        only true because `child_initializer` rebuilds the environment first.
        """
        import batcher.core.udf.isolation as isolation

        def explode(_limits):
            raise RuntimeError("rlimits are broken on this platform")

        monkeypatch.setattr(isolation, "_apply_limits", explode)
        with pytest.raises(RuntimeError):
            child_initializer(DEFAULT_ENV_ALLOWLIST, ResourceLimits(memory_bytes=1 << 30))
        assert os.environ.get("AWS_SECRET_ACCESS_KEY") is None


class TestShardDirectory:
    @posix_only
    def test_the_shard_directory_is_private(self) -> None:
        """The input shards ARE the query's data.

        They were written as `bcudf_<pid>_<n>_<g>.arrow` directly into world-writable
        `/dev/shm` under the default umask — mode 0644 — so any local user could read a
        running query's batches. On a shared box that is a data leak needing no query
        access at all.
        """
        path = shard_directory()
        try:
            assert os.path.isdir(path)
            assert os.stat(path).st_mode & 0o077 == 0, "shard directory is group/other readable"
        finally:
            with pytest.raises(OSError) if os.listdir(path) else _noop():
                os.rmdir(path)

    @posix_only
    def test_shards_written_into_it_are_private(self) -> None:
        import pyarrow as pa

        from batcher.core.udf.processes import _input_shards

        batches = [pa.record_batch({"a": pa.array([1, 2, 3], type=pa.int64())})]
        paths, _size = _input_shards(batches, 1)
        try:
            for path in paths:
                assert os.stat(path).st_mode & 0o077 == 0, f"{path} is readable by others"
        finally:
            for path in paths:
                os.remove(path)
            os.rmdir(os.path.dirname(paths[0]))


class TestConfigResolution:
    def test_the_default_is_environment_isolation(self) -> None:
        from batcher.config import active_config

        mode, allowed, limits = resolve_isolation(active_config())
        assert mode == "env"
        assert "PATH" in allowed
        # Default limits change nothing about how a UDF runs; only "strict" applies them.
        assert limits.memory_bytes == 0

    def test_opting_out_is_possible(self) -> None:
        import dataclasses

        from batcher.config import active_config

        config = active_config()
        opted_out = config.replace(
            execution=dataclasses.replace(config.execution, udf_isolation="none")
        )
        mode, _allowed, _limits = resolve_isolation(opted_out)
        assert mode == "none"


class _noop:
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False
