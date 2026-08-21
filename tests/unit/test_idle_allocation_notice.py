"""Using the allocation you were given, and being told when you are not.

`srun -N 64 python job.py` with no Ray running gives the job sixty-four nodes and runs the
whole query on whichever one the script landed on. It returns the right answer and uses a
sixty-fourth of the hardware it was billed for, and nothing anywhere said so — from Batcher's
side an unclustered process is a perfectly ordinary one.

The complement of the notice `dist` already gives when Ray *is* up but narrower than the
allocation. The assertions here are as much about the silence as the warning: a notice that
fires on a laptop, on a single-node allocation, or against an explicit `distributed=False` is
one every reader learns to skip, which costs the case it exists for.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean(clean_site_env, monkeypatch):
    """No scheduler signal from outside, and no Ray.

    `_ray_already_live` is a `sys.modules` lookup, so any earlier test in the process that
    imported Ray turns it True and this notice — whose whole condition is that no cluster is
    running — silently stops firing. Pinning it is what makes these tests about the notice
    rather than about which file pytest ran first.
    """
    from batcher.api.terminal import routing

    monkeypatch.setattr(routing, "_ray_already_live", lambda: False)
    monkeypatch.setattr(routing, "_IDLE_ALLOCATION_WARNED", False)


def test_a_multi_node_allocation_running_single_node_says_so(monkeypatch, caplog):
    # `srun -N 64 python job.py` with no Ray runs the whole query on one node. It returns the
    # right answer and uses a sixty-fourth of the hardware it was billed for, and nothing
    # anywhere said so -- from Batcher's side an unclustered process is an ordinary one.
    from batcher.api.terminal import routing

    monkeypatch.setenv("SLURM_JOB_ID", "1")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "gpu-[01-64]")
    with caplog.at_level("WARNING", logger="batcher.api"):
        assert routing.resolve_distributed("auto") is False
    assert "64 nodes" in caplog.text


def test_the_notice_is_given_once_not_once_per_query(monkeypatch, caplog):
    from batcher.api.terminal import routing

    monkeypatch.setenv("SLURM_JOB_ID", "1")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "gpu-[01-04]")
    with caplog.at_level("WARNING", logger="batcher.api"):
        for _ in range(3):
            routing.resolve_distributed("auto")
    assert caplog.text.count("no Ray cluster is running") == 1


@pytest.mark.parametrize(
    "env",
    [
        {},  # a laptop
        {"SLURM_JOB_ID": "1", "SLURM_JOB_NODELIST": "gpu-01"},  # one node is what was asked for
    ],
)
def test_nothing_is_said_when_there_is_nothing_to_say(monkeypatch, caplog, env):
    from batcher.api.terminal import routing

    for name, value in env.items():
        monkeypatch.setenv(name, value)
    with caplog.at_level("WARNING", logger="batcher.api"):
        routing.resolve_distributed("auto")
    assert "no Ray cluster is running" not in caplog.text


def test_an_explicit_single_node_choice_is_not_second_guessed(monkeypatch, caplog):
    # `distributed=False` is the user saying the one node is what they meant.
    from batcher.api.terminal import routing

    monkeypatch.setenv("SLURM_JOB_ID", "1")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "gpu-[01-64]")
    with caplog.at_level("WARNING", logger="batcher.api"):
        assert routing.resolve_distributed(False) is False
    assert caplog.text == ""


def test_the_allocation_is_looked_at_once_whatever_the_answer(monkeypatch):
    """A second look can only reach the same conclusion, and it is not free.

    On PBS, LSF or Grid Engine reading the job's shape means reading a host file. A
    single-node allocation never fires the notice, so a flag set only when it *does* fire
    would leave that file read on every terminal op for the life of the process.
    """
    from batcher.api.terminal import routing

    reads = []

    def counted():
        reads.append(1)
        from batcher._internal.site.scheduler import SchedulerJob

        return SchedulerJob(kind="pbs", nodes=("gpu01",))

    monkeypatch.setattr("batcher._internal.site.scheduler_job", counted)
    for _ in range(5):
        routing.resolve_distributed("auto")
    assert len(reads) == 1
