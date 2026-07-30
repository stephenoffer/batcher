"""Site detection — which GPU cloud, which scheduler, and where the fast local disk is.

The invariant across all three: an environment with no signal must leave every default exactly
where it was. Detection here decides where a spill lands and how a job's shape is read, and a
confident wrong answer is worse than no answer — spilling to a mount that turns out to be tmpfs
relieves no memory at all, and sizing a stage to a node's physical device count over-subscribes
the moment two jobs share the node.
"""

from __future__ import annotations

import os

import pytest

from batcher._internal.site import provider, scheduler, scratch

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Strip every signal these probes read, so the host's own environment cannot leak in."""
    for name in list(os.environ):
        if name.startswith(("SLURM", "RUNPOD", "COREWEAVE", "LAMBDA", "CRUSOE", "NEBIUS")):
            monkeypatch.delenv(name, raising=False)
    for name in (
        "BATCHER_PROVIDER",
        "BATCHER_SCRATCH_DIR",
        "KUBERNETES_SERVICE_HOST",
        "RAY_ADDRESS",
        "RAY_NODE_IP_ADDRESS",
        "NODE_NAME",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "AWS_EXECUTION_ENV",
        "GOOGLE_CLOUD_PROJECT",
        "AZURE_CLIENT_ID",
        "NVIDIA_VISIBLE_DEVICES",
    ):
        monkeypatch.delenv(name, raising=False)
    provider.reset_provider_probe()
    scratch.reset_scratch_probe()
    yield
    provider.reset_provider_probe()
    scratch.reset_scratch_probe()


# --- Provider -----------------------------------------------------------------------------


def test_no_marker_reports_unknown_and_changes_nothing():
    assert provider.detect_provider() == "unknown"
    site = provider.site_profile()
    assert site.known is False
    assert site.neocloud is False
    assert site.scratch_hints == ()


def test_a_neocloud_marker_is_identified_with_its_scratch_hint(monkeypatch):
    monkeypatch.setenv("COREWEAVE_NODE_NAME", "gpu-h100-042")
    monkeypatch.setenv("COREWEAVE_REGION", "LAS1")
    provider.reset_provider_probe()
    site = provider.site_profile()
    assert site.provider == "coreweave"
    assert site.region == "LAS1"
    assert site.known is True
    assert site.neocloud is True
    assert "/ephemeral" in site.scratch_hints


def test_a_neocloud_wins_over_the_hyperscaler_it_resells(monkeypatch):
    # A neocloud node inside Kubernetes carries generic cloud markers too; the specific
    # answer is the one with the useful defaults, so it must be checked first.
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("CRUSOE_VM_ID", "vm-123")
    provider.reset_provider_probe()
    assert provider.detect_provider() == "crusoe"


def test_a_hyperscaler_is_identified_but_is_not_a_neocloud(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    provider.reset_provider_probe()
    site = provider.site_profile()
    assert site.provider == "aws"
    assert site.known is True
    assert site.neocloud is False


def test_the_override_names_a_platform_this_module_has_not_seen(monkeypatch):
    monkeypatch.setenv("BATCHER_PROVIDER", "some-new-gpu-cloud")
    provider.reset_provider_probe()
    assert provider.detect_provider() == "some-new-gpu-cloud"
    site = provider.site_profile()
    assert site.provider == "some-new-gpu-cloud"
    # An unlisted platform gets no invented hints.
    assert site.scratch_hints == ()


def test_every_provider_entry_is_well_formed():
    names = [p.name for p in provider.PROVIDERS]
    assert len(names) == len(set(names))
    for spec in provider.PROVIDERS:
        assert spec.markers, spec.name
        assert spec.name == spec.name.lower()


def test_site_summary_is_flat(monkeypatch):
    monkeypatch.setenv("LAMBDA_INSTANCE_ID", "i-1")
    provider.reset_provider_probe()
    summary = provider.site_summary()
    assert summary["provider"] == "lambda"
    assert summary["neocloud"] is True
    assert summary["scheduler"] == "none"


# --- Scheduler ----------------------------------------------------------------------------


def test_nothing_scheduling_reports_none():
    job = scheduler.scheduler_job()
    assert job.kind == "none"
    assert job.nodes == ()
    assert job.multi_node is False
    assert job.total_gpus == 0


def test_slurm_allocation_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "912")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "gpu-[001-004]")
    monkeypatch.setenv("SLURM_GPUS_ON_NODE", "8")
    monkeypatch.setenv("SLURM_NTASKS", "4")
    monkeypatch.setenv("SLURM_PROCID", "2")
    monkeypatch.setenv("SLURM_LOCALID", "0")
    monkeypatch.setenv("SLURMD_NODENAME", "gpu-003")
    monkeypatch.setenv("SLURM_JOB_PARTITION", "h100")
    job = scheduler.scheduler_job()
    assert job.kind == "slurm"
    assert job.nodes == ("gpu-001", "gpu-002", "gpu-003", "gpu-004")
    assert job.multi_node is True
    assert job.total_gpus == 32
    assert (job.rank, job.local_rank, job.tasks) == (2, 0, 4)
    assert job.partition == "h100"


def test_the_allocation_grant_is_preferred_over_a_device_id_list(monkeypatch):
    # `SLURM_GPUS_ON_NODE` is what the job was granted; falling back to the id list keeps
    # older Slurm working, and both are the allocation rather than the node's hardware.
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "gpu-001")
    monkeypatch.setenv("SLURM_JOB_GPUS", "0,1,2")
    assert scheduler.scheduler_job().gpus_per_node == 3
    monkeypatch.setenv("SLURM_GPUS_ON_NODE", "8")
    assert scheduler.scheduler_job().gpus_per_node == 8


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("", ()),
        ("node1", ("node1",)),
        ("gpu-[001-003]", ("gpu-001", "gpu-002", "gpu-003")),
        ("gpu-[01-02,07]", ("gpu-01", "gpu-02", "gpu-07")),
        ("gpu-[1-2],login01", ("gpu-1", "gpu-2", "login01")),
        ("a[1-2],b[3-4]", ("a1", "a2", "b3", "b4")),
        # A duplicate in the list is not two nodes.
        ("n1,n1", ("n1",)),
    ],
)
def test_hostlist_expansion_covers_the_shapes_slurm_emits(spec, expected):
    assert scheduler.expand_nodelist(spec) == expected


def test_zero_padding_is_preserved_because_it_is_the_node_name():
    # `gpu-[008-010]` are nodes gpu-008..gpu-010, not gpu-8..gpu-10; an unpadded name does
    # not resolve, and the failure is a job that cannot reach three quarters of itself.
    assert scheduler.expand_nodelist("gpu-[008-010]") == ("gpu-008", "gpu-009", "gpu-010")


def test_a_malformed_range_is_passed_through_not_silently_dropped():
    # Yielding no nodes for a list we failed to parse would read as an empty allocation.
    assert scheduler.expand_nodelist("gpu-[5-2]") == ("gpu-5-2",)
    assert scheduler.expand_nodelist("gpu-[x-y]") == ("gpu-x-y",)


def test_kubernetes_reports_the_node_the_pod_landed_on(monkeypatch):
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
    monkeypatch.setenv("NODE_NAME", "gpu-node-7")
    monkeypatch.setenv("POD_NAMESPACE", "training")
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "0,1,2,3")
    job = scheduler.scheduler_job()
    assert job.kind == "kubernetes"
    assert job.node_name == "gpu-node-7"
    assert job.nodes == ("gpu-node-7",)
    assert job.gpus_per_node == 4
    assert job.partition == "training"


def test_the_outer_scheduler_wins_over_ray(monkeypatch):
    # A Ray worker inside a Slurm allocation is bounded by the allocation, which is what
    # will end it, so that is the kind reported.
    monkeypatch.setenv("RAY_ADDRESS", "auto")
    assert scheduler.scheduler_kind() == "ray"
    monkeypatch.setenv("SLURM_JOB_ID", "7")
    assert scheduler.scheduler_kind() == "slurm"


def test_a_pod_with_no_downward_api_reports_kind_without_inventing_identity(monkeypatch):
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
    job = scheduler.scheduler_job()
    assert job.kind == "kubernetes"
    assert job.node_name == ""
    assert job.nodes == ()


# --- Scratch ------------------------------------------------------------------------------


def test_no_local_volume_leaves_the_tempdir_default_alone(monkeypatch):
    monkeypatch.setattr(scratch, "SCRATCH_CANDIDATES", ())
    scratch.reset_scratch_probe()
    assert scratch.scratch_volumes() == ()
    assert scratch.local_scratch_root() is None


def test_a_measured_volume_is_preferred_and_ranked_by_device_class(monkeypatch, tmp_path):
    fast, slow = tmp_path / "nvme", tmp_path / "net"
    fast.mkdir()
    slow.mkdir()
    monkeypatch.setattr(scratch, "SCRATCH_CANDIDATES", (str(slow), str(fast)))
    monkeypatch.setattr(scratch, "_MIN_USEFUL_BYTES", 0)
    classes = {str(fast): "nvme", str(slow): "network"}
    monkeypatch.setattr(
        "batcher._internal.hardware.storage.device_class", lambda p: classes.get(p, "unknown")
    )
    scratch.reset_scratch_probe()
    volumes = scratch.scratch_volumes()
    assert [v.device_class for v in volumes] == ["nvme", "network"]
    assert scratch.local_scratch_root() == str(fast)


def test_a_tmpfs_is_never_chosen_as_scratch(monkeypatch, tmp_path):
    # The trap: /dev/shm is fast and often large, and spilling to it relieves nothing,
    # so the query OOMs at exactly the point it spilled to avoid that.
    shm = tmp_path / "shm"
    shm.mkdir()
    monkeypatch.setattr(scratch, "SCRATCH_CANDIDATES", (str(shm),))
    monkeypatch.setattr(scratch, "_MIN_USEFUL_BYTES", 0)
    monkeypatch.setattr("batcher._internal.hardware.storage.device_class", lambda p: "memory")
    scratch.reset_scratch_probe()
    assert scratch.scratch_volumes() == ()
    assert scratch.local_scratch_root() is None


def test_a_volume_too_small_to_matter_is_not_preferred(monkeypatch, tmp_path):
    small = tmp_path / "cfg"
    small.mkdir()
    monkeypatch.setattr(scratch, "SCRATCH_CANDIDATES", (str(small),))
    monkeypatch.setattr(scratch, "_MIN_USEFUL_BYTES", 1 << 60)
    monkeypatch.setattr("batcher._internal.hardware.storage.device_class", lambda p: "nvme")
    scratch.reset_scratch_probe()
    assert scratch.local_scratch_root() is None


def test_an_unwritable_candidate_is_rejected_before_it_is_ranked(monkeypatch, tmp_path):
    # `os.access` would pass here for root; only a write proves the mount takes bytes.
    monkeypatch.setattr(scratch, "SCRATCH_CANDIDATES", (str(tmp_path / "absent"),))
    scratch.reset_scratch_probe()
    assert scratch.local_scratch_root() is None


def test_the_operator_override_wins_over_measurement(monkeypatch, tmp_path):
    chosen = tmp_path / "chosen"
    chosen.mkdir()
    monkeypatch.setenv("BATCHER_SCRATCH_DIR", str(chosen))
    assert scratch.local_scratch_root() == str(chosen)
    # An override that is not usable falls back to the tempdir rather than to a guess.
    monkeypatch.setenv("BATCHER_SCRATCH_DIR", str(tmp_path / "absent"))
    monkeypatch.setattr(scratch, "SCRATCH_CANDIDATES", ())
    scratch.reset_scratch_probe()
    assert scratch.local_scratch_root() is None


def test_provider_hints_are_probed_before_the_generic_candidates(monkeypatch, tmp_path):
    hinted = tmp_path / "ephemeral"
    hinted.mkdir()
    monkeypatch.setattr(scratch, "SCRATCH_CANDIDATES", ())
    monkeypatch.setattr(scratch, "_MIN_USEFUL_BYTES", 0)
    monkeypatch.setattr(
        provider, "site_profile", lambda: provider.SiteProfile(scratch_hints=(str(hinted),))
    )
    monkeypatch.setattr("batcher._internal.hardware.storage.device_class", lambda p: "nvme")
    scratch.reset_scratch_probe()
    assert scratch.local_scratch_root() == str(hinted)


# --- The scratch decision reaches the paths that spill ------------------------------------


def test_the_spill_paths_prefer_the_measured_local_volume(monkeypatch, tmp_path):
    # The failure this closes: on a GPU node a bare tempdir is the container's overlay,
    # commonly under 100 GB, while the node's terabytes of NVMe sit under a provider mount.
    fast = tmp_path / "ephemeral"
    fast.mkdir()
    monkeypatch.setattr("batcher._internal.site.local_scratch_root", lambda: str(fast))

    from batcher.dist.flight_worker import _reduce_work_dir
    from batcher.dist.spill.scratch import _work_dir

    work, owned = _work_dir(None, "probe_")
    assert work.startswith(str(fast))
    assert owned is True
    assert _reduce_work_dir("probe_", None).startswith(str(fast))
    # An operator who named a directory has already decided.
    chosen = tmp_path / "chosen"
    chosen.mkdir()
    assert _reduce_work_dir("probe_", str(chosen)).startswith(str(chosen))


def test_no_local_volume_leaves_the_spill_paths_on_a_tempdir(monkeypatch, tmp_path):
    import tempfile

    monkeypatch.setattr("batcher._internal.site.local_scratch_root", lambda: None)
    from batcher.dist.flight_worker import _reduce_work_dir

    assert _reduce_work_dir("probe_", None).startswith(tempfile.gettempdir())


def test_the_spill_cost_model_prices_the_disk_the_engine_will_actually_use(monkeypatch, tmp_path):
    # The drift this closes: the spill paths resolve to the node's measured NVMe, so a cost
    # model still asking the tempdir prices the container's overlay — a factor of ten in the
    # wrong direction on exactly the machines where an out-of-core plan needs ranking.
    from batcher.kyber import storage_cost

    seen: list[str] = []
    monkeypatch.setattr(storage_cost, "device_class", lambda path: seen.append(path) or "network")
    monkeypatch.setattr("batcher._internal.site.local_scratch_root", lambda: "/ephemeral")
    assert storage_cost.spill_device_factor() == storage_cost.SPILL_DEVICE_FACTOR["network"]
    assert seen == ["/ephemeral"]
    # And with no local volume it falls back to the tempdir, as before.
    seen.clear()
    monkeypatch.setattr("batcher._internal.site.local_scratch_root", lambda: None)
    storage_cost.spill_device_factor()
    assert seen and seen[0] != "/ephemeral"
