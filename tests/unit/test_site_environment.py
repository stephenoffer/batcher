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

#: The real firmware probe, captured before the autouse fixture silences it. The DMI tests
#: below put it back deliberately; everything else in this file is about the environment and
#: must not be answered by whatever machine the suite happens to run on.
_REAL_DMI = provider.dmi_identity.__wrapped__


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Strip every signal these probes read, so the host's own environment cannot leak in."""
    for name in list(os.environ):
        if name.startswith(
            ("SLURM", "PBS_", "LSB_", "RUNPOD", "COREWEAVE", "LAMBDA", "CRUSOE", "NEBIUS")
        ):
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
    # The firmware is a *fallback* identity, and this suite is about the environment. Silence
    # it by default so a test asserting "no marker means unknown" is not answered by the host
    # the suite happens to run on; the DMI tests below re-enable it deliberately.
    monkeypatch.setattr(provider, "dmi_identity", lambda: ("", "", None))
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
    classes = {str(fast): "nvme", str(slow): "rotational"}
    monkeypatch.setattr(
        "batcher._internal.hardware.storage.device_class", lambda p: classes.get(p, "unknown")
    )
    scratch.reset_scratch_probe()
    volumes = scratch.scratch_volumes()
    assert [v.device_class for v in volumes] == ["nvme", "rotational"]
    assert scratch.local_scratch_root() == str(fast)


def test_a_network_volume_is_not_node_local_scratch(monkeypatch, tmp_path):
    # This module's contract is node-local *fast* storage, and a network mount is neither:
    # slower than the tempdir it would be chosen over for the random access an external merge
    # produces, and shared, so two workers spilling to it contend for one pipe. A deployment
    # that wants to spill over the network says so where the choice is visible.
    volume = tmp_path / "nfs"
    volume.mkdir()
    monkeypatch.setattr(scratch, "SCRATCH_CANDIDATES", (str(volume),))
    monkeypatch.setattr(scratch, "_MIN_USEFUL_BYTES", 0)
    monkeypatch.setattr("batcher._internal.hardware.storage.device_class", lambda p: "network")
    scratch.reset_scratch_probe()
    assert scratch.scratch_volumes() == ()
    assert scratch.local_scratch_root() is None


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

    # These create real directories, so they go under `tmp_path` and nowhere else: a run that
    # resolved the wrong root would otherwise litter the node's shared scratch mount.
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
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    from batcher.dist.flight_worker import _reduce_work_dir

    made = _reduce_work_dir("probe_", None)
    assert made.startswith(tempfile.gettempdir()) or made.startswith(str(tmp_path))


def test_the_spill_cost_model_prices_the_disk_the_engine_will_actually_use(monkeypatch, tmp_path):
    # The drift this closes: the spill paths resolve to the node's measured NVMe, so a cost
    # model still asking the tempdir prices the container's overlay — a factor of ten in the
    # wrong direction on exactly the machines where an out-of-core plan needs ranking.
    from batcher.kyber import storage_cost

    seen: list[str] = []
    # Pinned where the probe lives: the class table moved to layer 0 so Carbonite could read
    # the same figures without importing Kyber.
    monkeypatch.setattr(
        "batcher._internal.hardware.storage.device_class",
        lambda path: seen.append(path) or "network",
    )
    # Patched at its *definition* site (`site.scratch`), not the `site` re-export: the
    # resolution now runs inside `site.spill_scratch_dir`, which reads the module-level
    # name. A patch on the re-export silently stops applying and the test keeps passing
    # while testing nothing.
    monkeypatch.setattr("batcher._internal.site.scratch.local_scratch_root", lambda: "/ephemeral")
    assert storage_cost.spill_device_factor() == storage_cost.SPILL_DEVICE_FACTOR["network"]
    assert seen == ["/ephemeral"]
    # And with no local volume it falls back to the tempdir, as before.
    seen.clear()
    monkeypatch.setattr("batcher._internal.site.scratch.local_scratch_root", lambda: None)
    storage_cost.spill_device_factor()
    assert seen and seen[0] != "/ephemeral"


def test_a_slurm_allocation_wider_than_ray_is_reported_once(monkeypatch, caplog):
    # The silent failure: `srun -N 4 python job.py` runs on four nodes, a bare `ray.init()`
    # starts a one-node Ray on whichever one the script landed on, and the job uses a quarter
    # of the hardware it was billed for while returning the right answer.
    import logging
    import sys
    import types

    from batcher.dist.executors.ray_runtime import capacity as readiness

    monkeypatch.setattr(readiness, "_ALLOCATION_WARNED", False)
    monkeypatch.setenv("SLURM_JOB_ID", "7")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "gpu-[01-04]")
    fake_ray = types.SimpleNamespace(is_initialized=lambda: True, nodes=lambda: [{"Alive": True}])
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    with caplog.at_level(logging.WARNING):
        readiness.warn_once_if_allocation_is_wider_than_ray()
    assert "holds 4 nodes but Ray sees 1" in caplog.text
    # Once per process: repeating it per query trains the reader to skip it.
    caplog.clear()
    readiness.warn_once_if_allocation_is_wider_than_ray()
    assert caplog.text == ""


def test_a_ray_cluster_that_spans_the_allocation_says_nothing(monkeypatch, caplog):
    import logging
    import sys
    import types

    from batcher.dist.executors.ray_runtime import capacity as readiness

    monkeypatch.setattr(readiness, "_ALLOCATION_WARNED", False)
    monkeypatch.setenv("SLURM_JOB_ID", "7")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "gpu-[01-04]")
    monkeypatch.setitem(
        sys.modules,
        "ray",
        types.SimpleNamespace(is_initialized=lambda: True, nodes=lambda: [{"Alive": True}] * 4),
    )
    with caplog.at_level(logging.WARNING):
        readiness.warn_once_if_allocation_is_wider_than_ray()
    assert caplog.text == ""


def test_an_unscheduled_process_says_nothing(monkeypatch, caplog):
    import logging

    from batcher.dist.executors.ray_runtime import capacity as readiness

    monkeypatch.setattr(readiness, "_ALLOCATION_WARNED", False)
    with caplog.at_level(logging.WARNING):
        readiness.warn_once_if_allocation_is_wider_than_ray()
    assert caplog.text == ""


def test_the_file_cache_can_be_enabled_once_for_a_fleet(monkeypatch, tmp_path):
    # The right cache directory is a per-node fact — `/ephemeral` on one provider,
    # `/mnt/local_disk` on the next — so a literal path in a shared config is the wrong one
    # everywhere but the machine it was written for.
    from batcher.io._file_cache import FILE_CACHE_AUTO, resolve_cache_dir

    volume = tmp_path / "ephemeral"
    volume.mkdir()
    monkeypatch.setattr("batcher._internal.site.local_scratch_root", lambda: str(volume))
    resolved = resolve_cache_dir(FILE_CACHE_AUTO)
    assert resolved == str(volume / "batcher_file_cache")
    # An explicit path is used as written, and nothing configured stays off.
    assert resolve_cache_dir("/named/dir") == "/named/dir"
    assert resolve_cache_dir(None) is None
    assert resolve_cache_dir("") is None


def test_a_node_with_no_fast_disk_resolves_auto_to_no_cache(monkeypatch):
    # Caching onto the container overlay would compete for the disk the read it is caching
    # would otherwise never touch.
    from batcher.io._file_cache import FILE_CACHE_AUTO, resolve_cache_dir

    monkeypatch.setattr("batcher._internal.site.local_scratch_root", lambda: None)
    assert resolve_cache_dir(FILE_CACHE_AUTO) is None


def test_auto_and_its_resolved_path_are_one_cache(monkeypatch, tmp_path):
    # Keyed on the resolved path, so the two spellings do not produce two caches over the
    # same files — which would double the disk and halve the hit rate.
    from batcher.config import Config, config_context
    from batcher.io import _file_cache

    volume = tmp_path / "nvme"
    volume.mkdir()
    monkeypatch.setattr("batcher._internal.site.local_scratch_root", lambda: str(volume))
    monkeypatch.setattr(_file_cache, "_CACHES", {})
    with config_context(Config.from_dict({"memory": {"file_cache_dir": "auto"}})):
        first = _file_cache.get_file_cache()
    resolved = str(volume / "batcher_file_cache")
    with config_context(Config.from_dict({"memory": {"file_cache_dir": resolved}})):
        second = _file_cache.get_file_cache()
    assert first is not None
    assert first is second


def test_the_dashboard_snapshot_carries_the_site_where_there_is_one(monkeypatch):
    # The facts that explain a *default* rather than measure a run: why the spill went where
    # it did, why the shuffle was priced the way it was. Neither is visible anywhere else.
    from batcher.observe.system import system_snapshot

    monkeypatch.setenv("BATCHER_PROVIDER", "crusoe")
    provider.reset_provider_probe()
    monkeypatch.setattr("batcher._internal.site.local_scratch_root", lambda: "/ephemeral")
    site = system_snapshot()["site"]
    assert site["provider"] == "crusoe"
    assert site["neocloud"] is True
    assert site["scratch_dir"] == "/ephemeral"


def test_a_machine_with_no_site_facts_gets_no_site_section(monkeypatch):
    from batcher.observe.system import system_snapshot

    monkeypatch.setattr("batcher._internal.site.local_scratch_root", lambda: None)
    monkeypatch.setattr(
        "batcher._internal.hardware.fabric.rdma_summary",
        lambda: {
            "ports": 0,
            "active_ports": 0,
            "bandwidth_gbps": 0.0,
            "link_layers": {},
            "rdma_available": False,
            "partition": "",
            "devices": [],
            "numa_nodes": [],
        },
    )
    assert "site" not in system_snapshot()


def test_a_fabric_alone_is_enough_to_report_a_site(monkeypatch):
    from batcher.observe.system import system_snapshot

    monkeypatch.setattr("batcher._internal.site.local_scratch_root", lambda: None)
    monkeypatch.setattr(
        "batcher._internal.hardware.fabric.rdma_summary",
        lambda: {
            "ports": 8,
            "active_ports": 6,
            "bandwidth_gbps": 2400.0,
            "link_layers": {"InfiniBand": 6},
            "rdma_available": True,
            "partition": "",
            "devices": [],
            "numa_nodes": [],
        },
    )
    monkeypatch.setattr("batcher._internal.hardware.fabric.fabric_bandwidth_gbps", lambda: 2400.0)
    site = system_snapshot()["site"]
    assert site["fabric_ports"] == 6
    assert site["fabric_gbps"] == 2400.0


# --- The firmware, when the environment says nothing ----------------------------------------


def _dmi(tmp_path, **fields) -> str:
    root = tmp_path / "dmi"
    root.mkdir(exist_ok=True)
    for name, value in fields.items():
        (root / name).write_text(f"{value}\n")
    return str(root)


def test_the_firmware_names_a_platform_the_environment_did_not(monkeypatch, tmp_path):
    # A bare-metal node rented from a provider that exports no marker still knows what it is.
    monkeypatch.setattr(provider, "dmi_identity", _REAL_DMI)
    monkeypatch.setattr(
        provider, "DMI_ROOT", _dmi(tmp_path, sys_vendor="Amazon EC2", product_name="p5.48xlarge")
    )
    assert provider.detect_provider() == "aws"
    profile = provider.site_profile()
    assert profile.machine == "p5.48xlarge"
    assert profile.instance_type == "p5.48xlarge"
    assert profile.virtualized is False


def test_an_environment_marker_still_wins_over_the_firmware(monkeypatch, tmp_path):
    # DMI names the machine a node was *built* as, not the service renting it out: a GPU
    # cloud reselling EC2 capacity would read as `aws` while its own marker says who it is.
    monkeypatch.setattr(provider, "dmi_identity", _REAL_DMI)
    monkeypatch.setattr(provider, "DMI_ROOT", _dmi(tmp_path, sys_vendor="Amazon EC2"))
    monkeypatch.setenv("CRUSOE_VM_ID", "vm-1")
    assert provider.detect_provider() == "crusoe"


def test_a_hypervisor_vendor_reports_a_virtual_machine(monkeypatch, tmp_path):
    # An empty fabric probe is conclusive on bare metal and is not in a VM, where `/sys`
    # shows the hypervisor's view of the PCI tree.
    monkeypatch.setattr(provider, "dmi_identity", _REAL_DMI)
    monkeypatch.setattr(provider, "DMI_ROOT", _dmi(tmp_path, sys_vendor="QEMU"))
    assert provider.dmi_identity()[2] is True
    assert provider.site_profile().virtualized is True


def test_unreadable_firmware_says_nothing_rather_than_not_a_vm(monkeypatch, tmp_path):
    monkeypatch.setattr(provider, "dmi_identity", _REAL_DMI)
    monkeypatch.setattr(provider, "DMI_ROOT", str(tmp_path / "absent"))
    assert provider.dmi_identity() == ("", "", None)
    assert provider.detect_provider() == "unknown"
    assert provider.site_profile().virtualized is None
    assert "virtualized" not in provider.site_summary()


def test_an_unrecognized_firmware_vendor_is_not_a_platform(monkeypatch, tmp_path):
    monkeypatch.setattr(provider, "dmi_identity", _REAL_DMI)
    monkeypatch.setattr(
        provider, "DMI_ROOT", _dmi(tmp_path, sys_vendor="Supermicro", product_name="AS-4125GS")
    )
    # Bare metal from a builder nobody resells under that name: the machine is known, the
    # platform is not, and neither is invented from the other.
    assert provider.detect_provider() == "unknown"
    assert provider.dmi_identity()[1] == "AS-4125GS"
    assert provider.dmi_identity()[2] is False


# --- The other batch schedulers a GPU cluster runs under --------------------------------------


def test_pbs_reads_its_node_list_from_a_file(monkeypatch, tmp_path):
    # The one structural difference from Slurm: PBS writes hosts to a file, one line per task
    # *slot*, so a four-node job with eight tasks each lists every node eight times.
    nodefile = tmp_path / "nodes"
    nodefile.write_text("gpu01\n" * 8 + "gpu02\n" * 8)
    monkeypatch.setenv("PBS_JOBID", "912.head")
    monkeypatch.setenv("PBS_NODEFILE", str(nodefile))
    monkeypatch.setenv("PBS_NGPUS", "8")
    monkeypatch.setenv("PBS_NCPUS", "96")
    monkeypatch.setenv("PBS_NP", "16")
    monkeypatch.setenv("PBS_QUEUE", "gpu")
    job = scheduler.scheduler_job()
    assert job.kind == "pbs"
    assert job.nodes == ("gpu01", "gpu02"), "distinct names, not one per slot"
    assert job.multi_node is True
    assert job.total_gpus == 16
    assert (job.cpus_per_node, job.tasks, job.partition) == (96, 16, "gpu")


def test_lsf_prefers_its_host_file_over_the_inline_list(monkeypatch, tmp_path):
    # `LSB_HOSTS` overflows on a large job; the file is the one that stays correct at scale.
    hostfile = tmp_path / "hosts"
    hostfile.write_text("gpu07\ngpu07\ngpu08\n")
    monkeypatch.setenv("LSB_JOBID", "5150")
    monkeypatch.setenv("LSB_HOSTS", "wrong wrong")
    monkeypatch.setenv("LSB_DJOB_HOSTFILE", str(hostfile))
    monkeypatch.setenv("LSB_QUEUE", "gpuq")
    job = scheduler.scheduler_job()
    assert job.kind == "lsf"
    assert job.nodes == ("gpu07", "gpu08")
    assert job.partition == "gpuq"


def test_lsf_falls_back_to_the_inline_host_list(monkeypatch):
    monkeypatch.setenv("LSB_JOBID", "5150")
    monkeypatch.setenv("LSB_HOSTS", "gpu01 gpu01 gpu02")
    job = scheduler.scheduler_job()
    assert job.nodes == ("gpu01", "gpu02")


def test_an_unreadable_host_file_is_an_empty_allocation_not_a_crash(monkeypatch, tmp_path):
    monkeypatch.setenv("PBS_JOBID", "1")
    monkeypatch.setenv("PBS_NODEFILE", str(tmp_path / "absent"))
    job = scheduler.scheduler_job()
    assert job.kind == "pbs"
    assert job.nodes == ()
    assert job.multi_node is False


def test_slurm_still_wins_when_more_than_one_scheduler_is_in_the_environment(monkeypatch):
    # A Slurm job that submits through a PBS-compatible wrapper carries both; the outermost
    # allocation is the one that bounds the job and will end it.
    monkeypatch.setenv("PBS_JOBID", "1")
    monkeypatch.setenv("LSB_JOBID", "2")
    assert scheduler.scheduler_kind() == "pbs"
    monkeypatch.setenv("SLURM_JOB_ID", "3")
    assert scheduler.scheduler_kind() == "slurm"
