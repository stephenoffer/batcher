"""Every scheduler Batcher can run under, and the shape it reads out of each one.

Two failures motivate this file, and both are silent. A scheduler the detector does not know
reports `"none"`, so the job's *whole* shape — node list, device grant, core grant, rank —
falls back to probing the local machine, which on a shared node is wrong in the direction that
oversubscribes it. And a device count read as `1` where the pod actually holds eight sizes
every stage at an eighth of the hardware while returning perfectly correct results.

So the assertions here are about *shape*, not about detection alone: a test that only checked
`kind` would pass on a reader that returned an empty record.
"""

from __future__ import annotations

import json

import pytest

from batcher._internal.site import scheduler
from batcher._internal.site.scheduler import detect

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean(clean_site_env):
    """Every test here asserts what the environment says, so nothing may leak in from outside."""


# --- Detection order ------------------------------------------------------------------------


def test_an_unscheduled_process_reports_none_and_claims_no_shape():
    job = scheduler.scheduler_job()
    assert job.kind == "none"
    assert (job.nodes, job.node_count, job.total_gpus, job.rank) == ((), 0, 0, 0)
    assert job.multi_node is False
    assert job.shares_node is False


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"SLURM_JOB_ID": "1"}, "slurm"),
        ({"PBS_JOBID": "1.head"}, "pbs"),
        ({"LSB_JOBID": "1"}, "lsf"),
        ({"JOB_ID": "1", "SGE_TASK_ID": "undefined"}, "sge"),
        ({"FLUX_JOB_ID": "fABC"}, "flux"),
        ({"_CONDOR_SLOT": "slot1_2"}, "htcondor"),
        ({"AWS_BATCH_JOB_ID": "abc"}, "aws_batch"),
        ({"SM_CURRENT_HOST": "algo-1"}, "sagemaker"),
        ({"CLOUD_ML_JOB_ID": "job1"}, "vertex"),
        ({"AZUREML_RUN_ID": "run1"}, "azureml"),
        ({"DB_CLUSTER_ID": "0101-x"}, "databricks"),
        ({"SKYPILOT_TASK_ID": "sky-1"}, "skypilot"),
        ({"NOMAD_ALLOC_ID": "a1"}, "nomad"),
        ({"CONTAINER_ID": "container_1_0001_01_000001", "NM_HOST": "n1"}, "yarn"),
        ({"KUBERNETES_SERVICE_HOST": "10.0.0.1"}, "kubernetes"),
        ({"RAY_ADDRESS": "auto"}, "ray"),
    ],
)
def test_every_supported_scheduler_is_recognized_from_its_own_marker(monkeypatch, env, expected):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert scheduler.scheduler_kind() == expected
    assert scheduler.scheduler_job().kind == expected


def test_a_generic_marker_alone_is_not_a_scheduler(monkeypatch):
    # `JOB_ID` is what half of CI sets, and `CONTAINER_ID` is not exclusively YARN's. Both
    # need a second, platform-specific variable before they are believed — otherwise a
    # Jenkins-triggered laptop run claims to be a Grid Engine allocation.
    monkeypatch.setenv("JOB_ID", "4711")
    monkeypatch.setenv("CONTAINER_ID", "container_1_0001_01_000001")
    assert scheduler.scheduler_kind() == "none"


def test_the_outermost_scheduler_wins(monkeypatch):
    # A Ray worker in a Kubernetes pod in a Slurm allocation is bounded by the allocation,
    # which is what will end it.
    monkeypatch.setenv("RAY_ADDRESS", "auto")
    assert scheduler.scheduler_kind() == "ray"
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
    assert scheduler.scheduler_kind() == "kubernetes"
    monkeypatch.setenv("SLURM_JOB_ID", "7")
    assert scheduler.scheduler_kind() == "slurm"


def test_skypilot_is_read_before_the_backend_it_provisioned(monkeypatch):
    # SkyPilot on a Kubernetes backend carries both sets of markers, and only SkyPilot knows
    # the task spans four nodes; Kubernetes knows only about this pod.
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
    monkeypatch.setenv("SKYPILOT_TASK_ID", "sky-2024-01-01-task")
    monkeypatch.setenv("SKYPILOT_NUM_NODES", "4")
    job = scheduler.scheduler_job()
    assert job.kind == "skypilot"
    assert job.node_count == 4
    assert job.multi_node is True


def test_the_override_names_a_platform_this_table_has_never_seen(monkeypatch):
    monkeypatch.setenv("BATCHER_SCHEDULER", "housescheduler")
    job = scheduler.scheduler_job()
    assert (scheduler.scheduler_kind(), job.kind) == ("housescheduler", "housescheduler")
    # Naming a site is not claiming to know its shape.
    assert (job.nodes, job.gpus_per_node, job.rank) == ((), 0, 0)


def test_the_override_still_reads_the_shape_when_it_names_a_known_scheduler(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "9")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "n[1-2]")
    monkeypatch.setenv("BATCHER_SCHEDULER", "slurm")
    assert scheduler.scheduler_job().nodes == ("n1", "n2")


# --- Device grants --------------------------------------------------------------------------


def test_the_container_runtimes_all_marker_is_not_one_device(monkeypatch):
    # `NVIDIA_VISIBLE_DEVICES=all` is the NVIDIA container toolkit's default and therefore the
    # common case. Counting it as a single identifier reported one GPU on every eight-GPU pod
    # that had not been narrowed, and every stage sized against it ran at an eighth of the node.
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "all")
    assert scheduler.scheduler_job().gpus_per_node == 0


@pytest.mark.parametrize(
    ("var", "value", "expected"),
    [
        ("CUDA_VISIBLE_DEVICES", "0,1,2,3", 4),
        ("HIP_VISIBLE_DEVICES", "0,1", 2),  # AMD Instinct
        ("HABANA_VISIBLE_MODULES", "0,1,2,3,4,5,6,7", 8),  # Intel Gaudi
        ("NEURON_RT_VISIBLE_CORES", "0-15", 16),  # AWS Trainium, an inclusive range
        ("ZE_AFFINITY_MASK", "0.0,0.1", 2),  # Intel XPU sub-devices
        ("TPU_VISIBLE_DEVICES", "0,1,2,3", 4),
        ("NVIDIA_VISIBLE_DEVICES", "GPU-fe8b1c,GPU-9a02de", 2),  # device-plugin UUIDs
        ("CUDA_VISIBLE_DEVICES", "", 0),  # explicitly denied the GPU
        ("CUDA_VISIBLE_DEVICES", "-1", 0),
    ],
)
def test_every_accelerator_vendor_publishes_its_grant_under_its_own_name(
    monkeypatch, var, value, expected
):
    # A reader that knows only the NVIDIA spelling reports the whole node to a Gaudi, Neuron
    # or XPU pod that has been pinned to a fraction of it.
    monkeypatch.setenv(var, value)
    assert scheduler.visible_device_count() == expected


def test_a_framework_pin_outranks_the_container_runtimes_injection(monkeypatch):
    # The runtime injects every device it was asked to; Slurm, Ray and a device plugin then
    # narrow that to this process's share. The narrower answer is the true one.
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    assert scheduler.visible_device_count() == 2


# --- Slurm ------------------------------------------------------------------------------------


def test_slurm_reads_the_gpu_request_form_as_well_as_the_grant(monkeypatch):
    # `SLURM_GPUS_PER_NODE` is written `gpu:8` — the resource name carries no number, so a
    # plain int parse found nothing and the allocation reported no devices at all.
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "gpu-[01-02]")
    monkeypatch.setenv("SLURM_GPUS_PER_NODE", "gpu:8")
    job = scheduler.scheduler_job()
    assert (job.gpus_per_node, job.total_gpus) == (8, 16)


def test_a_slurm_step_without_a_node_list_still_knows_how_wide_it_is(monkeypatch):
    # Reading only the list made a 64-node allocation look single-node, which turns off every
    # cross-node decision the engine makes.
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    monkeypatch.setenv("SLURM_JOB_NUM_NODES", "64")
    monkeypatch.setenv("SLURM_GPUS_ON_NODE", "8")
    job = scheduler.scheduler_job()
    assert (job.nodes, job.node_count, job.total_gpus) == ((), 64, 512)
    assert job.multi_node is True


def test_slurm_publishes_the_per_task_core_grant_and_the_node_share(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "96")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "12")
    monkeypatch.setenv("SLURM_NTASKS_PER_NODE", "8")
    job = scheduler.scheduler_job()
    assert (job.cpus_per_node, job.cpus_per_task, job.local_size) == (96, 12, 8)
    assert job.shares_node is True


def test_a_slurm_array_task_is_identified_by_its_index(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "37")
    assert scheduler.scheduler_job().array_index == "37"


# --- PBS --------------------------------------------------------------------------------------


def test_pbs_reports_a_hostname_not_a_node_index(monkeypatch, tmp_path):
    # `PBS_NODENUM` is this node's *index* in the allocation. Reading it as the node name put
    # the string "0" where every caller expected a hostname.
    nodefile = tmp_path / "nodes"
    nodefile.write_text("gpu01\ngpu02\n")
    monkeypatch.setenv("PBS_JOBID", "912.head")
    monkeypatch.setenv("PBS_NODEFILE", str(nodefile))
    monkeypatch.setenv("PBS_NODENUM", "0")
    monkeypatch.setenv("HOSTNAME", "gpu02")
    assert scheduler.scheduler_job().node_name == "gpu02"


def test_pbs_divides_the_node_grant_by_the_tasks_sharing_the_node(monkeypatch, tmp_path):
    # Eight tasks on a 96-core node get 12 cores each. Sizing all eight to 96 oversubscribes
    # the node twelve-fold, which is what the per-slot node file exists to prevent.
    nodefile = tmp_path / "nodes"
    nodefile.write_text("gpu01\n" * 8)
    monkeypatch.setenv("PBS_JOBID", "1")
    monkeypatch.setenv("PBS_NODEFILE", str(nodefile))
    monkeypatch.setenv("HOSTNAME", "gpu01")
    monkeypatch.setenv("NCPUS", "96")
    job = scheduler.scheduler_job()
    assert (job.tasks_per_node, job.cpus_per_node, job.cpus_per_task) == (8, 96, 12)
    assert job.shares_node is True


# --- LSF --------------------------------------------------------------------------------------


def test_lsf_reads_the_per_host_slot_breakdown(monkeypatch):
    # `LSB_DJOB_NUMPROC` is the job-wide slot total, so using it per node over-counts by the
    # number of hosts. `LSB_MCPU_HOSTS` is the breakdown that stays right.
    monkeypatch.setenv("LSB_JOBID", "5150")
    monkeypatch.setenv("LSB_HOSTS", "gpu07 gpu07 gpu08 gpu08")
    monkeypatch.setenv("LSB_MCPU_HOSTS", "gpu07 16 gpu08 8")
    monkeypatch.setenv("HOSTNAME", "gpu07")
    job = scheduler.scheduler_job()
    assert job.nodes == ("gpu07", "gpu08")
    assert job.cpus_per_node == 16
    assert detect.allocated_cpus() == 16, "this host's own entry, not the job-wide minimum"


# --- Grid Engine ------------------------------------------------------------------------------


def test_grid_engine_reads_its_slot_column_rather_than_counting_lines(monkeypatch, tmp_path):
    # `PE_HOSTFILE` is `hostname nslots queue processors` per line — the slot count is a
    # column, not a repeat count, so reading it as a plain host list loses the task layout.
    pe = tmp_path / "pe_hostfile"
    pe.write_text("gpu01 8 all.q@gpu01 UNDEFINED\ngpu02 4 all.q@gpu02 UNDEFINED\n")
    monkeypatch.setenv("JOB_ID", "77")
    monkeypatch.setenv("PE_HOSTFILE", str(pe))
    monkeypatch.setenv("NSLOTS", "8")
    monkeypatch.setenv("QUEUE", "all.q")
    monkeypatch.setenv("SGE_TASK_ID", "undefined")
    job = scheduler.scheduler_job()
    assert job.kind == "sge"
    assert job.nodes == ("gpu01", "gpu02")
    assert (job.tasks, job.cpus_per_task, job.partition) == (12, 8, "all.q")


# --- Flux and HTCondor ------------------------------------------------------------------------


def test_flux_numbers_its_own_tasks(monkeypatch):
    monkeypatch.setenv("FLUX_JOB_ID", "fABCD1234")
    monkeypatch.setenv("FLUX_JOB_SIZE", "32")
    monkeypatch.setenv("FLUX_JOB_NNODES", "4")
    monkeypatch.setenv("FLUX_TASK_RANK", "9")
    monkeypatch.setenv("FLUX_TASK_LOCAL_ID", "1")
    job = scheduler.scheduler_job()
    assert (job.kind, job.node_count, job.tasks) == ("flux", 4, 32)
    assert (job.rank, job.local_rank, job.local_size) == (9, 1, 8)


def test_htcondor_reads_its_identity_out_of_the_job_ad(monkeypatch, tmp_path):
    # HTCondor exports the slot name and almost nothing else; the job's own identity and its
    # resource request live in the ClassAd file it writes beside the job.
    ad = tmp_path / "job.ad"
    ad.write_text('ClusterId = 4242\nProcId = 7\nRequestCpus = 4\nRequestGpus = 2\nOwner = "x"\n')
    monkeypatch.setenv("_CONDOR_SLOT", "slot1_3")
    monkeypatch.setenv("_CONDOR_JOB_AD", str(ad))
    job = scheduler.scheduler_job()
    assert (job.kind, job.job_id, job.array_index) == ("htcondor", "4242.7", "7")
    assert (job.cpus_per_task, job.gpus_per_node, job.partition) == (4, 2, "slot1_3")


def test_an_htcondor_gpu_assignment_beats_the_request(monkeypatch, tmp_path):
    ad = tmp_path / "job.ad"
    ad.write_text("RequestGpus = 4\n")
    monkeypatch.setenv("_CONDOR_SLOT", "slot1")
    monkeypatch.setenv("_CONDOR_JOB_AD", str(ad))
    monkeypatch.setenv("_CONDOR_ASSIGNED_GPUS", "CUDA0,CUDA1")
    assert scheduler.scheduler_job().gpus_per_node == 2


def test_an_unreadable_ad_file_is_an_empty_record_not_a_crash(monkeypatch, tmp_path):
    monkeypatch.setenv("_CONDOR_SLOT", "slot1")
    monkeypatch.setenv("_CONDOR_JOB_AD", str(tmp_path / "absent"))
    assert scheduler.scheduler_job().kind == "htcondor"


# --- Launcher ranks ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "env",
    [
        {"RANK": "5", "WORLD_SIZE": "8", "LOCAL_RANK": "1", "LOCAL_WORLD_SIZE": "4"},
        {
            "OMPI_COMM_WORLD_RANK": "5",
            "OMPI_COMM_WORLD_SIZE": "8",
            "OMPI_COMM_WORLD_LOCAL_RANK": "1",
            "OMPI_COMM_WORLD_LOCAL_SIZE": "4",
        },
        {"PMI_RANK": "5", "PMI_SIZE": "8", "MPI_LOCALRANKID": "1", "MPI_LOCALNRANKS": "4"},
    ],
)
def test_the_three_launcher_vocabularies_all_read(monkeypatch, env):
    # Without these, a four-node Kubernetes job reports rank 0 on every node, and anything
    # that shards by rank does the same quarter of the work four times.
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    ranks = scheduler.launcher_ranks()
    assert (ranks.rank, ranks.tasks, ranks.local_rank, ranks.local_size) == (5, 8, 1, 4)


def test_a_kubernetes_job_derives_its_node_count_from_the_launcher(monkeypatch):
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
    monkeypatch.setenv("NODE_NAME", "gpu-node-7")
    monkeypatch.setenv("WORLD_SIZE", "16")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    monkeypatch.setenv("RANK", "9")
    job = scheduler.scheduler_job()
    assert (job.node_count, job.rank, job.local_size) == (2, 9, 8)
    assert job.multi_node is True


def test_a_volcano_task_index_outranks_a_hand_set_rank(monkeypatch):
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("VC_TASK_INDEX", "3")
    assert scheduler.scheduler_job().rank == 3


# --- Managed job services ---------------------------------------------------------------------


def test_aws_batch_publishes_a_multi_node_shape_without_a_launcher(monkeypatch):
    monkeypatch.setenv("AWS_BATCH_JOB_ID", "abc123#0")
    monkeypatch.setenv("AWS_BATCH_JOB_NUM_NODES", "4")
    monkeypatch.setenv("AWS_BATCH_JOB_NODE_INDEX", "2")
    monkeypatch.setenv("AWS_BATCH_JQ_NAME", "gpu-queue")
    job = scheduler.scheduler_job()
    assert (job.kind, job.node_count, job.rank, job.partition) == ("aws_batch", 4, 2, "gpu-queue")
    assert job.multi_node is True


def test_sagemaker_ranks_a_host_by_its_position_in_the_host_list(monkeypatch):
    monkeypatch.setenv("SM_HOSTS", json.dumps(["algo-1", "algo-2", "algo-3"]))
    monkeypatch.setenv("SM_CURRENT_HOST", "algo-2")
    monkeypatch.setenv("SM_NUM_GPUS", "8")
    monkeypatch.setenv("SM_NUM_CPUS", "96")
    monkeypatch.setenv("TRAINING_JOB_NAME", "train-2024")
    job = scheduler.scheduler_job()
    assert (job.kind, job.nodes, job.rank) == ("sagemaker", ("algo-1", "algo-2", "algo-3"), 1)
    assert (job.gpus_per_node, job.total_gpus, job.cpus_per_node) == (8, 24, 96)


def test_vertex_reads_its_peer_list_out_of_the_cluster_spec(monkeypatch):
    spec = {
        "cluster": {
            "workerpool0": ["chief-0:2222"],
            "workerpool1": ["worker-0:2222", "worker-1:2222"],
        },
        "task": {"type": "workerpool1", "index": 1},
    }
    monkeypatch.setenv("CLUSTER_SPEC", json.dumps(spec))
    job = scheduler.scheduler_job()
    assert job.kind == "vertex"
    assert job.nodes == ("chief-0", "worker-0", "worker-1")
    assert (job.rank, job.node_name) == (2, "worker-1")


def test_a_malformed_cluster_spec_is_an_empty_shape_not_a_crash(monkeypatch):
    monkeypatch.setenv("CLOUD_ML_JOB_ID", "job-1")
    monkeypatch.setenv("CLUSTER_SPEC", "{not json")
    job = scheduler.scheduler_job()
    assert (job.kind, job.job_id, job.nodes) == ("vertex", "job-1", ())


def test_skypilot_publishes_the_shape_the_platform_underneath_does_not(monkeypatch):
    monkeypatch.setenv("SKYPILOT_TASK_ID", "sky-task")
    monkeypatch.setenv("SKYPILOT_NUM_NODES", "3")
    monkeypatch.setenv("SKYPILOT_NODE_RANK", "1")
    monkeypatch.setenv("SKYPILOT_NODE_IPS", "10.0.0.1\n10.0.0.2\n10.0.0.3")
    monkeypatch.setenv("SKYPILOT_NUM_GPUS_PER_NODE", "8")
    job = scheduler.scheduler_job()
    assert job.nodes == ("10.0.0.1", "10.0.0.2", "10.0.0.3")
    assert (job.rank, job.node_name, job.total_gpus) == (1, "10.0.0.2", 24)


def test_yarn_derives_the_application_id_from_the_container_id(monkeypatch):
    # The epoch field (`e17`) is optional, which is what a positional split gets wrong.
    monkeypatch.setenv("CONTAINER_ID", "container_e17_1699999999999_0042_01_000003")
    monkeypatch.setenv("NM_HOST", "worker-11")
    job = scheduler.scheduler_job()
    assert (job.kind, job.job_id, job.node_name) == (
        "yarn",
        "application_1699999999999_0042",
        "worker-11",
    )


# --- The allocation core bound ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"SLURM_CPUS_PER_TASK": "12"}, 12),
        ({"SLURM_CPUS_ON_NODE": "4(x2),8"}, 4),  # heterogeneous: the weakest grant binds
        ({"NCPUS": "16", "PBS_JOBID": "1.head"}, 16),
        # `NCPUS` is a name unrelated tooling sets, and this bound narrows every thread pool
        # in the process, so it is believed only inside a job of the scheduler that sets it.
        ({"NCPUS": "16"}, None),
        ({"PBS_NCPUS": "24"}, 24),
        ({"NSLOTS": "6"}, 6),
        # No name to match, so the weakest entry binds: under-parallelizing costs throughput
        # where the alternative oversubscribes a host the scheduler placed co-tenants on.
        ({"LSB_MCPU_HOSTS": "a 16 b 8"}, 8),
        # This host's own entry when the breakdown names it, which is the exact answer.
        ({"LSB_MCPU_HOSTS": "a 16 b 8", "HOSTNAME": "a"}, 16),
        ({"LSB_MCPU_HOSTS": "a 16 b 8", "HOSTNAME": "b.cluster.internal"}, 8),
        ({"LSB_HOSTS": "a a a a", "LSB_DJOB_NUMPROC": "4"}, 4),
        ({}, None),
        ({"SLURM_CPUS_ON_NODE": "weird"}, None),  # unparseable → no bound beats a wrong one
    ],
)
def test_the_allocation_core_bound_reads_every_scheduler_that_publishes_one(
    monkeypatch, env, expected
):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert detect.allocated_cpus() == expected


def test_a_job_carrying_two_schedulers_variables_takes_the_weaker_bound(monkeypatch):
    # A job submitted through a compatibility wrapper carries both, and the weakest bound is
    # the one that keeps the node's co-tenants whole.
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "64")
    monkeypatch.setenv("PBS_JOBID", "1.head")
    monkeypatch.setenv("NCPUS", "8")
    assert detect.allocated_cpus() == 8


def test_the_allocation_bound_caps_the_usable_core_count(monkeypatch):
    from batcher._internal.hardware import available_cpu_count, cpu

    monkeypatch.setattr(cpu.os, "cpu_count", lambda: 128)
    monkeypatch.setattr(cpu, "_affinity_count", lambda: 128)
    monkeypatch.setattr(cpu, "cfs_quota_count", lambda: None)
    assert available_cpu_count() == 128
    monkeypatch.setenv("NSLOTS", "8")
    assert available_cpu_count() == 8, "an unconfined Grid Engine node must still bound"


# --- Hostlists ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        # A suffix after the bracket is how an HPC site names the interconnect-side interface;
        # without it the group expanded to four nodes plus a phantom named "-ib".
        ("node[01-04]-ib", ("node01-ib", "node02-ib", "node03-ib", "node04-ib")),
        ("gpu-[1-2]x,login01", ("gpu-1x", "gpu-2x", "login01")),
    ],
)
def test_a_hostlist_group_may_carry_a_suffix(spec, expected):
    assert scheduler.expand_nodelist(spec) == expected


def test_a_name_may_carry_more_than_one_group():
    # Hierarchical naming at a large site. A single-group parse read `rack[1-2]node[1-3]` as
    # two nodes plus three, rather than as the six it names.
    assert scheduler.expand_nodelist("rack[1-2]node[1-3]") == (
        "rack1node1",
        "rack1node2",
        "rack1node3",
        "rack2node1",
        "rack2node2",
        "rack2node3",
    )


def test_a_comma_inside_a_group_does_not_split_the_name():
    # The failure this guards: `gpu-[01-02,07]` read as two entries yields `gpu-[01-02` and
    # `07]`, neither of which is a hostname.
    assert scheduler.expand_nodelist("gpu-[01-02,07],login01") == (
        "gpu-01",
        "gpu-02",
        "gpu-07",
        "login01",
    )


# --- The scheduler's own scratch directory ------------------------------------------------------


@pytest.fixture
def local_disk(monkeypatch):
    """Make every measured directory look like node-local NVMe.

    The suite's own `tmp_path` is on tmpfs on many hosts, and the scratch probe correctly
    refuses tmpfs — spilling to RAM relieves no memory pressure. Pinning the device class is
    what lets these tests be about the *wiring* rather than about the runner's mounts.
    """
    import batcher._internal.hardware.storage as storage

    monkeypatch.setattr(storage, "device_class", lambda path: "nvme")


@pytest.mark.parametrize("var", ["_CONDOR_SCRATCH_DIR", "SLURM_TMPDIR", "PBS_JOBFS"])
def test_a_spill_prefers_the_directory_the_scheduler_made_for_the_job(
    monkeypatch, tmp_path, local_disk, var
):
    # Node-local, private to the job, and removed when the job ends -- so a spill there cannot
    # outlive the job and fill a shared mount for the next tenant. Under HTCondor it is also
    # the only directory a job is guaranteed to be able to write to at all.
    from batcher._internal.site import scratch

    job_dir = tmp_path / "jobscratch"
    job_dir.mkdir()
    monkeypatch.setenv(var, str(job_dir))
    scratch.reset_scratch_probe()
    try:
        assert scratch.spill_scratch_dir() == str(job_dir)
    finally:
        scratch.reset_scratch_probe()


def test_a_job_scratch_directory_on_tmpfs_is_refused_like_any_other(monkeypatch, tmp_path):
    # Spilling to RAM relieves no memory pressure at all, so the query runs out of memory at
    # exactly the point it spilled to avoid that. The scheduler naming it changes nothing.
    import batcher._internal.hardware.storage as storage
    from batcher._internal.site import scratch

    monkeypatch.setattr(storage, "device_class", lambda path: "memory")
    monkeypatch.setenv("SLURM_TMPDIR", str(tmp_path))
    scratch.reset_scratch_probe()
    try:
        assert scratch.job_scratch_volume() is None
    finally:
        scratch.reset_scratch_probe()


def test_the_model_cache_is_not_moved_onto_a_directory_the_scheduler_deletes(
    monkeypatch, tmp_path, local_disk
):
    # `local_scratch_root` answers "the node's fast local volume", and its other caller caches
    # model weights. Pointing that at a per-job directory forces every job on the node to
    # re-download tens of gigabytes.
    from batcher._internal.site import scratch

    def no_mounts():
        return ()

    # `reset_scratch_probe` clears both memoized probes by name, so a stand-in has to answer
    # `cache_clear` too or the reset in the teardown below raises instead of resetting.
    no_mounts.cache_clear = lambda: None
    monkeypatch.setenv("SLURM_TMPDIR", str(tmp_path))
    monkeypatch.setattr(scratch, "scratch_volumes", no_mounts)
    scratch.reset_scratch_probe()
    try:
        assert scratch.local_scratch_root() is None
    finally:
        scratch.reset_scratch_probe()


def test_a_scheduler_scratch_directory_that_is_not_there_is_ignored(monkeypatch, tmp_path):
    monkeypatch.setenv("SLURM_TMPDIR", str(tmp_path / "absent"))
    assert detect.scheduler_scratch_dir() == ""


def test_tmpdir_is_not_treated_as_a_node_local_hint(monkeypatch, tmp_path):
    # Every scheduler sets it, and on a container it is the root overlay this probe exists to
    # avoid. Promoting it would rank it above a measured NVMe -- the exact inversion.
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    assert detect.scheduler_scratch_dir() == ""
