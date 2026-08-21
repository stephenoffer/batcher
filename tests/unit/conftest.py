"""Fixtures shared by the unit suite.

Only fixtures whose *definition* would otherwise be pasted into several modules live here.
Nothing is autouse: a fixture that silently rewrites the environment for 900 test modules is
how a suite starts passing for reasons nobody can name.
"""

from __future__ import annotations

import os

import pytest

from batcher._internal.site import scheduler

#: Prefixes whose whole family is stripped by `clean_site_env`. Broad on purpose: a scheduler
#: exports dozens of variables, and a test that clears only the marker still reads a node list,
#: a rank or a device count from whatever launched the suite.
SITE_ENV_PREFIXES: tuple[str, ...] = (
    "SLURM",
    "PBS_",
    "LSB_",
    "SGE_",
    "PE_HOSTFILE",
    "FLUX_",
    "_CONDOR_",
    "AWS_BATCH_",
    "SM_",
    "AZUREML_",
    "SKYPILOT_",
    "NOMAD_",
    "DB_CLUSTER",
    "DB_IS_DRIVER",
    "DB_DRIVER_IP",
    "RUNPOD",
    "COREWEAVE",
    "LAMBDA",
    "CRUSOE",
    "NEBIUS",
    "TOGETHER_",
    "VAST_",
)

#: Individual variables with no shared prefix. Every scheduler marker in `SCHEDULERS` and every
#: accelerator-visibility variable is spliced in from the modules' own tables, so a platform
#: added there cannot leak into a test that forgot to name it.
SITE_ENV_NAMES: tuple[str, ...] = (
    "BATCHER_PROVIDER",
    "BATCHER_SCHEDULER",
    "BATCHER_SCRATCH_DIR",
    "HOSTNAME",
    "NODE_NAME",
    "KUBERNETES_NODE_NAME",
    "SPEC_NODE_NAME",
    "POD_NAME",
    "POD_NAMESPACE",
    "JOB_NAME",
    "JOB_ID",
    "JOB_COMPLETION_INDEX",
    "NSLOTS",
    "NCPUS",
    "QUEUE",
    "NM_HOST",
    "HADOOP_YARN_HOME",
    "LOCAL_DIRS",
    "RAY_JOB_ID",
    "CLOUD_ML_JOB_ID",
    "TRAINING_JOB_NAME",
    "VC_TASK_INDEX",
    "VK_TASK_INDEX",
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "OMPI_COMM_WORLD_RANK",
    "OMPI_COMM_WORLD_SIZE",
    "OMPI_COMM_WORLD_LOCAL_RANK",
    "OMPI_COMM_WORLD_LOCAL_SIZE",
    "PMI_RANK",
    "PMI_SIZE",
    "PMIX_RANK",
    "MPI_LOCALRANKID",
    "MPI_LOCALNRANKS",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "AWS_EXECUTION_ENV",
    "GOOGLE_CLOUD_PROJECT",
    "AZURE_CLIENT_ID",
    *(m for spec in scheduler.SCHEDULERS for m in spec.markers),
    *(r for spec in scheduler.SCHEDULERS for r in spec.require),
    *scheduler.VISIBLE_DEVICE_COUNT_ENVS,
)


@pytest.fixture
def clean_site_env(monkeypatch):
    """Strip every environment signal the site and scheduler probes read.

    Request it from any test that asserts what Batcher detects about where it is running.
    Without it the answer depends on whatever launched the suite, which on a CI runner inside
    Kubernetes or on a developer's Slurm login node is not "nothing".
    """
    for name in list(os.environ):
        if name.startswith(SITE_ENV_PREFIXES):
            monkeypatch.delenv(name, raising=False)
    for name in SITE_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch
