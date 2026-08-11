"""Where this process is running — the provider, the scheduler, and the node's local disks.

`hardware` describes the machine. This describes the *site*: which GPU cloud it belongs to,
what launched the process, and which of the mounted filesystems is the fast local one. None of
it changes what a query computes, and all of it changes what a query should choose.

The four questions, one module each:

* `provider` — which GPU cloud this is, from environment markers only. A neocloud is not AWS
  with a different logo: the instance names, the scratch mount, the object-store endpoint, and
  the preemption signal all differ, and the defaults that are right on one are wrong on the next.
* `scheduler` — what launched this process: Slurm, Kubernetes, Ray, or nothing. A Slurm
  allocation already knows its node list and its per-node device count, and reading them beats
  discovering the same shape by probing.
* `scratch` — the node-local fast filesystem. Container roots are small overlays; the terabytes
  of NVMe a GPU node ships with are mounted somewhere else under a name that varies by provider,
  and a spill that defaults to `/tmp` finds the overlay.
* `model_cache` — where model weights land, which is the same overlay by default, wanted by
  every GPU worker on the node at once and at tens of gigabytes each.

Everything here is read from environment variables and the filesystem. **No metadata-service
call, ever** — a network round trip on a control-plane path is a hang waiting for a firewall,
and every fact worth having is already in the environment on every platform below.

A neutral utility: any layer may import `_internal`.
"""

from __future__ import annotations

from batcher._internal.site.container import (
    container_findings,
    in_container,
    memlock_limit_bytes,
    open_files_limit,
    shm_bytes,
    shm_root,
    usable_shm,
)
from batcher._internal.site.model_cache import (
    model_cache_root,
    reset_model_cache_probe,
    use_node_local_model_cache,
)
from batcher._internal.site.provider import (
    PROVIDERS,
    SiteProfile,
    detect_provider,
    reset_provider_probe,
    site_profile,
    site_summary,
)
from batcher._internal.site.scheduler import (
    SchedulerJob,
    expand_nodelist,
    scheduler_job,
    scheduler_kind,
)
from batcher._internal.site.scratch import (
    SCRATCH_CANDIDATES,
    ScratchVolume,
    local_scratch_root,
    reset_scratch_probe,
    scratch_volumes,
    spill_scratch_dir,
)

__all__ = [
    "PROVIDERS",
    "SCRATCH_CANDIDATES",
    "SchedulerJob",
    "ScratchVolume",
    "SiteProfile",
    "container_findings",
    "detect_provider",
    "expand_nodelist",
    "in_container",
    "local_scratch_root",
    "memlock_limit_bytes",
    "model_cache_root",
    "open_files_limit",
    "reset_model_cache_probe",
    "reset_provider_probe",
    "reset_scratch_probe",
    "scheduler_job",
    "scheduler_kind",
    "scratch_volumes",
    "shm_bytes",
    "shm_root",
    "site_profile",
    "site_summary",
    "spill_scratch_dir",
    "usable_shm",
    "use_node_local_model_cache",
]
