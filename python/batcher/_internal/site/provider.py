"""Which GPU cloud this is — read from the environment, never from a metadata service.

The GPU capacity market is not three hyperscalers. A fleet of H100s is as likely to be on
CoreWeave, Crusoe, Lambda, Nebius, RunPod, or Together as on EC2, and those platforms differ in
exactly the places a data engine has defaults for: the fast local disk is mounted somewhere
else, the object store is S3-compatible at a non-AWS endpoint, preemption arrives as a
Kubernetes eviction rather than a two-minute IMDS notice, and the node labels a scheduler reads
come from a different vocabulary. A default tuned for EC2 is not wrong on those platforms in a
way that fails loudly; it is wrong in a way that spills to a 20 GB container overlay while a
7 TB NVMe sits unused.

Detection is **environment variables only**. That is a deliberate constraint, not a limitation:

* A metadata-service probe is a network round trip on a control-plane path, and the failure
  mode when a firewall blackholes it is a multi-second hang on every query rather than an error.
* Every platform below exports something identifying into the container. Where one does not,
  `BATCHER_PROVIDER` names it, which is also the escape hatch for a platform not listed.

**An unrecognized environment is `"unknown"`, and `"unknown"` behaves exactly as the engine did
before this module existed.** No default is changed by a guess; a caller asks for a specific
fact (the scratch hint, the spot signal) and gets `None` when the site does not supply one.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass, field

__all__ = [
    "PROVIDERS",
    "SiteProfile",
    "detect_provider",
    "reset_provider_probe",
    "site_profile",
    "site_summary",
]


@dataclass(frozen=True, slots=True)
class _Provider:
    """One platform's markers, in the shape the detector needs.

    Attributes:
        name: The identifier this platform is reported under.
        markers: Environment variables whose presence identifies it. Any one is sufficient.
        instance_vars: Variables that name the instance or node type, most specific first.
        region_vars: Variables that name the region.
        scratch_hints: Mount points this platform puts fast node-local storage on, best first.
    """

    name: str
    markers: tuple[str, ...]
    instance_vars: tuple[str, ...] = ()
    region_vars: tuple[str, ...] = ()
    scratch_hints: tuple[str, ...] = ()


#: The platforms, ordered most specific first. A neocloud is checked before the hyperscaler it
#: may be reselling capacity from, because a CoreWeave node inside a Kubernetes cluster carries
#: Kubernetes markers too, and the more specific answer is the one with the useful defaults.
#:
#: `scratch_hints` are the mount points each platform documents for its local NVMe. They are
#: *hints*: `site.scratch` still verifies that the path exists, is writable, and is on a real
#: block device before preferring it, so a stale hint costs nothing.
PROVIDERS: tuple[_Provider, ...] = (
    _Provider(
        name="coreweave",
        markers=("COREWEAVE_NODE_NAME", "CW_NODE_NAME", "COREWEAVE_REGION"),
        instance_vars=("COREWEAVE_INSTANCE_TYPE", "NODE_INSTANCE_TYPE"),
        region_vars=("COREWEAVE_REGION",),
        scratch_hints=("/ephemeral", "/mnt/local"),
    ),
    _Provider(
        name="lambda",
        markers=("LAMBDA_INSTANCE_ID", "LAMBDA_API_KEY", "LAMBDALABS_INSTANCE_ID"),
        instance_vars=("LAMBDA_INSTANCE_TYPE",),
        region_vars=("LAMBDA_REGION",),
        scratch_hints=("/home/ubuntu/scratch", "/ephemeral"),
    ),
    _Provider(
        name="crusoe",
        markers=("CRUSOE_VM_ID", "CRUSOE_PROJECT_ID"),
        instance_vars=("CRUSOE_VM_TYPE",),
        region_vars=("CRUSOE_LOCATION",),
        scratch_hints=("/scratch", "/ephemeral"),
    ),
    _Provider(
        name="nebius",
        markers=("NEBIUS_PROJECT_ID", "NEBIUS_INSTANCE_ID"),
        instance_vars=("NEBIUS_PLATFORM",),
        region_vars=("NEBIUS_REGION",),
        scratch_hints=("/mnt/data", "/scratch"),
    ),
    _Provider(
        name="runpod",
        markers=("RUNPOD_POD_ID", "RUNPOD_API_KEY"),
        instance_vars=("RUNPOD_GPU_NAME",),
        region_vars=("RUNPOD_DC_ID",),
        scratch_hints=("/workspace", "/runpod-volume"),
    ),
    _Provider(
        name="together",
        markers=("TOGETHER_CLUSTER", "TOGETHER_JOB_ID"),
        instance_vars=("TOGETHER_INSTANCE_TYPE",),
        scratch_hints=("/scratch",),
    ),
    _Provider(
        name="vast",
        markers=("VAST_CONTAINERLABEL", "CONTAINER_API_KEY"),
        instance_vars=("VAST_GPU_NAME",),
        scratch_hints=("/workspace",),
    ),
    _Provider(
        name="oci",
        markers=("OCI_RESOURCE_PRINCIPAL_VERSION", "OCI_REGION"),
        region_vars=("OCI_REGION",),
        scratch_hints=("/mnt/localdisk", "/nvme"),
    ),
    _Provider(
        name="azure",
        markers=("AZURE_CLIENT_ID", "MSI_ENDPOINT", "AZURE_SUBSCRIPTION_ID"),
        region_vars=("AZURE_REGION", "REGION_NAME"),
        scratch_hints=("/mnt/resource", "/mnt"),
    ),
    _Provider(
        name="gcp",
        markers=("GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT", "CLOUDSDK_CORE_PROJECT"),
        region_vars=("CLOUDSDK_COMPUTE_REGION",),
        scratch_hints=("/mnt/disks/ssd0", "/mnt/local-ssd"),
    ),
    _Provider(
        name="aws",
        markers=("AWS_EXECUTION_ENV", "AWS_DEFAULT_REGION", "AWS_REGION"),
        region_vars=("AWS_REGION", "AWS_DEFAULT_REGION"),
        scratch_hints=("/mnt/nvme", "/opt/dlami/nvme"),
    ),
)

#: The override, checked before every marker. A platform this module has not seen is named
#: here, and it is also how a test pins the answer.
_PROVIDER_OVERRIDE = "BATCHER_PROVIDER"

#: Node-name variables shared across platforms. Kubernetes' downward API convention comes
#: first because on a GPU fleet the pod is usually where the process actually runs.
_NODE_VARS = ("BATCHER_NODE_NAME", "NODE_NAME", "KUBERNETES_NODE_NAME", "HOSTNAME")


def _first(names: tuple[str, ...]) -> str:
    """The first non-empty environment value among `names`, or `""`."""
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


@dataclass(frozen=True, slots=True)
class SiteProfile:
    """What this process's site is, as far as the environment says.

    Attributes:
        provider: Platform identifier from `PROVIDERS`, or `"unknown"`.
        instance_type: Instance or node type as the platform names it, `""` when unpublished.
        region: Region, `""` when unpublished.
        node_name: This node's name, `""` when unpublished.
        scratch_hints: Mount points this platform documents for fast local storage, best
            first. Hints only: nothing is used before it is verified to exist and be writable.
    """

    provider: str = "unknown"
    instance_type: str = ""
    region: str = ""
    node_name: str = ""
    scratch_hints: tuple[str, ...] = field(default_factory=tuple)

    @property
    def known(self) -> bool:
        """Whether the platform was identified at all.

        False means every default stays exactly where it was. An unidentified site is not a
        degraded one; it is the site the engine was written against before this existed.
        """
        return self.provider != "unknown"

    @property
    def neocloud(self) -> bool:
        """Whether this is a GPU-specialist cloud rather than a general-purpose hyperscaler.

        The distinction is operational rather than commercial: on these platforms a node is
        GPU-dense by construction, local NVMe is the norm rather than an option, and capacity
        is commonly leased rather than owned, so the defaults worth changing are different.
        """
        return self.provider in _NEOCLOUDS


#: Platforms whose entire product is GPU capacity. Listed rather than derived, because the
#: property that matters (GPU-dense nodes with local NVMe and leased capacity) is a fact about
#: each platform's offering, not something inferable from its name.
_NEOCLOUDS = frozenset({"coreweave", "lambda", "crusoe", "nebius", "runpod", "together", "vast"})


def detect_provider() -> str:
    """The platform this process is running on.

    Args:
        None.

    Returns:
        A name from `PROVIDERS`, the value of `BATCHER_PROVIDER` when set, or `"unknown"`.
        Never a guess: an environment with no marker reports unknown, and every caller treats
        that as "keep the default you had".
    """
    override = os.environ.get(_PROVIDER_OVERRIDE, "").strip().lower()
    if override:
        return override
    for provider in PROVIDERS:
        if any(os.environ.get(m, "").strip() for m in provider.markers):
            return provider.name
    return "unknown"


@functools.lru_cache(maxsize=1)
def site_profile() -> SiteProfile:
    """This process's site, assembled once.

    Memoized: the environment a process was launched with does not change under it, and the
    callers ask on every terminal op. `reset_provider_probe()` clears it for a test.

    Returns:
        The profile. An unrecognized site reports `provider="unknown"` with empty fields
        rather than a partially-guessed record.
    """
    name = detect_provider()
    spec = next((p for p in PROVIDERS if p.name == name), None)
    node = _first(_NODE_VARS)
    if spec is None:
        return SiteProfile(provider=name, node_name=node)
    return SiteProfile(
        provider=name,
        instance_type=_first(spec.instance_vars),
        region=_first(spec.region_vars),
        node_name=node,
        scratch_hints=spec.scratch_hints,
    )


def reset_provider_probe() -> None:
    """Forget the memoized site profile, so the next call re-reads the environment.

    A name currently bound to a test stand-in has no cache to clear and is skipped, so
    patching the probe out and resetting in either order is safe — the same contract
    `reset_hardware_probes` holds to.
    """
    clear = getattr(site_profile, "cache_clear", None)
    if clear is not None:
        clear()


def site_summary() -> dict:
    """A flat description of the site, for the decision log and the dashboard.

    Returns:
        Provider, instance type, region, node name, whether this is a GPU-specialist cloud,
        and what launched the process. Empty strings where the environment says nothing.
    """
    from batcher._internal.site.scheduler import scheduler_kind

    profile = site_profile()
    return {
        "provider": profile.provider,
        "instance_type": profile.instance_type,
        "region": profile.region,
        "node_name": profile.node_name,
        "neocloud": profile.neocloud,
        "scheduler": scheduler_kind(),
    }
