"""Where a node caches model weights — the other thing that must not land on the overlay.

`scratch` moved spilling off the container root for a reason that applies at least as hard to
model weights, and to more runs: a GPU node's root filesystem is a 20-100 GB overlay shared
with the image and every other tenant, and the default HuggingFace cache is a directory under
`$HOME` that sits on it. A node running eight GPU workers is running eight processes that each
want the same tens of gigabytes of weights, on the smallest and slowest filesystem the node
has, while its terabytes of NVMe sit empty.

What that looks like when it fails is not a clear error. It is `ENOSPC` partway through a
shard download, on one worker, after several minutes — or a run that works at four workers per
node and fills the disk at eight. Pointing the cache at the measured local volume fixes both,
and it does something else worth as much: the eight workers then *share* one copy, because
they share a filesystem, so the second worker's load is a page-cache hit rather than a
download.

**The cache directory is chosen once, and it is chosen before the hub client is imported.**
`huggingface_hub` reads its cache location into module constants at import time, so an
environment variable set afterwards is read by nothing and changes nothing. That failure is
silent and looks exactly like success, so `use_node_local_model_cache` refuses to claim an
effect it cannot have: it reports `None` when the client is already imported, rather than
setting a variable nobody will read.

Nothing here downloads, locks, or evicts. The hub client already serializes concurrent
downloads of the same file through its own lock files, and putting every worker on one
filesystem is what lets that lock do its job across the node instead of once per container
path.
"""

from __future__ import annotations

import functools
import os
import sys

from batcher._internal.logging import note_suppressed

__all__ = [
    "CACHE_ENVS",
    "MODEL_CACHE_DIRNAME",
    "model_cache_root",
    "reset_model_cache_probe",
    "use_node_local_model_cache",
]

#: The environment variables the HuggingFace stack reads its cache location from, in the order
#: it prefers them. Any one of them already set is an operator decision, and this module leaves
#: it alone — a fleet that mounts a shared model cache does so through exactly these.
CACHE_ENVS = ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME", "TRANSFORMERS_CACHE")

#: Subdirectory created under the node's scratch volume. Named rather than using the volume
#: root so the cache is separable from a spill directory on the same mount — an operator
#: clearing space needs to be able to tell the reusable gigabytes from the disposable ones.
MODEL_CACHE_DIRNAME = "batcher-model-cache"


def model_cache_root() -> str | None:
    """The directory this node should cache model weights in, or `None` to keep the default.

    Resolution order:

    1. Any of `CACHE_ENVS` already set — an operator who named a cache has made this decision,
       including the common one of mounting a shared network cache across the fleet.
    2. A subdirectory of the best measured node-local volume, which is the terabytes of NVMe
       the container root is not.
    3. `None` — no fast local storage is mounted, so the default cache is already the only
       answer available.

    Returns:
        A directory path, or `None`. The directory is not created here; `use_node_local_model_cache`
        does that, so a caller that only wants to *know* the answer does not leave one behind.
    """
    if any(os.environ.get(name, "").strip() for name in CACHE_ENVS):
        return None
    if _default_cache_populated():
        return None
    from batcher._internal.site.scratch import local_scratch_root

    root = local_scratch_root()
    return os.path.join(root, MODEL_CACHE_DIRNAME) if root else None


def _default_cache_populated() -> bool:
    """Whether the default cache already holds weights, which makes moving it a regression.

    A fleet that bakes its models into the image puts them exactly here. Redirecting the cache
    then does not save a download, it *forces* one — of tens of gigabytes, on every worker, for
    a model the node already had. That is worse than the overlay pressure this module exists to
    relieve, and it is invisible in the same way: the run simply starts slowly.
    """
    # `XDG_CACHE_HOME` moves the whole default, and an image that bakes its models in under a
    # relocated cache is exactly the case this check exists for — missing it would redirect and
    # force the re-download the check is here to prevent.
    base = os.environ.get("XDG_CACHE_HOME", "").strip() or os.path.join(
        os.path.expanduser("~"), ".cache"
    )
    try:
        return any(
            entry.name.startswith("models--")
            for entry in os.scandir(os.path.join(base, "huggingface", "hub"))
        )
    except OSError:
        return False


@functools.lru_cache(maxsize=1)
def use_node_local_model_cache() -> str | None:
    """Point the HuggingFace cache at this node's local disk, and report whether it took.

    Called on a worker immediately before the first model load, and before the hub client is
    imported. It is not an optimization that can be applied late: `huggingface_hub` reads its
    cache path into module constants at import, so setting the variable afterwards is read by
    nothing. Rather than claim an effect it cannot have, this reports `None` in that case.

    Memoized for the process, because the decision is per node and the caller is a per-worker
    build hook that runs for every stateful UDF, model-shaped or not.

    Returns:
        The cache directory now in force, or `None` when nothing was changed — an operator
        already named a cache, no local volume is mounted, the directory could not be created,
        or the hub client was imported before this ran.

    Examples:
        .. doctest::

            >>> from batcher._internal.site.model_cache import use_node_local_model_cache
            >>> use_node_local_model_cache()  # doctest: +SKIP
            '/mnt/local_disk/batcher-model-cache'
    """
    if "huggingface_hub" in sys.modules:
        # Setting the variable here would look like success and do nothing. Saying so is the
        # only honest answer, and a caller that wants the cache moves this call earlier.
        note_suppressed(
            "ml",
            "point the model cache at node-local storage (huggingface_hub was already "
            "imported, so its cache path is fixed for this process)",
            RuntimeError("hub client already imported"),
        )
        return None
    # Broad on purpose. The caller is the build hook every load-once UDF passes through, so a
    # query's success must not depend on a filesystem probe: whatever goes wrong under here —
    # an unreadable mount table, a directory that vanished between the probe and the create —
    # the right answer is the cache the process already had.
    try:
        root = model_cache_root()
        if root is None:
            return None
        os.makedirs(root, exist_ok=True)
    except Exception as exc:
        note_suppressed("ml", "choose a node-local model cache directory", exc)
        return None
    # Both spellings: `HF_HUB_CACHE` is what the current client reads, and `HF_HOME` is what an
    # older one and several sibling libraries read. Setting one and not the other splits the
    # cache in two on a node running a mix, which costs the download twice.
    os.environ["HF_HUB_CACHE"] = root
    os.environ.setdefault("HF_HOME", root)
    return root


def reset_model_cache_probe() -> None:
    """Forget the memoized decision, so the next call re-reads the environment and the mounts."""
    use_node_local_model_cache.cache_clear()
