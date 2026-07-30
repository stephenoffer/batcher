"""Filesystem resolution for IO sources and sinks — one cloud-agnostic backend.

Every file listing, glob, open, size, mkdir, and delete in Batcher goes through a
single `pyarrow.fs`-backed façade, so the *same* code path serves local disk, NFS /
on-prem mounts, S3 (incl. on-prem S3 like MinIO / Ceph via ``endpoint_override``),
GCS, Azure, and HDFS. The scheme is parsed from the path; `pyarrow.fs.FileSystem
.from_uri` constructs the right backend (reading credentials/region/endpoint from the
URI query string or the standard environment), and anything `pyarrow.fs` does not
support natively falls back to an fsspec backend wrapped behind the *same*
`pyarrow.fs` interface — so there is exactly one filesystem abstraction to reason
about. The façade exposes only what the IO bases need; the handles `open` returns are
accepted by every pyarrow reader.

On-prem / self-hosted object stores work without code changes — point at your
endpoint, e.g.
``read("s3://bucket/data/*.parquet?endpoint_override=https://minio.internal:9000")``
or set ``AWS_ENDPOINT_URL`` (and HDFS via ``hdfs://namenode:8020/path``).
"""

from __future__ import annotations

import functools
import os
from typing import Any

import pyarrow as pa
import pyarrow.fs as pafs

from batcher._internal.errors import IOError

# The `FileSystem` protocol and its `pyarrow.fs` adapter live in a sibling module (the
# interface, separate from this module's job of choosing a backend for a URI); re-exported
# here because every caller and test imports them by this path.
from batcher.io._backend import FileSystem, _ArrowFileSystem, _scheme

# The local-SSD read-through cache lives in a sibling module (its own responsibility);
# re-exported here since the filesystem is its only caller and tests import it by this path.
from batcher.io._file_cache import FileBytesCache, get_file_cache

__all__ = [
    "FileBytesCache",
    "FileSystem",
    "LocalFileSystem",
    "get_file_cache",
    "resolve_filesystem",
]

# Object stores where a single PUT is already atomic (no partial-read visibility),
# so a write goes straight to the destination — a temp-then-rename would only add a
# full-object server-side copy with no atomicity gain. Everything else (local, NFS,
# HDFS) gets temp-write-then-rename so a crash never leaves a truncated file.
_OBJECT_STORE_SCHEMES = frozenset(
    {"s3", "s3a", "gs", "gcs", "abfs", "abfss", "az", "azure", "wasb", "wasbs"}
)
# Cloud scheme aliases → the canonical scheme `from_uri` / fsspec understand.
_SCHEME_ALIASES = {"s3a": "s3", "gcs": "gs", "abfss": "abfs", "wasbs": "wasb"}


#: IO threads per usable core, and the band the result is clamped into. An IO thread spends
#: almost all of its life blocked on a socket, so the right count tracks how much the *link*
#: can carry rather than how much the CPU can compute — which is why it is deliberately an
#: oversubscription of cores and not a share of them. The floor keeps the previous behavior on
#: a small container; the ceiling stops a 192-core GPU node from opening connections faster
#: than any object store will accept them.
_IO_THREADS_PER_CORE = 4
_IO_THREADS_FLOOR = 32
_IO_THREADS_CEILING = 256


@functools.cache
def ensure_io_threads() -> None:
    """Lift pyarrow's IO thread pool above its 8-thread default, once per process.

    A wide object-store read is otherwise throttled to ~8 concurrent GETs, so a
    many-small-files scan can't saturate the link. Idempotent/cached; a no-op if the pool
    is already wider. `BATCHER_IO_THREADS` overrides the target.

    The target scales with the cores this process may use rather than sitting at a constant.
    A dense GPU node reads from object storage over a link two orders of magnitude faster than
    the small VM the old constant was chosen on, and 32 concurrent GETs leave most of it idle
    — while the same 32 on a four-core container are more connections than it can service.
    Both are the same mistake, made in opposite directions by one number."""
    from batcher._internal.hardware import available_cpu_count

    override = os.environ.get("BATCHER_IO_THREADS")
    if override:
        target = max(8, int(override))
    else:
        scaled = _IO_THREADS_PER_CORE * available_cpu_count()
        target = max(_IO_THREADS_FLOOR, min(_IO_THREADS_CEILING, scaled))
    if pa.io_thread_count() < target:
        pa.set_io_thread_count(target)
    cap_arrow_cpu_threads()


def cap_arrow_cpu_threads() -> None:
    """Cap pyarrow's CPU thread pool to the cores this process may actually use.

    pyarrow sizes its compute/decode pool to `os.cpu_count()` (host cores), so under a
    cgroup CPU quota (a Kubernetes/Ray pod) its kernels over-subscribe cores the container
    never gets, thrashing the scheduler. Only ever *lowers* the count. Result-invariant."""
    from batcher._internal.hardware import available_cpu_count

    usable = available_cpu_count()
    if pa.cpu_count() > usable:
        pa.set_cpu_count(usable)


class LocalFileSystem(_ArrowFileSystem):
    """The local filesystem (``pyarrow.fs.LocalFileSystem``), kept as a named type for
    callers/tests that construct it directly."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(pafs.LocalFileSystem(), "", atomic_rename=True)


def _local_prefix(path: str) -> str:
    """The ``file://`` prefix to strip for a local path (``""`` for a bare path)."""
    return "file://" if path.startswith("file://") else ""


def resolve_filesystem(
    path: str,
    *,
    filesystem: Any = None,
    storage_options: dict[str, str] | None = None,
) -> FileSystem:
    """Return the `pyarrow.fs`-backed façade for `path`, dispatching on its scheme.

    Local and ``file://`` paths use the local filesystem; ``s3``/``gs``/``hdfs``/
    ``abfs``/… are constructed by `pyarrow.fs.FileSystem.from_uri` (credentials,
    region, and on-prem ``endpoint_override`` come from the URI query string or the
    environment); an unknown scheme falls back to an fsspec backend wrapped behind the
    same `pyarrow.fs` interface, so third-party backends work with no code change.

    `filesystem` accepts an already-constructed filesystem — a `pyarrow.fs.FileSystem`,
    a `pyarrow.fs.PyFileSystem`, or an **fsspec** filesystem instance — and uses it
    verbatim. This is the "bring your own filesystem" path: a user who has already built
    an `S3FileSystem` with a custom retry strategy, or holds an authenticated fsspec
    handle, hands it in rather than re-expressing it as a URI. It wins over
    `storage_options`.

    `storage_options` is the portable credential dict every other engine speaks (fsspec,
    delta-rs, Polars, pandas): keys like ``key``/``secret``/``endpoint_url`` for S3,
    ``token`` for GCS, ``account_name``/``account_key`` for Azure. It is applied to the
    native backend, so a boto/gcsfs/adlfs-style config works without threading each value
    into the URI. Unlike a live `filesystem` object it is a plain dict, so it survives to a
    distributed worker unchanged — which is why the file sources thread *this*, not the
    handle, onto their splits.
    """
    if filesystem is not None:
        return _wrap_user_filesystem(path, filesystem)
    scheme = _scheme(path)
    if scheme in ("", "file") and not storage_options:
        prefix = _local_prefix(path)
        return _ArrowFileSystem(pafs.LocalFileSystem(), prefix, atomic_rename=True)
    # Cache the resolved object-store filesystem per (scheme, authority, query-options): every
    # `from_uri` re-walks the credential chain (an IMDS round-trip on S3) and opens a fresh
    # connection pool, and the scan path resolves a filesystem *per split / footer / open* —
    # precisely the many-small-files tax. The FS is stateless/thread-safe and identical for
    # every object key in the same bucket + config, so it is safe to memoize (the native store
    # cache does the same). The object *path* is dropped from the key so all keys share one FS.
    base = path.split("?", 1)[0]
    query = path.split("?", 1)[1] if "?" in path else ""
    authority = base.split("://", 1)[1].split("/", 1)[0] if "://" in base else ""
    reduced = f"{scheme}://{authority}" + (f"?{query}" if query else "")
    if storage_options:
        # A dict is unhashable and would defeat the lru_cache, so fold it into a hashable
        # key rather than dropping the cache — a scan still resolves the FS once, not per
        # split. Sorted so option order never splits the cache.
        return _resolve_uri_fs_opts(reduced, tuple(sorted(storage_options.items())))
    return _resolve_uri_fs(reduced)


def _wrap_user_filesystem(path: str, filesystem: Any) -> FileSystem:
    """Wrap a user-provided filesystem in the façade, so the rest of IO is unchanged.

    Accepts a native `pyarrow.fs.FileSystem`/`PyFileSystem` directly, and an fsspec
    filesystem instance via pyarrow's `FSSpecHandler`. The prefix is derived the same way
    the string paths' is — object stores strip the scheme, so the caller keeps passing full
    URIs and the façade maps them to in-store paths.
    """
    if isinstance(filesystem, pafs.FileSystem):
        fs = filesystem
    else:
        # Duck-typed fsspec: anything with `_strip_protocol` is an fsspec AbstractFileSystem.
        try:
            from pyarrow.fs import FSSpecHandler, PyFileSystem
        except ImportError as exc:  # pragma: no cover - pyarrow always ships these
            raise IOError("wrapping an fsspec filesystem needs pyarrow's FSSpecHandler") from exc
        if not hasattr(filesystem, "_strip_protocol"):
            raise IOError(
                "filesystem= must be a pyarrow.fs.FileSystem or an fsspec filesystem "
                f"instance, got {type(filesystem).__name__}"
            )
        fs = PyFileSystem(FSSpecHandler(filesystem))
    scheme = _scheme(path)
    prefix = f"{scheme}://" if scheme not in ("", "file") else _local_prefix(path)
    is_object_store = _SCHEME_ALIASES.get(scheme, scheme) in _OBJECT_STORE_SCHEMES
    return _ArrowFileSystem(
        fs, prefix, atomic_rename=not is_object_store, cacheable=is_object_store
    )


@functools.lru_cache(maxsize=128)
def _resolve_uri_fs_opts(uri: str, options: tuple[tuple[str, str], ...]) -> FileSystem:
    """`_resolve_uri_fs` for an explicit `storage_options` set — folded into `?query` so the
    one builder handles both the URI-carried and dict-carried config identically.

    A value may be an ``env:NAME`` / ``file:PATH`` reference, resolved here — on the machine
    building the filesystem, which on a distributed read is the worker. That keeps a secret
    key out of the `storage_options` dict that rides the split (only the reference travels),
    the same discipline the crypto-key and connector-credential paths already use. The
    cache keys on the *reference*, so the resolved secret never enters a cache key either."""
    from batcher.io.credentials import resolve_secret

    resolved = [(k, resolve_secret(v, what=f"storage option {k}") or "") for k, v in options]
    extra = "&".join(f"{k}={v}" for k, v in resolved)
    joined = f"{uri}{'&' if '?' in uri else '?'}{extra}" if extra else uri
    return _resolve_uri_fs(joined)


@functools.lru_cache(maxsize=128)
def _resolve_uri_fs(uri: str) -> FileSystem:
    """Build (once, cached) the `pyarrow.fs` façade for an object-store `uri` reduced to
    `scheme://authority?query` (see `resolve_filesystem`)."""
    scheme = _scheme(uri)
    # Canonicalize BEFORE `from_uri`, not after. `s3a://` (the Hadoop spelling) and
    # `gcs://` name backends pyarrow implements natively but does not answer to under
    # those aliases, so asking it first and aliasing only in the failure path meant every
    # Hadoop-ecosystem URI silently took the slower fsspec route — correct results, quietly
    # worse, with nothing to indicate it.
    canonical_uri = uri
    canonical_scheme = _SCHEME_ALIASES.get(scheme, scheme)
    if canonical_scheme != scheme:
        canonical_uri = f"{canonical_scheme}{uri[len(scheme) :]}"
    try:
        fs, in_path = pafs.FileSystem.from_uri(canonical_uri)
    except (ValueError, OSError, pa.ArrowInvalid, pa.ArrowNotImplementedError) as e:
        # "Scheme not implemented natively" (→ fsspec) vs "implemented, but you passed an
        # option it does not take". Both arrive as ArrowInvalid; falling back on the second
        # is harmful, because the fsspec path keeps the query string (`strip_query=False`,
        # for presigned URLs) so a rejected `?anonymous=true` becomes part of the KEY —
        # surfacing a config mistake as `NoSuchKey` on a path nobody asked for.
        if "query parameter" in str(e):
            # `from_uri` takes a narrow option set, but the `S3FileSystem` *constructor*
            # takes the ones an on-prem store actually needs — explicit keys, `anonymous`,
            # `force_virtual_addressing` (path- vs virtual-hosted style), timeouts, a proxy.
            # Build it directly rather than refusing: refusing left Ceph-behind-a-proxy and
            # path-style-only deployments with no in-URI escape hatch at all.
            if canonical_scheme == "s3":
                return _s3_with_options(uri)
            raise IOError(f"unsupported option in {scheme}:// URI: {e}") from e
        # A scheme pyarrow.fs doesn't implement natively → fsspec fallback.
        return _fsspec_backed(scheme, uri)
    path = uri
    # The prefix is `scheme://authority`; compute it from the path with any `?query`
    # (config like endpoint_override) removed, since pyarrow's in_path excludes both.
    # `from_uri` also strips a trailing slash from `in_path` (``…/dir/`` → ``…/dir``),
    # so subtracting `len(in_path)` from the *un*-trimmed base mis-slices the prefix by
    # the slash (``s3://`` → ``s3://r``) and every later `_p()` drops a real character.
    # Strip the trailing slash off `base` before the suffix math so the two align.
    base = path.split("?", 1)[0]
    trimmed = base.rstrip("/")
    if in_path and trimmed.endswith(in_path):
        # The common shape: the backend's path is a suffix of the URI.
        prefix, root = trimmed[: len(trimmed) - len(in_path)], ""
    else:
        # Azure: `in_path` (``container/key``) is not a suffix of
        # ``abfs://container@account…/key``, so no suffix arithmetic can recover it — the
        # old `len()` subtraction sliced mid-hostname and sent every read, list, and WRITE
        # to a wrong container/key. Split at the authority and let the backend say what it
        # prepends. `rstrip` because `from_uri` reports a bare authority as ``container/``
        # and the URL path already starts with "/".
        prefix, urlpath = _split_authority(trimmed)
        root = (in_path[: len(in_path) - len(urlpath)] if urlpath else in_path).rstrip("/")
    canonical = _SCHEME_ALIASES.get(scheme, scheme)
    is_object_store = canonical in _OBJECT_STORE_SCHEMES
    return _ArrowFileSystem(
        fs, prefix, atomic_rename=not is_object_store, cacheable=is_object_store, root=root
    )


def _split_authority(uri: str) -> tuple[str, str]:
    """Split ``scheme://authority/path`` into its ``scheme://authority`` and ``/path``.

    Returns the whole URI and ``""`` when there is no path component, so the caller's
    suffix arithmetic degenerates safely rather than slicing into the authority."""
    marker = uri.find("://")
    if marker < 0:
        return uri, ""
    slash = uri.find("/", marker + 3)
    return (uri, "") if slash < 0 else (uri[:slash], uri[slash:])


#: Query options `S3FileSystem.__init__` accepts, with the coercion each needs. Anything
#: outside this set is rejected by name so a typo is not silently ignored by the builder.
_S3_BOOL_OPTS = ("anonymous", "force_virtual_addressing", "background_writes")
_S3_INT_OPTS = ("connect_timeout", "request_timeout")

#: Attempts a throttled S3 request gets before the read fails. pyarrow's default is three,
#: chosen against a client opening a handful of connections; this engine opens as many as the
#: machine can drive — up to 256 concurrent GETs on a dense node — and a store's answer to that
#: is `503 SlowDown`, which is not a failure but a request to wait. Three attempts turns a
#: throttle into a failed query on exactly the scans worth running on such a machine.
#:
#: Raised rather than made unbounded: a store that is genuinely down should still fail the
#: query rather than retry it for minutes. `retry_max_attempts` in the URI overrides it, in
#: both directions.
_S3_DEFAULT_RETRY_ATTEMPTS = 8
_S3_STR_OPTS = (
    "access_key",
    "secret_key",
    "session_token",
    "role_arn",
    "session_name",
    "external_id",
    "region",
    "scheme",
    "endpoint_override",
)


def _s3_with_options(uri: str) -> FileSystem:
    """An `S3FileSystem` built from the URI query, for options `from_uri` will not take.

    The on-prem escape hatch: MinIO and Ceph RGW commonly need path-style addressing
    (``force_virtual_addressing=false``), a plain-HTTP endpoint, explicit keys, or a
    longer timeout — none of which `from_uri` accepts. An unknown option is an error
    naming the option, because `S3FileSystem` would otherwise ignore it silently and the
    user would debug a connection that quietly used none of their settings.

    Every filesystem built here also gets a retry budget sized for the fan-out this engine
    actually opens (`_S3_DEFAULT_RETRY_ATTEMPTS`), overridable with ``retry_max_attempts``."""
    from urllib.parse import parse_qsl, urlsplit

    parts = urlsplit(uri)
    opts: dict[str, object] = {}
    attempts = _S3_DEFAULT_RETRY_ATTEMPTS
    for key, value in parse_qsl(parts.query):
        if key == "retry_max_attempts":
            attempts = max(1, int(value))
        elif key in _S3_BOOL_OPTS:
            opts[key] = value.strip().lower() in ("1", "true", "yes", "on")
        elif key in _S3_INT_OPTS:
            opts[key] = int(value)
        elif key in _S3_STR_OPTS:
            opts[key] = value
        else:
            raise IOError(
                f"unknown s3:// option {key!r}. Supported: retry_max_attempts, "
                f"{', '.join(sorted(_S3_BOOL_OPTS + _S3_INT_OPTS + _S3_STR_OPTS))}"
            )
    opts["retry_strategy"] = pafs.AwsStandardS3RetryStrategy(max_attempts=attempts)
    try:
        fs = pafs.S3FileSystem(**opts)  # type: ignore[arg-type]
    except (ValueError, OSError, pa.ArrowInvalid) as exc:
        raise IOError(f"cannot open s3:// storage with the given options: {exc}") from exc
    # A directly-constructed S3FileSystem addresses objects as `bucket/key`, so the mapping
    # is the plain `s3://` prefix strip — the same shape `from_uri` produces.
    return _ArrowFileSystem(fs, "s3://", atomic_rename=False, cacheable=True)


def _fsspec_backed(scheme: str, path: str) -> FileSystem:
    """Wrap an fsspec backend behind the `pyarrow.fs` interface (the escape hatch for
    schemes pyarrow does not implement natively)."""
    try:
        import fsspec
        from pyarrow.fs import FSSpecHandler, PyFileSystem
    except ImportError as exc:
        raise IOError(
            f"reading {scheme}:// paths needs the cloud extra: pip install 'batcher-engine[cloud]'"
        ) from exc
    protocol = _SCHEME_ALIASES.get(scheme, scheme)
    try:
        fsspec_fs = fsspec.filesystem(protocol)
    except ImportError as exc:
        # fsspec knows the protocol but its driver package is absent. Name the driver, not
        # the `[cloud]` extra — that extra carries s3fs/gcsfs/adlfs, and a scheme outside
        # those (an in-house or third-party backend) is not fixed by installing it.
        raise IOError(
            f"reading {scheme}:// paths needs the fsspec driver for '{protocol}': {exc}"
        ) from exc
    except (ValueError, OSError) as exc:
        # Two very different failures share these types, and conflating them sends the user
        # the wrong way. Only "Protocol not known" means the scheme is unimplemented —
        # `wasb://`/`wasbs://` (legacy Azure Blob) land there. Everything else is a backend
        # that exists but could not be constructed: adlfs raising ValueError because no
        # Azure credentials were supplied, or the HDFS driver raising OSError because it
        # cannot load libjvm. Reporting either of those as "unsupported scheme" would send
        # someone hunting for a missing feature instead of fixing their credentials or JVM,
        # so pass the backend's own message through untouched.
        if "protocol not known" not in str(exc).lower():
            raise IOError(f"cannot open {scheme}:// storage: {exc}") from exc
        hint = (
            " — for legacy Azure Blob URIs use the current abfs:// / abfss:// spelling"
            if protocol in ("wasb", "wasbs")
            else ""
        )
        raise IOError(f"unsupported storage scheme {scheme}://: {exc}{hint}") from exc
    fs = PyFileSystem(FSSpecHandler(fsspec_fs))
    # The in-filesystem path is whatever fsspec's `_strip_protocol` produces — object
    # stores strip the scheme ("bucket/key"), but HTTP(S) keep the whole URL. Derive
    # the prefix from that so listing/globbing line up with the paths fsspec returns,
    # and keep the query (presigned-URL signatures live there) by not stripping it.
    stripped = fsspec_fs._strip_protocol(path)
    prefix = path[: len(path) - len(stripped)] if path.endswith(stripped) else ""
    return _ArrowFileSystem(
        fs, prefix, atomic_rename=protocol not in _OBJECT_STORE_SCHEMES, strip_query=False
    )
