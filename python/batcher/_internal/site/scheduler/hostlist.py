"""Turning a scheduler's compressed node list into node names.

Two shapes cover every scheduler in this package. Slurm and Flux compress an allocation into
a bracketed range notation, so a 512-node job arrives as one short string. PBS, LSF and Grid
Engine write a *file* instead, one line per task slot rather than per node.

Both are parsed here rather than by shelling out to `scontrol show hostnames`: a subprocess
per query is slow on a control-plane path, and the client binary is frequently absent from a
container that nevertheless has the environment.
"""

from __future__ import annotations

__all__ = ["expand_nodelist", "nodes_from_file", "nodes_from_pe_hostfile"]


def _split_entries(spec: str) -> list[str]:
    """Split a hostlist on the commas *between* names, ignoring those inside brackets.

    `gpu-[01-02,07],login01` is two names, not three. A plain split on commas is the shape of
    bug that turns one four-node group into two unresolvable hostnames.
    """
    out: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in spec:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            out.append("".join(current))
            current = []
            continue
        current.append(ch)
    out.append("".join(current))
    return [entry.strip() for entry in out if entry.strip()]


def _segments(entry: str) -> list[str | list[str]] | None:
    """One name split into literal text and the alternatives each bracket group expands to.

    Returns `None` for an unbalanced or empty group, which the caller passes through
    literally: a name we failed to parse is a name, and dropping it would read as a node the
    allocation does not have.
    """
    out: list[str | list[str]] = []
    rest = entry
    while rest:
        open_at = rest.find("[")
        if open_at < 0:
            out.append(rest)
            break
        close_at = rest.find("]", open_at)
        if close_at < 0:
            return None
        if open_at:
            out.append(rest[:open_at])
        parts = _expand_group(rest[open_at + 1 : close_at])
        if parts is None:
            return None
        out.append(parts)
        rest = rest[close_at + 1 :]
    return out


def _expand_group(spec: str) -> list[str] | None:
    """The values one bracket group stands for, preserving the zero padding Slurm uses.

    `001-003,007` yields `001 002 003 007`. Padding comes from the literal width of the lower
    bound, because that is what the node is actually named: `gpu-[008-010]` are nodes
    `gpu-008`..`gpu-010`, and an unpadded name does not resolve — the failure being a job that
    cannot reach three quarters of itself.

    Returns `None` for a group this cannot read (an inverted or non-numeric range), so the
    caller passes the whole name through literally rather than yielding no nodes for it.
    """
    out: list[str] = []
    for raw in spec.split(","):
        part = raw.strip()
        if not part:
            return None
        low, sep, high = part.partition("-")
        if not sep:
            out.append(part)
            continue
        if not (low.isdigit() and high.isdigit()):
            return None
        start, end = int(low), int(high)
        if end < start:
            return None
        out.extend(f"{n:0{len(low)}d}" for n in range(start, end + 1))
    return out or None


def _cross(segments: list[str | list[str]]) -> list[str]:
    """Every name a segmented entry stands for, in odometer order.

    A name may carry more than one bracket group — `rack[1-2]node[1-4]` is eight nodes, and a
    single-group parse read it as two plus four. Rarer than the one-group form, and it is
    exactly the hierarchical naming a large site uses.
    """
    names = [""]
    for segment in segments:
        if isinstance(segment, str):
            names = [name + segment for name in names]
        else:
            names = [name + value for name in names for value in segment]
    return names


def expand_nodelist(spec: str) -> tuple[str, ...]:
    """Expand a Slurm-style hostlist into individual node names.

    Args:
        spec: The hostlist, as `"gpu-[001-004,007],login01"`, `"node[01-04]-ib"` (the
            interconnect-side naming an HPC site uses), or `"rack[1-2]node[1-4]"`.

    Returns:
        Node names in the order the list gives them, deduplicated. A name this cannot parse
        is passed through literally rather than dropped, since yielding nothing for it would
        read as an allocation that does not include it. Empty for an empty spec, which
        callers read as "no node list published".
    """
    if not spec or not spec.strip():
        return ()
    out: list[str] = []
    for entry in _split_entries(spec.strip()):
        segments = _segments(entry)
        out.extend(_cross(segments) if segments is not None else [entry])
    return tuple(dict.fromkeys(out))


def nodes_from_file(path: str) -> tuple[str, ...]:
    """Node names from a scheduler's host file, deduplicated in first-seen order.

    PBS writes one line per *task slot*, so a four-node job with eight tasks each lists every
    node eight times. The distinct names in order are the allocation; the repetition is the
    task layout, which `tasks` already carries.

    Args:
        path: The host file named by the scheduler.

    Returns:
        The distinct node names, or `()` when the file cannot be read.
    """
    try:
        with open(path) as f:
            names = [line.strip() for line in f if line.strip()]
    except OSError:
        return ()
    return tuple(dict.fromkeys(names))


def nodes_from_pe_hostfile(path: str) -> tuple[tuple[str, ...], int]:
    """Nodes and total slots from a Grid Engine `PE_HOSTFILE`.

    Grid Engine's file is the one that is not a bare host list: each line is
    `hostname nslots queue processor-range`, so the slot count is a column rather than a
    repeat count. Reading it as a plain host file loses the task layout entirely.

    Args:
        path: The file named by `PE_HOSTFILE`.

    Returns:
        The distinct node names in order, and the summed slot count (`0` when no line
        published one). `((), 0)` when the file cannot be read.
    """
    try:
        with open(path) as f:
            lines = [line.split() for line in f if line.strip()]
    except OSError:
        return ((), 0)
    names: list[str] = []
    slots = 0
    for parts in lines:
        names.append(parts[0])
        if len(parts) > 1 and parts[1].isdigit():
            slots += int(parts[1])
    return (tuple(dict.fromkeys(names)), slots)
