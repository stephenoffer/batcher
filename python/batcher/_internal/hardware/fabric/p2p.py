"""Device-to-device reach: which pairs exchange on the fabric, and at what rate.

`device_links.device_topology` answers how far two devices sit apart *on the PCI bus*, which
is the whole answer on a PCIe-only node and the wrong axis on an NVLink one. Two devices under
different root complexes — `sys`, the worst class the bus has — exchange at 900 GB/s if a
coherent link joins them, and a group chosen by bus distance alone will pick the pair sharing
a switch over the pair sharing a fabric. That is a 20x error in the direction that looks
plausible: the placement is defensible on the topology it read, and it is not the topology the
traffic uses.

This module is the join of the two. `peer_matrix` overlays the live NVLink pairs onto the bus
matrix so `"nvlink"` is a class alongside `pix`/`pxb`/.../`sys`, `bandwidth_matrix` prices each
pair, and `peer_islands` reports the sets of devices a collective can stay inside. Everything
takes its inputs by argument with a live default, so a placement decision is testable against a
described node rather than only on the one the test happens to run on.

The conservative direction is preserved throughout: an unreadable pair reports `sys` and the
slowest bandwidth the node can offer, never the fabric. A group placed against a fabric that
is not there runs; a group kept off one that is there merely runs slower.

A neutral utility: any layer may import `_internal`.
"""

from __future__ import annotations

from collections.abc import Sequence

from batcher._internal.hardware.fabric.pcie import PCIE_CLASSES

__all__ = [
    "NVLINK_CLASS",
    "P2P_CLASSES",
    "bandwidth_matrix",
    "bisection_gbps",
    "fabric_fraction",
    "host_staged_pairs",
    "island_of",
    "p2p_capable",
    "peer_bandwidth_gbps",
    "peer_class",
    "peer_group_class",
    "peer_islands",
    "peer_matrix",
    "peer_summary",
    "tightest_peer_group",
]

#: The class a pair on a coherent device fabric reports. Named rather than spelled at each
#: site because it is compared against `PCIE_CLASSES` entries in five places, and a typo
#: there degrades silently to "not on the fabric" rather than raising.
NVLINK_CLASS = "nvlink"

#: Every device-to-device class, closest first: the fabric, then the bus classes in
#: `PCIE_CLASSES` order. A caller ranks with `P2P_CLASSES.index`, so this order *is* the cost
#: model's ordering and adding a class in the wrong position silently reprices every pair.
P2P_CLASSES = (NVLINK_CLASS, *PCIE_CLASSES)

#: Classes whose pairs copy directly, device to device, with no bounce through host memory:
#: the fabric, and the bus classes where the transfer stays below a host bridge. From `phb`
#: outward the copy reaches the CPU and turns around, which is a staged transfer whatever the
#: driver calls it.
_DIRECT_CLASSES = frozenset({NVLINK_CLASS, "pix", "pxb"})

#: Fraction of a link's rate a pair actually achieves, by class. A peer copy under one switch
#: runs near the link rate; one that reaches the root complex shares the host bridge with every
#: other device on it; one crossing the socket contends with all coherent traffic on the
#: machine. Multipliers rather than absolute figures, so a node's *measured* PCIe rate stays
#: the basis and nothing here invents a bandwidth number.
_CLASS_EFFICIENCY: dict[str, float] = {
    "pix": 1.0,
    "pxb": 0.8,
    "phb": 0.5,
    "node": 0.4,
    "sys": 0.25,
}


def peer_matrix(
    pcie_matrix: Sequence[Sequence[str]] | None = None,
    nvlink_pairs: Sequence[tuple[int, int]] | None = None,
) -> tuple[tuple[str, ...], ...]:
    """The device-to-device class matrix, with the fabric overlaid on the bus.

    `m[i][j]` is `"nvlink"` when the pair shares an active link, otherwise the PCIe class
    between them. The diagonal is `"nvlink"` on a fabric node and `"pix"` otherwise, which is
    the same statement either way — a device is as close to itself as the node's best class
    allows — and it keeps `peer_bandwidth_gbps` from pricing a device's own memory at the bus.

    Args:
        pcie_matrix: The bus matrix, or `None` to take `device_topology()` live.
        nvlink_pairs: Active `(low, high)` fabric pairs, or `None` to take `p2p_pairs()` live.

    Returns:
        An n-by-n matrix in device-index order, empty when the bus topology is unreadable.
        Fabric pairs naming a device the bus matrix does not have are ignored rather than
        growing the matrix: the two probes disagreeing about the device count is a reason to
        distrust the extra entry, not to invent a row for it.
    """
    from batcher._internal.hardware.fabric.device_links import device_topology
    from batcher._internal.hardware.fabric.nvlink import p2p_pairs

    base = device_topology() if pcie_matrix is None else pcie_matrix
    if not base:
        return ()
    rows = [list(row) for row in base]
    n = len(rows)
    pairs = p2p_pairs() if nvlink_pairs is None else nvlink_pairs
    linked = False
    for a, b in pairs:
        if 0 <= a < n and 0 <= b < n and a != b and b < len(rows[a]) and a < len(rows[b]):
            rows[a][b] = rows[b][a] = NVLINK_CLASS
            linked = True
    if linked:
        for i in range(n):
            rows[i][i] = NVLINK_CLASS
    return tuple(tuple(row) for row in rows)


def peer_class(a: int, b: int, matrix: Sequence[Sequence[str]] | None = None) -> str:
    """The class of one device pair, or `"sys"` when it cannot be read.

    Args:
        a: One device index.
        b: The other.
        matrix: A `peer_matrix`, or `None` to take one live.

    Returns:
        A name from `P2P_CLASSES`. An index the matrix does not have reports `"sys"`, the
        coarsest class, on the standing rule that an unknown distance is never assumed short.
    """
    m = peer_matrix() if matrix is None else matrix
    if a < 0 or b < 0 or a >= len(m) or b >= len(m) or b >= len(m[a]):
        return "sys"
    return m[a][b]


def peer_bandwidth_gbps(
    peer_cls: str, *, nvlink_gbps: float = 0.0, pcie_gbps: float = 0.0
) -> float:
    """What a pair of this class actually exchanges at, in gigabytes per second.

    A fabric pair runs at the device model's published per-device NVLink rate. A bus pair runs
    at the host link's rate derated for how much of the machine the transfer crosses — under
    one switch it is the link, at the root complex it shares the bridge, across the socket it
    contends with everything coherent on the node.

    Args:
        peer_cls: A class from `P2P_CLASSES`.
        nvlink_gbps: The device model's NVLink rate, `0.0` when unknown.
        pcie_gbps: The negotiated host link rate, `0.0` when unknown.

    Returns:
        Gigabytes per second, `0.0` when the relevant rate is unknown. Zero means "no opinion"
        and callers must treat it as such: a cost model dividing by it produces an infinity
        that reads as a refusal rather than as the missing probe it is.
    """
    if peer_cls == NVLINK_CLASS:
        return max(0.0, nvlink_gbps)
    return max(0.0, pcie_gbps) * _CLASS_EFFICIENCY.get(peer_cls, _CLASS_EFFICIENCY["sys"])


def bandwidth_matrix(
    matrix: Sequence[Sequence[str]] | None = None,
    *,
    nvlink_gbps: float = 0.0,
    pcie_gbps: float = 0.0,
) -> tuple[tuple[float, ...], ...]:
    """Every device pair's exchange rate, in gigabytes per second.

    The matrix a transfer planner divides bytes by. The diagonal is included and carries the
    same rate as a pair of that class, which is deliberate: a "transfer" from a device to
    itself is not free in a planner that models it, it simply never appears.

    Args:
        matrix: A `peer_matrix`, or `None` to take one live.
        nvlink_gbps: The device model's NVLink rate.
        pcie_gbps: The negotiated host link rate.

    Returns:
        An n-by-n matrix of rates, empty when the class matrix is.
    """
    m = peer_matrix() if matrix is None else matrix
    return tuple(
        tuple(peer_bandwidth_gbps(c, nvlink_gbps=nvlink_gbps, pcie_gbps=pcie_gbps) for c in row)
        for row in m
    )


def p2p_capable(a: int, b: int, matrix: Sequence[Sequence[str]] | None = None) -> bool:
    """Whether devices `a` and `b` can copy directly, without staging through host memory.

    True on the fabric and below a single PCIe switch; false once the path reaches a host
    bridge, where the bytes cross into host memory and back however the copy is spelled.

    Args:
        a: One device index.
        b: The other.
        matrix: A `peer_matrix`, or `None` to take one live.

    Returns:
        True for a direct pair. A device paired with itself is direct. An unreadable pair is
        false, so a planner allocates the staging buffer it may turn out not to need rather
        than discovering mid-transfer that it does.
    """
    if a == b:
        return True
    return peer_class(a, b, matrix) in _DIRECT_CLASSES


def host_staged_pairs(
    matrix: Sequence[Sequence[str]] | None = None,
) -> tuple[tuple[int, int], ...]:
    """Every device pair whose exchange has to bounce through host memory.

    The set a staging budget is sized for, and the figure that says whether a device-to-device
    plan is worth making at all: a node where every pair is staged gains nothing from one.

    Args:
        matrix: A `peer_matrix`, or `None` to take one live.

    Returns:
        Ascending `(low, high)` pairs, each once. Empty on a fully direct node and on an
        unreadable one — the two are distinguished by whether the matrix itself is empty.
    """
    m = peer_matrix() if matrix is None else matrix
    return tuple(
        (a, b) for a in range(len(m)) for b in range(a + 1, len(m)) if not p2p_capable(a, b, m)
    )


def peer_islands(
    matrix: Sequence[Sequence[str]] | None = None, *, classes: Sequence[str] = (NVLINK_CLASS,)
) -> tuple[tuple[int, ...], ...]:
    """Groups of devices mutually reachable through links of the given classes.

    The unit a collective should be placed inside. On an eight-device DGX every device is in
    one island; on a pair of four-device boards there are two, and a collective spanning them
    runs at the slower of the two links for every step.

    Args:
        matrix: A `peer_matrix`, or `None` to take one live.
        classes: Which classes count as being on the same island. The default is the fabric
            alone; passing `("nvlink", "pix")` asks the same question of switch-local pairs.

    Returns:
        Islands as ascending index tuples, ordered by their lowest member. Every device
        appears exactly once, so a device with no qualifying link is its own island of one —
        the honest shape, and one a caller can size against without a separate absence check.
    """
    m = peer_matrix() if matrix is None else matrix
    n = len(m)
    wanted = frozenset(classes)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a in range(n):
        for b in range(a + 1, min(n, len(m[a]))):
            if m[a][b] in wanted:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return tuple(tuple(sorted(members)) for _, members in sorted(groups.items()))


def fabric_fraction(devices: Sequence[int], matrix: Sequence[Sequence[str]] | None = None) -> float:
    """What fraction of a group's pairs exchange on the coherent fabric.

    The one figure that says whether a set of devices is a group at all: `1.0` is a single
    NVLink domain, `0.0` is a set of boards that happen to share a chassis. It is the input to
    a fan-out width decision — a collective spread past its fabric pays for every extra device
    twice, once in the copy and once in the round it adds to the schedule.

    Args:
        devices: Device indices taking part.
        matrix: A `peer_matrix`, or `None` to take one live.

    Returns:
        `0.0` through `1.0`; `0.0` for fewer than two devices, where there are no pairs.
    """
    members = sorted(set(devices))
    pairs = [(a, b) for i, a in enumerate(members) for b in members[i + 1 :]]
    if not pairs:
        return 0.0
    m = peer_matrix() if matrix is None else matrix
    return sum(1 for a, b in pairs if peer_class(a, b, m) == NVLINK_CLASS) / len(pairs)


def island_of(device: int, islands: Sequence[Sequence[int]] | None = None) -> tuple[int, ...]:
    """The island `device` belongs to, or an empty tuple when it is in none.

    Args:
        device: A device index.
        islands: `peer_islands` output, or `None` to take it live.

    Returns:
        The island's members ascending, empty when the device is not in the topology at all.
    """
    for island in peer_islands() if islands is None else islands:
        if device in island:
            return tuple(island)
    return ()


def tightest_peer_group(
    size: int, matrix: Sequence[Sequence[str]] | None = None
) -> tuple[int, ...]:
    """The `size` devices that exchange fastest with each other, fabric included.

    The NVLink-aware answer to `device_links.tightest_device_group`, which ranks on the bus
    alone and so prefers two devices sharing a switch over two sharing a fabric. Same greedy
    shape — every device as a seed, nearest neighbours added, best worst-case pair kept — so
    the two agree exactly on a node with no fabric.

    Args:
        size: How many devices the group needs.
        matrix: A `peer_matrix`, or `None` to take one live.

    Returns:
        Device indices ascending, empty when the topology is unreadable or holds fewer than
        `size` devices. Empty is "no opinion", not a refusal.
    """
    m = peer_matrix() if matrix is None else matrix
    if size <= 0 or len(m) < size:
        return ()
    rank = {name: i for i, name in enumerate(P2P_CLASSES)}
    fallback = len(P2P_CLASSES)
    best: tuple[int, tuple[int, ...]] | None = None
    for seed in range(len(m)):
        order = sorted(range(len(m)), key=lambda j: (rank.get(m[seed][j], fallback), j))
        group = tuple(sorted(order[:size]))
        worst = max(rank.get(m[a][b], fallback) for a in group for b in group)
        if best is None or worst < best[0]:
            best = (worst, group)
    return best[1] if best is not None else ()


def peer_group_class(group: Sequence[int], matrix: Sequence[Sequence[str]] | None = None) -> str:
    """The worst pair class inside a group, which is what bounds a collective on it.

    Args:
        group: Device indices.
        matrix: A `peer_matrix`, or `None` to take one live.

    Returns:
        A name from `P2P_CLASSES`, or `""` when the topology is unreadable or the group has
        fewer than two devices to compare.
    """
    m = peer_matrix() if matrix is None else matrix
    if len(group) < 2 or any(i >= len(m) for i in group):
        return ""
    rank = {name: i for i, name in enumerate(P2P_CLASSES)}
    fallback = len(P2P_CLASSES)
    return max((m[a][b] for a in group for b in group), key=lambda c: rank.get(c, fallback))


def bisection_gbps(
    group: Sequence[int],
    matrix: Sequence[Sequence[str]] | None = None,
    *,
    nvlink_gbps: float = 0.0,
    pcie_gbps: float = 0.0,
) -> float:
    """What a group can move across its own worst cut, in gigabytes per second.

    An all-to-all moves roughly half its bytes across any bisection of the group, so this is
    the figure a redistribution's duration divides by — not the sum of every link, which is
    what a naive aggregate reports and which over-states a group split across two boards by
    the ratio of fabric to bus.

    Computed over the balanced cut that is cheapest to describe and worst in practice: the low
    half of the group against the high half, summing each crossing pair's rate. It is a bound
    rather than a measurement, and a conservative one, which is the useful direction.

    Args:
        group: Device indices.
        matrix: A `peer_matrix`, or `None` to take one live.
        nvlink_gbps: The device model's NVLink rate.
        pcie_gbps: The negotiated host link rate.

    Returns:
        Gigabytes per second, `0.0` for a group of fewer than two devices or when no rate is
        known.
    """
    members = sorted(set(group))
    if len(members) < 2:
        return 0.0
    m = peer_matrix() if matrix is None else matrix
    half = len(members) // 2
    left, right = members[:half], members[half:]
    total = 0.0
    for a in left:
        for b in right:
            total += peer_bandwidth_gbps(
                peer_class(a, b, m), nvlink_gbps=nvlink_gbps, pcie_gbps=pcie_gbps
            )
    return total


def peer_summary(matrix: Sequence[Sequence[str]] | None = None) -> dict:
    """The node's device-to-device shape as one record, for a report or a gauge.

    Args:
        matrix: A `peer_matrix`, or `None` to take one live.

    Returns:
        `devices`, `islands` (as lists), `largest_island`, `fabric_pairs` (pairs on the
        coherent fabric), `staged_pairs` (pairs needing a host bounce), and `class` (the
        node-wide worst pair). Zeroed on a node whose topology is unreadable.
    """
    m = peer_matrix() if matrix is None else matrix
    islands = peer_islands(m)
    n = len(m)
    fabric = sum(
        1 for a in range(n) for b in range(a + 1, n) if peer_class(a, b, m) == NVLINK_CLASS
    )
    return {
        "devices": n,
        "islands": [list(i) for i in islands],
        "largest_island": max((len(i) for i in islands), default=0),
        "fabric_pairs": fabric,
        "staged_pairs": len(host_staged_pairs(m)),
        "class": peer_group_class(tuple(range(n)), m),
    }
