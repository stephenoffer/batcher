#!/usr/bin/env python3
"""Draw `graph_isolated_nodes.svg` -- the same edge table, before and after a node table.

Source of truth: `docs/user-guide/analyze/graphs.md`, section "Isolated nodes are invisible
unless you say otherwise". `bg.Graph.from_edges` over the one edge `(1, 2)` has two nodes and
an average degree of 1.0; `.with_nodes` over the node table `[1, 2, 3, 4]` has four nodes and
an average degree of 0.5. An edge table cannot express a node with no edges, because such a
node appears in no row. Both numbers come from that page's executed example.

Drawn as a before and after because the graph's edges do not change at all: only which
nodes exist does, and with it every per-node average.
"""

from __future__ import annotations

from _authoring import (
    AMBER_DEEP,
    BLUE,
    FONT,
    GREY,
    arrow,
    band,
    code,
    heading,
    label,
    note,
    svg,
    write,
)

W, H = 980, 450

NODE_Y = 196
LOWER_Y = 276


def node(x: float, y: float, name: str, isolated: bool = False) -> str:
    """A graph node. An isolated one is dashed and amber, so shape and colour both mark it."""
    stroke = AMBER_DEEP if isolated else BLUE
    dash = "stroke-dasharray: 5 4;" if isolated else ""
    # Inline style, because the `surface` class sets a stroke that would win over an attribute.
    return (
        f'<circle cx="{x}" cy="{y}" r="22" class="surface" '
        f'style="stroke: {stroke}; stroke-width: 2.4; {dash}"/>'
        f'<text x="{x}" y="{y + 5}" text-anchor="middle" font-family="{FONT}" font-size="15" '
        f'font-weight="800" class="t-title">{name}</text>'
    )


def stat(x: float, y: float, name: str, value: str) -> str:
    """A measured number under its function name."""
    return (
        note(x, y, name, anchor="middle")
        + f'<text x="{x}" y="{y + 30}" text-anchor="middle" font-family="{FONT}" '
        f'font-size="24" font-weight="800" class="t-title">{value}</text>'
    )


body: list[str] = [
    # --- before -------------------------------------------------------------------------
    band(20, 20, 400, 360, "EDGES ONLY", "grey"),
    code(40, 56, ["src  dst", "  1    2"], 120, 12.5),
    note(176, 76, "Graph.from_edges(...)"),
    note(176, 94, "one edge row"),
    node(120, NODE_Y, "1"),
    node(320, NODE_Y, "2"),
    arrow(142, NODE_Y, 294, NODE_Y),
    label(220, NODE_Y - 12, "edge", anchor="middle", size=12),
    note(220, LOWER_Y, "a node with no edges is in", anchor="middle"),
    note(220, LOWER_Y + 18, "no row, so it does not exist", anchor="middle"),
    f'<path d="M 40 314 L 400 314" stroke="{GREY}" stroke-width="1"/>',
    stat(130, 338, "num_nodes()", "2"),
    stat(310, 338, "average_degree", "1.0"),
    # --- the change -----------------------------------------------------------------------
    arrow(426, NODE_Y, 552, NODE_Y),
    label(489, NODE_Y - 14, "with_nodes()", anchor="middle", size=12),
    note(489, NODE_Y + 24, "node: 1, 2, 3, 4", anchor="middle"),
    # --- after --------------------------------------------------------------------------
    band(560, 20, 400, 360, "EDGES PLUS A NODE TABLE", "blue"),
    code(580, 56, ["src  dst", "  1    2"], 120, 12.5),
    note(716, 76, "the same edge row,"),
    note(716, 94, "plus every node"),
    node(660, 160, "1"),
    node(860, 160, "2"),
    arrow(682, 160, 834, 160),
    label(760, 148, "edge", anchor="middle", size=12),
    node(660, 244, "3", isolated=True),
    node(860, 244, "4", isolated=True),
    heading(760, 238, "ISOLATED", anchor="middle", kind="amber"),
    note(760, 256, "no edges", anchor="middle"),
    f'<path d="M 580 314 L 940 314" stroke="{GREY}" stroke-width="1"/>',
    stat(670, 338, "num_nodes()", "4"),
    stat(850, 338, "average_degree", "0.5"),
    note(
        490,
        418,
        "Both answers are correct for their question. Attach the node table when the "
        "denominator should include every node.",
        anchor="middle",
    ),
]

write("graph_isolated_nodes", svg(W, H, "".join(body)))
print("wrote graph_isolated_nodes.svg")
