#!/usr/bin/env python3
"""Render the documentation architecture diagrams to PNG with Graphviz.

Each diagram is defined here as a Graphviz body; this script wraps it in a shared
style header and shells out to `dot` to produce `<name>.png` next to this file. The
PNGs are committed so the Sphinx build needs no Graphviz; rerun this only when a
diagram changes.

Usage:
    python docs/_static/diagrams/render.py        # needs `dot` (brew install graphviz)
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DPI = "192"  # retina-crisp; furo scales images to the column width

# Shared palette — control plane (Python) is blue, data plane (Rust) is amber,
# neutral plan/engine nodes are slate. A professional two-tone scheme; the white
# canvas reads fine on light + dark (the docs frame these PNGs in a white card).
BLUE, BLUE_BG, BLUE_EDGE = "#93c5fd", "#eff6ff", "#2563eb"  # blue
ORANGE, ORANGE_BG, ORANGE_EDGE = "#fcd34d", "#fffbeb", "#d97706"  # amber
GREY, GREY_BG = "#cbd5e1", "#f1f5f9"  # slate
INK = "#334155"

HEADER = f"""
  graph [bgcolor="white", fontname="Helvetica", nodesep=0.35, ranksep=0.5];
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12,
        color="{GREY}", fillcolor="{GREY_BG}", fontcolor="{INK}", penwidth=1.3,
        margin="0.28,0.16"];
  edge [color="#5f6368", penwidth=1.4, arrowsize=0.8, fontname="Helvetica",
        fontsize=10, fontcolor="#5f6368"];
"""

DIAGRAMS: dict[str, str] = {}


def diagram(name: str) -> "callable":
    def register(fn):
        DIAGRAMS[name] = f"digraph {name} {{\n{HEADER}\n{fn()}\n}}\n"
        return fn

    return register


@diagram("two_planes")
def _two_planes() -> str:
    return f"""
  rankdir=TB; compound=true;
  subgraph cluster_py {{
    label="Python — control plane"; labeljust="l"; margin=16;
    fontname="Helvetica-Bold"; fontsize=13; fontcolor="{BLUE_EDGE}";
    style="rounded,filled"; fillcolor="{BLUE_BG}"; color="{BLUE}";
    node [color="{BLUE}", fillcolor="white"];
    edge [color="{BLUE_EDGE}"];
    p1 [label="Dataset / SQL\\nlazy, immutable"];
    p2 [label="Kyber\\noptimize → PhysicalPlan"];
    p3 [label="Carbonite\\ncheck feasibility, allocate"];
    p4 [label="Core\\ndrive execution, measure"];
    p1 -> p2 -> p3 -> p4;
  }}
  subgraph cluster_rs {{
    label="Rust — data plane"; labeljust="l"; margin=16;
    fontname="Helvetica-Bold"; fontsize=13; fontcolor="{ORANGE_EDGE}";
    style="rounded,filled"; fillcolor="{ORANGE_BG}"; color="{ORANGE}";
    node [color="{ORANGE}", fillcolor="white"];
    r1 [label="bc-py — FFI boundary\\nzero-copy Arrow"];
    r2 [label="bc-interp\\ninterpreter · parallel · JIT"];
    r3 [label="bc-runtime\\nmergeable agg / join / sort / window"];
    r4 [label="bc-codegen\\nCranelift JIT for expressions"];
    r5 [label="bc-sketches\\nHLL / KLL / Count-Min"];
    r6 [label="bc-transport\\nArrow Flight shuffle"];
    r1 -> r2 -> r3 -> r4 -> r5 -> r6 [style=invis];
  }}
  p4 -> r1 [ltail=cluster_py, lhead=cluster_rs, constraint=false,
            label="JSON IR +\\nArrow batches  ", style=dashed, penwidth=1.6];
"""


@diagram("pipeline_breakers")
def _pipeline_breakers() -> str:
    return f"""
  rankdir=TB;
  pipe [label="Scan → Filter → Project", fillcolor="{BLUE_BG}", color="{BLUE}",
        xlabel=""];
  build [label="HashJoin build", fillcolor="{ORANGE_BG}", color="{ORANGE}"];
  agg [label="Aggregate", fillcolor="{ORANGE_BG}", color="{ORANGE}"];
  note1 [shape=note, fillcolor="#fffbe6", color="#e8c14a",
         label="streams — never materializes"];
  note2 [shape=note, fillcolor="#fffbe6", color="#e8c14a",
         label="breaker: build the hash table"];
  note3 [shape=note, fillcolor="#fffbe6", color="#e8c14a",
         label="breaker: partial aggregate"];
  pipe -> build [label=" "];
  build -> agg [label=" "];
  {{ rank=same; pipe; note1; }}
  {{ rank=same; build; note2; }}
  {{ rank=same; agg; note3; }}
  pipe -> note1 [style=invis];
  build -> note2 [style=invis];
  agg -> note3 [style=invis];
"""


@diagram("layer_stack")
def _layer_stack() -> str:
    layers = [
        ("User API", "bt.read() · bt.col() · ds.filter() · ds.join() · ds.collect()", "blue"),
        ("Dataset API", "lazy operations that build logical plans", "blue"),
        ("Logical Plan", "tree of operators: Scan · Filter · Project · Join · Aggregate", "grey"),
        ("Kyber — optimizer", "rule + cost passes; learned cardinality; adaptive re-opt", "blue"),
        ("Physical Plan", "concrete mergeable operators (single-node == distributed)", "grey"),
        ("Execution Engine", "plan compilation · task scheduling · progress tracking", "orange"),
        ("Carbonite", "memory · caching · spill-to-disk · data transfer", "orange"),
        ("Ray (optional, distributed)", "scheduling + metadata; bulk data over Arrow Flight", "grey"),
    ]
    palette = {
        "blue": (BLUE, BLUE_BG),
        "orange": (ORANGE, ORANGE_BG),
        "grey": (GREY, GREY_BG),
    }
    lines = ["  rankdir=TB; node [width=4.6];"]
    for i, (title, sub, kind) in enumerate(layers):
        color, bg = palette[kind]
        lines.append(
            f'  n{i} [label=<<b>{title}</b><br/>'
            f'<font point-size="10">{sub}</font>>, '
            f'fillcolor="{bg}", color="{color}"];'
        )
    for i in range(len(layers) - 1):
        lines.append(f"  n{i} -> n{i + 1};")
    return "\n".join(lines)


@diagram("data_flow")
def _data_flow() -> str:
    steps = [
        ('1 · User code', 'ds.filter(col("x") &gt; 10).select("a", "b")', "blue"),
        ("2 · Logical plan", "Project(a, b) → Filter(x &gt; 10) → Scan(ds)", "blue"),
        ("3 · Kyber optimizes", "push filter below project · prune columns · pick scan", "blue"),
        ("4 · Physical plan", "Project → Filter → Scan, as mergeable operators", "grey"),
        ("5 · Execution engine", "compile to task graph · schedule morsels · stream", "orange"),
        ("6 · Carbonite", "manage memory + spill · move batches over Arrow Flight", "orange"),
        ("7 · Rust data plane", "executes the operators over Arrow batches", "orange"),
        ("8 · Results", "collected and returned to the caller", "grey"),
    ]
    palette = {
        "blue": (BLUE, BLUE_BG),
        "orange": (ORANGE, ORANGE_BG),
        "grey": (GREY, GREY_BG),
    }
    lines = ["  rankdir=TB; node [width=5.0];"]
    for i, (title, sub, kind) in enumerate(steps):
        color, bg = palette[kind]
        lines.append(
            f'  s{i} [label=<<b>{title}</b><br/>'
            f'<font point-size="10">{sub}</font>>, '
            f'fillcolor="{bg}", color="{color}"];'
        )
    for i in range(len(steps) - 1):
        lines.append(f"  s{i} -> s{i + 1};")
    return "\n".join(lines)


@diagram("carbonite_loop")
def _carbonite_loop() -> str:
    return f"""
  rankdir=LR;
  kyber [label="Kyber\\ndecides", fillcolor="{BLUE_BG}", color="{BLUE}"];
  carbonite [label="Carbonite\\nprotects", fillcolor="{ORANGE_BG}", color="{ORANGE}"];
  core [label="Core\\nexecutes", fillcolor="{GREY_BG}", color="{GREY}"];
  kyber -> carbonite [label="plan +\\nestimated cost"];
  carbonite -> core [label="allocations"];
  core -> kyber [label="measured cardinalities,\\npeak memory", constraint=false,
                 color="{BLUE_EDGE}", fontcolor="{BLUE_EDGE}", style=dashed];
"""


def main() -> int:
    dot = shutil.which("dot")
    if not dot:
        print("error: `dot` not found — install Graphviz (brew install graphviz)", file=sys.stderr)
        return 1
    for name, body in DIAGRAMS.items():
        src = HERE / f"{name}.dot"
        png = HERE / f"{name}.png"
        src.write_text(body)
        subprocess.run([dot, "-Tpng", f"-Gdpi={DPI}", "-o", str(png), str(src)], check=True)
        print(f"rendered {png.relative_to(HERE.parent.parent.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
