#!/usr/bin/env python3
"""Draw `install_extras_stack.svg`: the core `batcher-engine` wheel and the optional extras.

Source of truth: `pyproject.toml` (`[project] dependencies` and
`[project.optional-dependencies]`, including the `lakehouse`, `streaming`, `multimodal`
and `all` bundles) and `docs/getting-started/installation.md` (one wheel carrying the
Python control plane and the precompiled Rust engine; Python 3.11 or newer; extras add
integrations without changing the core API; a missing extra raises
`MissingDependencyError` carrying the install command). The extras drawn are the ones
that page's table lists, less `numpy`, which is also a required dependency of the core.

Layout: a two-layer stack, extras above the core they plug into.
"""

from __future__ import annotations

from _authoring import arrow, band, card, heading, hero, label, note, pill, svg, write

W, H = 980, 560

# (column x, row y, heading, pill rows)
GROUPS = [
    (48, 62, "CLUSTERS", [["ray"]]),
    (282, 62, "OBJECT STORES", [["cloud"]]),
    (516, 62, "LAKEHOUSE", [["delta", "iceberg", "hudi"]]),
    (750, 62, "STREAMING", [["kafka", "kinesis", "pubsub"], ["pulsar", "eventhubs"]]),
    (48, 160, "MEDIA DECODE", [["image", "audio", "video"]]),
    (282, 160, "ML FRAMEWORKS", [["torch", "tensorflow", "jax"]]),
    (516, 160, "LLMS, EMBEDDINGS", [["st", "vllm", "sglang"]]),
    (750, 160, "DATAFRAMES", [["pandas", "polars"]]),
]


def pill_row(x: float, y: float, names: list[str]) -> str:
    """Lay pills left to right with an even gap, using the pill's own width rule."""
    out = ""
    for name in names:
        out += pill(x, y, name, "grey")
        x += 14 + 6.7 * len(name) + 8
    return out


body = [
    band(20, 20, 940, 250, 'OPTIONAL EXTRAS  ·  pip install "batcher-engine[ray,cloud]"', "grey")
]
for x, y, title, rows in GROUPS:
    body.append(heading(x, y + 22, title, kind="grey"))
    for i, names in enumerate(rows):
        body.append(pill_row(x, y + 50 + i * 28, names))

body += [
    note(490, 250, "Bundles: lakehouse, streaming, multimodal, all.", anchor="middle"),
    arrow(490, 336, 490, 278),
    label(504, 312, "each plugs into the same API"),
    band(20, 344, 940, 196, "CORE WHEEL  ·  pip install batcher-engine", "blue"),
    card(48, 386, 280, 84, "Python control plane", "Dataset, SQL, optimizer"),
    hero(350, 386, 300, 84, "Rust engine", "precompiled, no toolchain"),
    heading(684, 404, "REQUIRED", kind="grey"),
    pill_row(684, 434, ["pyarrow", "numpy"]),
    pill_row(684, 462, ["sqlglot", "psutil"]),
    note(
        490,
        512,
        "Python 3.11+. A feature whose extra is missing raises MissingDependencyError.",
        anchor="middle",
    ),
]

write("install_extras_stack", svg(W, H, "".join(body)))
print("wrote install_extras_stack.svg")
