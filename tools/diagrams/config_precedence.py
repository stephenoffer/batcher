#!/usr/bin/env python3
"""Draw `config_precedence.svg`: the five layers that resolve the active `Config`, highest first.

Source of truth: `python/batcher/config/config.py`. Its module docstring states the order,
"``config_context`` > programmatic ``set_config`` > ``BATCHER_*`` env vars > a JSON file at
``BATCHER_CONFIG_FILE`` > defaults", and `_initial_config` layers defaults, then the file,
then the environment, once at import into the default of the `_active` `ContextVar`.
`set_config` sets that `ContextVar`; `config_context` sets it for a `with` block and resets
the token on exit. `set_option` (`options.py`) goes through `set_config`, and
`option_context` and `tenant` are built on the same `ContextVar` as `config_context`.
"""

from __future__ import annotations

from _authoring import BLUE, GREY, arrow, card, heading, label, note, svg, tint, write

W, H = 980, 530

X, CARD_W, CARD_H, GAP = 130, 500, 66, 14
TOP = 70
GROUP_GAP = 30  # extra room between the runtime and import-time groups
ROWS = [TOP + i * (CARD_H + GAP) + (GROUP_GAP if i >= 2 else 0) for i in range(5)]

layers = [
    ("config_context(cfg)", "also option_context and tenant", "blue"),
    ("set_config(cfg)", "also set_option", "blue"),
    ("BATCHER_* variables", "Config.from_env", None),
    ("BATCHER_CONFIG_FILE", "Config.from_file", None),
    ("Built-in defaults", "the dataclass field values", None),
]

body: list[str] = []
for y, (title, sub, kind) in zip(ROWS, layers, strict=True):
    body.append(
        tint(X, y, CARD_W, CARD_H, title, sub, kind)
        if kind
        else card(X, y, CARD_W, CARD_H, title, sub)
    )

bottom = ROWS[-1] + CARD_H
mids = [y + CARD_H / 2 for y in ROWS]
NX = 700  # notes column
body += [
    # The direction of precedence, drawn as an arrow so it does not rely on stacking order alone.
    arrow(76, bottom, 76, TOP + 4, "amber"),
    label(76, TOP - 16, "wins", anchor="middle", size=12),
    note(76, bottom + 22, "loses", anchor="middle"),
    # When each layer is read: a bracket per group, and one note per row.
    f'<path d="M 668 {ROWS[0] + 6} L 676 {ROWS[0] + 6} L 676 {ROWS[1] + CARD_H - 6} '
    f'L 668 {ROWS[1] + CARD_H - 6}" fill="none" stroke="{BLUE}" stroke-width="2"/>',
    f'<path d="M 668 {ROWS[2] + 6} L 676 {ROWS[2] + 6} L 676 {ROWS[4] + CARD_H - 6} '
    f'L 668 {ROWS[4] + CARD_H - 6}" fill="none" stroke="{GREY}" stroke-width="2"/>',
    heading(X, TOP - 16, "SET AT RUNTIME"),
    note(NX, mids[0] + 4, "Innermost with block, restored on exit"),
    note(NX, mids[1] + 4, "Process-wide until changed"),
    heading(X, ROWS[2] - 16, "READ ONCE AT IMPORT", kind="grey"),
    note(NX, mids[2] + 4, "Overlays the file"),
    note(NX, mids[3] + 4, "Overlays the defaults"),
    note(NX, mids[4] - 5, "What reset_option restores, not"),
    note(NX, mids[4] + 13, "the environment's values"),
]

write("config_precedence", svg(W, H, "".join(body)))
print("wrote config_precedence.svg")
