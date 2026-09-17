#!/usr/bin/env python3
"""Draw `skew_hot_key_salting.svg`: a hot join key on a cluster, before and after salting.

Source of truth:

* `docs/user-guide/operate/tuning/skew.md`, "Hot keys on a cluster": every row of a hot key
  crosses the shuffle to one reducer; a heavy-hitters sketch marks a value hot when it covers
  `distributed.skew_join_fraction` of the rows (10% by default); the probe side's hot rows
  then fan out across several reducers and the build side's matching rows are copied to each;
  cold keys hash exactly as before, so the joined relation is unchanged.
* `python/batcher/dist/skew.py`: `salt_factor` (the fan-out is sized from the hot fraction
  and the shuffle width, at least 2 and capped) and the module docstring's statement that
  salting only moves a hot key's work between reducers, never the joined relation.
* `python/batcher/config/config.py`: `skew_join_fraction = 0.10`, `skew_join_salt = 0`.

The reducer bars are relative shapes, not measurements. The sentinel `-1` is the page's own
example of a hot key. Load is carried by bar length *and* by a text label, never by color.

Form: before/after at the same scale, four reducers each, so the eye compares bar lengths.
"""

from __future__ import annotations

from _authoring import (
    AMBER_DEEP,
    BLUE_MID,
    FONT,
    GREY,
    arrow,
    band,
    label,
    note,
    svg,
    tint,
    write,
)

W, H = 980, 556

BAR_X0 = 196  # left edge of the load bars inside each panel, relative to the panel
BAR_MAX = 214


def reducer_row(px: float, y: float, name: str, hot: float, cold: float, text: str) -> str:
    """One reducer: its name, a stacked load bar (hot rows, then cold), and a label."""
    x0 = px + BAR_X0
    hw = BAR_MAX * hot
    cw = BAR_MAX * cold
    out = (
        f'<text x="{px + 104}" y="{y + 15}" text-anchor="end" font-family="{FONT}" '
        f'font-size="12.5" font-weight="700" class="t-title">{name}</text>'
        f'<rect x="{x0 - 76}" y="{y}" width="{BAR_MAX + 76}" height="22" rx="5" '
        f'fill="none" stroke="#cbd5e1" stroke-dasharray="3 3"/>'
    )
    if hw:
        out += (
            f'<rect x="{x0 - 76}" y="{y}" width="{hw}" height="22" rx="5" fill="{AMBER_DEEP}" '
            f'fill-opacity="0.85"/>'
        )
    out += (
        f'<rect x="{x0 - 76 + hw}" y="{y}" width="{cw}" height="22" rx="5" fill="{BLUE_MID}" '
        f'fill-opacity="0.45"/>'
        f'<text x="{px + 104}" y="{y + 36}" text-anchor="end" font-family="{FONT}" '
        f'font-size="10.5" class="t-sub">{text}</text>'
    )
    return out


body: list[str] = [
    band(20, 20, 456, 482, "BEFORE: HASH(KEY) ONLY", "grey"),
    band(504, 20, 456, 482, "AFTER: HOT KEY SALTED", "blue"),
]

for px, title, sub in (
    (44, "probe side", "customer_id = -1 is most rows"),
    (528, "probe side", "-1 marked hot by the sketch"),
):
    body.append(tint(px, 60, 408, 52, title, sub, "amber"))
    body.append(arrow(px + 204, 116, px + 204, 156))
    body.append(label(px + 216, 142, "shuffle", size=11.5))

# ---- Before: every -1 row reaches the same reducer --------------------------------------------
body.append(note(248, 184, "equal keys hash to one place, by construction", anchor="middle"))
rows_before = (
    ("reducer 0", 0.92, 0.08, "all -1 rows"),
    ("reducer 1", 0.0, 0.2, "cold keys"),
    ("reducer 2", 0.0, 0.18, "cold keys"),
    ("reducer 3", 0.0, 0.22, "cold keys"),
)
for i, (name, hot, cold, text) in enumerate(rows_before):
    body.append(reducer_row(44, 208 + i * 56, name, hot, cold, text))

body.append(
    f'<text x="248" y="448" text-anchor="middle" font-family="{FONT}" font-size="12" '
    f'font-weight="700" fill="{AMBER_DEEP}">one reducer carries the key</text>'
)
body.append(note(248, 470, "No bucket count or re-hash separates", anchor="middle"))
body.append(note(248, 486, "rows that share a key.", anchor="middle"))

# ---- After: hot rows fan out, build rows copied, cold keys unchanged --------------------------
body.append(note(732, 184, "hot rows fan out; cold keys hash as before", anchor="middle"))
rows_after = (
    ("reducer 0", 0.31, 0.08, "-1 share + build copy"),
    ("reducer 1", 0.31, 0.2, "-1 share + build copy"),
    ("reducer 2", 0.3, 0.18, "-1 share + build copy"),
    ("reducer 3", 0.0, 0.22, "cold keys"),
)
for i, (name, hot, cold, text) in enumerate(rows_after):
    body.append(reducer_row(528, 208 + i * 56, name, hot, cold, text))

body.append(
    f'<text x="732" y="448" text-anchor="middle" font-family="{FONT}" font-size="12" '
    f'font-weight="700" fill="{BLUE_MID}">the key&apos;s load is split</text>'
)
body.append(note(732, 470, "Matching build rows are copied to each share,", anchor="middle"))
body.append(note(732, 486, "so the joined relation is unchanged.", anchor="middle"))

# ---- Legend and the learned part ----------------------------------------------------------------
body += [
    f'<rect x="44" y="524" width="22" height="12" rx="3" fill="{AMBER_DEEP}" fill-opacity="0.85"/>',
    note(74, 534, "rows of the hot key"),
    f'<rect x="226" y="524" width="22" height="12" rx="3" fill="{BLUE_MID}" fill-opacity="0.45"/>',
    note(256, 534, "cold-key rows"),
    f'<line x1="372" y1="520" x2="372" y2="540" stroke="{GREY}"/>',
    note(
        390,
        534,
        "A key is hot when it covers skew_join_fraction of the rows, 10% by default.",
    ),
]

write("skew_hot_key_salting", svg(W, H, "".join(body)))
print("wrote skew_hot_key_salting.svg")
