#!/usr/bin/env python3
"""Draw `secret_reference_flow.svg` -- a key reference travels, the key is resolved on the worker.

Source of truth: `docs/user-guide/trust/secrets.md` and `crates/bc-secrets/src/lib.rs`. A
reference (`env:NAME`, `file:PATH`, `cmd:NAME`) travels in the plan IR in place of the
secret and is resolved in the data plane against the executing machine's own environment,
mounted files, or the operator's `BATCHER_SECRET_COMMAND` helper, whose stdout is the secret.
Resolution is cached for `BATCHER_SECRET_TTL_SECONDS` (default 300) because key references
resolve per batch. An inline literal key emits a `SecurityWarning`, and under
`BATCHER_REQUIRE_KEY_REFS=1` raises `PlanError` at plan-build time. A missing reference fails
naming the reference, never the key.

Drawn for the expression-layer key references (`aes_encrypt` and friends), which is the path
that caches. Connector credentials resolve once per connection with no cache, and the footer
says so rather than drawing a second flow.
"""

from __future__ import annotations

from _authoring import (
    arrow,
    band,
    card,
    code,
    heading,
    label,
    note,
    pill,
    svg,
    tint,
    write,
)

W, H = 980, 560

KEY_BY_REFERENCE = ["ds.select(c=bt.aes_encrypt(", '    bt.col("ssn"),', '    "env:AES_KEY"))']

body: list[str] = [
    # --- the driver ---------------------------------------------------------------------
    band(20, 20, 280, 470, "THE DRIVER", "grey"),
    note(38, 62, "builds the plan"),
    code(36, 84, KEY_BY_REFERENCE, 248, 12),
    arrow(160, 190, 160, 316, "amber"),
    label(172, 250, "an inline key"),
    label(172, 266, "instead"),
    tint(36, 322, 248, 76, "SecurityWarning", "the key is now in the plan", "amber"),
    note(38, 426, "With BATCHER_REQUIRE_KEY_REFS=1"),
    note(38, 444, "it is a PlanError at plan-build"),
    note(38, 462, "time instead."),
    # --- what travels -------------------------------------------------------------------
    band(318, 20, 250, 470, "WHAT TRAVELS", "blue"),
    arrow(284, 104, 334, 104),
    label(309, 92, "lowers", anchor="middle", size=11.5),
    card(334, 62, 218, 76, "the plan IR", "shipped to every worker"),
    pill(443, 170, "env:AES_KEY", "blue", anchor="middle"),
    note(443, 200, "the reference only,", anchor="middle"),
    note(443, 218, "never the key", anchor="middle"),
    heading(443, 270, "NEVER SEE THE KEY", anchor="middle", kind="grey"),
    note(443, 296, "plan logs", anchor="middle"),
    note(443, 316, "the profile", anchor="middle"),
    note(443, 336, "explain()", anchor="middle"),
    note(443, 356, "the FFI boundary", anchor="middle"),
    note(443, 426, "A missing reference fails,", anchor="middle"),
    note(443, 444, "naming the reference,", anchor="middle"),
    note(443, 462, "never the key.", anchor="middle"),
    # --- on each worker -----------------------------------------------------------------
    band(586, 20, 374, 470, "ON EACH WORKER", "blue"),
    arrow(552, 91, 604, 91),
    label(578, 79, "arrives", anchor="middle", size=11.5),
    card(604, 62, 338, 58, "resolve the reference", "bc-secrets, in the data plane"),
    arrow(662, 120, 662, 152),
    arrow(773, 120, 773, 152),
    arrow(884, 120, 884, 152),
    label(717, 142, "by scheme", anchor="middle", size=11.5),
    tint(604, 156, 110, 66, "env:", "variable"),
    tint(718, 156, 110, 66, "file:", "mounted file"),
    tint(832, 156, 110, 66, "cmd:", "helper stdout"),
    arrow(773, 222, 773, 254),
    label(785, 244, "the secret"),
    card(604, 258, 338, 58, "cached per process", "BATCHER_SECRET_TTL_SECONDS, 300 s"),
    arrow(773, 316, 773, 344),
    label(785, 336, "per batch"),
    card(604, 348, 338, 58, "the kernel uses the key", "on the machine that runs it"),
    note(612, 436, "cmd: runs the operator's BATCHER_SECRET_COMMAND"),
    note(612, 454, "with NAME as its argument. Unset, it fails."),
    note(
        490,
        530,
        "Connector passwords and storage_options take the same references, resolved once "
        "per connection and not cached.",
        anchor="middle",
    ),
]

write("secret_reference_flow", svg(W, H, "".join(body)))
print("wrote secret_reference_flow.svg")
