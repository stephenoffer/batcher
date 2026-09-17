#!/usr/bin/env python3
"""Draw `hardening_boundary.svg` -- what Batcher enforces inside the process, and what not.

Source of truth: `docs/user-guide/trust/hardening.md`. Batcher authorizes and does not
authenticate; a `Principal` is asserted by the caller and any code in the process can
construct one, so the trust boundary is the process. Inside it the page's settings are
`governance.mode="strict"`, `governance.audit_path`, `require_verified_principal`,
`execution.udf_isolation="strict"`, `execution.max_concurrent_queries` and
`BATCHER_REQUIRE_KEY_REFS=1`, and two protections are automatic: artifacts on disk are
created owner-only, and value-derived statistics are dropped for masked or invisible columns
inside a `security()` block. Its "Requirements and limitations" list is the outer layer:
no authentication, no multi-tenancy, no encryption at rest, UDF isolation that is not a
sandbox and covers the process path only, and admission that is per process.

Drawn as nested layers rather than a flow because the page's argument is containment: each
control is only as strong as the layer around it.
"""

from __future__ import annotations

from _authoring import (
    AMBER_DEEP,
    band,
    heading,
    mark,
    note,
    svg,
    tint,
    write,
)

W, H = 980, 610

PLATFORM = (
    ("Authentication", "at your network edge"),
    ("Tenant isolation", "one process per domain"),
    ("Sandboxing", "containers for untrusted UDFs"),
    ("Encryption at rest", "an encrypted volume"),
)

ENFORCED = (
    ("Governance required", 'governance.mode="strict"'),
    ("Durable audit trail", "governance.audit_path"),
    ("Verified identities", "require_verified_principal"),
    ("Key references only", "BATCHER_REQUIRE_KEY_REFS=1"),
    ("UDF process ceilings", 'udf_isolation="strict"'),
    ("Admission control", "max_concurrent_queries"),
    ("Owner-only artifacts", "0700 dirs, 0600 files, always"),
    ("Governed statistics", "no min/max on masked columns"),
)

NOT_A_BOUNDARY = (
    ("Code in the process can", "construct any Principal."),
    ("A UDF on a thread reads", "the engine's environment."),
    ("Admission bounds one", "process, not the cluster."),
)

body: list[str] = [
    band(20, 20, 940, 570, "YOUR PLATFORM PROVIDES THESE; BATCHER DOES NOT", "grey"),
]
for i, (title, sub) in enumerate(PLATFORM):
    body.append(tint(42 + 228 * i, 56, 212, 64, title, sub, "amber"))

body += [
    # The process boundary: dashed, because it is a boundary Batcher assumes, not one it draws.
    f'<rect x="42" y="146" width="896" height="424" rx="16" fill="none" stroke="{AMBER_DEEP}" '
    f'stroke-width="2.2" stroke-dasharray="9 6"/>',
    heading(62, 174, "THE PROCESS IS THE TRUST BOUNDARY", kind="amber"),
    band(62, 192, 856, 222, "BATCHER ENFORCES INSIDE IT", "blue"),
]
for i, (title, sub) in enumerate(ENFORCED):
    row, col = divmod(i, 4)
    body.append(tint(80 + 208 * col, 230 + 88 * row, 196, 68, title, sub))

body += [
    heading(62, 448, "AND CANNOT GUARANTEE, EVEN THERE", kind="grey"),
]
for i, (first, second) in enumerate(NOT_A_BOUNDARY):
    x = 72 + 290 * i
    body += [
        mark(x + 10, 490, False),
        note(x + 32, 486, first),
        note(x + 32, 503, second),
    ]
body.append(
    note(
        490,
        552,
        "Authenticate at the edge, pass the identity into bt.security(), "
        "and run untrusted UDFs in a container.",
        anchor="middle",
    )
)

write("hardening_boundary", svg(W, H, "".join(body)))
print("wrote hardening_boundary.svg")
