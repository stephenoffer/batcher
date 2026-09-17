#!/usr/bin/env python3
"""Draw `credential_resolution.svg` -- the order a path's filesystem and credentials come from.

Source of truth: `python/batcher/io/filesystem.py`. `resolve_filesystem` returns a caller's
`filesystem=` verbatim before looking at anything else ("It wins over `storage_options`").
Otherwise `_resolve_uri_fs` resolves `env:`/`file:`/`cmd:` references in `storage_options`
(`_resolved_options`, on the machine building the filesystem), folds the options into the
URI query (`_with_query`) and asks `pyarrow.fs.FileSystem.from_uri`; a scheme pyarrow does not
implement natively falls back to `_fsspec_backed`, which takes the options as keyword
arguments instead. What the URI and the options leave unset is the provider SDK's default
credential chain, which the page's credentials table describes per store
(`docs/user-guide/moving-data/cloud-storage.md`).

Layout: a vertical ladder of the checks in order, with each exit drawn to the right.
"""

from __future__ import annotations

from _authoring import arrow, band, card, label, note, step, svg, tint, write

W, H = 980, 544
CX = 260  # centre of the ladder column
CW = 330  # ladder card width
X0 = CX - CW / 2

body: list[str] = [
    band(20, 20, 940, 450, "HOW A PATH GETS ITS FILESYSTEM AND CREDENTIALS", "grey"),
    # ---- 1: a filesystem object ------------------------------------------------------
    step(54, 91, 1),
    card(X0, 62, CW, 58, "filesystem= passed?", "a pyarrow or fsspec filesystem"),
    arrow(X0 + CW + 4, 91, 600, 91),
    label(510, 79, "yes", anchor="middle", size=11.5),
    tint(606, 62, 330, 58, "Used verbatim", "wins over storage_options", "amber"),
    arrow(CX, 124, CX, 164),
    label(CX + 12, 150, "no", size=11.5),
    # ---- 2: does pyarrow implement the scheme -----------------------------------------
    step(54, 199, 2),
    card(X0, 170, CW, 58, "Scheme native to pyarrow?", "s3, gs, abfs, hdfs, and aliases"),
    arrow(X0 + CW + 4, 199, 600, 199),
    label(510, 187, "no", anchor="middle", size=11.5),
    tint(606, 170, 330, 58, "fsspec backend", "storage_options as keyword arguments"),
    note(771, 248, "oss, cos, obs, oci, swift, lakefs", anchor="middle"),
    arrow(CX, 232, CX, 272),
    label(CX + 12, 258, "yes", size=11.5),
    # ---- 3: explicit per-path settings -------------------------------------------------
    step(54, 307, 3),
    card(X0, 278, CW, 58, "Native backend from the URI", "?query options, storage_options added"),
    note(606, 302, "Explicit keys, an endpoint_override or a role_arn"),
    note(606, 320, "set here apply to this path only."),
    arrow(CX, 340, CX, 380),
    label(CX + 12, 366, "anything left unset", size=11.5),
    # ---- 4: the SDK's own chain --------------------------------------------------------
    step(54, 415, 4),
    card(X0, 386, CW, 58, "The provider SDK's chain", "environment, instance or role identity"),
    note(606, 400, "AWS_ACCESS_KEY_ID, AZURE_STORAGE_*,"),
    note(606, 418, "GOOGLE_APPLICATION_CREDENTIALS: the"),
    note(606, 436, "variables each vendor's own tooling reads."),
    # ---- the secret-reference footnote ---------------------------------------------------
    note(
        490,
        504,
        "An env:, file: or cmd: value in storage_options resolves on the machine that opens the "
        "connection,",
        anchor="middle",
    ),
    note(
        490,
        524,
        "so a distributed read ships the reference to each worker and never the secret.",
        anchor="middle",
    ),
]

write("credential_resolution", svg(W, H, "".join(body)))
print("wrote credential_resolution.svg")
