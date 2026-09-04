#!/usr/bin/env python3
"""Fail when the installed engine is older than the Rust it is supposed to contain.

Nothing else catches this. A stale `python/batcher/_native.abi3.so` passes every lint gate,
passes `just docs`, and passes almost all of `just test-py` -- because most tests exercise
control-plane behaviour that a ten-hour-old binary still gets right. The suite goes green
while validating an engine that does not match the source tree.

That is not hypothetical. On 2026-08-26 the installed binary was built at 12:52 and four
commits touching **187 distinct files under `crates/`** landed after it, including three
Rust-side fixes. Every Python suite run on the box for the rest of that day validated the
12:52 engine, and every benchmark taken measured it. The only reason anyone noticed was that
one of those commits (`87d82730`) added a test case the old binary happens to fail -- a red
test that reads as a defect in the fix rather than as a stale build, which is the worse of
the two ways to find out.

The check is a timestamp against a commit date, so it is coarse on purpose:

* It compares the artifact's mtime to the newest commit touching `crates/`. A rebuild always
  moves the mtime forward, so a fresh build cannot be reported stale.
* It says nothing about *uncommitted* Rust in the working tree. A tree with local `crates/`
  edits is stale by definition and no timestamp can tell you whether they are in the binary;
  that is what the working-tree warning below is for.
* An absent artifact is not an error here. `just check` and `cargo test` need no extension
  module, and reporting a missing build as staleness would fail every Rust-only workflow.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = ROOT / "python" / "batcher" / "_native.abi3.so"
CRATES = ROOT / "crates"


def _newest_crates_commit() -> tuple[str, datetime, str] | None:
    """`(sha, when, subject)` of the newest commit touching `crates/`, or None."""
    result = subprocess.run(
        ["git", "log", "-1", "--format=%h%x00%cI%x00%s", "--", "crates"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    sha, iso, subject = result.stdout.strip().split("\0", 2)
    return sha, datetime.fromisoformat(iso), subject


def _dirty_crates() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--", "crates"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return [line[3:] for line in result.stdout.splitlines() if line.strip()]


def main() -> int:
    if not ARTIFACT.exists():
        print("lint-build-freshness: no built extension module — nothing to check")
        return 0

    newest = _newest_crates_commit()
    if newest is None:
        print("lint-build-freshness: could not read git history for `crates/` — skipped")
        return 0

    sha, committed, subject = newest
    # Both sides carry a timezone and are compared as instants. They are *displayed* in
    # the commit's own offset: printing one in UTC and the other in local time makes a
    # correct comparison look like a seven-hour discrepancy to whoever reads the failure.
    built = datetime.fromtimestamp(ARTIFACT.stat().st_mtime, tz=UTC).astimezone(committed.tzinfo)

    dirty = _dirty_crates()
    if dirty:
        print(f"lint-build-freshness: {len(dirty)} uncommitted file(s) under `crates/` —")
        print("  a timestamp cannot tell whether those are in the binary. Rebuild before")
        print("  trusting a Rust-dependent test or any benchmark number.")

    if built >= committed:
        print(
            f"lint-build-freshness: clean (built {built:%Y-%m-%d %H:%M}, "
            f"newest crates commit {sha} {committed:%Y-%m-%d %H:%M})"
        )
        return 0

    print(
        f"lint-build-freshness: the installed engine predates the Rust it should contain.\n"
        f"  built            {built:%Y-%m-%d %H:%M}\n"
        f"  newest crates/   {sha} {committed:%Y-%m-%d %H:%M}  {subject}\n"
        "  Every Rust-side change since that build is absent from the binary the Python\n"
        "  suite exercises, so a green run says nothing about it and a benchmark measures\n"
        "  the old engine. Rebuild with `just build` — and note that it overwrites the\n"
        "  artifact in place, so any process holding it mapped will take a Bus error;\n"
        "  coordinate before rebuilding a shared checkout."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
