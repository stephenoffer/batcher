# Rule: Working in a Tree Other Agents Are Also Editing

Most of this codebase is written by agents, often several at once, in one shared
working tree. Every failure below is one that actually happened here — they are not
hypotheticals, and each one wasted real time or nearly shipped a wrong claim.

## Assume the tree is shared

**Your session snapshot's `Status: (clean)` is a point-in-time reading and is
routinely false by the time you act on it.** Three agents in one session were told the
tree was clean and each found 40–60 modified files they had not touched; `HEAD` moved
twice mid-task. Before you reason about the state of the repo, run `git status` and
`git log --oneline -3` yourself. Treat anything you did not write as someone's
in-flight work.

## Never mutate shared global state

These commands affect files outside your task and have already clobbered another
agent's work here:

- **`git stash` / `git stash pop`** — stashes the *whole* tree, including edits you
  cannot see. One agent's stash swept up another's finished work; an orphaned
  `stash@{0}` outlived the session. If you need a baseline, read it with
  `git show HEAD:path` or copy the file aside — never stash.
- **`git checkout .` / `git checkout <branch>` / `git reset`** — same problem, worse.
- **Repo-wide autofix**: `ruff check --fix python`, `ruff format python`,
  `cargo fmt --all`. One agent's repo-wide `--fix` silently rewrote 13 findings inside
  other agents' half-finished files. **Scope every fix and format command to the paths
  you own** (`ruff format path/to/your/file.py`).
- **`git commit -a`** — commits everyone's work under your message.

## `just build` crashes every other session's running tests

`maturin develop` overwrites `python/batcher/_native.abi3.so` **in place**. Every Python
process that has already imported the engine holds that file memory-mapped, so replacing it
pulls the pages out from under running code. The result is not a test failure — it is a
`Fatal Python error: Bus error` (exit 135) or a segfault, with a 300-line dump of loaded
extension modules and no indication of the cause.

This happened three times in one session here, twice killing a full-suite run partway
(64 tests in, and 18 tests in) and once corrupting a benchmark. Each time the tell was the
same: compare the `.so`'s mtime against the run's start.

```
ls -la python/batcher/_native.abi3.so   # mtime inside your run window?  that is why
```

What to do about it:

- **Don't diagnose a Bus error / exit 135 / 139 as your bug** until you have checked that
  mtime. It almost never is.
- **Prefer targeted suites over the full run** while others are active — a two-minute run
  has a small window, a twenty-minute one is nearly certain to be hit.
- **Re-run before reporting.** A crashed run has no result, not a bad one.
- If you must run the whole suite, do it right after your own build, and expect to repeat.

The same swap is why a benchmark can silently measure the *other* agent's build: check
`bt.versions()["engine_profile"]`, which the suite now does for you.

## The pre-commit hook is repo-wide, so someone else's half-done refactor blocks you

`lint-structure` runs over the whole tree, not your staged paths. An agent mid-way through
shrinking an oversized file — say `stream/mod.rs` at 804 code lines against a limit of 800,
on its way down from 826 — fails the hook for *every* session trying to commit, including
ones that touched nothing near it.

Diagnose it before assuming your change is at fault: the FAIL line names the file, and
`git status <that file>` plus `git show HEAD:<that file> | wc -l` tells you whether someone
is actively working it down.

Then **wait and retry** — a loop around `git commit` is the whole fix, and their next save
usually clears it. Do not:

- **`--no-verify`.** The gate is the point, and skipping it is how an oversized file or a
  broken layer contract lands.
- **"Just fix" their file.** It is four lines; it is also the middle of someone's refactor,
  and your trim will collide with theirs.

The same applies to `MAP.md`: it goes stale the moment any session adds a module, so
regenerate it *inside* the retry loop rather than before it.

## Prove "pre-existing", never assume it

Reporting another agent's in-flight breakage as "pre-existing and unrelated" is the
most common false statement in this repo's agent reports, and it is how a real
regression gets waved through. Before you write that phrase, prove all three:

1. `git status <file>` — is it dirty? If yes it is probably someone's live work, not
   a pre-existing condition.
2. Does it fail at `HEAD`? (`git show HEAD:<file>` / check out a clean copy elsewhere.)
3. Is it outside your diff and unreachable from anything you moved?

**A failure count taken mid-refactor is not a fact.** One `pytest` run here showed 10
failures including eight in join/semijoin optimizer rules; every one of the eight was
another agent's transient package-split state and cleared on re-run. Package-izing
`module.py` → `module/` transiently breaks *every* import in the tree — a red suite
during that window tells you nothing. **Re-run at the end, after things settle, and
report that number.**

## Moving a file has couplings you must chase

A move is never just a move. Each of these has bitten a refactor here:

- **`STRUCTURE_ALLOW` / `DIR_ALLOW` keys in `tools/lint_structure.py`** are keyed by
  path. Move an allowlisted file and a *new* violation appears unless you re-path the
  key — and its stated reason may no longer be true.
- **Registration order is run order.** Kyber rules run in the order their modules are
  imported. A naive package split reordered 283 of 302 rules while every rule still
  existed and every name still resolved. Preserve import position, and verify.
- **Monkeypatch targets follow the name.** A test patching `module.attr` silently
  becomes a no-op when `attr` moves to `module.sub` — the patch stops applying and the
  test keeps passing while testing nothing.
- **Docs and guardrails cite paths.** `just lint-guardrails` catches stale paths in the
  agent docs; run it after any move.

## Prove equivalence with a diff, not an assertion

For any refactor claiming to preserve behavior:

```
just surface-save /tmp/before.json     # optimizer rule ORDER, IR tags, public API,
...refactor...                         # IO registry, FFI signatures
just surface-diff /tmp/before.json     # exits 1 on any observable change
```

An empty diff is evidence; "I only moved code" is not. If the diff is non-empty and
the change is intended, say so explicitly in your report. The agents in this repo whose
work needed no rework were exactly the ones that reported a diff.

## If you are orchestrating other agents

- **Give each agent a disjoint file set**, and say so in the prompt. Overlapping sets
  produce clobbering, not collaboration.
- **Prefer worktree isolation** (`isolation: "worktree"`) for anything touching more
  than a couple of files. The one thing that would have prevented every incident above.
- **Require an equivalence proof** in the prompt, not just "run the tests."
- **Ask what they did *not* touch.** The most useful line in an agent report is the
  scope boundary.
- **Run the gate yourself at the end.** Agent-reported gate results are snapshots from
  inside a moving tree; only the final serialized run counts.
