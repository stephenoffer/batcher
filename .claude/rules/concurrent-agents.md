# Rule: Working in a Tree Other Agents Are Also Editing

Most of this codebase is written by agents, often several at once, in one shared
working tree. Every failure below has happened here.

## Assume the tree is shared

**Your session snapshot's `Status: (clean)` is a point-in-time reading and is
routinely false by the time you act on it.** Before you reason about the state of the repo, run `git status` and
`git log --oneline -3` yourself. Treat anything you did not write as someone's
in-flight work.

## Never mutate shared global state

These commands affect files outside your task:

- **`git stash` / `git stash pop`** — stashes the *whole* tree, including edits you
  cannot see. If you need a baseline, read it with
  `git show HEAD:path` or copy the file aside — never stash.
- **`git checkout .` / `git checkout <branch>` / `git reset`** — same problem, worse.
- **`git checkout -- <path>`, even one named path.** It discards uncommitted work in whatever
  it touches; the narrower form only buys a smaller blast radius. Run
  `git status <the exact paths>` immediately before, every time, and if anything is dirty that
  you did not write, do not run it. **A scope check covers the scope it was run against** — a
  clean `git status benchmarks/` says nothing about `crates/`.
- **Repo-wide autofix**: `ruff check --fix python`, `ruff format python`,
  `cargo fmt --all` rewrite other agents' half-finished files. **Scope every fix and format command to the paths
  you own** (`ruff format path/to/your/file.py`).
- **`git commit -a`** — commits everyone's work under your message.
- **The index is shared too, so `git add` your paths is not enough.** `git commit` writes
  whatever is staged, including files *another agent staged*, and a blocked commit of yours
  leaves your own files staged for the next one to sweep up. Use
  **`git commit --only <paths>`** (or `git commit <paths>`), which commits exactly those paths
  and ignores the rest of the index. If you use plain `git commit`, read
  `git diff --cached --stat` immediately before it and confirm every line is yours.

- **`--only` bounds the *paths*, not the *hunks inside them*.** This is the trap on the far
  side of the fix above, and it is easy to walk into precisely because `--only` feels safe.
  A path you own one line of carries **every** uncommitted change in that file, including
  another agent's — and committing half of someone's change can leave HEAD not compiling.

  So the check is per *file*, not per commit: `git diff <path>` every entry on your list and
  confirm you wrote all of it. The trap is the file you add **last**, to fix some small thing
  the gate complained about — that is the one you inspect a single hunk of rather than the
  whole diff. A cheap tell after the fact is `git show --numstat`: a file you touched once
  showing `+153 -50` is not a file you touched once.

  **Make that check mechanical, because reading it does not work.** A script that refuses:

  ```bash
  # In the commit-retry loop, BEFORE `git commit`:
  foreign=$(git diff -- "$path" | grep -E '^[+-]' | grep -vE '^[+-][+-]' \
            | grep -vc 'a pattern matching only your own lines')
  [ "$foreign" -eq 0 ] || { echo "$path carries $foreign foreign line(s)"; sleep 60; continue; }
  ```

  If you cannot express
  "your own lines" as a pattern, the general form is to copy each file aside the moment you
  first edit it and diff the working tree against that copy at commit time: anything that
  appears is someone else's, no judgement required.

  When the wait is long, check whether the file is *blocking* or *broken*. If the uncommitted
  content in a shared file is the missing half of a change whose other half is already at
  HEAD — so HEAD does not compile — the right move is to commit that file **alone**, verbatim, as a labelled
  repair, and only then land your own change on top. Restore your own lines out of the file
  first so the repair commit is purely theirs.

  To back a path out of a commit you already made, without touching the working tree:
  `git restore --source=HEAD~1 --staged <path>` then `git commit --amend --no-edit`. Their
  content stays in the tree and goes back to being unstaged. Guard any `--amend` with
  `git rev-parse HEAD` against the hash you meant to amend — if another agent committed on
  top, you would be rewriting *their* commit.

## `just build` crashes every other session's running tests

`maturin develop` overwrites `python/batcher/_native.abi3.so` **in place**. Every Python
process that has already imported the engine holds that file memory-mapped, so replacing it
pulls the pages out from under running code. The result is not a test failure — it is a
`Fatal Python error: Bus error` (exit 135) or a segfault, with a 300-line dump of loaded
extension modules and no indication of the cause.

The tell: compare the `.so`'s mtime against the run's start.

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

### The long-lived Ray cluster holds a stale copy too

The same rebuild breaks the **shared Ray cluster**, and it looks nothing like the local case.
Ray's workers imported the engine when the cluster started, so after any `just build` they hold
the old `.so` memory-mapped for the rest of their lives. Every distributed test then dies —
`SystemExit: 1` from a worker, a raylet stack dump, or a nine-minute hang — while the identical
test passes single-node.

The tell is that it is not your test. Check with the smallest possible query before reading a
line of your own code:

```
python -c "import batcher as bt; \
  print(bt.from_pydict({'a':[1,2]}).agg(s=bt.col('a').sum()).collect(distributed=True).to_pydict())"
```

If a two-row sum crashes, nothing about your operator is under test. Confirm by running the same
file against a fresh cluster, which needs no coordination with anyone and takes one env var:

```
RAY_ADDRESS=local python -m pytest tests/integration/test_your_thing.py -q
```

A file that passes under that variable was never failing on its own account. Prefer this over `ray stop`: the cluster is shared, another session
may have work on it, and a fresh instance answers the question without touching theirs.

### You do not have to wait: build into a sandbox instead

`just build` is what clobbers the shared `.so`. Building the crate does not. So an FFI or
Rust-side change can be tested end to end, with the real Python suite, while another session
is mid-run — which otherwise blocks the whole `bc-py` surface for as long as they are active:

```
cargo build --release -p bc-py --features pyo3/extension-module   # -> target/release/lib_native.so
SB=<scratchpad>/sandbox && rm -rf $SB && mkdir -p $SB
git archive HEAD python tests pyproject.toml benchmarks .github | tar -x -C $SB
tar -c crates | tar -x -C $SB                                      # see below: some tests read these
cp target/release/lib_native.so $SB/python/batcher/_native.abi3.so
PYTHONPATH=$SB/python python -m pytest $SB/tests/differential -q
```

**Stage more than `python tests`, or you get spurious failures with misleading names.** A
number of tests read repo files *deliberately*, so a constant and its source cannot drift, and
under `PYTHONPATH` they resolve the repo root as the **sandbox** root — so a directory you did
not stage is simply absent:

| Also stage | Or else |
|---|---|
| `crates/` (the **working tree's**) | `test_diff_execution_mode_matrix::test_the_fixture_actually_shards` raises `FileNotFoundError`. It parses `crates/bc-interp/src/stream/parallel.rs` for `MIN_ROWS_TO_SHARD` rather than hardcoding a number that would let the file quietly stop testing the sharded path. |
| `benchmarks/` | `test_benchmark_isolation.py` fails to **collect** — `ModuleNotFoundError: No module named 'harness'`. |
| `pyproject.toml` | `test_python_floor` and `test_optional_guard_is_shared` fail: the declared Python floor and the extras an optional-guard names both live there. |
| `.github` | `test_python_floor::test_the_release_workflow_builds_on_the_declared_floor` fails. It reads `.github/workflows/release.yml` and cross-checks the pinned `python-version` against the floor. |
| `tools/` | Four `tests/unit` modules fail to **collect** with `ModuleNotFoundError: No module named 'tools'` — `test_agentic_runner`, `test_audit_health_detectors`, `test_ir_contract`, `test_lint_tests`. They import the linters they are about, which is the point of them. A collection error takes the whole module down, so this reads as a broken `PYTHONPATH` rather than an unstaged directory. **Staging it is not enough for `test_agentic_runner`:** four of its cases create `git worktree`s, and a `git archive`d sandbox is not a repository, so they fail there and pass in the tree. Run that file where the `.git` is. |

`crates/` is the one copied from the working tree rather than `git archive`d, because the
`.so` you just built came from the working tree and staging HEAD's sources would reintroduce
exactly the drift the test exists to catch. It is 7.5 MB. The rest are not compiled into the
`.so`, so HEAD's copies are right and keep the "committed state, not their WIP" property.

**`python/` needs the same treatment whenever the control plane is part of what you are
measuring**: `git archive HEAD python` stages a control plane without any uncommitted change you
are testing. The rule generalises past both directories: **`git archive HEAD` is right for the
files that are merely *present*, and wrong for any file whose behaviour is the subject of the measurement.**
Decide per directory, and when a reading contradicts something you verified against the installed
build minutes earlier, suspect the staging before you suspect the engine.

This table is incomplete; the only complete enumeration is running the suite in the sandbox.
A failure here is often an *error* rather than a failure (a module that fails to collect), and
reads as a broken interpreter or `PYTHONPATH` rather than an unstaged directory.

A partial regeneration — tests that climb above their own directory to reach a repo file:

```
grep -rlE 'parents\[|\.parent\.parent' tests --include='*.py'
```

Check what each joins onto that root; anything outside `python/` and `tests/` needs
staging or a skip guard.

**`grep`, not `rg`, and the reason generalises past this recipe.** `rg` in a Claude Code
session is a *shell function* from the shell snapshot, not a binary on `PATH`. A bare `rg`
works because the shell resolves it; `... | xargs rg ...` does not, because `xargs` execs
directly — and it fails to **stderr while exiting 0**, so the pipeline prints an empty list and
reports success. Any recipe in
`.claude/rules/` that pipes into `xargs rg` has it.

**The grep is a floor, not a proof**: it cannot see an *import* dependency (such as
`test_benchmark_isolation` on `benchmarks/`). Run the suite in the sandbox and see what breaks.

Three things make this work and are worth keeping: `bc-py`'s `[lib] name = "_native"` is a
plain `cdylib`, so the file cargo produces *is* the extension module under a different name;
`git archive HEAD` gives you the committed tree rather than the other session's half-finished
edits, so their broken imports do not become your test failures; and `PYTHONPATH` wins over the
installed package, so nothing outside the scratchpad is touched. Confirm you are in the sandbox
with `bt.versions()["engine_profile"]` and `bt._native.__file__`.

The same swap is why a benchmark can silently measure the *other* agent's build: check
`bt.versions()["engine_profile"]`, which the suite now does for you.

Two ways to shoot yourself with the sandbox:

- **One sandbox, two concurrent runs.** Copying the `.so` in is the same in-place overwrite
  `just build` does, so a second run started while the first is going kills it — a fatal
  interpreter error, a 300-line list of loaded extension modules, and no test results. Name
  the sandbox per invocation (`pybox-$$` or a caller-supplied tag) and the problem is gone.
- **Deleting a sandbox a run is still using.** `pytest` `chdir`s back to its start path at the
  end and dies with `FileNotFoundError` on a directory you cleaned up. Check for a live
  process before `rm -rf`.

### A full suite gets OOM-killed on a busy box; chunk it

The head node is shared. A whole-directory `pytest tests/differential` can be **killed** partway,
reporting nothing — the same `Killed` you would see from a hang, and easy to misread as your
change.

Run the directory in chunks of ~40 files, one process each. **The reason that survives a
hardware change is attribution, not headroom**: a chunk that is killed names itself, where a
whole-directory run takes the entire result down with it and tells you nothing.

Read the box (`nproc; free -g`); never trust a figure written here.

An OOM-killed backgrounded benchmark can report exit 0 with an empty log, because python's
stdout never flushed — so it looks like a benchmark that produced no output rather than one
that died. `sudo dmesg -T | grep -i "killed process"` names it in one line.

**Write the loop carefully, because the obvious spelling is broken here and fails silently.**
The shell is **zsh**, which does *not* word-split an unquoted parameter expansion, so this —

```
ls dir/*.py | xargs -n 40 | while read -r chunk; do python -m pytest $chunk -q; done
```

— hands pytest **one enormous non-existent filename** per chunk and prints `no tests ran in
0.14s`, exit 0, once per chunk — which reads as "the chunking worked".

In zsh use `${=chunk}` to force splitting, or sidestep the shell:

```python
for i in range(0, len(files), 40):
    subprocess.run([sys.executable, "-m", "pytest", *files[i : i + 40], "-q"], cwd=root)
```

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

## A retry loop must own its message and its path list

A retry loop around `git commit` that re-reads a message file and a path list edited
*underneath it* between attempts lands neither the message the author wrote nor the files they
meant — and neither is recoverable once another session commits on top.

Three rules keep the loop honest:

- **Snapshot the message before the loop starts** (`cp msg.txt msg.lock`) and point `-F` at the
  copy. A `cat >>` that appends to the live file mid-run, or a `pkill` that truncates it, then
  cannot reach the commit.
- **Freeze the path list too.** Editing the script a running loop is reading is how a path that
  no longer exists (or one that now belongs to another session) ends up in `--only`.
- **Never `pkill` your own loop by a pattern that can match the shell running it.** A kill that
  lands mid-heredoc truncates the message file.

And verify after: `git log -1 --format=%s | wc -c`. A subject over ~72 characters means the
message did not survive.

## Prove "pre-existing", never assume it

Reporting another agent's in-flight breakage as "pre-existing and unrelated" is the
most common false statement in this repo's agent reports, and it is how a real
regression gets waved through. Before you write that phrase, prove all three:

1. `git status <file>` — is it dirty? If yes it is probably someone's live work, not
   a pre-existing condition.
2. Does it fail at `HEAD`? (`git show HEAD:<file>` / check out a clean copy elsewhere.)
3. Is it outside your diff and unreachable from anything you moved?

**A failure count taken mid-refactor is not a fact.** Package-izing
`module.py` → `module/` transiently breaks *every* import in the tree — a red suite
during that window tells you nothing. **Re-run at the end, after things settle, and
report that number.**

## Moving a file has couplings you must chase

A move is never just a move:

- **`STRUCTURE_ALLOW` / `DIR_ALLOW` keys in `tools/lint_structure.py`** are keyed by
  path. Move an allowlisted file and a *new* violation appears unless you re-path the
  key — and its stated reason may no longer be true.
- **Registration order is run order.** Kyber rules run in the order their modules are
  imported, so a naive package split can reorder rules while every name still resolves.
  Preserve import position, and verify.
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
the change is intended, say so explicitly in your report.

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
