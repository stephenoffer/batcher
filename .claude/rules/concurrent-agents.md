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
- **`git checkout -- <path>`, even one named path.** This entry used to list only the
  whole-tree forms above, which made a *named* path read as the safe alternative. It is not:
  the operation discards uncommitted work in whatever it touches, and the only thing the
  narrower form buys is a smaller blast radius. An agent reverting its own instrumentation
  ran it on three `crates/` files without looking at them first, having checked
  `git status benchmarks/` carefully an hour earlier and then classified itself as "working
  in benchmarks". **A scope check covers the scope it was run against and expires outside
  it** — the check was real, it was about a different set of files, and its result was
  carried forward to files it never covered.

  Nothing was lost, and the reason is worth knowing because it is not judgement: the other
  session's work had already landed in a commit, so `checkout --` restored it *from* `HEAD`
  rather than destroying it. Anyone still holding **uncommitted** edits in those paths would
  have lost them. So: `git status <the exact paths>` immediately before, every time, and if
  anything is dirty that you did not write, do not run it.
- **Repo-wide autofix**: `ruff check --fix python`, `ruff format python`,
  `cargo fmt --all`. One agent's repo-wide `--fix` silently rewrote 13 findings inside
  other agents' half-finished files. **Scope every fix and format command to the paths
  you own** (`ruff format path/to/your/file.py`).
- **`git commit -a`** — commits everyone's work under your message.
- **The index is shared too, so `git add` your paths is not enough.** `git commit` writes
  whatever is staged, including files *another agent staged*, and a blocked commit of yours
  leaves your own files staged for the next one to sweep up. Both have happened here: one
  commit landed carrying 52 lines of another session's docs page, and an earlier one swallowed
  an unrelated test fix left over from a commit the pre-commit hook had rejected. Use
  **`git commit --only <paths>`** (or `git commit <paths>`), which commits exactly those paths
  and ignores the rest of the index. If you use plain `git commit`, read
  `git diff --cached --stat` immediately before it and confirm every line is yours.

- **`--only` bounds the *paths*, not the *hunks inside them*.** This is the trap on the far
  side of the fix above, and it is easy to walk into precisely because `--only` feels safe.
  A path you own one line of carries **every** uncommitted change in that file, including
  another agent's. It has happened: a commit adding a one-line `fill: None` to a test fixture
  in `bc-interp/src/par.rs` swept up 203 lines of a neighbouring session's half-finished
  aggregation work — without the `agg_par.rs` those lines call into — and **the Rust
  workspace stopped compiling at HEAD** until someone else committed the missing half.

  So the check is per *file*, not per commit: `git diff <path>` every entry on your list and
  confirm you wrote all of it. The trap is the file you add **last**, to fix some small thing
  the gate complained about — that is the one you inspect a single hunk of rather than the
  whole diff. A cheap tell after the fact is `git show --numstat`: a file you touched once
  showing `+153 -50` is not a file you touched once.

  **Make that check mechanical, because reading it does not work.** The session that wrote
  this entry then did the same thing twice more within the hour, both times on the file added
  last: `tools/lint_ir_contract.py` (which shipped a checker referencing constants that were
  not committed, so `lint-ir-contract` failed at HEAD) and `python/batcher/plan/ir_tags.py`
  (which happened to commit the missing constants and repair it, by luck rather than intent).
  Three instances, one cause, and an entry warning about it in between two of them.

  What did work, in the same session, was a script that refused:

  ```bash
  # In the commit-retry loop, BEFORE `git commit`:
  foreign=$(git diff -- "$path" | grep -E '^[+-]' | grep -vE '^[+-][+-]' \
            | grep -vc 'a pattern matching only your own lines')
  [ "$foreign" -eq 0 ] || { echo "$path carries $foreign foreign line(s)"; sleep 60; continue; }
  ```

  It held for an hour across thirty attempts and was never once tempting to override, where
  the same judgement applied by hand failed three times out of four. If you cannot express
  "your own lines" as a pattern, the general form is to copy each file aside the moment you
  first edit it and diff the working tree against that copy at commit time: anything that
  appears is someone else's, no judgement required.

  When the wait is long, check whether the file is *blocking* or *broken*. Twice in that
  session the uncommitted content in a shared file was not work-in-progress at all, it was
  the missing half of a change whose other half was already at HEAD — so HEAD did not
  compile, and the right move was to commit that file **alone**, verbatim, as a labelled
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

A file that goes from a nine-minute hang to `9 passed in 15s` under that variable was never
failing on its own account. Prefer this over `ray stop`: the cluster is shared, another session
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
not stage is simply absent. Measured by running the sandbox with and without each:

| Also stage | Or else |
|---|---|
| `crates/` (the **working tree's**) | `test_diff_execution_mode_matrix::test_the_fixture_actually_shards` raises `FileNotFoundError`. It parses `crates/bc-interp/src/stream/parallel.rs` for `MIN_ROWS_TO_SHARD` rather than hardcoding a number that would let the file quietly stop testing the sharded path. |
| `benchmarks/` | `test_benchmark_isolation.py` fails to **collect** — `ModuleNotFoundError: No module named 'harness'`. |
| `pyproject.toml` | `test_python_floor` and `test_optional_guard_is_shared` fail: the declared Python floor and the extras an optional-guard names both live there. |
| `.github` | `test_python_floor::test_the_release_workflow_builds_on_the_declared_floor` fails. It reads `.github/workflows/release.yml` and cross-checks the pinned `python-version` against the floor. |
| `tools/` | Four `tests/unit` modules fail to **collect** with `ModuleNotFoundError: No module named 'tools'` — `test_agentic_runner`, `test_audit_health_detectors`, `test_ir_contract`, `test_lint_tests`. They import the linters they are about, which is the point of them. A collection error takes the whole module down, so this reads as a broken `PYTHONPATH` rather than an unstaged directory (found 2026-09-12, running `tests/unit` in a sandbox staged by the recipe above). **Staging it is not enough for `test_agentic_runner`:** four of its cases create `git worktree`s, and a `git archive`d sandbox is not a repository, so they fail there and pass in the tree. Run that file where the `.git` is. |

`crates/` is the one copied from the working tree rather than `git archive`d, because the
`.so` you just built came from the working tree and staging HEAD's sources would reintroduce
exactly the drift the test exists to catch. It is 7.5 MB. The rest are not compiled into the
`.so`, so HEAD's copies are right and keep the "committed state, not their WIP" property.

**`python/` needs the same treatment whenever the control plane is part of what you are
measuring, and this recipe did not say so.** A session instrumented the engine's executor-routing
decision, ran it in a sandbox staged the way above, and read `prefer_materializing_aggregate=false`
— which looks exactly like a broken hint and sent it looking for a wire that was not broken. The
flag is computed in `kyber/optimizer/facade.py`, the change that makes it fire on that shape was
**uncommitted**, and `git archive HEAD python` had staged a control plane that genuinely does not
set it. Re-staged from the working tree, the same run read `prefer=true` and the real cause was
one field further on.

The rule generalises past both directories: **`git archive HEAD` is right for the files that are
merely *present*, and wrong for any file whose behaviour is the subject of the measurement.**
Decide per directory, and when a reading contradicts something you verified against the installed
build minutes earlier, suspect the staging before you suspect the engine.

Two things about this table are worth more than the entries.

**The failure modes are not equally readable, and the worst one is not the loudest.** A
`FileNotFoundError` at least names the file it wanted. `ModuleNotFoundError: No module named
'harness'` inside a file called `test_benchmark_isolation` reads as a broken interpreter or a
bad `PYTHONPATH`, reports as an *error* rather than a failure, and takes the whole module down
instead of one test — so the natural first guess is that the recipe is wrong rather than
incomplete. One or two failures is the worst possible size for any of them: small enough to
wave through as flake, large enough to cost an hour if you chase it.

**This list is known-incomplete by construction, and is not a specification.** It was assembled
by two sessions in one afternoon who were each looking for something else, and every entry was
found by a test failing rather than by anyone enumerating what tests read. `.github` is the
one to look at before trusting your judgement over a measurement: the test that needs it is
*about* `pyproject.toml`, names `.github` only inside a path expression, and reads for all the
world as though the workflow were context — so "that one isn't load-bearing" is the plausible
guess and it is wrong. It was asserted here, on the strength of a sandbox run whose staged set
had been read off an `ls` that hides dotfiles, and corrected by someone who ran the arm
properly.

A partial regeneration — tests that climb above their own directory to reach a repo file:

```
grep -rlE 'parents\[|\.parent\.parent' tests --include='*.py'
```

Check what each of the 26 joins onto that root; anything outside `python/` and `tests/` needs
staging or a skip guard.

**`grep`, not `rg`, and the reason generalises past this recipe.** `rg` in a Claude Code
session is a *shell function* from the shell snapshot, not a binary on `PATH`. A bare `rg`
works because the shell resolves it; `... | xargs rg ...` does not, because `xargs` execs
directly — and it fails to **stderr while exiting 0**, so the pipeline prints an empty list and
reports success. That is the same defect this paragraph exists to warn about, one layer out:
the regeneration step would have certified the absence it could not see. Any recipe in
`.claude/rules/` that pipes into `xargs rg` has it.

**And the grep is a floor, not a proof — one of the four entries proves it.**
`test_benchmark_isolation` fails in an unstaged sandbox with `ModuleNotFoundError: No module
named 'harness'`, which is an *import* dependency on `benchmarks/`, not a path read. It happens
to appear in the scan only because it also computes a path; a version of it that imported and
nothing more would be invisible, and nothing in the scan's design would say so. The only
enumeration that needs no judgement is **run the suite in the sandbox and see what breaks**,
which is how all four of these were found, twice by accident.

Three things make this work and are worth keeping: `bc-py`'s `[lib] name = "_native"` is a
plain `cdylib`, so the file cargo produces *is* the extension module under a different name;
`git archive HEAD` gives you the committed tree rather than the other session's half-finished
edits, so their broken imports do not become your test failures; and `PYTHONPATH` wins over the
installed package, so nothing outside the scratchpad is touched. Confirm you are in the sandbox
with `bt.versions()["engine_profile"]` and `bt._native.__file__`.

This was used to build, benchmark and then **reject** the dictionary-preserving boundary change
(`competitor_technique_review.md` item 6) during a session where another agent held 20-27
pytest processes against the installed `.so` for its entire duration.

The same swap is why a benchmark can silently measure the *other* agent's build: check
`bt.versions()["engine_profile"]`, which the suite now does for you.

Two ways to shoot yourself with the sandbox, both of which happened in one session:

- **One sandbox, two concurrent runs.** Copying the `.so` in is the same in-place overwrite
  `just build` does, so a second run started while the first is going kills it — a fatal
  interpreter error, a 300-line list of loaded extension modules, and no test results. Name
  the sandbox per invocation (`pybox-$$` or a caller-supplied tag) and the problem is gone.
- **Deleting a sandbox a run is still using.** `pytest` `chdir`s back to its start path at the
  end and dies with `FileNotFoundError` on a directory you cleaned up. Check for a live
  process before `rm -rf`.

### A full suite gets OOM-killed on a busy box; chunk it

The head node is shared. With three other sessions running suites, a whole-directory
`pytest tests/differential` was **killed** twice, ~20% in, reporting nothing — the same
`Killed` you would see from a hang, and easy to misread as your change.

Run the directory in chunks of ~40 files, one process each. **The reason that survives a
hardware change is attribution, not headroom**: a chunk that is killed names itself, where a
whole-directory run takes the entire result down with it and tells you nothing.

**The head node is not one machine, so no number written here is the number you have.** This
paragraph said "30 GB" for long enough that one session throttled itself on it; it was then
corrected to "184 GB with ~170 available and 96 cores", which was a real reading of a real
c5d.24xlarge and is wrong on the box this was next read on — `nproc` 16, `free -g` 30 GiB total
and 16 available (2026-09-08). A correction that replaces one hardcoded figure with another
inherits the defect it fixed. Read the box:

```
nproc; free -g
```

The consequence is not academic. A TPC-H **sf10** run with three engines preloading Arrow was
OOM-killed on the small box at 22.6 GB RSS, and the wrapper reported exit 0 with an empty log,
because the driver was backgrounded and python's stdout never flushed — so it looks like a
benchmark that produced no output rather than one that died. `sudo dmesg -T | grep -i "killed
process"` names it in one line. The same applies to any memory-hungry benchmark: a 10 M-row x
9-column A/B was killed until it was cut to 4 M.

**Write the loop carefully, because the obvious spelling is broken here and fails silently.**
The shell is **zsh**, which does *not* word-split an unquoted parameter expansion, so this —

```
ls dir/*.py | xargs -n 40 | while read -r chunk; do python -m pytest $chunk -q; done
```

— hands pytest **one enormous non-existent filename** per chunk and prints `no tests ran in
0.14s`, exit 0, once per chunk. Two sessions hit it on the same afternoon; one lost a
debugging cycle and the other misattributed it to a different bug in its own loop. It is the
worst-placed instance of that failure in this file, because the recipe is offered as the fix
for *an OOM that already reported nothing* — so "no tests ran" reads as "the chunking worked".

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

The contention above makes a retry loop around `git commit` the standard fix, and it has now
produced two malformed commits here. Both failed the same way: the loop re-read a message file
and a path list that had been *edited underneath it* between attempts, so what eventually
landed was neither the message the author wrote nor the files they meant.

One of the two lost its subject line entirely and committed with a paragraph of body text as
the summary; the other committed against a stale path list and carried two unrelated changes
together. Neither is recoverable once another session commits on top, because rewriting shared
history in this tree is worse than the defect.

Three rules keep the loop honest:

- **Snapshot the message before the loop starts** (`cp msg.txt msg.lock`) and point `-F` at the
  copy. A `cat >>` that appends to the live file mid-run, or a `pkill` that truncates it, then
  cannot reach the commit.
- **Freeze the path list too.** Editing the script a running loop is reading is how a path that
  no longer exists (or one that now belongs to another session) ends up in `--only`.
- **Never `pkill` your own loop by a pattern that can match the shell running it.** That is what
  truncated the message file here: the kill landed mid-heredoc.

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
