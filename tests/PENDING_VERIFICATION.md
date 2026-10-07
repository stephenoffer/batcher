# Pending verification

This file lists every public capability that is implemented and unit-tested against fakes, but
has never run against the real system it talks to. Each one shipped from the public API review
(the `AP-NNN` numbers) on the understanding that it would be verified later, once the service,
library, credentials or hardware it needs is available.

A capability stays on this list until a live run passes. When one does, delete its entry in the
same commit that records the run. A connector's documentation carries a warning pointing here
for as long as its entry exists.

## How to run the live tests

Every entry below has a smoke test under `tests/integration/live/`, skipped unless its
environment variable is set. To verify an entry, provision the service, export the variable the
entry names, and run the file it names:

```bash
BATCHER_LIVE_<NAME>=... python -m pytest tests/integration/live/test_live_<name>.py -q
```

A pass on a real service is evidence; a pass on the fakes in `tests/unit/` is not.

## Distributed paths that need a recorded cluster run

`CLAUDE.md` requires a recorded cluster run, in `benchmarks/BENCHMARK_RESULTS.md`, for any
change under `dist/`. Entries here that touch `dist/` say so.

## Entries
