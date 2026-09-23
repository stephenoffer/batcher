# Memory and caching

`cache()` is an execution hint. The result is identical with it, without it, and at every storage level, so the only thing it can change is what the query costs. Read `cache_stats()` as a difference across a span of work rather than as an absolute, because the counters are lifetime figures for the process. Spilling is the same bargain for memory: under a small budget the engine goes out of core instead of failing, and the answer does not move.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/operations/memory_and_caching.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/operations/memory_and_caching.py
```


## Asking a storage level where it puts things

{py:obj}`StorageLevel <batcher.StorageLevel>` names where a cached result lives, and two
predicates answer that without matching on the member name. Use them when the decision is
about the medium rather than the exact level, such as sizing a spill directory only when
one will be written.

```python
from batcher import StorageLevel

for level in StorageLevel:
    print(level.name, level.uses_memory, level.uses_disk)
```

`MEMORY_AND_DISK` answers `True` to both, which is the case a name comparison usually gets
wrong: it is neither `MEMORY_ONLY` nor `DISK_ONLY`, and a branch written against those two
names silently skips it.

## Reading the spill budget the engine will actually use

{py:meth}`Config.spill_budget_bytes <batcher.Config>` resolves the configured spill
envelope against the machine, so it reports the figure the engine enforces rather than the
one you wrote. Read it before sizing a job, because a fraction-of-RAM setting means
something different on every box.

```python
from batcher.config import Config

budget = Config().spill_budget_bytes()
print(budget > 0)
```

## See also

- {doc}`inspecting_a_query`: reading a plan, timing a query, and checking what the engine actually ran.
- {doc}`observability`: verbosity, logging, and execution statistics.
- {doc}`/user-guide/operate/tuning/performance`: measuring and tuning a query that is correct but slow.
- {doc}`/user-guide/operate/running/observability`: what the engine records about a run, and where.
