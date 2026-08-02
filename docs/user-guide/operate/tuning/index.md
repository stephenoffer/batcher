# Making it fast

These pages cover the levers that change how long a correct query takes. Work them in the
order below: read the plan before you tune anything, because the operator you would have
guessed at is usually not the one costing the time.

| Page | What it covers |
|---|---|
| {doc}`Performance and memory <performance>` | The levers that matter most, and the memory envelope they run inside |
| {doc}`Caching results <caching>` | Why a plan runs twice, and when to make it run once |
| {doc}`Reading query plans <explain-plans>` | `explain()`, what each operator line means, and how to find the expensive one |
| {doc}`Best practices <best-practices>` | The patterns that follow from keeping per-row work out of Python |

## See also

- {doc}`/user-guide/operate/running/index`: keeping a job healthy once it is fast enough.
- {doc}`/benchmarks/index`: what these levers measure out at against other engines.

```{toctree}
:hidden:

performance
caching
explain-plans
best-practices
```
