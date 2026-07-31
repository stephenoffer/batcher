# Operations cookbook

This section holds 7 runnable recipes for running the engine, ordered the way you meet them: configure it, watch it, then deal with what it tells you.

Every page embeds a complete, self-contained script from the [`examples/operations/`](https://github.com/batcher/batcher/tree/main/examples/operations) directory. The scripts build their own in-memory data and assert on their own output, and `tests/docs/test_examples.py` runs all of them, so a page that stops matching the engine fails the suite instead of drifting.

| Recipe | What it shows |
|---|---|
| {doc}`configuration` | Options, scoped overrides, and profiles |
| {doc}`environment` | What is installed, what the engine sees, and what to paste into a bug report |
| {doc}`inspecting_a_query` | Reading a plan, timing a query, and checking what actually ran |
| {doc}`observability` | Verbosity, logging, and execution statistics |
| {doc}`memory_and_caching` | Caching a reused branch, and spilling under a tight budget |
| {doc}`error_handling` | Catching the failure you meant to catch |
| {doc}`streaming_basics` | The same operators, run incrementally |

## See also

- {doc}`../../configuration/index`: the configuration guide and the field-by-field reference.
- {doc}`/user-guide/operate/explain-plans`: reading a plan and its measured profile in depth.
- {doc}`/user-guide/operate/observability`: progress, structured logs, and the web dashboard.
- {doc}`/user-guide/operate/troubleshooting`: the symptom-to-cause table when one of these goes wrong.

```{toctree}
:hidden:

configuration
environment
inspecting_a_query
observability
memory_and_caching
error_handling
streaming_basics
```
