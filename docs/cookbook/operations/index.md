# Operations cookbook

Running the engine: configuration, plan inspection, memory, observability, error handling, and streaming basics.

Every page here embeds a complete, self-contained script from the
[`examples/operations/`](https://github.com/batcher/batcher/tree/main/examples/operations) directory.
The scripts build their own in-memory data and assert on their own output, and
`tests/docs/test_examples.py` runs all of them, so a page that stops matching the
engine fails the suite instead of drifting.

| Recipe | What it shows |
|---|---|
| {doc}`configuration` | Configuring the engine: options, scoped overrides, and profiles |
| {doc}`environment` | What is installed, what the engine sees, and what to paste into a bug report |
| {doc}`error_handling` | The exception hierarchy: catching the failure you meant to catch |
| {doc}`inspecting_a_query` | Reading a plan, timing a query, and checking what the engine actually ran |
| {doc}`memory_and_caching` | Bounded memory: caching a reused branch and spilling under a tight budget |
| {doc}`observability` | Watching a query run: verbosity, logging, and execution statistics |
| {doc}`streaming_basics` | Batch as the bounded case of streaming: the same operators, incrementally |

```{toctree}
:hidden:

configuration
environment
error_handling
inspecting_a_query
memory_and_caching
observability
streaming_basics
```
