"""`_internal` — layer 0: what every layer above may depend on, and nothing more.

This is the bottom of the import matrix (`.claude/rules/architecture.md`): it imports
only `config`, so anyone may import it. That is precisely why it exists — the four
layer-3 subsystems (`kyber`, `carbonite`, `core`, `governance`) cannot import each
other, so when two of them need the same helper the answer is to lift it *down* to
here. Copy-pasting it into both is the one genuinely wrong way to share, and it has
happened before.

What lives here is cross-cutting machinery with no domain of its own: the exception
hierarchy (`errors`), the observability event bus (`events`), logging (`logging`),
hardware detection (`hardware`), install paths (`paths`), producer/consumer overlap
(`prefetch`), and the generic `Registry[T]` behind every extension point
(`registry`).

`native` is the load-bearing one: it is **the** single accessor for the compiled
engine. Always `from batcher._internal.native import engine` — never
`import batcher._native` directly. The static import graph cannot see into a compiled
extension, so a direct import is attributed to the root `batcher` package, which
re-exports `api`, which imports every subsystem — forging a phantom
`core -> batcher -> api -> kyber` cycle that silently breaks the layer-independence
contract. That is not hypothetical: it is what broke all six independence directions
once already.

Not part of the public API — nothing here is reachable from `import batcher as bt`.
"""
