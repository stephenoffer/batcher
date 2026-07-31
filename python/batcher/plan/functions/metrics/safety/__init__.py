"""Safety monitors for the text an LLM application reads and writes.

Two families. `injection` scores what arrived — an instruction hidden in a retrieved document, a
jailbreak framing in a user message, characters that render as nothing. `leakage` scores what
left — a credential, an encoded blob, a link built to carry the conversation somewhere else.

All of them are surface heuristics rather than classifiers, and each says so in its own
documentation. They exist to size a problem across a corpus and to alert on a change, not to
stand between a document and a tool call. Every one is an aggregate `Expr`, so a monitor over a
hundred million rows is one scan and breaks down by source or tenant with `group_by`.

The public names are re-exported by the parent `metrics` facade and reachable as `bt.<name>`.
"""
