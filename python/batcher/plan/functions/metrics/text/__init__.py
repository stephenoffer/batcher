"""Metrics that score generated or collected text, with no model in the loop.

Nine families, grouped by the question they answer: `overlap` compares a generation against a
reference, `retrieval` compares it against the context it was supposed to be grounded in,
`diversity` catches a model that has started repeating itself, `quality` and `script` measure the
surface of the raw text, `length` measures how much of it there is and how hard it is to read,
`formatting` checks whether it took the shape it was asked for, `tone` measures how it reads, and
`pii_safety` flags what should never have been in it.

Every name here is an aggregate `Expr` over one or two text columns, so an eval over a million
rows is one scan through the engine rather than a Python loop over examples. `_text` holds the
tokenization and ratio shape they share.

The public names are re-exported by the parent `metrics` facade and reachable as `bt.<name>`;
this package is the organization, not a second import path.
"""
