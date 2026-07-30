"""Prompt construction and context budgeting, as row-wise expressions.

Two halves of one job. `assembly` builds the prompt from a row's columns — templates, tags,
chat and instruction formats, retrieved context. `budget` keeps the result inside a model's
context window without running a tokenizer per row.

Everything here returns a row-wise string (or numeric) expression for `select` /
`with_columns`, so a prompt over a hundred million rows is built in the data plane.
"""

from __future__ import annotations

from batcher.plan.functions.prompt.assembly import (
    chatml_prompt,
    instruction_prompt,
    join_context,
    render_template,
    tagged_fields,
    wrap_tag,
)
from batcher.plan.functions.prompt.budget import (
    fits_context,
    prompt_token_estimate,
    truncate_middle,
    truncate_to_token_budget,
)

__all__ = [
    "chatml_prompt",
    "fits_context",
    "instruction_prompt",
    "join_context",
    "prompt_token_estimate",
    "render_template",
    "tagged_fields",
    "truncate_middle",
    "truncate_to_token_budget",
    "wrap_tag",
]
