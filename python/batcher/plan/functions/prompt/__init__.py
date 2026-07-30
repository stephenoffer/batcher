"""Prompt construction and context budgeting, as row-wise expressions.

Three halves of one job, which is one more than there should be. `assembly` builds the prompt
from a row's columns — templates, tags, chat and instruction formats, retrieved context.
`budget` keeps the result inside a model's context window without running a tokenizer per row.
`chat` reads the other direction, pulling turns, answers, and completeness out of the
list-of-`{role, content}`-structs column a chat log or a fine-tuning set arrives as.

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
from batcher.plan.functions.prompt.chat import (
    conversation_turns,
    ends_with_role,
    last_message,
    render_messages,
)

__all__ = [
    "chatml_prompt",
    "conversation_turns",
    "ends_with_role",
    "fits_context",
    "instruction_prompt",
    "join_context",
    "last_message",
    "prompt_token_estimate",
    "render_messages",
    "render_template",
    "tagged_fields",
    "truncate_middle",
    "truncate_to_token_budget",
    "wrap_tag",
]
