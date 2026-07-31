"""String free functions, in two halves: building text and reading structure out of it.

`building` assembles a string from columns (`concat`, `concat_ws`, `format_string`).
`extraction` runs the other direction, recovering the fragment a model wrapped its answer in.
They share a package because they share a namespace in the public API, not because they share
an implementation.
"""

from __future__ import annotations

from batcher.plan.functions.string.building import concat, concat_ws, format_string
from batcher.plan.functions.string.extraction import (
    extract_after,
    extract_between,
    extract_boxed,
    extract_choice,
    extract_citations,
    extract_code_block,
    extract_first_number,
    extract_json,
    extract_json_array,
    extract_last_number,
    extract_reasoning,
    extract_tag,
    is_refusal,
    strip_reasoning,
)

__all__ = [
    "concat",
    "concat_ws",
    "extract_after",
    "extract_between",
    "extract_boxed",
    "extract_choice",
    "extract_citations",
    "extract_code_block",
    "extract_first_number",
    "extract_json",
    "extract_json_array",
    "extract_last_number",
    "extract_reasoning",
    "extract_tag",
    "format_string",
    "is_refusal",
    "strip_reasoning",
]
