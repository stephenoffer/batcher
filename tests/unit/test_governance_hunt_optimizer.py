"""The Kyber optimizer must never undo a governance rewrite.

`enforce` runs *before* the optimizer and lowers a column mask to an expression at the
scan (``Project(mask, Filter(row_access, Scan))``). That is only a real access boundary
if no later optimizer pass can hoist the mask above a use of the raw column, strip it as
"dead", or reorder the row filter away. A pass that did any of those would be a silent
governance bypass — the highest-value thing to pin here — so these tests run the actual
optimizer over a governed plan and assert the protection is still in the output.

These are plan-shape tests (no engine): they inspect the optimized `LogicalPlan`/IR.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher.governance import (
    MatchesAttribute,
    Principal,
    Redact,
    SecurityCatalog,
    enforce,
)
from batcher.kyber.optimizer import Optimizer
from batcher.plan.expr_ir import Col
from batcher.plan.logical import Project, Projection, Scan
from batcher.plan.schema import SchemaRef

pytestmark = pytest.mark.unit

TABLE = "/data/customers.parquet"
COLUMNS = ["id", "ssn", "region", "salary"]
_SCHEMA = SchemaRef.from_arrow(
    pa.schema([(c, pa.int64() if c in ("id", "salary") else pa.string()) for c in COLUMNS])
)
ANALYST = Principal("ana", roles=["analyst"], attrs={"region": "EU"})


def _governed_catalog() -> SecurityCatalog:
    # `region` is deliberately NOT granted: the row filter must still be able to reference
    # it (a policy runs with the catalog's authority, not the caller's).
    return (
        SecurityCatalog()
        .grant("analyst", on=TABLE, select=["id", "ssn", "salary"])
        .mask_column(TABLE, "ssn", Redact(show_last=4))
        .filter_rows(TABLE, MatchesAttribute("region", "region"), name="region_scope")
    )


def _optimize(plan):
    return Optimizer(sources=[]).logical_rewrite(plan)


def _find_masks(ir: object) -> list[dict]:
    """Every ``str`` IR node with ``fn == 'mask'`` anywhere in the IR document."""
    found: list[dict] = []
    if isinstance(ir, dict):
        if ir.get("e") == "str" and ir.get("fn") == "mask":
            found.append(ir)
        for v in ir.values():
            found.extend(_find_masks(v))
    elif isinstance(ir, (list, tuple)):
        for v in ir:
            found.extend(_find_masks(v))
    return found


def _ssn_reaches_output_unmasked(ir: dict) -> bool:
    """True if the *top* projection emits a bare, unmasked ``ssn`` column."""
    if ir.get("op") != "project":
        return False
    for item in ir["exprs"]:
        if item["alias"] == "ssn":
            expr = item["expr"]
            # Acceptable only if it is (or wraps) a mask; a bare `col('ssn')` is a bypass.
            return not _find_masks(expr)
    return False


def test_optimizer_keeps_the_mask_when_the_masked_column_is_selected():
    """Selecting `ssn` on top of the governed scan must still read it through the mask."""
    governed, _ = enforce(Scan(0, _SCHEMA), [TABLE], ANALYST, _governed_catalog())
    user = Project(governed, (Projection("ssn", Col("ssn")),))
    ir = _optimize(user).to_ir()
    assert _find_masks(ir), "the mask expression was optimized away entirely"
    assert not _ssn_reaches_output_unmasked(ir), "raw ssn reached the output — a bypass"


def test_optimizer_keeps_the_row_filter_referencing_an_ungranted_column():
    """The row filter over the ungranted `region` must survive optimization/pruning."""
    governed, _ = enforce(Scan(0, _SCHEMA), [TABLE], ANALYST, _governed_catalog())
    # A user query that reads only `id` — column pruning will try to drop `region`.
    user = Project(governed, (Projection("id", Col("id")),))
    ir = _optimize(user).to_ir()
    # The predicate `region == 'EU'` (as its column reference) must still be in the plan.
    assert '"name": "region"' in str(ir).replace("'", '"'), (
        "the governance row filter on `region` was pruned away — a bypass"
    )


def test_optimizer_does_not_hoist_the_mask_above_the_row_filter_or_scan():
    """The mask must remain a projection above the scan/filter, never sink below or vanish."""
    governed, _ = enforce(Scan(0, _SCHEMA), [TABLE], ANALYST, _governed_catalog())
    user = Project(governed, (Projection("ssn", Col("ssn")),))
    opt = _optimize(user)
    ir = opt.to_ir()
    masks = _find_masks(ir)
    assert len(masks) == 1
    # The masked input must be the source `ssn` column (possibly via a folded cast), so the
    # raw value is consumed only inside the mask, never emitted alongside it.
    masked_input = masks[0]["input"]
    while masked_input.get("e") == "cast":
        masked_input = masked_input["input"]
    assert masked_input == {"e": "col", "name": "ssn"}
