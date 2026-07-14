# Bug-hunt ledger

A systematic hunt for high-impact defects across the whole engine, continuing the
contract-loop audit recorded in `audit_ledger.md`. That audit went deep on Kyber,
Carbonite and the contract loop; this one sweeps the areas it did not reach — the
Rust data plane, the SQL front-end, the `plan` and `api` layers, `io`, `dist`, `ml`,
and `governance`.

Every entry is a defect that was **reproduced** before it was fixed and is **pinned by a
test** that fails without the fix. Entries are numbered `B<n>` and never reused, so the
count is a count of *distinct* defects.

Severity: **S1** wrong results / data loss / security bypass · **S2** crash, hang, or
resource leak on a reachable path · **S3** silently degraded plan, estimate, or
performance · **S4** contract/hygiene defect with a real failure mode.

---

## Fixed

| # | Sev | Defect | Where | Test |
|---|-----|--------|-------|------|
| B1 | S2 | Five differential test modules imported `tests.differential.conftest`, which is not an importable package (the repo convention, used by ~690 other files, is `from conftest import`). All five failed at **collection**, so `pytest` aborted the entire run with 5 collection errors — and 279 tests, including `test_diff_operator_matrix.py` (the operator x flag x path cross-product that `CLAUDE.md` names as the safety net against exactly the `sort(descending=True)`-under-spill class of bug), had **never executed** on this branch. | `tests/differential/test_diff_{merge,metadata_answer_equals_execution,operator_matrix,spill_paths,shuffle_key_identity}.py` | the 279 tests themselves, now collected and green |
| B2 | S4 | `test_an_iceberg_identity_distinguishes_catalog_and_row_filter` passed the *relative* catalog URI `sqlite:///x`, and `IcebergSource.identity()` resolves `latest` through a real catalog connection — so every run of the suite bootstrapped a 20 KB SQLite catalog into the **repository root** as a file named `x`. Test pollution that a `git add -A` would commit. | `tests/io/test_lakehouse.py:53` | same test, now on `tmp_path` |
| B3 | S3 | `assert_same`/`_coerce` — the multiset comparator ~690 differential tests depend on — coerced every `int` to `float64` before comparing. Above 2^53 that is lossy, so two distinct int64 results collapse to one float image: `assert_same` accepted `9007199254740993` as equal to `9007199254740992`, meaning **no differential test over large integers could see an off-by-one**. Integers now stay exact; an integral *float* (DuckDB widening a column) is canonicalized to int instead, preserving the int/float tolerance while keeping `1` vs `1.5` distinct. | `tests/differential/conftest.py:38` | verified: off-by-one now caught, `1`==`1.0` still tolerated, `1`!=`1.5` still caught |
</content>
