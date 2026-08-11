# TPC-H

This page covers the scripts that run the TPC-H benchmark queries over the real sf1 dataset,
plus the scripts that measure what those queries cost.

Every table comes from the public mirror in `s3://ray-benchmark-data`, cached locally with
its canonical column names restored. The fact tables are read as a bounded prefix so the
suite stays a release check rather than a benchmark, which means the results are correct for
the slice rather than equal to the published TPC-H answers. The scripts assert on structural
properties that hold at any scale, not on magic numbers.

## The query shapes

The 22 queries between them cover the shape vocabulary a relational engine has to answer:
one filtered scan, several multi-table joins, a correlated minimum, a correlated average,
two set-difference queries, and a handful that need conditional aggregation.

```python
# docs: skip
import datetime as dt

from _common import tpch
import batcher as bt
from batcher import col

lineitem = tpch("lineitem")
cutoff = dt.date(1998, 12, 1) - dt.timedelta(days=90)

disc_price = col("l_extendedprice") * (1 - col("l_discount"))

report = (
    lineitem.filter(col("l_shipdate") <= bt.lit(cutoff))
    .group_by("l_returnflag", "l_linestatus")
    .agg(
        sum_qty=col("l_quantity").sum(),
        sum_disc_price=disc_price.sum(),
        count_order=bt.count(),
    )
    .sort("l_returnflag", "l_linestatus")
)
```

Two rewrites recur across the suite and are worth recognizing. A correlated subquery becomes
a grouped aggregate joined back to the rows it came from, which is what Q2 and Q17 do with a
minimum and an average. And an `EXISTS` becomes a semi join, which is what keeps Q4 counting
orders rather than lines.

## Cost

Q6 is the query with no joins and no grouping, so it isolates the read path. Comparing a
wide read against a projected one, and a filtered one against an unfiltered one, shows what
projection and predicate pushdown actually buy.

Q9 is the opposite: its filter is a substring match that no statistic can help with, so the
only way to cut work is to apply the expensive predicate to the smallest relation first.
`examples/tpch/join_order_matters.py` runs the same five-table query in two orders and
asserts they return identical rows.

## Every script on this page

The table below lists the TPC-H scripts in path order.

<!-- library-table: tpch -->
| Script | Shows |
| --- | --- |
| `examples/tpch/aggregate_pipeline.py` | A full reporting pipeline over TPC-H, from scan to written report |
| `examples/tpch/full_suite_verification.py` | Verifying every cached TPC-H table against the source it came from |
| `examples/tpch/join_order_matters.py` | The same five-table query, written in two join orders |
| `examples/tpch/q01_pricing_summary.py` | TPC-H Q1 - the pricing summary report over real `lineitem` data |
| `examples/tpch/q01_variants.py` | Q1 written four ways, all returning the same answer |
| `examples/tpch/q02_minimum_cost_supplier.py` | TPC-H Q2 - the cheapest supplier for a part, via a correlated minimum |
| `examples/tpch/q03_incremental.py` | Q3 recomputed incrementally as new orders arrive |
| `examples/tpch/q03_shipping_priority.py` | TPC-H Q3 - unshipped orders with the highest revenue |
| `examples/tpch/q04_order_priority_checking.py` | TPC-H Q4 - order priority, counted with a semi join |
| `examples/tpch/q05_local_supplier_volume.py` | TPC-H Q5 - revenue by nation, where customer and supplier share that nation |
| `examples/tpch/q06_forecasting_revenue_change.py` | TPC-H Q6 - the single-table scan query: three predicates and one sum |
| `examples/tpch/q06_variants_and_pushdown.py` | Q6 four ways, and what each costs |
| `examples/tpch/q07_volume_shipping.py` | TPC-H Q7 - trade volume between two nations, in both directions |
| `examples/tpch/q08_national_market_share.py` | TPC-H Q8 - one nation's share of a market, as a ratio of two conditional sums |
| `examples/tpch/q09_product_type_profit.py` | TPC-H Q9 - profit by nation and year, from a substring match on part name |
| `examples/tpch/q10_returned_item_reporting.py` | TPC-H Q10 - the customers costing you the most in returns |
| `examples/tpch/q11_important_stock.py` | TPC-H Q11 - the parts holding most of the inventory value, against a computed threshold |
| `examples/tpch/q12_shipping_modes.py` | TPC-H Q12 - late deliveries split by order priority, using conditional sums |
| `examples/tpch/q13_customer_distribution.py` | TPC-H Q13 - how many customers have how many orders, including the zeros |
| `examples/tpch/q14_promotion_effect.py` | TPC-H Q14 - what share of a month's revenue came from promotional parts |
| `examples/tpch/q15_top_supplier.py` | TPC-H Q15 - the supplier with the highest quarterly revenue, via a reused subquery |
| `examples/tpch/q16_parts_supplier_relationship.py` | TPC-H Q16 - how many suppliers can supply each part variant, after an exclusion |
| `examples/tpch/q17_small_quantity_revenue.py` | TPC-H Q17 - revenue from unusually small orders, against a per-part average |
| `examples/tpch/q18_large_volume_customer.py` | TPC-H Q18 - the orders whose total quantity crosses a threshold |
| `examples/tpch/q19_discounted_revenue.py` | TPC-H Q19 - three unrelated product filters OR'd into one scan |
| `examples/tpch/q20_potential_part_promotion.py` | TPC-H Q20 - suppliers holding excess stock, through two levels of subquery |
| `examples/tpch/q21_suppliers_kept_orders_waiting.py` | TPC-H Q21 - the supplier who was the only one late on a multi-supplier order |
| `examples/tpch/q22_global_sales_opportunity.py` | TPC-H Q22 - customers with a healthy balance who have never ordered |
| `examples/tpch/query_suite_smoke.py` | Running every TPC-H example's core query in one pass, as a smoke check |
| `examples/tpch/scan_and_project_costs.py` | What each TPC-H table costs to scan, and how much a projection saves |
<!-- /library-table -->
