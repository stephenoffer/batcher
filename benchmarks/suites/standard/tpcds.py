"""TPC-DS — a representative subset of the 99-query decision-support benchmark.

The full TPC-DS suite is large; this is a curated, commonly-cited subset (brand/
category roll-ups, a window-function revenue ratio, a correlated-subquery returns
query) that exercises multi-join, grouping, windowing, and correlated subqueries
while touching only the tables ``sources.py`` fetches. Each query is standard TPC-DS
SQL with the validation-default substitution parameters; expanding to all 99 is
mechanical once a query's tables are added to ``sources.TPCDS_TABLES``.
"""

from __future__ import annotations

from registry import suite

tpcds = suite("tpcds", dataset="tpcds")

QUERIES: dict[str, str] = {
    "tpcds-q3": """
        SELECT dt.d_year, item.i_brand_id AS brand_id, item.i_brand AS brand,
               SUM(ss_ext_sales_price) AS sum_agg
        FROM date_dim dt, store_sales, item
        WHERE dt.d_date_sk = store_sales.ss_sold_date_sk
          AND store_sales.ss_item_sk = item.i_item_sk
          AND item.i_manufact_id = 128 AND dt.d_moy = 11
        GROUP BY dt.d_year, item.i_brand, item.i_brand_id
        ORDER BY dt.d_year, sum_agg DESC, brand_id
        LIMIT 100
    """,
    "tpcds-q42": """
        SELECT dt.d_year, item.i_category_id, item.i_category, SUM(ss_ext_sales_price) AS s
        FROM date_dim dt, store_sales, item
        WHERE dt.d_date_sk = store_sales.ss_sold_date_sk
          AND store_sales.ss_item_sk = item.i_item_sk
          AND item.i_manager_id = 1 AND dt.d_moy = 11 AND dt.d_year = 2000
        GROUP BY dt.d_year, item.i_category_id, item.i_category
        ORDER BY s DESC, dt.d_year, item.i_category_id, item.i_category
        LIMIT 100
    """,
    "tpcds-q52": """
        SELECT dt.d_year, item.i_brand_id AS brand_id, item.i_brand AS brand,
               SUM(ss_ext_sales_price) AS ext_price
        FROM date_dim dt, store_sales, item
        WHERE dt.d_date_sk = store_sales.ss_sold_date_sk
          AND store_sales.ss_item_sk = item.i_item_sk
          AND item.i_manager_id = 1 AND dt.d_moy = 11 AND dt.d_year = 2000
        GROUP BY dt.d_year, item.i_brand, item.i_brand_id
        ORDER BY dt.d_year, ext_price DESC, brand_id
        LIMIT 100
    """,
    "tpcds-q55": """
        SELECT i_brand_id AS brand_id, i_brand AS brand, SUM(ss_ext_sales_price) AS ext_price
        FROM date_dim, store_sales, item
        WHERE date_dim.d_date_sk = store_sales.ss_sold_date_sk
          AND store_sales.ss_item_sk = item.i_item_sk
          AND i_manager_id = 28 AND d_moy = 11 AND d_year = 1999
        GROUP BY i_brand, i_brand_id
        ORDER BY ext_price DESC, i_brand_id
        LIMIT 100
    """,
    "tpcds-q98": """
        SELECT i_item_id, i_item_desc, i_category, i_class, i_current_price,
               SUM(ss_ext_sales_price) AS itemrevenue,
               SUM(ss_ext_sales_price) * 100
                   / SUM(SUM(ss_ext_sales_price)) OVER (PARTITION BY i_class) AS revenueratio
        FROM store_sales, item, date_dim
        WHERE ss_item_sk = i_item_sk AND i_category IN ('Sports', 'Books', 'Home')
          AND ss_sold_date_sk = d_date_sk
          AND d_date BETWEEN date '1999-02-22' AND date '1999-02-22' + interval '30' day
        GROUP BY i_item_id, i_item_desc, i_category, i_class, i_current_price
        ORDER BY i_category, i_class, i_item_id, i_item_desc, revenueratio
    """,
    "tpcds-q19": """
        SELECT i_brand_id AS brand_id, i_brand AS brand, i_manufact_id, i_manufact,
               SUM(ss_ext_sales_price) AS ext_price
        FROM date_dim, store_sales, item, customer, customer_address, store
        WHERE d_date_sk = ss_sold_date_sk AND ss_item_sk = i_item_sk AND i_manager_id = 8
          AND d_moy = 11 AND d_year = 1998 AND ss_customer_sk = c_customer_sk
          AND c_current_addr_sk = ca_address_sk
          AND substr(ca_zip, 1, 5) <> substr(s_zip, 1, 5) AND ss_store_sk = s_store_sk
        GROUP BY i_brand, i_brand_id, i_manufact_id, i_manufact
        ORDER BY ext_price DESC, i_brand, i_brand_id, i_manufact_id, i_manufact
        LIMIT 100
    """,
    "tpcds-q1": """
        WITH customer_total_return AS (
            SELECT sr_customer_sk AS ctr_customer_sk, sr_store_sk AS ctr_store_sk,
                   SUM(sr_return_amt) AS ctr_total_return
            FROM store_returns, date_dim
            WHERE sr_returned_date_sk = d_date_sk AND d_year = 2000
            GROUP BY sr_customer_sk, sr_store_sk
        )
        SELECT c_customer_id
        FROM customer_total_return ctr1, store, customer
        WHERE ctr1.ctr_total_return > (
                  SELECT AVG(ctr_total_return) * 1.2 FROM customer_total_return ctr2
                  WHERE ctr1.ctr_store_sk = ctr2.ctr_store_sk)
          AND s_store_sk = ctr1.ctr_store_sk AND s_state = 'TN'
          AND ctr1.ctr_customer_sk = c_customer_sk
        ORDER BY c_customer_id
        LIMIT 100
    """,
}

for _name, _query in QUERIES.items():
    tpcds.sql(_name, _query)
