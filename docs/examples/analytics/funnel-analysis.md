# Funnel analysis

How many people who viewed a product went on to add it to a cart, then to check out,
then to pay? A funnel is four counts, and getting them right is harder than it looks
for two reasons: the steps have to happen *in order*, and the obvious way to enforce
that order is a self-join that blows up.

## The data

Four users, four steps, timestamps in minutes past 10:00. Note `u3`: it has a
`purchase` event that lands *before* its `view`. That is a real thing, and it happens when
events arrive from two different clients, or when someone comes back to a page after
buying.

```python
import datetime as dt

import batcher as bt
from batcher import col, count_if

t0 = dt.datetime(2024, 3, 1, 10, 0)


def at(minutes: int) -> dt.datetime:
    return t0 + dt.timedelta(minutes=minutes)


events = bt.from_pydict(
    {
        "user": ["u1", "u1", "u1", "u1", "u2", "u2", "u3", "u3", "u4"],
        "step": [
            "view", "cart", "checkout", "purchase",
            "view", "cart",
            "view", "purchase",
            "view",
        ],
        "ts": [at(0), at(2), at(5), at(7), at(0), at(30), at(1), at(0), at(3)],
    }
)
print(events.count())
# 9
```

## The trap, part one: counting steps instead of paths

:::{warning}
`GROUP BY step` has no idea what order anything happened in, so it produces a funnel that
*widens*: two purchases and one checkout. A funnel that widens is measuring the wrong
thing.
:::

```python
naive = (
    events.group_by("step")
    .agg(users=col("user").n_unique())
    .sort("users", descending=True)
)
print(naive.to_pydict())
# {'step': ['view', 'cart', 'purchase', 'checkout'], 'users': [4, 2, 2, 1]}
```

Two purchases, one checkout. `u3` purchased at minute 0 and viewed at minute 1, so its
`purchase` belongs to some earlier journey, not this one.

Here is what the two funnels on this page produce, side by side. The ordered one is the
answer built further down:

| Step | `GROUP BY step` | Ordered funnel |
| --- | --- | --- |
| view | 4 | 4 |
| cart | 2 | 2 |
| checkout | 1 | 1 |
| purchase | 2 | 1 |

## The trap, part two: the self-join

The usual fix is to join the event table to itself once per step:

```python
# docs: skip
views = events.filter(col("step") == "view").rename({"ts": "t_view"})
carts = events.filter(col("step") == "cart").rename({"ts": "t_cart"})
# ...and so on, then join views to carts to checkouts to purchases on user.
```

That works on four users. On real data it does not. A user with 12 views and 3 carts
produces 36 rows out of the first join before you have filtered anything. Add two more
steps and the intermediate blows past the size of the input by orders of magnitude.
The join is doing a cross product inside each user, and then you throw almost all of it
away.

You do not need the cross product. You need one number per (user, step): the earliest
time that user reached that step.

## One pass, then a pivot

Timestamps do not aggregate directly, so carry the event time as epoch seconds: an `Int64`,
which `min` is happy to reduce. `pivot` then spreads the steps across columns, giving one row
per user with a null wherever a step never happened.

```python
per_user = (
    events.with_columns(t=col("ts").dt.epoch())
    .pivot(index=["user"], on="step", values="t", aggregate="min")
    .sort("user")
)
print(per_user.to_pydict())
# {'user': ['u1', 'u2', 'u3', 'u4'],
#  'cart': [1709287320, 1709289000, None, None],
#  'checkout': [1709287500, None, None, None],
#  'purchase': [1709287620, None, 1709287200, None],
#  'view': [1709287200, 1709287200, 1709287260, 1709287380]}
```

One shuffle on `user`, one row per user out. Memory is bounded by the number of users
times the number of steps, not by the number of events.

## Counting the funnel

Now every step condition is an ordinary comparison on that one row, and a step only
counts if the whole prefix ahead of it happened in order.

:::{tip}
Nulls do the work for free: `None > 1709287200` is null, which `count_if` does not count.
A user who never carted therefore drops out of every downstream step without a single
explicit null check.
:::

The two tabs build the same plan, the same shuffle, and the same answer. The
`MIN(CASE WHEN ...)` idiom in the SQL tab is what `pivot` lowers to.

::::{tab-set}
:::{tab-item} DataFrame
```python
funnel = per_user.group_by().agg(
    viewed=count_if(col("view").is_not_null()),
    carted=count_if(col("cart") > col("view")),
    checked_out=count_if((col("checkout") > col("cart")) & (col("cart") > col("view"))),
    purchased=count_if(
        (col("purchase") > col("checkout"))
        & (col("checkout") > col("cart"))
        & (col("cart") > col("view"))
    ),
)
print(funnel.to_pydict())
# {'viewed': [4], 'carted': [2], 'checked_out': [1], 'purchased': [1]}
```
:::

:::{tab-item} SQL
```python
sql_funnel = bt.sql(
    """
    WITH per_user AS (
      SELECT user,
             MIN(CASE WHEN step = 'view'     THEN EXTRACT(EPOCH FROM ts) END) AS t_view,
             MIN(CASE WHEN step = 'cart'     THEN EXTRACT(EPOCH FROM ts) END) AS t_cart,
             MIN(CASE WHEN step = 'checkout' THEN EXTRACT(EPOCH FROM ts) END) AS t_checkout,
             MIN(CASE WHEN step = 'purchase' THEN EXTRACT(EPOCH FROM ts) END) AS t_purchase
      FROM events
      GROUP BY user
    )
    SELECT
      COUNT(*) FILTER (WHERE t_view IS NOT NULL) AS viewed,
      COUNT(*) FILTER (WHERE t_cart > t_view) AS carted,
      COUNT(*) FILTER (WHERE t_checkout > t_cart AND t_cart > t_view) AS checked_out,
      COUNT(*) FILTER (
        WHERE t_purchase > t_checkout AND t_checkout > t_cart AND t_cart > t_view
      ) AS purchased
    FROM per_user
    """,
    events=events,
)
print(sql_funnel.to_pydict())
# {'viewed': [4], 'carted': [2], 'checked_out': [1], 'purchased': [1]}
```
:::
::::

Four viewed, two carted, one checked out, one purchased. Monotonically non-increasing, which a funnel must be. `u3` is gone from
the purchase count, because its purchase preceded its view.

## Reading it as a table

`unpivot` turns the one wide row back into one row per step, which is what a chart
wants:

```python
long = funnel.unpivot(
    on=["viewed", "carted", "checked_out", "purchased"],
    variable_name="step",
    value_name="users",
)
print(long.to_pydict())
# {'step': ['viewed', 'carted', 'checked_out', 'purchased'], 'users': [4, 2, 1, 1]}
```

:::{dropdown} Variations worth knowing: expiry windows and last-touch funnels
Real funnels expire: a cart three weeks after a view is not a conversion. Add the bound to
the comparison rather than filtering the events. `(col("cart") > col("view")) &
(col("cart") - col("view") < 3600)` restricts the step to an hour, and because the times are
epoch seconds the arithmetic is just arithmetic.

`aggregate="max"` in the pivot gives the *last* time a user hit each step instead of the
first, which measures a different (and usually less flattering) funnel. Pick one and say
which.
:::

## See also

:::{seealso}
- {doc}`Sessionization <sessionization>`: scoping the funnel to a single visit.
- {doc}`Retention curves <retention-curves>`: the same one-row-per-user shape, over days.
- {doc}`Aggregations <../../user-guide/aggregations>`: `count_if` and the rest of `agg`.
- {doc}`Pivoting <../../user-guide/pivoting>`: `pivot` and `unpivot`, the two moves this page turns on.
- {doc}`Joins <../../user-guide/joins>`: what the self-join you are avoiding would have cost.
- {doc}`Join algorithms <../../deep-dives/join-algorithms>`: why the intermediate blows up, in detail.
- {doc}`Dataset API <../../api/dataset>`: `pivot`, `unpivot`, `group_by`.
:::
