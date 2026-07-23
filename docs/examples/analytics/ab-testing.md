# A/B testing

Variant B shipped. Did it convert better than A? The query is a `GROUP BY variant` and a
division, and there are two ways to get the division wrong that will hand you the opposite
answer with a straight face.

## The data

Two tables, which is the point. `assignments` is the randomization: one row per user, the
arm they were bucketed into, and a segment. `events` is what they did. Some users generated
several events, and one user, `u4`, generated none at all.

```python
import batcher as bt
from batcher import col, count_if

assignments = bt.from_pydict(
    {
        "user": ["u1", "u2", "u3", "u4", "u5", "u6", "u7", "u8"],
        "variant": ["A", "A", "A", "A", "B", "B", "B", "B"],
        "country": ["us", "us", "uk", "uk", "us", "us", "uk", "uk"],
    }
)
events = bt.from_pydict(
    {
        "user": ["u1", "u1", "u2", "u3", "u5", "u5", "u5", "u6", "u7", "u8", "u8"],
        "converted": [True, True, False, True, True, True, True, False, True, False, True],
    }
)
print(assignments.count(), events.count())
# 8 11
```

## The trap: dividing by the wrong denominator

:::{warning}
The experimental unit is the *user*, because that is what was randomized. Divide by events
instead and the inner join quietly drops every user who did nothing, while one enthusiastic
user inflates the other arm's denominator. Two bugs, both invisible, both pushing the same
way.
:::

Join the events to the assignments, count conversions, divide by rows. This is the query
everyone writes first:

```python
per_event = (
    events.join(assignments, on="user")
    .group_by("variant")
    .agg(events=bt.count(), conversions=count_if(col("converted")))
    .with_columns(rate=(col("conversions") / col("events")).round(3))
    .sort("variant")
)
print(per_event.to_pydict())
# {'variant': ['A', 'B'], 'events': [4, 7], 'conversions': [3, 5], 'rate': [0.75, 0.714]}
```

A wins, 75% to 71%. Ship A.

Except `u5` fired three converting events on its own, so B's denominator is inflated by one
enthusiastic user. And `u4`, who was assigned to A and did nothing, is not in the events
table at all, so it never reaches the denominator. The inner join silently dropped it.

## Collapse to the unit of randomization first

Aggregate the events to one row per user, then join *from* the assignment table (a left
join, so the users who did nothing survive), then fill their nulls.

::::{tab-set}
:::{tab-item} DataFrame
```python
per_user = events.group_by("user").agg(converted=col("converted").bool_or())

per_variant = (
    assignments.join(per_user, on="user", how="left")
    .with_columns(converted=col("converted").fill_null(False))
    .group_by("variant")
    .agg(users=bt.count(), conversions=count_if(col("converted")))
    .with_columns(rate=(col("conversions") / col("users")).round(3))
    .sort("variant")
)
print(per_variant.to_pydict())
# {'variant': ['A', 'B'], 'users': [4, 4], 'conversions': [2, 3], 'rate': [0.5, 0.75]}
```
:::

:::{tab-item} SQL
```python
sql_variant = bt.sql(
    """
    WITH per_user AS (
      SELECT user, BOOL_OR(converted) AS converted
      FROM events
      GROUP BY user
    )
    SELECT a.variant,
           COUNT(*) AS users,
           COUNT(*) FILTER (WHERE COALESCE(u.converted, FALSE)) AS conversions
    FROM assignments a
    LEFT JOIN per_user u ON a.user = u.user
    GROUP BY a.variant
    ORDER BY a.variant
    """,
    assignments=assignments,
    events=events,
)
print(sql_variant.to_pydict())
# {'variant': ['A', 'B'], 'users': [4, 4], 'conversions': [2, 3]}
```
:::
::::

B wins, 75% to 50%. The result flips:

| Query | Denominator | A | B | Winner |
| --- | --- | --- | --- | --- |
| Per event, inner join | events | 3/4 = 0.75 | 5/7 = 0.714 | A |
| Per user, left join | assigned users | 2/4 = 0.5 | 3/4 = 0.75 | B |

Both arms now have four users, which is what randomization promised and what the first
query destroyed. `bool_or` is the right reducer for a boolean: converted-at-least-once.

:::{important}
Note the direction of the join. `assignments` is the left side and `per_user` is the right,
never the other way. The exposure denominator comes from the assignment log, not from the behaviour log. Anything
computed from the behaviour log conditions on having behaved, which is the whole thing you
are trying to measure.
:::

## Is 50% vs 75% real?

With four users per arm: no. But do the arithmetic explicitly rather than eyeballing it. A
two-proportion z-test is two lines over a two-row result. The aggregation happened in the
engine, so what comes back to Python is a summary you can do scalar math on.

```python
import math

rows = per_variant.to_pydict()
n_a, c_a = rows["users"][0], rows["conversions"][0]
n_b, c_b = rows["users"][1], rows["conversions"][1]

p_pool = (c_a + c_b) / (n_a + n_b)
se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
z = (c_b / n_b - c_a / n_a) / se
print(round(z, 3))
# 0.73
```

A z of 0.73 is nowhere near the 1.96 you would need at 95%. The 25-point gap is one user.
Eight users cannot resolve a 25-point effect, and this test says so.

## Simpson's paradox, in one table

Before you believe any pooled number, break it by the segments you know about. Randomization
balances segments *in expectation*. In a real experiment with real traffic it often does not,
and then the pooled average lies.

```python
by_segment = (
    assignments.join(per_user, on="user", how="left")
    .with_columns(converted=col("converted").fill_null(False))
    .group_by("country", "variant")
    .agg(users=bt.count(), conversions=count_if(col("converted")))
    .with_columns(rate=(col("conversions") / col("users")).round(3))
    .sort("country", "variant")
)
print(by_segment.to_pydict())
# {'country': ['uk', 'uk', 'us', 'us'], 'variant': ['A', 'B', 'A', 'B'],
#  'users': [2, 2, 2, 2], 'conversions': [1, 2, 1, 1], 'rate': [0.5, 1.0, 0.5, 0.5]}
```

B's entire pooled advantage comes from the UK. In the US the two arms are identical. The
pooled 75%-vs-50% is not a story about the variant, it is a story about two UK users. On a
real experiment this is where you check whether the assignment is actually balanced across
segments, and if it is not, whether that is a bug in the bucketing.

## See also

:::{seealso}
- [Funnel analysis](funnel-analysis.md): the one-row-per-user collapse, in more detail.
- [Retention curves](retention-curves.md): another denominator that has to come from the
  cohort rather than from the behaviour log.
- [Sampling](../../user-guide/sampling.md): `sample` is a stable seeded content hash, so a
  holdout is reproducible and identical single-node or distributed.
- [Aggregations](../../user-guide/aggregations.md): `count_if`, `bool_or`, and the rest.
- [Joins](../../user-guide/joins.md): the left join that keeps `u4` in the denominator.
- [Join algorithms](../../deep-dives/join-algorithms.md): what the left join does with the
  rows that have no match.
- [Expressions API](../../api/expressions.md): `count_if`, `bool_or`, `fill_null`.
:::
