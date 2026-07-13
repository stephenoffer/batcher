# Sessionization

A click stream is a flat list of events. A *session* is a burst of them: everything a
user did before they went away for a while. There is no session column in the data. You have
to derive it, and the derivation is where people go wrong.

## The data

Eight page views, two users. `u1` browses for five minutes, disappears for forty, and
comes back. `u2` visits once, then again ninety minutes later. With a 30-minute
inactivity gap that is four sessions.

```python
import datetime as dt

import batcher as bt
from batcher import col, lag

t0 = dt.datetime(2024, 1, 1, 9, 0)


def at(minutes: int) -> dt.datetime:
    return t0 + dt.timedelta(minutes=minutes)


events = bt.from_pydict(
    {
        "user": ["u1", "u1", "u1", "u1", "u1", "u2", "u2", "u2"],
        "ts": [at(0), at(2), at(5), at(45), at(47), at(0), at(90), at(92)],
        "page": ["home", "search", "item", "home", "checkout", "home", "home", "item"],
    }
)
print(events.count())
# 8
```

## The trap

:::{warning}
"Session = user + calendar day" merges every visit a user made that day and splits anyone
still browsing at midnight. It undercounts sessions and overstates depth, and the error
does not average out.
:::

It is the shortcut everyone reaches for, because it is one `GROUP BY` and it needs no
window function:

```python
by_day = (
    events.group_by("user", day=col("ts").dt.truncate("day"))
    .agg(hits=bt.count())
    .sort("user")
)
print(by_day.to_pydict())
# {'user': ['u1', 'u2'], 'day': [datetime.datetime(2024, 1, 1, 0, 0),
#  datetime.datetime(2024, 1, 1, 0, 0)], 'hits': [5, 3]}
```

Two sessions, not four. Every one of `u1`'s two visits got merged, and the forty-minute
gap in the middle vanished. Sessions per user, pages per session, session length: all wrong.
In the other direction, a user still browsing at 23:58 gets *split* at midnight, so the
same bug both merges and splits depending on the clock.

Against the gap-based answer built below:

| | `user` + calendar day | 30-minute inactivity gap |
| --- | --- | --- |
| Sessions in total | 2 | 4 |
| `u1` | one session, 5 pages | two sessions, 3 pages and 2 pages |
| `u2` | one session, 3 pages | two sessions, 1 page and 2 pages |

The other classic wrong answer is a self-join to find each event's predecessor
(`e2.ts < e1.ts` and no event between them). It is correct and it is quadratic in the
number of events per user. Do not.

## Gap, flag, cumulative sum

Three steps, three window expressions, one shuffle on `user`.

First, the gap to the previous event. `lag` over the user's events in time order. Timestamps
do not subtract, so work in epoch seconds.

```python
gaps = events.with_columns(
    gap_s=col("ts").dt.epoch()
    - lag(col("ts").dt.epoch(), 1).over(partition_by=["user"], order_by=["ts"])
)
print(gaps.sort("user", "ts").to_pydict()["gap_s"])
# [None, 120, 180, 2400, 120, None, 5400, 120]
```

The nulls are the first event of each user, which has no predecessor.

Second, flag the session boundaries. An event starts a session if there is nothing before it,
or if the gap exceeds the threshold.

:::{important}
That first `is_null()` check is not a defensive nicety. Drop it and every user's first
event silently falls into session 0.
:::

```python
GAP_SECONDS = 30 * 60

flagged = gaps.with_columns(
    is_new=(col("gap_s").is_null() | (col("gap_s") > GAP_SECONDS)).cast("int64")
)
print(flagged.sort("user", "ts").to_pydict()["is_new"])
# [1, 0, 0, 1, 0, 1, 1, 0]
```

Third, number the sessions. A running sum of the flag, in time order, within the user. Each
boundary bumps the counter; everything between keeps the number. The SQL tab writes out all
three steps as CTEs, one window function per projection, and lands on the same plan.

::::{tab-set}
:::{tab-item} DataFrame
```python
sessions = flagged.with_columns(
    session=col("is_new").cum_sum(partition_by=["user"], order_by=["ts"])
)
print(sessions.sort("user", "ts").select("user", "page", "session").to_pydict())
# {'user': ['u1', 'u1', 'u1', 'u1', 'u1', 'u2', 'u2', 'u2'],
#  'page': ['home', 'search', 'item', 'home', 'checkout', 'home', 'home', 'item'],
#  'session': [1, 1, 1, 2, 2, 1, 2, 2]}
```
:::

:::{tab-item} SQL
```python
sql_sessions = bt.sql(
    """
    WITH e AS (
      SELECT user, ts, page, EXTRACT(EPOCH FROM ts) AS t FROM events
    ),
    g AS (
      SELECT user, ts, page, t,
             LAG(t) OVER (PARTITION BY user ORDER BY ts) AS prev_t
      FROM e
    ),
    f AS (
      SELECT user, ts, page,
             CASE WHEN prev_t IS NULL OR t - prev_t > 1800 THEN 1 ELSE 0 END AS is_new
      FROM g
    )
    SELECT user, page,
           SUM(is_new) OVER (PARTITION BY user ORDER BY ts) AS session
    FROM f
    ORDER BY user, ts
    """,
    events=events,
)
print(sql_sessions.to_pydict()["session"])
# [1, 1, 1, 2, 2, 1, 2, 2]
```
:::
::::

:::{note}
Each window function gets its own projection in the SQL, and the arithmetic happens in the
next CTE. That is not a Batcher quirk so much as good hygiene; it is also how the plan
looks internally either way.
:::

`(user, session)` is now a key you can group on like any other.

## Session-level metrics

An aggregate is not a scalar expression, so `col("t").max() - col("t").min()` will not
type-check inside `agg`. Emit the two aggregates, then subtract them in the projection
that follows, which is what SQL does with `MAX(t) - MIN(t)` anyway.

```python
summary = (
    sessions.with_columns(t=col("ts").dt.epoch())
    .group_by("user", "session")
    .agg(
        pages=bt.count(),
        distinct_pages=col("page").n_unique(),
        started=col("t").min(),
        ended=col("t").max(),
    )
    .with_columns(duration_s=col("ended") - col("started"))
    .sort("user", "session")
)
print(summary.select("user", "session", "pages", "duration_s").to_pydict())
# {'user': ['u1', 'u1', 'u2', 'u2'], 'session': [1, 2, 1, 2], 'pages': [3, 2, 1, 2],
#  'duration_s': [300, 120, 0, 120]}
```

Four sessions, and `u2`'s first one has a duration of zero: a single-page visit. Those are
real and you should not filter them out just because they look like noise. A bounce rate is
exactly the fraction of sessions with `pages == 1`.

## The shortcut

:::{tip}
If all you want is the per-session aggregate and not the session id on every row,
`session_window` does the whole thing in one call. It composes the same window and
group-by operators, so the result is identical.
:::

```python
windowed = events.session_window("ts", "30m", partition_by=["user"], pages=bt.count())
print(windowed.sort("user", "session_start").to_pydict()["pages"])
# [3, 2, 1, 2]
```

Reach for the three-step version when you need the session id attached to each event: to join
sessions to conversions, say, or to look at within-session page sequences.

:::{dropdown} Picking the gap: why 30 minutes is arbitrary
Thirty minutes is the web-analytics convention and it is arbitrary. Look at the
distribution of `gap_s` before you commit: the inter-event gaps within a visit and the
gaps between visits are usually two separate humps, and the threshold belongs in the
valley between them. `col("gap_s").quantile(0.95)` on your own data is a better argument
than "everyone uses 30".
:::

## See also

:::{seealso}
- [Funnel analysis](funnel-analysis.md): scope a funnel to one session by grouping on
  `(user, session)` instead of `user`.
- [Time-series rollups](time-series-rollups.md): the other side of the clock, where the
  bucket is fixed and the gaps are the problem.
- [Window functions](../../user-guide/window-functions.md): `lag`, frames, and `cum_sum`.
- [Streaming](../../user-guide/streaming.md): the same sessions computed incrementally,
  with a watermark bounding how long a session stays open.
- [Window internals](../../deep-dives/window-internals.md): the partition-sort-scan the
  three window expressions above share.
- [Expressions API](../../api/expressions.md): `lag`, `cum_sum`, `dt.epoch`.
:::
