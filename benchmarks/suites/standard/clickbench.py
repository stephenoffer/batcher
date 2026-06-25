"""ClickBench — the 43-query single-table analytics benchmark (ClickHouse ``hits``).

The standard ClickBench statements over the public ``hits`` dataset, written once as
SQL and fanned across every SQL-capable engine. Columns follow the ClickBench schema
(``sources.py`` reads the dataset as-is and normalizes date columns to
``timestamp[us]``). Engines whose SQL planner cannot express a query (e.g. the regex
back-reference in Q28) surface as ``n/a``/``PARTIAL``, never a wrong answer.
"""

from __future__ import annotations

from registry import suite

clickbench = suite("clickbench", dataset="clickbench")

# Q29 sums a sliding window of 90 derived columns; build it rather than spell it out.
_Q29 = "SELECT " + ", ".join(f"SUM(ResolutionWidth + {i})" for i in range(90)) + " FROM hits"

QUERIES: dict[str, str] = {
    "cb-q00": "SELECT COUNT(*) FROM hits",
    "cb-q01": "SELECT COUNT(*) FROM hits WHERE AdvEngineID <> 0",
    "cb-q02": "SELECT SUM(AdvEngineID), COUNT(*), AVG(ResolutionWidth) FROM hits",
    "cb-q03": "SELECT AVG(UserID) FROM hits",
    "cb-q04": "SELECT COUNT(DISTINCT UserID) FROM hits",
    "cb-q05": "SELECT COUNT(DISTINCT SearchPhrase) FROM hits",
    "cb-q06": "SELECT MIN(EventDate), MAX(EventDate) FROM hits",
    "cb-q07": (
        "SELECT AdvEngineID, COUNT(*) FROM hits WHERE AdvEngineID <> 0 "
        "GROUP BY AdvEngineID ORDER BY COUNT(*) DESC"
    ),
    "cb-q08": (
        "SELECT RegionID, COUNT(DISTINCT UserID) AS u FROM hits "
        "GROUP BY RegionID ORDER BY u DESC LIMIT 10"
    ),
    "cb-q09": (
        "SELECT RegionID, SUM(AdvEngineID), COUNT(*) AS c, AVG(ResolutionWidth), "
        "COUNT(DISTINCT UserID) FROM hits GROUP BY RegionID ORDER BY c DESC LIMIT 10"
    ),
    "cb-q10": (
        "SELECT MobilePhoneModel, COUNT(DISTINCT UserID) AS u FROM hits "
        "WHERE MobilePhoneModel <> '' GROUP BY MobilePhoneModel ORDER BY u DESC LIMIT 10"
    ),
    "cb-q11": (
        "SELECT MobilePhone, MobilePhoneModel, COUNT(DISTINCT UserID) AS u FROM hits "
        "WHERE MobilePhoneModel <> '' GROUP BY MobilePhone, MobilePhoneModel "
        "ORDER BY u DESC LIMIT 10"
    ),
    "cb-q12": (
        "SELECT SearchPhrase, COUNT(*) AS c FROM hits WHERE SearchPhrase <> '' "
        "GROUP BY SearchPhrase ORDER BY c DESC LIMIT 10"
    ),
    "cb-q13": (
        "SELECT SearchPhrase, COUNT(DISTINCT UserID) AS u FROM hits WHERE SearchPhrase <> '' "
        "GROUP BY SearchPhrase ORDER BY u DESC LIMIT 10"
    ),
    "cb-q14": (
        "SELECT SearchEngineID, SearchPhrase, COUNT(*) AS c FROM hits WHERE SearchPhrase <> '' "
        "GROUP BY SearchEngineID, SearchPhrase ORDER BY c DESC LIMIT 10"
    ),
    "cb-q15": ("SELECT UserID, COUNT(*) FROM hits GROUP BY UserID ORDER BY COUNT(*) DESC LIMIT 10"),
    "cb-q16": (
        "SELECT UserID, SearchPhrase, COUNT(*) FROM hits GROUP BY UserID, SearchPhrase "
        "ORDER BY COUNT(*) DESC LIMIT 10"
    ),
    "cb-q17": (
        "SELECT UserID, SearchPhrase, COUNT(*) FROM hits GROUP BY UserID, SearchPhrase LIMIT 10"
    ),
    "cb-q18": (
        "SELECT UserID, extract(minute FROM EventTime) AS m, SearchPhrase, COUNT(*) FROM hits "
        "GROUP BY UserID, m, SearchPhrase ORDER BY COUNT(*) DESC LIMIT 10"
    ),
    "cb-q19": "SELECT UserID FROM hits WHERE UserID = 435090932899640449",
    "cb-q20": "SELECT COUNT(*) FROM hits WHERE URL LIKE '%google%'",
    "cb-q21": (
        "SELECT SearchPhrase, MIN(URL), COUNT(*) AS c FROM hits "
        "WHERE URL LIKE '%google%' AND SearchPhrase <> '' "
        "GROUP BY SearchPhrase ORDER BY c DESC LIMIT 10"
    ),
    "cb-q22": (
        "SELECT SearchPhrase, MIN(URL), MIN(Title), COUNT(*) AS c, COUNT(DISTINCT UserID) "
        "FROM hits WHERE Title LIKE '%Google%' AND URL NOT LIKE '%.google.%' "
        "AND SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY c DESC LIMIT 10"
    ),
    "cb-q23": "SELECT * FROM hits WHERE URL LIKE '%google%' ORDER BY EventTime LIMIT 10",
    "cb-q24": (
        "SELECT SearchPhrase FROM hits WHERE SearchPhrase <> '' ORDER BY EventTime LIMIT 10"
    ),
    "cb-q25": (
        "SELECT SearchPhrase FROM hits WHERE SearchPhrase <> '' ORDER BY SearchPhrase LIMIT 10"
    ),
    "cb-q26": (
        "SELECT SearchPhrase FROM hits WHERE SearchPhrase <> '' "
        "ORDER BY EventTime, SearchPhrase LIMIT 10"
    ),
    "cb-q27": (
        "SELECT CounterID, AVG(length(URL)) AS l, COUNT(*) AS c FROM hits WHERE URL <> '' "
        "GROUP BY CounterID HAVING COUNT(*) > 100000 ORDER BY l DESC LIMIT 25"
    ),
    "cb-q28": (
        "SELECT REGEXP_REPLACE(Referer, '^https?://(?:www\\.)?([^/]+)/.*$', '\\1') AS k, "
        "AVG(length(Referer)) AS l, COUNT(*) AS c, MIN(Referer) FROM hits WHERE Referer <> '' "
        "GROUP BY k HAVING COUNT(*) > 100000 ORDER BY l DESC LIMIT 25"
    ),
    "cb-q29": _Q29,
    "cb-q30": (
        "SELECT SearchEngineID, ClientIP, COUNT(*) AS c, SUM(IsRefresh), AVG(ResolutionWidth) "
        "FROM hits WHERE SearchPhrase <> '' GROUP BY SearchEngineID, ClientIP "
        "ORDER BY c DESC LIMIT 10"
    ),
    "cb-q31": (
        "SELECT WatchID, ClientIP, COUNT(*) AS c, SUM(IsRefresh), AVG(ResolutionWidth) "
        "FROM hits WHERE SearchPhrase <> '' GROUP BY WatchID, ClientIP ORDER BY c DESC LIMIT 10"
    ),
    "cb-q32": (
        "SELECT WatchID, ClientIP, COUNT(*) AS c, SUM(IsRefresh), AVG(ResolutionWidth) "
        "FROM hits GROUP BY WatchID, ClientIP ORDER BY c DESC LIMIT 10"
    ),
    "cb-q33": "SELECT URL, COUNT(*) AS c FROM hits GROUP BY URL ORDER BY c DESC LIMIT 10",
    "cb-q34": "SELECT 1, URL, COUNT(*) AS c FROM hits GROUP BY 1, URL ORDER BY c DESC LIMIT 10",
    "cb-q35": (
        "SELECT ClientIP, ClientIP - 1, ClientIP - 2, ClientIP - 3, COUNT(*) AS c FROM hits "
        "GROUP BY ClientIP, ClientIP - 1, ClientIP - 2, ClientIP - 3 ORDER BY c DESC LIMIT 10"
    ),
    "cb-q36": (
        "SELECT URL, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 "
        "AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND DontCountHits = 0 "
        "AND IsRefresh = 0 AND URL <> '' GROUP BY URL ORDER BY PageViews DESC LIMIT 10"
    ),
    "cb-q37": (
        "SELECT Title, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 "
        "AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND DontCountHits = 0 "
        "AND IsRefresh = 0 AND Title <> '' GROUP BY Title ORDER BY PageViews DESC LIMIT 10"
    ),
    "cb-q38": (
        "SELECT URL, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 "
        "AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND IsRefresh = 0 "
        "AND IsLink <> 0 AND IsDownload = 0 GROUP BY URL "
        "ORDER BY PageViews DESC LIMIT 10 OFFSET 1000"
    ),
    "cb-q39": (
        "SELECT TraficSourceID, SearchEngineID, AdvEngineID, "
        "CASE WHEN (SearchEngineID = 0 AND AdvEngineID = 0) THEN Referer ELSE '' END AS Src, "
        "URL AS Dst, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 "
        "AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND IsRefresh = 0 "
        "GROUP BY TraficSourceID, SearchEngineID, AdvEngineID, Src, Dst "
        "ORDER BY PageViews DESC LIMIT 10 OFFSET 1000"
    ),
    "cb-q40": (
        "SELECT URLHash, EventDate, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 "
        "AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND IsRefresh = 0 "
        "AND TraficSourceID IN (-1, 6) AND RefererHash = 3594120000172545465 "
        "GROUP BY URLHash, EventDate ORDER BY PageViews DESC LIMIT 10 OFFSET 100"
    ),
    "cb-q41": (
        "SELECT WindowClientWidth, WindowClientHeight, COUNT(*) AS PageViews FROM hits "
        "WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' "
        "AND IsRefresh = 0 AND DontCountHits = 0 AND URLHash = 2868770270353813622 "
        "GROUP BY WindowClientWidth, WindowClientHeight "
        "ORDER BY PageViews DESC LIMIT 10 OFFSET 10000"
    ),
    "cb-q42": (
        "SELECT DATE_TRUNC('minute', EventTime) AS M, COUNT(*) AS PageViews FROM hits "
        "WHERE CounterID = 62 AND EventDate >= '2013-07-14' AND EventDate <= '2013-07-15' "
        "AND IsRefresh = 0 AND DontCountHits = 0 GROUP BY DATE_TRUNC('minute', EventTime) "
        "ORDER BY M LIMIT 10 OFFSET 1000"
    ),
}

for _name, _query in QUERIES.items():
    clickbench.sql(_name, _query)
