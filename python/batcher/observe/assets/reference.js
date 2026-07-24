/* The dashboard's knowledge base — everything it can explain, as data.
 *
 * A newcomer's problem here is not that the numbers are hidden; it is that "hash_join spilled
 * 2.4 GiB at 68% of operator time" assumes a vocabulary they have not been given. So every
 * term the engine uses, every operator it can run, and every metric it reports has an entry
 * here, and the UI is wired so that any of them can be asked about in place.
 *
 * Kept separate from `learn.js` (which renders it) because this is content, not mechanism:
 * it is edited by whoever changes the engine's vocabulary, not by whoever changes a popover.
 *
 * Writing rules for entries, so the voice stays consistent:
 *   - `what` answers "what is this?" in one sentence, with no other jargon in it.
 *   - `why` answers "why do I care?" — tie it to a decision the reader can make.
 *   - `fix` is imperative and specific. "Add a filter earlier" beats "consider optimizing".
 *   - Never say "simply", "just", or "obviously". If it were obvious, no entry was needed.
 */

'use strict';

const REFERENCE = (() => {
  /* ---------- vocabulary ----------
   * Terms a reader meets on the page and has no reason to already know. `see` cross-links
   * to related entries, which is what turns a set of definitions into something browsable. */

  const TERMS = {
    pipeline: {
      what: 'One query shape. Re-running the same query, even over completely different data, adds a run to the same pipeline.',
      why: 'It is how the dashboard groups runs so you can see a trend instead of a list. Two queries that differ only in a literal value are the same pipeline.',
      see: ['run', 'plan signature'],
    },
    run: {
      what: 'One execution of one pipeline, start to finish.',
      why: 'Runs are what you compare. A single run tells you what happened; a run against its baseline tells you whether that was normal.',
      see: ['pipeline', 'baseline'],
    },
    'plan signature': {
      what: 'A fingerprint of the query’s shape, ignoring the specific values it filtered on.',
      why: 'It is what decides whether two runs belong to the same pipeline, and therefore whether their steps can be compared to each other at all.',
      see: ['pipeline'],
    },
    plan: {
      what: 'The sequence of steps the engine chose in order to answer your query.',
      why: 'You write what you want; the engine decides how to get it. The plan is that decision, and it is where a slow query is usually explained.',
      see: ['operator', 'optimizer'],
    },
    operator: {
      what: 'One step in a plan — a scan, a filter, a join, a sort.',
      why: 'Time is spent in operators, so "which operator" is the first question to ask about a slow query.',
      see: ['plan', 'operator time'],
    },
    optimizer: {
      what: 'The component that rewrites your query into a faster plan with the same answer.',
      why: 'It decides join order, which filters run early, and which columns are read at all. Its choices show up in the Decisions panel.',
      see: ['plan', 'cardinality estimate'],
    },
    explain: {
      what: 'The plan written out as a text tree, with each step nested inside the one it feeds.',
      why: 'Every SQL engine ships one and it is the form that pastes into an issue and searches with the browser. The graph shows a plan\u2019s shape; the tree shows what is nested inside what.',
      fix: 'Open a run and go to Query. Tick \u201cShow the plan as written\u201d to see it before the optimizer touched it.',
      see: ['plan', 'logical plan', 'optimizer'],
    },
    'plan diff': {
      what: 'The plan you wrote compared against the plan that actually ran.',
      why: 'It is the only view that shows what the optimizer contributed. One rewrite \u2014 a pushdown, say \u2014 moves one step and drags every step it passed into a new position, so the move that caused it is reported as the finding and the rest as its consequences.',
      fix: 'Open a run, go to Query, then \u201cWhat the optimizer changed\u201d.',
      see: ['optimizer', 'predicate pushdown', 'logical plan'],
    },
    'flame graph': {
      what: 'Every step drawn as a block whose width is the time it took, stacked by how deep it sits in the plan.',
      why: 'A ranked list loses the structure and a graph loses the cost; this shows both at once. A parent block is deliberately not the sum of its children, because steps run at the same time and stacking them would invent a total larger than the query took.',
      see: ['operator time', 'critical path'],
    },
    'data starvation': {
      what: 'A worker sitting idle because nothing has arrived for it to process.',
      why: 'On the Live page it shows as a device swinging between idle and saturated rather than holding a steady load. The two average to the same number and need opposite fixes: a starved device is waiting on reading, decoding, or shuffling, so a bigger batch or a faster device changes nothing.',
      fix: 'Widen the read parallelism or move decode work off the critical path before touching the model or the batch size.',
      see: ['blocked time', 'backpressure'],
    },
    'blocked time': {
      what: 'Time a worker spent waiting for its next input rather than computing.',
      why: 'The single clearest signal that the bottleneck is upstream of the expensive part. Rising blocked time means the pipeline cannot keep the workers fed.',
      see: ['data starvation', 'backpressure', 'flow control'],
    },
    'critical path': {
      what: 'The chain of steps that feed one another from the start of the query to its end.',
      why: 'Its total is the floor on how fast the query can possibly be. Speeding up a step that is not on it changes the total by nothing at all.',
      see: ['operator time', 'wall-clock'],
    },
    'wall-clock': {
      what: 'Elapsed real time — what a stopwatch beside the machine would have read.',
      why: 'It is the number a user actually waits. Compare it against operator time to see how much parallelism you are getting.',
      see: ['operator time', 'critical path'],
    },
    'operator time': {
      what: 'The sum of time spent across every step, added together.',
      why: 'Steps run at the same time on different cores, so this routinely exceeds wall-clock. A ratio of 8:1 against wall-clock means roughly eight cores were busy.',
      see: ['wall-clock', 'parallelism'],
    },
    parallelism: {
      what: 'Operator time divided by wall-clock — roughly how many cores were kept busy.',
      why: 'Close to 1 on a multi-core machine would mean the query ran essentially single-file. Read it from the run\u2019s own operator-time and duration totals rather than from a per-step CPU figure.',
      fix: 'Per-step CPU is shown as \u201cnot measured\u201d on tiers that do not sample the OS clock per batch \u2014 an em dash there means the engine did not measure it, not that the step used no CPU.',
      see: ['operator time', 'wall-clock', 'morsel'],
    },
    morsel: {
      what: 'The unit of work the engine schedules — a batch of rows, 16,384 by default.',
      why: 'Work is handed to cores a morsel at a time, which is what lets a query spread across cores without you partitioning anything.',
      see: ['parallelism', 'batch'],
    },
    batch: {
      what: 'A block of rows moving through the engine together, in columnar Arrow form.',
      why: 'The engine never processes one row at a time. Costs are per batch, which is why a query over 10 rows and one over 10,000 can take nearly the same time.',
      see: ['morsel', 'arrow'],
    },
    arrow: {
      what: 'The columnar memory format every part of the engine speaks.',
      why: 'It is why data crosses between components without being copied or converted, and why an operator can work on a whole column at once.',
      see: ['batch'],
    },
    spill: {
      what: 'Writing intermediate data to disk because it did not fit in the memory budget.',
      why: 'It keeps a large query alive instead of failing it, but disk is far slower than memory, so a spilling step is usually the slow one.',
      fix: 'Give the query more memory, cut rows earlier with a filter, or reduce the columns carried into the step.',
      see: ['memory budget', 'hash join'],
    },
    'memory budget': {
      what: 'The ceiling the engine holds itself to for a query.',
      why: 'It is what decides whether a step spills. Raising it can turn a spilling query into a fast one — or push the machine into swapping.',
      see: ['spill'],
    },
    selectivity: {
      what: 'Rows out divided by rows in, for one step.',
      why: '1% means the step threw away 99% of what it read. A very selective step that runs late is work you paid for and discarded.',
      see: ['predicate pushdown', 'cardinality estimate'],
    },
    'predicate pushdown': {
      what: 'Moving a filter as early in the plan as it can legally go.',
      why: 'Rows removed early are rows every later step does not have to touch. It is the single highest-leverage rewrite the optimizer makes.',
      see: ['selectivity', 'optimizer'],
    },
    'cardinality estimate': {
      what: 'The optimizer’s prediction of how many rows a step will produce.',
      why: 'Every plan choice rests on it. A bad estimate is the usual root cause of a badly chosen join order or build side.',
      fix: 'Run the query once more — the engine records real cardinalities and feeds them back, so the second plan is built on measurements.',
      see: ['estimate error', 'optimizer'],
    },
    'estimate error': {
      what: 'Actual rows divided by estimated rows, for one step.',
      why: 'Anything beyond about 10x means the plan was chosen on a bad guess and may not be the plan you would want.',
      see: ['cardinality estimate'],
    },
    'hash join': {
      what: 'Joining by loading one side into a hash table, then streaming the other side past it.',
      why: 'It is the engine’s workhorse join, and it is where memory pressure usually appears: the loaded side has to fit.',
      fix: 'Ensure the smaller side is the one being built. If both sides are large, filter before the join rather than after.',
      see: ['build side', 'spill'],
    },
    'build side': {
      what: 'The input a hash join loads into memory before streaming the other past it.',
      why: 'The build side must fit in the budget. Building the larger side is a common cause of an avoidable spill.',
      see: ['hash join', 'spill'],
    },
    baseline: {
      what: 'The median of this pipeline’s other runs.',
      why: 'It turns a duration into a judgement. 400ms means nothing on its own; 400ms against a 90ms baseline means something happened.',
      see: ['p95', 'regression'],
    },
    regression: {
      what: 'A run meaningfully slower than the same pipeline’s baseline.',
      why: 'It is the difference between a query that is slow and a query that got slow — only the second one has a cause you can go and find.',
      see: ['baseline'],
    },
    p50: {
      what: 'The median: half of runs finished faster than this, half slower.',
      why: 'It describes the typical run and ignores the occasional disaster, which is what makes it a fair baseline.',
      see: ['p95', 'baseline'],
    },
    p95: {
      what: 'The duration 95% of runs came in under.',
      why: 'It is what your slowest users actually experience. Needs a couple of dozen runs before it means much.',
      see: ['p50'],
    },
    throughput: {
      what: 'Rows processed per second.',
      why: 'It makes runs over different data comparable — a slower run over ten times the rows is not a regression.',
      see: ['wall-clock'],
    },
    'adaptive re-optimization': {
      what: 'Re-planning partway through a query, using row counts it has now actually measured.',
      why: 'The optimizer’s opening guess can be wrong; once a step has run, its true size is known and the rest of the plan can be rebuilt around it.',
      see: ['cardinality estimate', 'pipeline breaker'],
    },
    'pipeline breaker': {
      what: 'A step that must see all of its input before it can emit anything — a sort, or a join’s build side.',
      why: 'These are where memory accumulates, where spilling happens, and where the engine can safely stop and re-plan.',
      see: ['adaptive re-optimization', 'spill'],
    },
    shuffle: {
      what: 'Redistributing rows across workers so that matching keys end up together.',
      why: 'It is the expensive part of any distributed join or grouping, because it moves data over the network.',
      see: ['partition'],
    },
    partition: {
      what: 'One slice of a dataset that a single worker handles.',
      why: 'Partition count sets the parallelism ceiling, and uneven partitions leave most workers waiting on one.',
      see: ['shuffle', 'skew'],
    },
    skew: {
      what: 'One partition or key holding far more rows than the others.',
      why: 'The query can only finish when its slowest worker does, so one heavy partition sets the total time no matter how many cores you add.',
      fix: 'Split the hot key, or add a salt to the grouping key to spread it.',
      see: ['partition', 'shuffle'],
    },
    projection: {
      what: 'Choosing which columns to read or carry forward.',
      why: 'Columnar storage means unread columns cost nothing. Narrowing a projection is often the cheapest available speedup.',
      see: ['columnar'],
    },
    columnar: {
      what: 'Storing each column together, rather than storing each row together.',
      why: 'A query touching 3 of 200 columns reads 3 columns’ worth of bytes, and the engine can work on a whole column in one pass.',
      see: ['projection', 'arrow'],
    },
    'jit compilation': {
      what: 'Compiling an expression to machine code at runtime instead of interpreting it.',
      why: 'It removes per-row interpreter overhead. The engine compiles once per operator and reuses it for every batch.',
      see: ['operator'],
    },
    'flow control': {
      what: 'Slowing a producer so a slow consumer is never overwhelmed.',
      why: 'It is what keeps a distributed shuffle from filling memory faster than the receiver can drain it.',
      see: ['backpressure', 'credit'],
    },
    morsel_size: {
      what: 'How many rows travel through the engine together — 16,384 by default.',
      why: 'Bigger morsels amortise per-batch overhead; smaller ones spread work more evenly across cores. It is a scheduling knob, not a correctness one.',
      see: ['morsel', 'batch'],
    },
    'out-of-core': {
      what: 'Processing more data than fits in memory by spilling the overflow to disk.',
      why: 'It is what lets a query over a terabyte finish on a laptop — slower, but alive.',
      see: ['spill', 'memory budget'],
    },
    'adaptive execution': {
      what: 'Changing the plan mid-query using the row counts it has now actually measured.',
      why: 'The opening plan is a guess; once a step has run, its true size is known and the rest can be rebuilt around it. This is the engine\u2019s core advantage.',
      see: ['adaptive re-optimization', 'cardinality estimate'],
    },
    'stage boundary': {
      what: 'A point where the engine must finish one phase before the next can start.',
      why: 'It is where memory is highest, where spilling happens, and where the plan can safely be re-examined.',
      see: ['pipeline breaker', 'shuffle'],
    },
    'hash table': {
      what: 'A lookup structure that finds matching rows in near-constant time.',
      why: 'It is what makes a hash join fast, and what has to fit in memory — its size is set by the build side.',
      see: ['hash join', 'build side'],
    },
    'null handling': {
      what: 'How the engine treats missing values in comparisons and grouping.',
      why: 'A null is not equal to anything, including another null, which is why a join or filter on a nullable column can drop rows you expected to keep.',
      see: ['selectivity'],
    },
    'type coercion': {
      what: 'Widening narrow numeric types to a common type at the boundary.',
      why: 'Int32 becomes Int64, Float32 becomes Float64, once, at the edge — so the rest of the engine never branches on width.',
      see: ['arrow'],
    },
    'columnar scan': {
      what: 'Reading only the columns a query touches, skipping the rest.',
      why: 'A query over 3 of 200 columns reads 3 columns\u2019 worth of bytes — the whole reason a columnar store is fast on wide tables.',
      see: ['columnar', 'projection'],
    },
    'runtime filter': {
      what: 'A filter built during execution from one side of a join and pushed to the other.',
      why: 'The join learns which keys actually occur and hands that back to the scan, so the scan skips rows that could never match.',
      see: ['predicate pushdown', 'hash join'],
    },
    'spill file': {
      what: 'The on-disk overflow a stateful operator writes when it exceeds its memory budget.',
      why: 'Reading it back is far slower than memory, so its presence is usually why a step is the slow one.',
      see: ['spill', 'out-of-core'],
    },
    'work stealing': {
      what: 'An idle core taking a pending morsel from a busy one.',
      why: 'It is how the engine keeps every core fed without a central scheduler deciding who does what.',
      see: ['morsel', 'parallelism'],
    },
    'catalog': {
      what: 'The engine\u2019s record of what tables and columns exist and their types.',
      why: 'Planning consults it to resolve names and choose types before any data is read.',
      see: ['logical plan'],
    },
    'watermark': {
      what: 'A moving marker in a stream past which no earlier data is expected.',
      why: 'It is how a streaming aggregation knows a window is complete and can emit its result.',
      see: ['pipeline breaker'],
    },
    'exchange': {
      what: 'A step that moves rows between workers so matching keys meet.',
      why: 'It is the network-heavy part of a distributed join or grouping — the same idea as a shuffle, seen as a plan step.',
      see: ['shuffle', 'partition'],
    },
    'vectorized aggregation': {
      what: 'Computing a group summary a column at a time rather than a row at a time.',
      why: 'It is why a sum over a million rows costs a handful of passes, not a million additions.',
      see: ['vectorized', 'aggregate'],
    },
    'streaming aggregation': {
      what: 'Grouping rows as they arrive, when the input is already ordered by the key.',
      why: 'It never has to hold every group at once, so it does not spill — the cheap case a sort before it can unlock.',
      see: ['aggregate', 'sort key'],
    },
    'top-n': {
      what: 'Keeping only the N largest or smallest rows without fully sorting.',
      why: 'A sort orders everything; a top-N keeps a running set of N, so a limit after a sort is far cheaper than the sort alone.',
      see: ['sort', 'limit'],
    },
    'semi join': {
      what: 'Keeping rows from one side that have a match on the other, without duplicating them.',
      why: 'It is what an EXISTS or IN becomes — a filter driven by a second table, not a full join.',
      see: ['hash join', 'runtime filter'],
    },
    'anti join': {
      what: 'Keeping rows from one side that have no match on the other.',
      why: 'The mirror of a semi join — what NOT EXISTS becomes.',
      see: ['semi join'],
    },
    'lazy evaluation': {
      what: 'Building the whole plan before running any of it, so the optimizer sees everything.',
      why: 'It is why chaining ten operations costs one optimized pass, not ten — nothing runs until a terminal step asks for rows.',
      see: ['optimizer', 'plan'],
    },
    'terminal operation': {
      what: 'The call that actually makes a lazy plan run — collect, write, iterate.',
      why: 'Everything before it only describes work; this is where the engine is finally handed the plan to execute.',
      see: ['lazy evaluation'],
    },
    'predicate': {
      what: 'A condition that is true or false for a row — the test inside a filter or join.',
      why: 'The optimizer moves predicates as early as it can, because a row removed early is a row nothing later has to touch.',
      see: ['predicate pushdown', 'selectivity'],
    },
    'schema': {
      what: 'The columns a dataset has and their types.',
      why: 'It is resolved before any data is read, which is how a mistyped column name fails at plan time rather than mid-run.',
      see: ['catalog', 'type coercion'],
    },
    'query id': {
      what: 'The unique identifier for a single run.',
      why: 'It is what you paste to a colleague, or into a link, to point at exactly the execution you are looking at.',
      see: ['run'],
    },
    'pipeline breaker': {
      what: 'A step that must see all of its input before it can emit anything — a sort, or a join’s build side.',
      why: 'These are where memory accumulates, where spilling happens, and where the engine can safely stop and re-plan.',
      see: ['adaptive re-optimization', 'spill'],
    },
    vectorized: {
      what: 'Processing a whole column at once rather than a value at a time.',
      why: 'It is why the engine is fast on wide scans: one loop over 16,384 values costs far less than 16,384 loops.',
      see: ['batch', 'columnar'],
    },
    'late materialization': {
      what: 'Reading a column only once the rows that need it are known.',
      why: 'A filter that keeps 1% of rows means the other 99% of the wide columns are never read at all.',
      see: ['projection', 'selectivity'],
    },
    'zone map': {
      what: 'A per-block record of the smallest and largest value in a column.',
      why: 'It lets a scan skip whole blocks without reading them, which is why filtering on a sorted column is so much cheaper.',
      see: ['scan', 'predicate pushdown'],
    },
    'build side': {
      what: 'The input a hash join loads into memory before streaming the other past it.',
      why: 'The build side must fit in the budget. Building the larger side is a common cause of an avoidable spill.',
      see: ['hash join', 'spill'],
    },
    'probe side': {
      what: 'The input a hash join streams past an already-built hash table.',
      why: 'It is never held in memory, so it can be arbitrarily large — which is why you want the *smaller* side built.',
      see: ['hash join', 'build side'],
    },
    'broadcast join': {
      what: 'Sending a small input to every worker so no shuffle is needed.',
      why: 'It turns a network-bound join into a local one, but only when the broadcast side is genuinely small.',
      see: ['shuffle', 'hash join'],
    },
    'operator id': {
      what: 'The position of a step in the plan, counted from the top.',
      why: 'It is stable across renderings, so the same step carries the same number in the graph, the table and the logs.',
      see: ['operator', 'plan'],
    },
    'critical path': {
      what: 'The chain of steps that feed one another from the start of the query to its end.',
      why: 'Its total is the floor on how fast the query can possibly be. Speeding up a step that is not on it changes the total by nothing at all.',
      see: ['operator time', 'wall-clock'],
    },
    backpressure: {
      what: 'A slow consumer telling a fast producer to wait.',
      why: 'It is what keeps memory bounded in a streaming pipeline. A step blocked on backpressure is not slow — it is waiting for the step after it.',
      see: ['credit', 'shuffle'],
    },
    credit: {
      what: 'Permission to send one more batch. A producer with no credits waits.',
      why: 'It is the mechanism behind backpressure, and it is why a shuffle cannot flood a slow receiver.',
      see: ['backpressure', 'shuffle'],
    },
    'buffer pool': {
      what: 'The memory the engine manages itself, rather than leaving to the allocator.',
      why: 'Owning the pool is what lets the engine know it is near the limit and spill deliberately, instead of being killed by the OS.',
      see: ['memory budget', 'spill'],
    },
    'peak memory': {
      what: 'The high-water mark of memory held during a run.',
      why: 'Averages hide the moment that matters. What decides whether a query survives is its peak, not its mean.',
      see: ['memory budget', 'spill'],
    },
    'cardinality': {
      what: 'How many rows there are — or how many distinct values a column has.',
      why: 'Nearly every plan decision is a bet on a cardinality, which is why a bad estimate shows up as a bad plan.',
      see: ['cardinality estimate', 'selectivity'],
    },
    'group key': {
      what: 'The column or columns a grouping collapses rows on.',
      why: 'Its distinct count sets the size of the hash table, and therefore whether the grouping fits in memory.',
      see: ['cardinality', 'spill'],
    },
    'sort key': {
      what: 'The column or columns an ordering is defined by.',
      why: 'If it matches how the data is already stored, the sort can be skipped entirely.',
      see: ['sort', 'zone map'],
    },
    'plan cache': {
      what: 'Reuse of a previously optimized plan for a query of the same shape.',
      why: 'It removes planning time from a repeated query, which matters most for small queries where planning was a large share of the total.',
      see: ['plan signature', 'optimizer'],
    },
    'physical plan': {
      what: 'The plan after the optimizer has chosen concrete algorithms — which join, which aggregation strategy.',
      why: 'It is what actually ran. The logical plan says what was asked for; this says how it was answered.',
      see: ['plan', 'optimizer'],
    },
    'logical plan': {
      what: 'The plan as meaning: what to read, filter, join and return, with no algorithm chosen yet.',
      why: 'Two very different physical plans can share one logical plan, which is exactly what the optimizer is choosing between.',
      see: ['physical plan', 'plan'],
    },
  };

  /* ---------- operators ----------
   * Each entry answers the three questions someone actually has when a step is highlighted:
   * what does it do, why might it be the slow one, and what do I do about it. */

  const OPERATORS = {
    scan: {
      label: 'Read source',
      what: 'Reads rows from a file, table, or in-memory dataset.',
      slow: 'Reading more columns than the query needs, or reading files that a filter could have skipped entirely.',
      fix: 'Select only the columns you use, and filter on a partitioned or sorted column so whole files can be skipped.',
      terms: ['projection', 'columnar'],
    },
    filter: {
      label: 'Filter rows',
      what: 'Keeps only the rows matching a condition.',
      slow: 'Rarely slow itself. It becomes a problem when it sits late in the plan, after expensive work was already done on rows it then discards.',
      fix: 'Check the plan: if a very selective filter runs after a join, the optimizer could not push it down — often because of a function applied to the column.',
      terms: ['selectivity', 'predicate pushdown'],
    },
    project: {
      label: 'Compute columns',
      what: 'Derives new columns and drops ones no longer needed.',
      slow: 'Expensive per-row expressions, or a user function the engine cannot compile.',
      fix: 'Prefer built-in expressions over user functions — built-ins compile to machine code and run per batch.',
      terms: ['jit compilation', 'projection'],
    },
    aggregate: {
      label: 'Group & aggregate',
      what: 'Groups rows by a key and computes a summary per group — sum, count, average.',
      slow: 'A grouping key with very many distinct values, which makes the hash table large enough to spill.',
      fix: 'Group by fewer or coarser keys, filter before grouping, or raise the memory budget.',
      terms: ['spill', 'skew', 'pipeline breaker'],
    },
    hash_join: {
      label: 'Join',
      what: 'Matches rows from two inputs on a key, by building a hash table from one side.',
      slow: 'Building the larger side, a bad row-count estimate leading to the wrong side being chosen, or a key with heavy skew.',
      fix: 'Filter both inputs before the join. If the estimate was badly off, re-run — the engine feeds measured counts back into the next plan.',
      terms: ['build side', 'estimate error', 'spill'],
    },
    sort: {
      label: 'Sort',
      what: 'Orders rows by one or more columns.',
      slow: 'It must see every row before emitting the first, so it holds the whole input and is a frequent spiller.',
      fix: 'Sort after filtering rather than before, and if you only need the top rows, use a limit so the engine can keep just those.',
      terms: ['pipeline breaker', 'spill'],
    },
    limit: {
      label: 'Limit',
      what: 'Stops after a set number of rows.',
      slow: 'Almost never. If it is, the steps beneath it could not stop early.',
      fix: 'Nothing to do here — look at the step feeding it.',
      terms: [],
    },
    distinct: {
      label: 'Deduplicate',
      what: 'Removes duplicate rows.',
      slow: 'Like grouping, it holds every distinct value seen so far, so high-cardinality input makes it heavy.',
      fix: 'Deduplicate on the narrowest set of columns that gives a correct answer.',
      terms: ['spill', 'pipeline breaker'],
    },
    union: {
      label: 'Combine',
      what: 'Concatenates the rows of two inputs.',
      slow: 'Rarely. Cost normally belongs to the inputs.',
      fix: 'Look at the branches feeding it.',
      terms: [],
    },
    unnest: {
      label: 'Expand lists',
      what: 'Turns one row holding a list into one row per element of that list.',
      slow: 'It multiplies rows. A column averaging 50 elements makes the plan beneath it 50 times wider, and every later step pays that.',
      fix: 'Filter before expanding rather than after, and expand as late in the plan as the query allows.',
      terms: ['selectivity', 'batch'],
    },
    window: {
      label: 'Window function',
      what: 'Computes a value per row over a surrounding frame — a running total, a rank.',
      slow: 'It sorts within each partition, so it carries a sort’s costs plus its own.',
      fix: 'Partition by a column that divides the data evenly, and narrow the frame if the whole partition is not needed.',
      terms: ['pipeline breaker', 'skew'],
    },
  };

  /* ---------- metrics ----------
   * `good` and `bad` give a reader a yardstick. A number with no sense of scale is a number
   * nobody can act on, and "is 340ms bad?" is the most common unanswered question here. */

  const METRICS = {
    duration: { label: 'Duration', what: 'Wall-clock time from start to finish.',
      good: 'Steady against the pipeline baseline.', bad: 'More than about 1.5x the baseline.', term: 'wall-clock' },
    rows: { label: 'Rows', what: 'Rows the query returned.',
      good: 'Roughly stable run to run.', bad: 'A sudden change usually means the input changed, not the engine.', term: 'run' },
    read: { label: 'Rows read', what: 'Rows pulled from sources before any filtering.',
      good: 'Close to what the query needs.', bad: 'Orders of magnitude above rows returned — filters are running too late.', term: 'selectivity' },
    throughput: { label: 'Throughput', what: 'Rows processed per second.',
      good: 'Holds steady as data grows.', bad: 'Falls as data grows — something is scaling worse than linearly.', term: 'throughput' },
    parallelism: { label: 'Parallelism', what: 'Operator time over wall-clock — cores kept busy.',
      good: 'Near your core count.', bad: 'Near 1 on a multi-core machine.', term: 'parallelism' },
    rows_in: { label: 'Rows in', what: 'Rows a step consumed.',
      good: 'Close to what the step needs.',
      bad: 'Far above rows out on an early step — work is being read and thrown away.', term: 'cardinality' },
    rows_out: { label: 'Rows out', what: 'Rows a step produced.',
      good: 'Falling as you move up a plan with filters in it.',
      bad: 'Rising sharply past a join — the key is probably not unique.', term: 'cardinality' },
    selectivity: { label: 'Selectivity', what: 'Rows out over rows in, for one step.',
      good: 'Low on an early filter — it is doing its job.',
      bad: 'Low on a late filter — everything beneath it worked on rows it discarded.', term: 'selectivity' },
    spill_bytes: { label: 'Spilled bytes', what: 'How much a step wrote to disk.',
      good: 'Zero.', bad: 'Anything, on a latency-sensitive query.', term: 'spill' },
    result_bytes: { label: 'Result size', what: 'Bytes a step handed to the next one.',
      good: 'Small relative to what it read.',
      bad: 'Large and growing up the plan — wide rows are being carried further than needed.', term: 'projection' },
    build_rows: { label: 'Build rows', what: 'Rows loaded into a join’s hash table.',
      good: 'The smaller of the two inputs.',
      bad: 'Larger than the probe side — the build side was chosen the wrong way round.', term: 'build side' },
    critical_path: { label: 'On the critical path', what: 'Whether a step is on the chain that sets the total.',
      good: 'Not something to fix — something to check before optimising.',
      bad: 'Time spent speeding up a step that is off it changes the total by nothing.', term: 'critical path' },
    queued: { label: 'Queued', what: 'Time before a run started executing.',
      good: 'Near zero.',
      bad: 'Large — the engine was busy, so the query was not slow, it was waiting.', term: 'wall-clock' },
    operators: { label: 'Steps', what: 'How many operators the plan has.',
      good: 'Whatever the query needs.',
      bad: 'Not meaningful alone — a longer plan is often a faster one.', term: 'plan' },
    est_rows: { label: 'Expected rows', what: 'How many rows the planner predicted this step would produce.',
      good: 'Close to the actual count.', bad: 'Far from actual — the plan rested on a bad guess.', term: 'cardinality estimate' },
    result_size: { label: 'Result size', what: 'Bytes a step handed to the next.',
      good: 'Small relative to what it read.', bad: 'Growing up the plan — wide rows carried too far.', term: 'projection' },
    backend: { label: 'Backend', what: 'Which execution path ran the step — interpreter, parallel, or JIT.',
      good: 'JIT on a hot numeric path.', bad: 'Not a problem to fix; a note on how it ran.', term: 'jit compilation' },
    pruned: { label: 'Pruned', what: 'How much a scan skipped without reading.',
      good: 'High — the sort key or partition matched the filter.', bad: 'Low on a large scan — nothing could be skipped.', term: 'zone map' },
    cpu_util: { label: 'CPU per step', what: 'Fraction of a step\u2019s allocated cores kept busy.',
      good: 'High, on a step that should parallelise.',
      bad: 'An em dash means the engine did not measure it on this tier \u2014 not that the step was idle.',
      term: 'parallelism' },
    peak_memory: { label: 'Peak memory', what: 'The high-water mark of memory held during the run.',
      good: 'Comfortably inside the budget.', bad: 'At the budget — the next step up in data will spill.', term: 'memory budget' },
    spill: { label: 'Spilled', what: 'Bytes written to disk because they did not fit in memory.',
      good: 'Zero.', bad: 'Anything, if the query is latency-sensitive.', term: 'spill' },
    steps: { label: 'Steps', what: 'Operators in the plan.',
      good: 'Whatever the query needs.', bad: 'Not meaningful on its own — a longer plan is often a better one.', term: 'plan' },
    est_error: { label: 'Estimate error', what: 'Actual rows over predicted rows.',
      good: 'Within about 2x.', bad: 'Beyond 10x — the plan rested on a bad guess.', term: 'estimate error' },
    p50: { label: 'p50', what: 'Median run duration.', good: 'Flat over time.', bad: 'Trending up.', term: 'p50' },
    p95: { label: 'p95', what: 'The duration 95% of runs beat.',
      good: 'Within a few times p50.', bad: 'Far above p50 — the tail is where users live.', term: 'p95' },
  };

  /* ---------- recipes ----------
   * Task-shaped entries for the Learn view. A newcomer arrives with a task ("why is this
   * slow"), not with a term, so the reference has to be reachable from that direction too. */

  const RECIPES = [
    { task: 'See what the optimizer did to my query',
      steps: ['Open a run and go to the Query tab.',
              'Choose \u201cWhat the optimizer changed\u201d.',
              'The finding at the top is the rewrite; the fold below it is every step that rewrite moved past.',
              'If it says the original plan was not recorded, that is not the same as \u201cnothing changed\u201d.'] },
    { task: 'Read the plan as text, the way EXPLAIN shows it',
      steps: ['Open a run, go to Query, and stay on Explain.',
              'Tick \u201cShow the plan as written\u201d for the plan before optimization.',
              'Use \u201cCopy as text\u201d to paste the whole tree into an issue.',
              'Press x from anywhere in a run to jump straight here.'] },
    { task: 'Watch a long job while it runs',
      steps: ['Open the Live tab, or press g then r.',
              'Partition progress shows a percentage only where the engine reports a total.',
              'GPU gauges carry their target bands, so a reading says whether it is good without you knowing the bands.',
              'Check blocked time: if it is high, the bottleneck is upstream of the model.'] },
    { task: 'Find out whether rows were silently dropped',
      steps: ['Open the Live tab while the job runs.',
              'Any rows skipped under on_read_error="skip" are reported with their reason.',
              'This is the one failure mode that leaves no error behind, so it is worth looking for deliberately.'] },
    { task: 'Scrape the engine into Prometheus',
      steps: ['Start the dashboard with bt.start_ui().',
              'Point your scraper at /metrics on its port.',
              'The same numbers are at /api/metrics as JSON, and /api lists every route.'] },
    { task: 'Find out why one run was slow',
      steps: ['Open the pipeline, then the run.',
              'Read the verdict sentence at the top — it names the slowest step.',
              'Open the Steps tab and look at the step with the longest bar.',
              'Click that step: the inspector explains what it does and why it may be slow.'] },
    { task: 'Tell a regression from a naturally slow query',
      steps: ['Open the pipeline and look at the run history chart.',
              'A flat line means the query is simply expensive.',
              'A step change means something moved — compare a run either side of it.',
              'Use Compare to see which step accounts for the difference.'] },
    { task: 'Work out whether a query is memory-bound',
      steps: ['Check the Spilled tile on the run.',
              'Anything above zero means it ran out of memory somewhere.',
              'In the Steps tab, spilling steps are outlined and flagged.',
              'Filter earlier, carry fewer columns, or raise the budget.'] },
    { task: 'Check whether cores are being used',
      steps: ['Compare Parallelism against your core count.',
              'Near 1 means the query effectively ran single-file.',
              'A single dominant step on the critical path is the usual cause.'] },
    { task: 'Find out why a query used so much memory',
      steps: ['Open the run and read the Peak memory tile.',
              'In Steps, switch to Table and sort by peak memory.',
              'Pipeline breakers — sorts, joins, groupings — are where memory accumulates.',
              'Reduce what reaches them: filter earlier, or carry fewer columns.'] },
    { task: 'Work out whether a filter is running early enough',
      steps: ['Open the run and switch Steps to Graph.',
              'Find the filter and read the row count on its outgoing edge.',
              'Compare that against the counts on edges below it.',
              'A very selective filter sitting above a join means work was done and discarded.'] },
    { task: 'Decide whether a plan change or the data caused a slowdown',
      steps: ['Open the pipeline and find a fast run and a slow one.',
              'Use Compare to put them side by side.',
              'If the steps match but their row counts grew, the data changed.',
              'If the steps themselves differ, the optimizer chose differently.'] },
    { task: 'See what the optimizer decided and why',
      steps: ['Open the run and go to Findings.',
              'The decisions list names each rewrite the optimizer applied.',
              'Each one names the rule that fired.',
              'The plan you see in Steps is the result of those decisions in order.'] },
    { task: 'Follow one query through the logs',
      steps: ['Open Logs.',
              'Click any structured field on a line to filter to it.',
              'Filter on the query id to see only that run.',
              'Click a timestamp to copy a link straight to that line.'] },
    { task: 'Find when a problem started',
      steps: ['Open Logs and read the volume histogram at the top.',
              'A spike in the red band is where errors began.',
              'Click that bar to narrow to the window.',
              'Clear the time filter to widen again.'] },
    { task: 'Judge whether a pipeline is getting slower over time',
      steps: ['Open the pipeline and read Duration over time.',
              'A rising line with steady row counts is a real regression.',
              'A rising line tracking row growth is just more data.',
              'Step history shows which step the increase belongs to.'] },
    { task: 'Read a plan you did not write',
      steps: ['Switch Steps to Graph — data flows upward from the scans.',
              'Every operator name is clickable for what it does.',
              'Edge labels are the rows flowing between steps.',
              'A number that jumps by orders of magnitude is where rows multiplied.'] },
    { task: 'Tell whether a query is CPU-bound or waiting on memory',
      steps: ['Open the run and read the Spilled tile.',
              'Any spill means it ran out of memory somewhere.',
              'No spill but slow means the time went to CPU work — look at the dominant step.',
              'The step inspector says which, per step.'] },
    { task: 'Understand why the engine chose the plan it did',
      steps: ['Open the run and go to Findings.',
              'The Decisions list names each rewrite, in order.',
              'Each names the rule that fired and what it changed.',
              'The plan in Steps is the sum of those decisions.'] },
    { task: 'Check whether re-running a query would help',
      steps: ['Look for an "estimate off" flag on any step.',
              'A large miss means the plan was built on a bad guess.',
              'The engine records the real counts on this run.',
              'Re-running feeds them back, so the next plan is built on measurements.'] },
    { task: 'Find the step that sets the floor on total time',
      steps: ['Open the run and switch Steps to Graph.',
              'The critical path is the highlighted chain of steps.',
              'Its total is the fastest the query could possibly be.',
              'Speeding up anything off it changes the total by nothing.'] },
    { task: 'Compare the machine two runs ran on',
      steps: ['Open System to see cores, memory, and the current settings.',
              'A 400ms aggregate means something different on 4 cores than on 96.',
              'Check the settings a run used before comparing its timings to another.'] },
    { task: 'Read the logs for a run that failed',
      steps: ['Open Logs and filter to Errors only.',
              'Click a structured field to narrow to one query id.',
              'Use "show surrounding lines" to see what led up to it.',
              'The lines a level filter hid are often what explains it.'] },
    { task: 'Spot a query that scans far more than it returns',
      steps: ['Open the run and switch Steps to Table.',
              'Compare Rows in on the scan against the final Rows.',
              'Orders of magnitude apart means filters ran too late.',
              'A "scanned N to keep M" finding names it directly.'] },
    { task: 'Tell whether the cores were actually used',
      steps: ['Read the time bar on the run header.',
              'A wide "steps" portion with a low operator-time-to-duration ratio means single-file.',
              'One dominant step on the critical path is the usual cause.',
              'The parallelism metric puts a number on it.'] },
    { task: 'Find every run of one query shape',
      steps: ['Open Pipelines — each card is one shape.',
              'A query re-run over new data lands on the same card.',
              'Open the card for its full run history.',
              'Step history shows every run as a column.'] },
    { task: 'Reproduce a run for a bug report',
      steps: ['Open the run and use Download this run as JSON.',
              'It carries the plan, the timings, and the decisions.',
              'The query id in the file points at exactly this execution.',
              'Paste the id or the link so a colleague opens the same view.'] },
    { task: 'Learn what a step type does',
      steps: ['Open Learn and read the Plan steps section.',
              'Each operator says what it does and when it is slow.',
              'Or click any step name in the plan for the same explanation.',
              'A dotted term anywhere on the page is clickable too.'] },
    { task: 'Share what you are looking at',
      steps: ['Every view is in the URL — copy the address bar.',
              'Or press ? and use Copy link.',
              'The link reopens the same pipeline, run, and tab.'] },
  ];

  /* ═══════════ coming from another engine ═══════════
   *
   * Almost nobody arrives here without having read a Spark UI, an Airflow grid, a DuckDB
   * EXPLAIN, or a Ray dashboard first, and the fastest way to make a new tool legible is to
   * say which familiar thing each part of it *is*. This maps the panel a reader knows to
   * the panel here.
   *
   * Two rules for anything added below. **Only describe features of the other tool that are
   * long-standing and easily checked** — a Spark UI tab, an Airflow view, a documented
   * command. And **never claim Batcher is faster, or better, here**: this is a map for
   * finding your way around, not a scoreboard. The code-checked competitive comparison lives
   * in `docs/internals/competitive_architecture.md`, and it is the only place that decides
   * what Batcher may claim.
   */
  const COMPARISONS = [
    { tool: 'Apache Spark',
      familiar: 'The Spark UI — Jobs, Stages, SQL/DataFrame, Executors, Storage.',
      rows: [
        ['Stages tab', 'A run’s Steps view, switched to Stages. A stage here is the work between two pipeline breakers, which is the same idea Spark shuffles at.'],
        ['SQL / DataFrame tab', 'The Query tab: Explain for the plan tree, and the optimizer diff for what Catalyst’s equivalent did to it.'],
        ['Executors tab', 'The System page. Batcher is one process by default, so this reports the machine rather than a list of executors.'],
        ['Event timeline', 'Not offered. The engine records how long each operator took, not when it started, so a timeline placed on a clock would be invented rather than measured.'],
      ] },
    { tool: 'Apache Airflow',
      familiar: 'The Grid, Graph, and Gantt views over DAG runs.',
      rows: [
        ['Grid view', 'The Step history matrix on a pipeline page — one column per run, one row per step, shaded against that step’s own median.'],
        ['Graph view', 'The Plan graph. The difference: Airflow’s nodes are tasks you wrote, these are operators the optimizer chose.'],
        ['DAG runs list', 'The Runs table on a pipeline page.'],
        ['Gantt view', 'Not offered, for the same reason as Spark’s event timeline — no measured start offsets. The Stages view answers the underlying question of what ran together.'],
      ] },
    { tool: 'Ray Data',
      familiar: 'The Ray dashboard’s per-operator progress and resource panels.',
      rows: [
        ['Operator progress', 'The Live page: partitions finished per stage, with a real denominator where the engine reports one.'],
        ['Resource usage', 'The Live page’s accelerator gauges, plus the System page for the host.'],
        ['Backpressure', 'The blocked-time reading on the Live page. Rising blocked time means the workers are outrunning the pipeline feeding them.'],
        ['Object store', 'Nothing to show. Batcher’s bulk data moves over Arrow Flight, deliberately bypassing the Ray object store.'],
      ] },
    { tool: 'DuckDB',
      familiar: 'EXPLAIN and EXPLAIN ANALYZE at the SQL prompt.',
      rows: [
        ['EXPLAIN', 'The Query tab, Explain, with "Show the plan as written" ticked.'],
        ['EXPLAIN ANALYZE', 'The Query tab, Explain, as it renders by default — the plan that ran, annotated with each step’s measured time and rows.'],
        ['The profiling tree', 'The same view, or the Flame rendering under Steps if you would rather see the cost distribution than the nesting.'],
      ] },
    { tool: 'Polars',
      familiar: '`.explain()`, `.profile()`, and the optimization flags on `collect()`.',
      rows: [
        ['explain(optimized=False)', 'The Query tab, Explain, showing the plan as written.'],
        ['explain()', 'The Query tab, Explain, showing the plan as run.'],
        ['profile()', 'The Steps view, in any of its five renderings.'],
        ['Which optimizations fired', 'The optimizer diff — the panel with no direct equivalent elsewhere, because it shows the two plans against each other rather than one at a time.'],
      ] },
  ];

  const termKeys = Object.keys(TERMS).sort();
  /* A cross-reference resolves to a glossary term *or* an operator.
   *
   * The two vocabularies overlap in the reader's head — someone following a link from "zone
   * map" to "scan" wants the scan operator's page, and does not know or care that the
   * reference stores operators in a different object. Operators are adapted to the term
   * shape so every consumer (popover, Learn page, search) handles one type. */
  const lookup = (word) => {
    const key = String(word || '').toLowerCase();
    if (TERMS[key]) return TERMS[key];
    const op = OPERATORS[key];
    if (!op) return null;
    return {
      what: op.what,
      why: `Slow when: ${op.slow}`,
      fix: op.fix,
      see: op.terms || [],
      isOperator: true,
      label: op.label,
    };
  };
  const operator = (kind) => OPERATORS[kind] || null;
  const metric = (key) => METRICS[key] || null;

  /* Free-text search across all three sets, for the command palette and the Learn view. */
  function search(query) {
    const q = String(query || '').trim().toLowerCase();
    if (!q) return [];
    const hits = [];
    for (const [word, entry] of Object.entries(TERMS)) {
      if (word.includes(q) || entry.what.toLowerCase().includes(q)) {
        hits.push({ kind: 'term', key: word, label: word, blurb: entry.what });
      }
    }
    for (const [kind, entry] of Object.entries(OPERATORS)) {
      const hay = `${kind} ${entry.label} ${entry.what} ${entry.slow} ${entry.fix}`.toLowerCase();
      if (hay.includes(q)) {
        hits.push({ kind: 'operator', key: kind, label: entry.label, blurb: entry.what });
      }
    }
    for (const recipe of RECIPES) {
      if (recipe.task.toLowerCase().includes(q)) {
        hits.push({ kind: 'recipe', key: recipe.task, label: recipe.task, blurb: recipe.steps[0] });
      }
    }
    // Someone typing "spark" or "airflow" is asking where the thing they know lives here,
    // which is a question this reference can answer directly.
    for (const entry of COMPARISONS) {
      if (entry.tool.toLowerCase().includes(q) || entry.familiar.toLowerCase().includes(q)) {
        hits.push({ kind: 'comparison', key: entry.tool,
                    label: `Coming from ${entry.tool}`, blurb: entry.familiar });
      }
    }
    return hits.slice(0, 24);
  }

  return { TERMS, OPERATORS, METRICS, RECIPES, COMPARISONS, termKeys, lookup, operator,
           metric, search };
})();
