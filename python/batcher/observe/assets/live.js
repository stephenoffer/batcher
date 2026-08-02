/* The live view — what the engine is doing *right now*, at whatever scale it is doing it.
 *
 * Everything else on this dashboard is retrospective: a run finished, here is its profile.
 * That is the wrong shape for the two cases Batcher exists for. A distributed job and a
 * batch-inference job run for minutes or hours, and the question during those hours is not
 * "how did it go" but "is it healthy, and when will it finish".
 *
 * The engine already publishes what that needs — `PARTITION`, `GPU`, `INFER`, `POOL`, and
 * `SKIPPED` events, folded by `observe/inference.py` into a bounded per-job snapshot. Until
 * now the activity store dropped every one of them on the floor and the dashboard had no
 * idea they existed. This renders them.
 *
 * ── Sampling honesty ────────────────────────────────────────────────────────────
 * The engine reports *current* readings, not history: one GPU utilization figure per
 * sample, not a series. The trend lines here are therefore built from readings this page
 * took while it was open, and they say so. A page opened two minutes ago has two minutes of
 * trend and does not pretend to have the job's whole history — and after a reload the
 * series legitimately starts again, which is what "sampled by this page" means.
 */

'use strict';

const LIVE = (() => {
  const { esc, ms, count, bytes, pct, duration } = UI;
  const $ = (id) => document.getElementById(id);

  /* Readings this page has taken, per metric key. Bounded: a job left open overnight must
   * not grow an array until the tab dies. 240 samples at the live poll rate is ~4 minutes
   * of detail, which is the window in which "is it degrading" is answerable. */
  const HISTORY_MAX = 240;
  const history = new Map();

  function sample(key, value) {
    if (value == null || !Number.isFinite(value)) return;
    let series = history.get(key);
    if (!series) { series = []; history.set(key, series); }
    series.push({ x: Date.now() / 1000, y: value });
    if (series.length > HISTORY_MAX) series.splice(0, series.length - HISTORY_MAX);
  }
  const seriesFor = (key) => history.get(key) || [];
  const forget = () => history.clear();

  /* GPU utilization bands, from the same guidance `observe/inference.py` judges against —
   * imported as numbers rather than re-derived, so the gauge and the diagnostic beneath it
   * can never disagree about what "under-used" means. */
  const UTIL_SEVERE = 30, UTIL_LOW = 70, UTIL_TARGET = 85;
  const MEM_HIGH = 0.9;

  /* ---------- the whole view ---------- */

  /* Record this poll's readings, whether or not the page is on screen.
   *
   * Kept apart from `render` on purpose: the trend lines are built from samples taken over
   * time, so sampling only while the panel is visible would restart the series every time
   * someone navigated away and back — and a chart that resets when you look at it is worse
   * than no chart. This is called from the poll loop; `render` only draws.
   */
  function observe(snapshot) {
    if (!snapshot || !snapshot.query_id) return;
    sample('rows_per_sec', snapshot.rows_per_sec);
    sample('latency', snapshot.inference?.latency_ms);
    sample('blocked', snapshot.inference?.blocked_ms);
    for (const [device, g] of Object.entries(snapshot.gpu || {})) sample(`gpu:${device}`, g.util_pct);
    if (snapshot.partitions?.done) partitionRate(snapshot.partitions.done);
  }

  function render(snapshot, running, opts = {}) {
    const host = $('live-body');
    if (!host) return;
    const hasJob = snapshot && snapshot.query_id;
    if (!hasJob && !running.length) {
      forget();
      host.innerHTML = UI.emptyState({
        glyph: 'run',
        title: 'Nothing is running',
        body: 'This page fills while a query is in flight. It is built for the long ones — a ' +
              'distributed job or a batch-inference pass — where the question is not how it ' +
              'went but whether it is healthy and when it will land. Start one and it appears ' +
              'here within a second.',
        code: 'import batcher as bt\n\n' +
              'ds = bt.read_parquet("images/*.parquet")\n' +
              'ds.ml.infer(model, batch_size=64).write.parquet("out/")',
        hint: 'Partition progress, GPU load, and dropped rows only appear for work that ' +
              'reports them — a single-node scan has none of those, and nothing is invented.',
      });
      return;
    }

    host.innerHTML =
      inFlight(running, opts) +
      (hasJob ? progress(snapshot) : '') +
      (hasJob ? diagnostics(snapshot.diagnostics || []) : '') +
      (hasJob ? accelerators(snapshot) : '') +
      (hasJob ? throughput(snapshot) : '') +
      (hasJob ? pipeline(snapshot) : '');

    if (opts.onOpenRun) {
      CHARTS.onPick(host, (id) => opts.onOpenRun(id));
    }
    drawTrends(snapshot);
  }

  /* ---------- running queries ---------- */

  /* Every query in flight, with the progress the engine can actually report.
   *
   * Rows *seen* rather than a percentage: the engine knows how many rows have crossed an
   * operator, and for most sources it does not know how many there will be. A progress bar
   * needs a denominator, and inventing one is how a job sits at "97%" for an hour. Where a
   * real denominator exists — partition counts on a distributed stage — the bar appears. */
  function inFlight(running, opts) {
    if (!running.length) return '';
    return `<section class="panel"><div class="panel-head"><h2>In flight ` +
      `<span class="badge is-live">${running.length}</span></h2>` +
      `<span class="hint">updating every second</span></div><div class="panel-body">` +
      running.map((q) => {
        const elapsed = Math.max(0, Date.now() / 1000 - q.started_wall);
        const steps = q.n_stages ? `${q.n_done} of ${q.n_stages} steps` : 'planning';
        const stepShare = q.n_stages ? q.n_done / q.n_stages : 0;
        return `<article class="live-run" data-pick="${esc(q.query_id)}" role="button" tabindex="0" ` +
          `aria-label="Open the run ${esc(DAG.friendlyKind(q.label))}">` +
          `<div class="live-run-head"><span class="pulse-dot" aria-hidden="true"></span>` +
          `<b>${esc(DAG.friendlyKind(q.label))}</b>` +
          `<span class="dim">running for ${esc(duration(elapsed))}</span></div>` +
          `<div class="live-run-bar" role="img" aria-label="${esc(steps)}">` +
          `<i style="width:${(stepShare * 100).toFixed(1)}%"></i></div>` +
          `<div class="live-run-stats">` +
          stat('steps done', steps) +
          stat('rows seen', count(q.rows_seen)) +
          stat('bytes seen', bytes(q.bytes_seen)) +
          `</div></article>`;
      }).join('') + `</div></section>`;
  }

  const stat = (label, value) =>
    `<span class="live-stat"><span class="live-stat-label">${esc(label)}</span>` +
    `<span class="live-stat-value">${esc(String(value))}</span></span>`;

  /* ---------- partition progress ---------- */

  /* The one place a real percentage exists. A distributed stage knows how many partitions
   * it split into, so `done / total` is a measurement rather than an estimate — and where
   * the total is unknown the count is shown bare, with no denominator invented for it. */
  function progress(s) {
    const p = s.partitions || {};
    const stages = Object.entries(p.stages || {});
    if (!p.done && !stages.length) return '';
    const eta = estimate(s);
    return `<section class="panel"><div class="panel-head"><h2>Partitions</h2>` +
      `<span class="hint">${esc(s.label || s.query_id)}</span></div><div class="panel-body">` +
      (p.total
        ? `<div class="live-headline"><span class="live-headline-num">${p.done}` +
          `<span class="live-headline-of"> of ${p.total}</span></span>` +
          `<span class="live-headline-label">partitions finished</span>` +
          (eta ? `<span class="live-eta">${esc(eta)}</span>` : '') + `</div>` +
          `<div class="live-progress" role="progressbar" aria-valuenow="${p.done}" ` +
          `aria-valuemin="0" aria-valuemax="${p.total}">` +
          `<i style="width:${((p.fraction || 0) * 100).toFixed(1)}%"></i></div>`
        : `<div class="live-headline"><span class="live-headline-num">${p.done}</span>` +
          `<span class="live-headline-label">partitions finished</span>` +
          `<span class="live-eta is-quiet">no total reported, so no percentage</span></div>`) +
      (stages.length > 1
        ? `<div class="live-stages">` + stages.map(([name, st]) => (
            `<div class="live-stage"><span class="live-stage-name">${esc(name)}</span>` +
            `<span class="live-stage-track"><i style="width:${st.total ? ((st.done / st.total) * 100).toFixed(1) : 0}%"></i></span>` +
            `<span class="live-stage-count mono">${st.done}${st.total ? ` / ${st.total}` : ''}</span>` +
            `<span class="live-stage-rows mono dim">${esc(count(st.rows))} rows</span></div>`)).join('') +
          `</div>`
        : '') + `</div></section>`;
  }

  /* A finish time, only when every input to it is measured.
   *
   * Needs a partition total, some partitions already done, and a rate. Missing any of
   * those, there is no honest estimate and none is shown — a countdown derived from one
   * sample is a guess wearing a clock's clothes. */
  function estimate(s) {
    const p = s.partitions || {};
    if (!p.total || !p.done || p.done >= p.total) return '';
    const rate = partitionRate(p.done);
    if (!rate) return 'measuring the rate…';
    return `about ${duration((p.total - p.done) / rate)} left at the current rate`;
  }

  /* Partitions per second, measured across this page's own samples rather than assumed. */
  let lastPartitions = null;
  function partitionRate(done) {
    const now = Date.now() / 1000;
    if (lastPartitions && done > lastPartitions.done) {
      const rate = (done - lastPartitions.done) / Math.max(0.001, now - lastPartitions.at);
      sample('partition_rate', rate);
      lastPartitions = { done, at: now };
    } else if (!lastPartitions) {
      lastPartitions = { done, at: now };
    }
    const series = seriesFor('partition_rate');
    if (!series.length) return 0;
    // Median of the recent samples, not the last one: partition completion is bursty, and
    // the last interval alone swings an estimate between "2 minutes" and "40 minutes".
    return UI.median(series.slice(-8).map((p) => p.y));
  }

  /* ---------- diagnostics ---------- */

  /* The engine's own verdicts, rendered as they arrive. These come from
   * `InferenceProgress.diagnostics`, which knows the bands; the UI does not re-judge them,
   * it presents them — two thresholds for the same reading is how a dashboard ends up
   * disagreeing with the log line beside it. */
  function diagnostics(list) {
    if (!list.length) return '';
    return `<section class="panel"><div class="panel-head"><h2>Findings ` +
      `<span class="badge${list.some((d) => d.severity !== 'info') ? ' is-warn' : ''}">${list.length}</span></h2>` +
      `<span class="hint">judged by the engine, not by this page</span></div>` +
      `<div class="panel-body insights">` + list.map((d) => (
        `<div class="insight sev-${esc(d.severity === 'warning' ? 'warn' : d.severity)}">` +
        `<div class="insight-head"><span class="insight-sev">${esc(d.severity)}</span>` +
        `<span class="insight-title">${esc(d.message)}</span>` +
        `<span class="insight-rule mono">${esc(d.code)}</span></div></div>`)).join('') +
      `</div></section>`;
  }

  /* ---------- accelerators ---------- */

  /* Per-device load, with the bands drawn *on* the gauge.
   *
   * A bare "72%" does not say whether 72% is good, and for a GPU that is the entire
   * question — the hardware is the expensive part of the job and the target band is where
   * it earns out. Starvation gets its own line rather than a colour, because an oscillating
   * device and a merely slow one average to the same number and need opposite fixes. */
  function accelerators(s) {
    const gpus = Object.entries(s.gpu || {});
    if (!gpus.length) return '';
    const bands = [
      { from: 0, to: UTIL_SEVERE / 100, tone: 'critical' },
      { from: UTIL_SEVERE / 100, to: UTIL_LOW / 100, tone: 'warn' },
      { from: UTIL_TARGET / 100, to: 1, tone: 'good' },
    ];
    return `<section class="panel"><div class="panel-head"><h2>Accelerators ` +
      `<span class="badge">${gpus.length}</span></h2>` +
      `<span class="hint">target band is ${UTIL_TARGET}% and above</span></div>` +
      `<div class="panel-body live-gpus">` + gpus.map(([device, g]) => {
        const memTone = g.mem_fraction > MEM_HIGH ? 'critical' : g.mem_fraction > 0.7 ? 'good' : '';
        const utilTone = g.util_pct < UTIL_SEVERE ? 'critical' : g.util_pct < UTIL_LOW ? 'warn' : 'good';
        return `<article class="live-gpu">` +
          `<header class="live-gpu-head"><b>${esc(device)}</b>` +
          (g.starved
            ? `<span class="chip is-warn">starved — swinging between idle and saturated</span>`
            : `<span class="chip is-quiet">steady</span>`) + `</header>` +
          CHARTS.gauge({ value: g.util_pct, max: 100, label: 'utilization',
                         format: 'plain', bands, tone: utilTone }) +
          CHARTS.gauge({ value: g.mem_used_bytes, max: g.mem_total_bytes || 1, label: 'VRAM',
                         format: 'bytes', tone: memTone }) +
          `<div class="live-gpu-trend" data-trend="gpu:${esc(device)}"></div>` +
          `<p class="live-gpu-note">${g.mem_total_bytes
            ? `${esc(bytes(g.mem_used_bytes))} of ${esc(bytes(g.mem_total_bytes))} in use ` +
              `(${esc(pct(g.mem_fraction))})`
            : 'VRAM total not reported by this device'}</p>` +
          `</article>`;
      }).join('') + `</div></section>`;
  }

  /* ---------- throughput ---------- */

  function throughput(s) {
    const inf = s.inference || {};
    if (!s.rows_per_sec && !inf.batches) return '';
    return `<section class="panel"><div class="panel-head"><h2>Throughput</h2>` +
      `<span class="hint">sampled by this page while it has been open</span></div>` +
      `<div class="panel-body">` +
      `<div class="statstrip">` +
      tile('Rows / sec', count(s.rows_per_sec), 'smoothed') +
      tile('Rows so far', count(s.total_rows), 'this job') +
      tile('Batches', count(inf.batches), 'inference micro-batches') +
      tile('Latency', ms(inf.latency_ms), 'per batch, smoothed') +
      tile('Blocked', ms(inf.blocked_ms), 'waiting for input') +
      tile('Actors', `${(s.pool || {}).size || 0}`, `${(s.pool || {}).pending || 0} queued`) +
      `</div>` +
      `<div class="live-trend" data-trend="rows_per_sec" data-trend-label="rows per second" ` +
      `data-trend-format="rate"></div>` +
      `</div></section>`;
  }

  const tile = (label, value, note) =>
    `<div class="stat"><span class="stat-label">${esc(label)}</span>` +
    `<span class="stat-value">${esc(String(value))}</span>` +
    `<span class="stat-note">${esc(note)}</span></div>`;

  /* ---------- pipeline health & data loss ---------- */

  /* Two things that only show up at scale, and that a summary at the end would hide.
   *
   * Blocked time rising means the workers are finishing faster than the pipeline can feed
   * them, so the fix is upstream, not a bigger model or a bigger GPU. Skipped rows are
   * silent data loss: `on_read_error="skip"` is doing exactly what it was asked to, and
   * the only failure mode is nobody noticing. */
  function pipeline(s) {
    const skipped = s.skipped || {};
    const inf = s.inference || {};
    const starving = inf.blocked_ms > 0 && inf.latency_ms > 0 &&
      inf.blocked_ms / (inf.blocked_ms + inf.latency_ms) > 0.2;
    if (!skipped.total && !starving) return '';
    return `<section class="panel"><div class="panel-head"><h2>Worth knowing</h2></div>` +
      `<div class="panel-body">` +
      (starving
        ? `<p class="live-warn">Workers spent <b>${esc(pct(inf.blocked_ms / (inf.blocked_ms + inf.latency_ms)))}</b> ` +
          `of their time waiting for input rather than computing. The bottleneck is upstream ` +
          `of the model — reading, decoding, or shuffling — so a larger batch or a faster ` +
          `device will not help.</p>`
        : '') +
      (skipped.total
        ? `<div class="live-skipped"><p class="live-warn">` +
          `<b>${esc(count(skipped.total))} rows were dropped</b> and the job carried on. That is ` +
          `<span class="mono">on_read_error="skip"</span> working as asked — but the result is ` +
          `missing data, so it is worth being deliberate about.</p>` +
          CHARTS.bars(Object.entries(skipped.by_reason || {})
            .map(([reason, n]) => ({ label: reason, value: n, tone: 'warn' }))
            .sort((a, b) => b.value - a.value), { format: 'count', dense: true }) +
          `</div>`
        : '') + `</div></section>`;
  }

  /* ---------- trends ---------- */

  /* Drawn after the markup lands, because the chart layer measures nothing and needs its
   * host element to exist. Every trend is this page's own samples, which is why the caption
   * says so — a series that restarts on reload must never look like the job restarting. */
  function drawTrends(snapshot) {
    for (const host of document.querySelectorAll('[data-trend]')) {
      const key = host.dataset.trend;
      const values = seriesFor(key);
      if (values.length < 3) {
        host.innerHTML = `<p class="hint">Collecting samples…</p>`;
        continue;
      }
      const label = host.dataset.trendLabel || key;
      CHARTS.timeSeries(host, [{ key, label, values }], {
        width: 620, height: 150, format: host.dataset.trendFormat || 'plain',
        xFormat: 'clock', label: `${label}, sampled by this page`,
      });
    }
    void snapshot;
  }

  return { observe, render, forget };
})();
