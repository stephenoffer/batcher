/* Batcher dashboard shell — state, routing, polling, and the global controls.
 *
 * Framework-free: five files served off disk, so the dashboard ships inside a Python wheel
 * and starts with `bt.start_ui()` rather than an `npm install`. `ui.js` holds the shared
 * machinery, `views.js` the renderers, `dag.js` the plan explorer; this decides *when* they
 * run and wires the shell around them.
 *
 * TWO RULES KEEP IT CALM:
 *   1. Render only what changed. `paint()` hashes the data before drawing. Without it a 1 Hz
 *      poll rebuilt the DOM every second, discarding scroll position, closing tooltips and
 *      restarting every animation on a dashboard whose queries had all finished.
 *   2. Poll only as fast as the data can change: 1 s while a query is in flight, 5 s idle,
 *      never while the tab is hidden, and never for a finished run's detail — that document
 *      is immutable once written.
 */

'use strict';

const POLL_ACTIVE_MS = 1000;
const POLL_IDLE_MS = 5000;
const SYSTEM_EVERY_MS = 30000;
const LEVELS = { DEBUG: 10, INFO: 20, WARNING: 30, ERROR: 40, CRITICAL: 50 };
//: How many log rows to put in the DOM at once. Generous enough that scrolling feels
//: unbounded, small enough that a repaint stays under a frame.
const LOG_RENDER_CAP = 800;
const $ = (id) => document.getElementById(id);

/* Bind a handler, tolerating a missing element. `boot` wires ~40 controls, and an exception
 * in any one of them aborts the rest — leaving a dashboard that renders nothing and reports
 * nothing. A control that is absent should cost that control, not the page. */
function on(target, event, handler) {
  const el = typeof target === 'string' ? $(target) : target;
  if (el) el.addEventListener(event, handler);
  return el;
}

/** The nearest ancestor matching `selector`, or null — never throws on a missing element. */
const up = (el, selector) => (el && el.closest ? el.closest(selector) : null);

const state = {
  // The three levels of the hierarchy. `pipeline` is a signature, `selected` a run id;
  // either being set is what makes that level's page meaningful.
  view: 'pipelines', pipeline: null, selected: null,
  tab: 'steps', stepsView: 'plan', queryView: 'explain', compareWith: null,
  logCursor: 0, logLines: [], lastSystemAt: 0,
  logRange: null, logFields: new Map(), logPending: 0, logHistoGeom: null,
  detail: null, detailId: null, compare: null,
  queries: [], pipelines: [], summary: {}, system: {}, operators: [], live: null,
  pipelineSort: 'time', pipelineFilter: '', pipelineLayout: 'cards',
  paused: false, lastError: null, loaded: false, report: null, reportFor: null,
  linkedShown: null,
  // Query-viewer state. `explainCollapsed` is a set of op_ids, not a copy of the tree, so
  // it survives the plan being re-fetched and can never describe a shape that changed.
  explainCollapsed: new Set(), explainNeedle: '', explainOriginal: false,
  irOriginal: false, diffShowAll: false,
};

const hashes = new Map();
function paint(key, data, fn) {
  const s = JSON.stringify(data);
  if (hashes.get(key) === s) return false;
  hashes.set(key, s);
  fn();
  return true;
}
const invalidate = (...keys) => keys.forEach((k) => hashes.delete(k));

/* ---------- fetch ---------- */

async function getJSON(path) {
  const res = await fetch(path, { headers: { Accept: 'application/json' } });
  if (!res.ok) throw new Error(`${path} → ${res.status}`);
  return res.json();
}

/* The connection status, tolerant of a single hiccup.
 *
 * A poll can fail transiently — a navigation aborts an in-flight fetch, the engine is busy
 * for a beat — and a full-width red "lost contact" banner on the strength of one miss is
 * alarmist. So the pill flips to a quiet "reconnecting" on the first failure and only the
 * *second consecutive* failure raises the banner; any success clears both at once. */
let connFailures = 0;
function setConnected(ok, message) {
  const el = $('conn');
  if (ok) connFailures = 0; else connFailures += 1;
  const down = connFailures >= 2;
  el.textContent = ok ? (state.paused ? 'paused' : 'live') : (down ? 'disconnected' : 'reconnecting');
  el.className = `pill ${ok ? (state.paused ? '' : 'is-live') : (down ? 'is-down' : 'is-warn')}`;
  $('error-banner').hidden = !down;
  if (down && message) $('error-text').textContent = message;
}

/* One poll, in two tiers.
 *
 * Everything used to be fetched every second, which meant a session with 100 retained runs
 * recomputed every cross-run analysis on the server — percentiles, the operator rollup, the
 * health report, the failure grouping — once a second, forever, to redraw panels whose
 * inputs change only when a query *finishes*. The fast tier is what genuinely moves while a
 * query runs; the slow tier is the rest, refreshed on a longer beat and immediately whenever
 * the run count changes, so nothing is ever stale after something happens.
 */
const ANALYSIS_EVERY_MS = 6000;

async function poll() {
  if (state.paused) { setTimeout(poll, POLL_IDLE_MS); return; }
  let running = 0;
  try {
    const [summary, queries, logs, live] = await Promise.all([
      getJSON('/api/summary'), getJSON('/api/queries'),
      getJSON(`/api/logs?since=${state.logCursor}`), getJSON('/api/live'),
    ]);
    running = summary.n_running || 0;
    Object.assign(state, { summary, queries: queries.queries, loaded: true,
                           live: live && live.query_id ? live : null });

    paint('kpis', summary, () => VIEWS.kpis(summary, flash));
    ingestLogs(logs);
    renderLive();
    await pollAnalysis(summary);

    if (state.selected) await loadDetail(state.selected);
    renderCrumbs();
    setConnected(true);
    state.lastError = null;
  } catch (err) {
    state.lastError = String(err.message || err);
    setConnected(false, state.lastError);
  } finally {
    const idle = running === 0;
    $('idle-hint').hidden = !idle || state.paused;
    setTimeout(poll, idle ? POLL_IDLE_MS : POLL_ACTIVE_MS);
  }
}

/* The cross-run analyses. Refetched on a slow beat, or at once when the number of finished
 * runs changes — the only event that can alter any of them. */
let lastAnalysisAt = 0;
let lastRunCount = -1;
async function pollAnalysis(summary) {
  const settled = (summary.n_queries || 0) - (summary.n_running || 0);
  const due = Date.now() - lastAnalysisAt > ANALYSIS_EVERY_MS || settled !== lastRunCount;
  if (!due) return;
  lastAnalysisAt = Date.now();
  lastRunCount = settled;

  const wants = [getJSON('/api/pipelines'), getJSON('/api/health'), getJSON('/api/timeseries'),
                 getJSON('/api/operators'), getJSON('/api/failures')];
  if (Date.now() - state.lastSystemAt > SYSTEM_EVERY_MS) wants.push(getJSON('/api/system'));
  const [pipelines, health, series, operators, failures, system] = await Promise.all(wants);
  Object.assign(state, { pipelines: pipelines.pipelines, operators: operators.operators });

  paint('health', health, () => VIEWS.health(health));
  paint('series', series, () => VIEWS.throughput(series));
  paint('rollup', operators.operators, () => VIEWS.operatorRollup(operators.operators));
  paint('split', pipelines.pipelines, () => VIEWS.timeSplit(pipelines.pipelines));
  paint('failures', failures.groups, () => {
    $('failure-count').textContent = failures.groups.reduce((n, g) => n + g.count, 0);
    VIEWS.failures(failures.groups);
  });
  paint('attention', [state.queries.filter((q) => q.status === 'error').map((q) => q.query_id),
                      state.detail?.insights],
        () => VIEWS.attention(state.queries, state.detail?.insights));
  renderPipelineList();
  renderPipelinePage();
  if (system) {
    state.lastSystemAt = Date.now();
    state.system = system;
    paint('system', system, () => VIEWS.system(system));
  }
}

/* The live page, plus the dot on its nav tab.
 *
 * The dot is the whole reason a person on another page ever comes here: work that only
 * matters while it is happening has to announce itself, or the page it lives on is only
 * ever found after the fact. */
function renderLive() {
  // Sample first, unconditionally: the trends are built from readings taken over time, and
  // sampling only while the page is visible would restart them on every navigation.
  LIVE.observe(state.live);
  const runningRuns = state.queries.filter((q) => q.status === 'running');
  const dot = $('live-dot');
  if (dot) dot.hidden = !runningRuns.length;
  const badge = $('live-status');
  if (badge) {
    badge.hidden = !runningRuns.length && !state.live;
    badge.className = `verdict ${runningRuns.length ? 'is-live' : ''}`;
    badge.textContent = runningRuns.length
      ? `${runningRuns.length} running` : 'nothing in flight';
  }
  // Drawing, unlike sampling, is skipped while the page is off screen — a live view builds
  // gauges and a chart on every poll, and doing that for a hidden section is pure cost.
  if (state.view !== 'live') return;
  paint('live', [state.live, runningRuns.map((q) => [q.query_id, q.rows_seen, q.n_done])], () => {
    LIVE.render(state.live, runningRuns, { onOpenRun: openRun });
  });
}

function flash(el) {
  el.classList.remove('did-change');
  void el.offsetWidth;
  el.classList.add('did-change');
}

document.addEventListener('visibilitychange', () => { if (!document.hidden && !state.paused) poll(); });

/* ---------- level 1: all pipelines ---------- */

function renderPipelineList() {
  paint('pipelines', [state.pipelines, state.pipelineSort, state.pipelineFilter,
                      state.pipelineLayout, UI.getPref('pinned'), state.loaded], () => {
    if (!state.loaded) {
      $('pipeline-cards').innerHTML = UI.skeleton(4);
      $('pipelines-empty').hidden = true;
      return;
    }
    if (!state.pipelines.length) {
      $('pipeline-cards').innerHTML = '';
      $('pipeline-table').innerHTML = '';
      $('pipelines-empty').hidden = false;
      $('pipelines-empty').className = '';
      $('pipelines-empty').innerHTML = UI.emptyState({
        glyph: 'pipeline',
        title: 'Waiting for the first query',
        body: 'This dashboard is connected and listening. Run a query in the process it is ' +
              'watching and it appears here within a second. Every distinct query <em>shape</em> ' +
              'becomes a pipeline, so re-running the same query builds a history you can ' +
              'compare against rather than a pile of separate entries.',
        code: 'import batcher as bt\n\n' +
              'ds = bt.read_parquet("events.parquet")\n' +
              'ds.filter(bt.col("amount") > 100).group_by("region").sum("amount").collect()',
        actions: [
          { id: 'tour', label: 'Take the 6-step tour', primary: true },
          { id: 'learn', label: 'Read the reference' },
        ],
        hint: 'Nothing to run yet? The reference explains what every panel will show.',
      });
      return;
    }
    const shown = VIEWS.pipelines(state.pipelines, {
      sort: state.pipelineSort, needle: state.pipelineFilter,
      onOpen: openPipeline,
      onPin: (sig) => { UI.togglePin(sig); invalidate('pipelines'); renderPipelineList();
                        UI.toast(UI.isPinned(sig) ? 'Pipeline pinned' : 'Pin removed'); },
      onRename: renamePipeline,
    });
    VIEWS.pipelineTable(shown || state.pipelines, openPipeline);
    const lanes = state.pipelineLayout !== 'table';
    $('pipeline-cards').hidden = !lanes;
    $('pipeline-table').hidden = lanes;
    $('lay-cards').classList.toggle('is-on', lanes);
    $('lay-table').classList.toggle('is-on', !lanes);
    $('lay-cards').setAttribute('aria-pressed', String(lanes));
    // Offered once, and only now: a tour of an empty dashboard points at nothing, so it
    // waits until there is real work on screen to point at.
    LEARN.maybeOfferTour(state.pipelines.length > 0);
    $('lay-table').setAttribute('aria-pressed', String(!lanes));
  });
}

/* Persist a pipeline's name. The dashboard's one write — it POSTs to the registry endpoint,
 * updates the in-memory copy optimistically so the rename is instant, and re-renders. If the
 * write fails (a read-only mount, say), it says so rather than pretending it stuck. */
async function renamePipeline(signature, name) {
  const p = state.pipelines.find((x) => x.signature === signature);
  if (p) p.name = name;                     // optimistic; the next poll confirms from disk
  invalidate('pipelines', 'pipeline-page');
  renderPipelineList();
  renderPipelinePage();
  renderCrumbs();
  try {
    const res = await fetch('/api/pipeline/meta', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pipeline_id: signature, name }),
    });
    if (!res.ok) throw new Error(String(res.status));
    UI.toast(name ? `Named “${name}”` : 'Name cleared', 'good');
  } catch (err) {
    UI.toast('Could not save the name — the dashboard could not write to disk', 'warn');
  }
}

/** A pipeline's display name: a person's, or one generated from its plan shape. */
const pipelineDisplayName = (p) => (p ? DAG.pipelineName(p.plan_shape, p.name) : '');

/* ---------- level 2: one pipeline ---------- */

const currentPipeline = () => state.pipelines.find((p) => p.signature === state.pipeline) || null;

/** The runs belonging to the pipeline being viewed, after its status chips. */
function pipelineRuns() {
  const statuses = [...document.querySelectorAll('#view-pipeline .status-chip.is-on')]
    .map((c) => c.dataset.status);
  let runs = state.queries.filter((q) => q.signature === state.pipeline);
  if (statuses.length) runs = runs.filter((q) => statuses.includes(q.status));
  return runs;
}

async function loadPipelineReport(signature) {
  if (state.reportFor === signature) return;
  try {
    state.report = await getJSON(`/api/pipeline?signature=${encodeURIComponent(signature)}`);
    state.reportFor = signature;
    VIEWS.pipelineReport(state.report);
    renderPipelineDag(state.report);
    VIEWS.runGrid(state.report.grid, {
      onOpenRun: openRun,
      // Clicking a step row focuses that step in the pipeline's own graph, so the matrix and
      // the plan are two views of one thing rather than two unrelated pictures.
      onSelectStep: (opId) => { PIPE_DAG.select(opId); $('p-dag')?.scrollIntoView({ block: 'center' }); },
    });
  } catch { /* the pipeline aged out between the list and this fetch */ }
}

function renderPipelinePage() {
  const p = currentPipeline();
  if (!p) return;
  loadPipelineReport(p.signature);
  const runs = pipelineRuns();
  paint('pipeline-page', [p, runs.map((r) => [r.query_id, r.status, r.total_ms])], () => {
    VIEWS.pipelineDetail(p, runs, {
      onOpenRun: openRun,
      onCompare: (runId, against) => openRun(runId, against),
      onRename: renamePipeline,
    });
  });
}

/* The pipeline's plan graph. A second DAG instance rather than a shared one: the run page
 * and the pipeline page are visible at different times but keep independent selections,
 * pan, and zoom, and sharing a singleton made each navigation reset the other. */
function renderPipelineDag(report) {
  const legend = PIPE_DAG.render($('p-dag'), report.dag, {
    onHover: showPipelineNodeTip,
    onLeave: hideTip,
    onZoom: (k) => { const el = $('p-dag-zoom'); if (el) el.textContent = `${Math.round(k * 100)}%`; },
    onSelect: () => {},
  }, `pipeline:${report_signature(report)}`);
  const host = $('p-dag-legend');
  if (host) {
    host.innerHTML = legend.map((e) => (
      `<span><i class="swatch${e.swatch === 'ramp' ? ' ramp' : ''}${e.cls === 'crit' ? ' crit' : ''}"` +
      `${e.color ? ` style="background:${e.color}"` : ''}></i>${UI.esc(e.label)}</span>`)).join('') +
      `<span class="dim">colour is <b>typical</b> share of operator time</span>`;
  }
}

const report_signature = () => state.pipeline || '';

/* The pipeline graph's tooltip reports the distribution, not a single run's number — that
 * is the whole difference between this graph and the run's. */
function showPipelineNodeTip(event, node, share) {
  const p = node.percentiles || {};
  showTip(event, `<b>${UI.esc(DAG.friendlyKind(node.kind))}</b>` +
    (node.detail ? ` <span class="mono" style="opacity:.7">${UI.esc(node.detail)}</span>` : '') +
    `<div class="t-row"><span>typical</span><span>${UI.ms(node.mean_ms)}</span></div>` +
    `<div class="t-row"><span>p95</span><span>${UI.ms(p.p95)}</span></div>` +
    `<div class="t-row"><span>slowest</span><span>${UI.ms(node.max_ms)}</span></div>` +
    `<div class="t-row"><span>share of plan</span><span>${UI.pct(share)}</span></div>` +
    `<div class="t-row"><span>on critical path</span><span>${UI.pct(node.critical_share)} of runs</span></div>` +
    `<div class="t-hint">measured over ${node.samples} run(s)</div>`);
}

/* ---------- navigating the hierarchy ---------- */

function openPipeline(signature) {
  const p = state.pipelines.find((x) => x.signature === signature);
  if (p) noteVisit('pipeline', signature, pipelineDisplayName(p));
  state.pipeline = signature;
  state.selected = null;
  state.detail = null;
  state.detailId = null;
  invalidate('pipeline-page');
  state.reportFor = null;
  UI.writeRoute({ view: 'pipeline', pipeline: signature, run: '', cmp: '' });
  switchView('pipeline');
  renderPipelinePage();
}

function openRun(id, compareWith = null) {
  const run = state.queries.find((q) => q.query_id === id);
  // Opening a run always establishes which pipeline it belongs to, so the breadcrumb and
  // the prev/next arrows work even when the run was reached from the palette or a link.
  if (run?.signature) state.pipeline = run.signature;
  if (run) noteVisit('run', id, `${DAG.friendlyKind(run.label)} ${UI.clock(run.started_wall)}`);
  state.selected = id;
  state.compareWith = compareWith;
  state.compare = null;
  invalidate('detail');
  UI.writeRoute({ view: 'run', pipeline: state.pipeline || '', run: id, cmp: compareWith || '' });
  switchView('run');
  if (compareWith) switchTab('compare');
  // While the detail is in flight, say "Loading…" rather than "Select a run" — the placeholder
  // is for having chosen nothing, not for a choice that is a beat from arriving.
  if (!state.detail || state.detailId !== id) {
    $('d-title').textContent = 'Loading run…';
    const empty = $('d-empty');
    if (empty) { empty.hidden = false; empty.textContent = 'Loading this run…'; }
  }
  // Load the run's detail *now*, not on the next poll. Idle polling is every 5 s, so without
  // this a click left the run page sitting on "Select a run" for up to five seconds — the
  // single worst piece of navigation latency in the app.
  loadDetail(id);
}

function goUp() {
  if (state.view === 'run' && state.pipeline) openPipeline(state.pipeline);
  else switchView('pipelines');
}

/** Move to the neighbouring run within this pipeline, oldest → newest. */
function stepRun(direction) {
  const runs = [...state.queries.filter((q) => q.signature === state.pipeline)].reverse();
  if (!runs.length) return;
  const at = runs.findIndex((r) => r.query_id === state.selected);
  const next = runs[Math.max(0, Math.min(runs.length - 1, at + direction))];
  if (next && next.query_id !== state.selected) openRun(next.query_id);
}

/* Every other run of this pipeline as a row of little bars, height by duration. Sideways
 * movement without leaving the page — the question "was this one unusual?" answered by
 * looking rather than by clicking back up a level. */
function renderRelated(d) {
  const host = $('related');
  // Oldest → newest, matching every other trend on the page. The API returns them in
  // insertion order, but sort explicitly so this does not depend on that.
  const siblings = (d.siblings || []).slice().sort((a, b) => a.started_wall - b.started_wall);
  if (siblings.length < 2) { host.hidden = true; return; }
  host.hidden = false;
  // The same "bars whose height is a duration" the chart layer already draws. Built there
  // rather than here so the run rail, the pipeline trend, and any future strip share one
  // implementation instead of three that drift.
  host.innerHTML = `<span class="rail-label">This pipeline's runs</span>` +
    CHARTS.strip(siblings.map((r) => ({
      id: r.query_id,
      value: r.total_ms || 0,
      current: r.query_id === d.query_id,
      tone: r.status === 'error' ? 'error' : '',
      label: `${UI.clock(r.started_wall)} ·${r.status === 'error' ? ' failed ·' : ''}`,
    })), { height: 34, format: 'ms', onPick: true });
  // `data-pick` carries the run id; route it through the same opener every other panel uses.
  CHARTS.onPick(host, openRun);
}

function renderRunPosition() {
  const runs = [...state.queries.filter((q) => q.signature === state.pipeline)].reverse();
  const at = runs.findIndex((r) => r.query_id === state.selected);
  $('run-position').textContent = at < 0 ? '—' : `${at + 1} / ${runs.length}`;
  $('run-prev').disabled = at <= 0;
  $('run-next').disabled = at < 0 || at >= runs.length - 1;
  // The compare picker offers the other runs of this pipeline, newest first.
  const options = runs.filter((r) => r.query_id !== state.selected).reverse();
  $('compare-pick').innerHTML = '<option value="">Choose a run…</option>' +
    options.map((r) => (
      `<option value="${UI.esc(r.query_id)}"${r.query_id === state.compareWith ? ' selected' : ''}>` +
      `${UI.clock(r.started_wall)} · ${UI.ms(r.total_ms)}</option>`)).join('');
}

/* ---------- cross-references ---------- */

/* Every panel that names something links to it with a data attribute; this one listener
 * turns all of them into navigation. Delegated rather than bound per render, so a panel
 * that redraws does not leak listeners, and a new link works the moment it is emitted. */
function installCrossReferences() {
  document.addEventListener('click', (event) => {
    const el = event.target.closest('[data-run], [data-pipe], [data-op]');
    if (!el) return;
    if (el.dataset.run) { event.preventDefault(); openRun(el.dataset.run); }
    else if (el.dataset.pipe) { event.preventDefault(); openPipeline(el.dataset.pipe); }
    else if (el.dataset.op != null) {
      event.preventDefault();
      focusStep(Number(el.dataset.op));
    }
  });
  // Keyboard parity for the rows that are buttons in spirit but divs in markup.
  document.addEventListener('keydown', (event) => {
    if (event.key !== 'Enter' && event.key !== ' ') return;
    const el = event.target.closest('[data-pipe], [data-op]');
    if (!el) return;
    event.preventDefault();
    el.click();
  });
}

/* ---------- recently viewed ---------- */
/* The trail *behind* you. The breadcrumb says where you are in the hierarchy; this says
 * where you have been, which is the other half of not getting lost. */

const RECENT_MAX = 6;

function noteVisit(kind, id, label) {
  const trail = (UI.getPref('recent') || []).filter((r) => r.id !== id);
  trail.unshift({ kind, id, label, at: Date.now() });
  UI.setPref('recent', trail.slice(0, RECENT_MAX));
  renderRecentRail();
}

function renderRecentRail() {
  const trail = UI.getPref('recent') || [];
  const host = $('recent-rail');
  if (!host) return;
  host.hidden = trail.length < 2;
  host.innerHTML = trail.length < 2 ? '' :
    `<span class="rail-label">Recent</span>` + trail.map((r) => (
      `<button class="rail-item${r.id === state.selected || r.id === state.pipeline ? ' is-current' : ''}" ` +
      `type="button" data-${r.kind === 'run' ? 'run' : 'pipe'}="${UI.esc(r.id)}">${UI.esc(r.label)}</button>`)).join('');
}

/* ---------- breadcrumb ---------- */

//: The breadcrumb belongs only to the pipeline/run drill-down. Every top-level destination
//: reached from the nav must hide it, or the trail from a run you left keeps claiming you
//: are still inside it while you read the Logs. 'live' and 'learn' were missing.
const TOP_LEVEL_VIEWS = new Set(['pipelines', 'live', 'logs', 'system', 'learn']);

function renderCrumbs() {
  const crumbs = $('crumbs');
  if (TOP_LEVEL_VIEWS.has(state.view)) {
    crumbs.hidden = true;
    return;
  }
  const p = currentPipeline();
  const parts = [`<button class="crumb" data-go="pipelines" type="button">All pipelines</button>`];
  if (p) {
    parts.push(state.view === 'pipeline'
      ? `<span class="crumb is-current" aria-current="page">${UI.esc(pipelineDisplayName(p))}</span>`
      : `<button class="crumb" data-go="pipeline" type="button">${UI.esc(pipelineDisplayName(p))}</button>`);
  }
  if (state.view === 'run' && state.detail) {
    parts.push(`<span class="crumb is-current" aria-current="page">Run at ${UI.clock(state.detail.started_wall)}</span>`);
  }
  crumbs.hidden = false;
  crumbs.innerHTML = parts.join('<span class="crumb-sep" aria-hidden="true">›</span>') +
    `<button class="crumb-up" data-go="up" type="button" title="Up one level (u)">↑ up</button>`;
  for (const b of crumbs.querySelectorAll('[data-go]')) {
    b.addEventListener('click', () => {
      if (b.dataset.go === 'pipelines') switchView('pipelines');
      else if (b.dataset.go === 'pipeline' && state.pipeline) openPipeline(state.pipeline);
      else goUp();
    });
  }
}

/* ---------- run detail ---------- */

async function loadDetail(id) {
  const stale = state.detailId !== id || state.detail?.status === 'running';
  if (stale) {
    try { state.detail = await getJSON(`/api/query/${encodeURIComponent(id)}`); }
    catch { return; }
    state.detailId = id;
  }
  paint('detail', [state.detail, state.compareWith], () => renderDetail(state.detail));
  if (state.compareWith && !state.compare) {
    try {
      state.compare = await getJSON(
        `/api/compare?a=${encodeURIComponent(state.compareWith)}&b=${encodeURIComponent(id)}`);
      VIEWS.comparison(state.compare, 'baseline', 'this run');
    } catch { /* the baseline aged out */ }
  }
}

function renderDetail(d) {
  $('d-empty').hidden = true;
  $('d-title').textContent = DAG.friendlyKind(d.label) || d.query_id;
  $('d-sub').textContent = `${UI.ago(d.started_wall)} · ${UI.clock(d.started_wall)}`;
  VIEWS.verdict(d);
  VIEWS.timeBar(d);
  updateDocumentTitle();
  $('d-story').hidden = false;
  $('d-story').innerHTML = VIEWS.story(d);
  VIEWS.statStrip(d);
  renderPlan(d);
  renderStepsView(d);
  renderQueryView(d);
  VIEWS.operators(d.dag?.nodes || [], focusStep);
  VIEWS.insights(d.insights || [], d.query_id);
  VIEWS.adaptive(d);
  VIEWS.decisions(d.decisions || []);
  renderRunLogs(d);
  VIEWS.meta(d);
  $('raw-json').textContent = JSON.stringify(d, null, 2);
  VIEWS.comparison(state.compare, 'baseline', 'this run');
  renderRunPosition();
  renderRelated(d);
  renderCrumbs();
}

/** Show a step in the graph, from wherever it was named. Every panel that mentions an
 *  operator routes here, so "show me this one" means the same thing everywhere. */
function focusStep(opId) {
  switchStepsView('plan');
  requestAnimationFrame(() => DAG.select(Number(opId)));
}

/* The step renderings other than the graph. Each is cheap, but a run with 200 operators
 * draws three of them for nothing if the reader is looking at the fourth — so only the
 * visible one is built, and switching rebuilds it. */
function renderStepsView(d) {
  const nodes = d.dag?.nodes || [];
  if (state.stepsView === 'timeline') VIEWS.timeline(nodes);
  else if (state.stepsView === 'stages') PLAN.stages('stages', d.dag, focusStep);
  else if (state.stepsView === 'flame') PLAN.flame('flame', nodes, focusStep);
}

/* The plan as a document: the text tree, the optimizer's diff, or the raw IR. */
function renderQueryView(d) {
  const diff = d.plan_diff || {};
  const badge = $('diff-count');
  if (badge) {
    const primary = (diff.changes || []).filter((c) => c.primary).length;
    badge.hidden = !primary;
    badge.textContent = primary;
  }
  if (state.queryView === 'explain') renderExplain(d);
  else if (state.queryView === 'diff') {
    PLAN.diff('plan-diff', diff, { showAll: state.diffShowAll });
  } else {
    PLAN.ir('ir', state.irOriginal ? d.profile?.logical_ir : d.profile?.optimized_ir);
  }
}

function renderExplain(d) {
  const rows = state.explainOriginal ? d.logical_explain : d.explain;
  const note = $('explain-note');
  if (note) {
    note.innerHTML = state.explainOriginal
      ? 'The plan <b>as written</b>, before the optimizer touched it. It carries no timings — ' +
        'this shape never ran.'
      : 'The plan <b>as run</b>, with each step’s measured time. Click a row to open it in ' +
        'the graph; the bar is its share of total step time.';
  }
  PLAN.explain('explain', rows || [], {
    needle: state.explainNeedle,
    collapsed: state.explainCollapsed,
    // The graph and the text tree share one selection: picking a step in either must light
    // it in the other, or they read as two unrelated views of unrelated plans.
    selected: DAG.selectedNode()?.op_id ?? null,
  });
}

/* Selection is shared between the graph and the ranked cost table: choosing a step in one
 * must highlight it in the other, or the two panels read as unrelated views of unrelated
 * data. Both routes land here. */
function selectStep(node, stats) {
  renderInspector(node, stats);
  VIEWS.costliest(state.detail?.dag?.nodes, selectFromCostTable,
                  node ? node.op_id : null);
}
function selectFromCostTable(opId) {
  DAG.selectOnly(opId);
  // Bring the step into view, but only when the user has not asked us to stop moving the
  // viewport out from under them — auto-pan is helpful until it fights you.
  if (UI.getPref('dagMotion') !== 'none') DAG.reveal(opId);
}

function renderPlan(d) {
  const legend = DAG.render($('dag'), d.dag, {
    onHover: showNodeTip, onLeave: hideTip,
    minimap: UI.getPref('showMinimap') ? $('minimap') : null,
    onZoom: (k) => { $('dag-zoom').textContent = `${Math.round(k * 100)}%`; },
    onSelect: selectStep,
  }, d.query_id);
  VIEWS.costliest(d.dag ? d.dag.nodes : [], selectFromCostTable, null);
  $('minimap').hidden = !UI.getPref('showMinimap');
  const host = $('dag-legend');
  if (!d.dag || !d.dag.nodes.length) {
    host.innerHTML = '<span>No plan graph for this run.</span>';
    $('inspector').innerHTML = '';
    return;
  }
  host.innerHTML = legend.map((e) => (
    `<span><i class="swatch${e.swatch === 'ramp' ? ' ramp' : ''}${e.cls === 'crit' ? ' crit' : ''}"` +
    `${e.color ? ` style="background:${e.color}"` : ''}></i>${UI.esc(e.label)}</span>`)).join('');
  selectStep(null);
}

function renderInspector(node, stats) {
  const host = $('inspector');
  if (!node) {
    const nodes = state.detail?.dag?.nodes || [];
    const critical = new Set(state.detail?.dag?.critical_path || []);
    const chain = nodes.filter((n) => critical.has(n.op_id));
    const chainMs = chain.reduce((s, n) => s + (n.elapsed_ms || 0), 0);
    host.innerHTML =
      `<div class="insp-empty"><h3>${VIEWS.term('critical path', 'Critical path')}</h3>` +
      (chain.length
        ? `<p class="insp-lede">These ${chain.length} steps feed each other end to end, so their ` +
          `<b>${UI.ms(chainMs)}</b> is the floor on this query's time. Speeding up anything off ` +
          `this chain changes nothing.</p><ol class="chain">` +
          chain.map((n) => (
            `<li><span class="chain-name">${UI.esc(DAG.friendlyKind(n.kind))}</span>` +
            `<span class="chain-val mono">${UI.ms(n.elapsed_ms)}</span></li>`)).join('') + `</ol>`
        : `<p class="insp-lede">No timing recorded for this plan.</p>`) +
      `<p class="insp-hint">Click any step to inspect it, or use the arrow keys to walk it.</p></div>`;
    return;
  }
  const rows = [
    ['Rows in', node.rows_in ? UI.count(node.rows_in) : '—'],
    ['Rows out', UI.count(node.rows_out)],
    ['Selectivity', UI.pct(node.selectivity)],
    ['Time', UI.ms(node.elapsed_ms)],
    ['Share', stats ? UI.pct(node.elapsed_ms / stats.opTotal) : '—'],
    ['Rows expected', node.est_rows == null ? 'not estimated' : UI.count(node.est_rows)],
    ['Actual vs expected', node.est_error == null ? '—' : `${node.est_error.toFixed(2)}x`],
    ['Estimate source', node.provenance || '—'],
    ['Algorithm', node.algorithm || '—'],
    ['Backend', node.backend || 'cpu'],
    ['Threads', node.threads || '—'],
    ['CPU', UI.cpuMeasured(node) ? UI.pct(node.cpu_util) : 'not measured'],
    ['Output size', UI.bytes(node.result_bytes)],
    ['Peak memory', UI.bytes(node.peak_rss_bytes)],
    ['Spilled', node.spilled ? UI.bytes(node.spill_bytes) : 'no'],
  ];
  const bad = node.est_error != null && (node.est_error > 10 || node.est_error < 0.1);
  host.innerHTML =
    `<div class="insp-head"><h3>${UI.esc(DAG.friendlyKind(node.kind))}</h3>` +
    `<span class="insp-tag mono" data-copyable tabindex="0" aria-label="Copy this step id">op ${node.op_id}</span></div>` +
    (node.detail ? `<p class="insp-detail mono">${UI.esc(node.detail)}</p>` : '') +
    (node.on_critical_path ? `<p class="chip is-crit">on the critical path</p>` : '') +
    (bad ? `<p class="chip is-warn">the planner misjudged this by ${node.est_error.toFixed(0)}x</p>` : '') +
    (node.spilled ? `<p class="chip is-serious">spilled ${UI.bytes(node.spill_bytes)}</p>` : '') +
    `<dl class="insp-rows">` + rows.map(([k, v]) => (
      `<div class="meta-row"><dt>${UI.esc(k)}</dt><dd>${UI.esc(String(v))}</dd></div>`)).join('') + `</dl>` +
    // Selecting a step should answer "what is this and why is it slow", not only restate its
    // numbers — that question is the whole reason someone clicked it.
    LEARN.explainOperator(node.kind);
}

function showNodeTip(event, node, share) {
  // The hover card carries the whole per-step picture a mouse user would otherwise have to
  // click through for: rows in and out, how selective the step was, spill, and how far the
  // planner's estimate was off — the one number that explains a wrong plan.
  const row = (label, value) => `<div class="t-row"><span>${label}</span><span>${value}</span></div>`;
  const parts = [
    `<b>${UI.esc(DAG.friendlyKind(node.kind))}</b>` +
      (node.detail ? ` <span class="mono" style="opacity:.7">${UI.esc(node.detail)}</span>` : ''),
  ];
  if (node.rows_in) parts.push(row('rows in', UI.count(node.rows_in)));
  parts.push(row('rows out', UI.count(node.rows_out)));
  if (node.selectivity != null && node.rows_in) {
    parts.push(row('kept', UI.pct(node.selectivity)));
  }
  parts.push(row('time', `${UI.ms(node.elapsed_ms)} · ${UI.pct(share)}`));
  if (node.spilled) parts.push(row('spilled', `<span class="is-serious">${UI.bytes(node.spill_bytes)}</span>`));
  if (node.est_error != null && (node.est_error > 10 || node.est_error < 0.1)) {
    parts.push(row('estimate', `<span class="is-warn">${node.est_error.toFixed(0)}x off</span>`));
  }
  if (node.on_critical_path) parts.push(`<div class="t-hint"><span class="crit-dot"></span>on the critical path</div>`);
  parts.push('<div class="t-hint">click to inspect</div>');
  showTip(event, parts.join(''));
}

function renderRunLogs(d) {
  const start = d.started_wall;
  const end = start + (d.total_ms || 0) / 1000 + 0.25;
  const lines = state.logLines.filter((l) => l.wall >= start - 0.25 && l.wall <= end);
  $('run-logs').innerHTML = lines.length
    ? `<p class="hint">Lines written while this run was executing.</p>` +
      lines.map((l, i) => VIEWS.logLine(l, i)).join('')
    : '<p class="empty">No log lines during this run. Raise the engine’s verbosity to see ' +
      'optimizer decisions — <span class="mono">observability.verbosity = "verbose"</span>.</p>';
}

/* ---------- logs ---------- */


function ingestLogs(payload) {
  if (payload.cursor === state.logCursor && state.logLines.length) return;
  state.logCursor = payload.cursor;
  const arrived = payload.lines.length;
  // Stamp each line with its position in the unfiltered stream, so a context view can pull
  // the neighbours a filter hid.
  let seq = state.logLines.length;
  for (const line of payload.lines) line.seq = seq++;
  state.logLines.push(...payload.lines);
  if (state.logLines.length > 5000) state.logLines.splice(0, state.logLines.length - 5000);

  // Reading a specific line while new ones land underneath is how a log viewer loses the
  // reader's place. When they are not following the tail, buffer behind a button instead.
  const following = $('log-follow')?.checked && !state.logRange;
  if (!following && arrived) {
    state.logPending += arrived;
    const btn = $('log-new');
    if (btn) {
      btn.hidden = false;
      btn.textContent = `${state.logPending} new line${state.logPending === 1 ? '' : 's'} — jump to newest`;
    }
    return;   // hold the render; the buffer button is the only thing that moved
  }
  invalidate('logs');
  renderLogs();
}

function filteredLogs() {
  const min = LEVELS[$('log-level').value] || 20;
  const raw = $('log-filter').value.trim();
  let test = () => true;
  if (raw) {
    if (UI.getPref('logRegex')) {
      try { const re = new RegExp(raw, 'i'); test = (t) => re.test(t); $('log-filter').classList.remove('is-bad'); }
      catch { $('log-filter').classList.add('is-bad'); test = () => true; }
    } else {
      const n = raw.toLowerCase();
      test = (t) => t.toLowerCase().includes(n);
    }
  }
  return state.logLines.filter((l) => (LEVELS[l.level] || 20) >= min &&
    test(`${l.logger} ${l.message} ${JSON.stringify(l.fields)}`));
}

/* Extra narrowing that lives only in the browser: a time window picked off the histogram
 * and a text filter over the already-loaded lines. Kept out of `filteredLogs()` so the
 * histogram keeps showing the shape of the *fetched* set rather than collapsing to whatever
 * is currently selected — a histogram that only shows the selection cannot be used to widen
 * it again. */
function logsAfterLocalFilters(lines) {
  let out = lines;
  if (state.logRange) {
    const [from, to] = state.logRange;
    out = out.filter((l) => l.wall >= from && l.wall < to);
  }
  const needle = ($('log-in-results')?.value || '').trim().toLowerCase();
  if (needle) {
    out = out.filter((l) => (
      String(l.message || '').toLowerCase().includes(needle) ||
      String(l.logger || '').toLowerCase().includes(needle) ||
      Object.entries(l.fields || {}).some(([k, v]) =>
        `${k}=${v}`.toLowerCase().includes(needle))
    ));
  }
  for (const [key, value] of state.logFields) {
    out = out.filter((l) => String((l.fields || {})[key] ?? '') === value);
  }
  return out;
}

function renderLogs() {
  const fetched = filteredLogs();
  const lines = logsAfterLocalFilters(fetched);
  const key = [fetched.length, lines.length, $('log-level').value, $('log-filter').value,
               UI.getPref('logRegex'), state.logRange ? state.logRange.join(',') : '',
               $('log-in-results')?.value || '', [...state.logFields].join('|')];
  paint('logs', key, () => {
    $('log-count').textContent = fetched.length;
    // The histogram is built from the level/search-filtered set, not the locally narrowed
    // one, so the bar you clicked stays visible and clickable-away.
    state.logHistoGeom = VIEWS.logHistogram(fetched, { onPickRange: pickLogRange });
    renderLogFilterChips();
    const shown = $('log-shown');
    if (shown) {
      shown.textContent = lines.length === fetched.length
        ? `${fetched.length} line${fetched.length === 1 ? '' : 's'}`
        : `${lines.length} of ${fetched.length}`;
    }
    const host = $('logs');
    // Only the newest slice is put in the DOM. A busy session retains thousands of lines,
    // and rendering all of them costs a repaint on every poll to draw rows nobody has
    // scrolled to. The cap is stated when it bites, because a list that silently stops is
    // a list that claims to be complete.
    const shownLines = lines.length > LOG_RENDER_CAP ? lines.slice(-LOG_RENDER_CAP) : lines;
    const truncated = lines.length - shownLines.length;
    host.innerHTML = lines.length
      ? (truncated
          ? `<p class="log-truncated">Showing the newest ${LOG_RENDER_CAP.toLocaleString()} of ` +
            `${lines.length.toLocaleString()} matching lines. Narrow the filter or pick a ` +
            `window on the histogram to see the earlier ones.</p>`
          : '') +
        shownLines.map((l, i) => VIEWS.logLine(l, i)).join('')
      : '<p class="empty">No lines match. Lower the level, clear the search, or raise the engine’s ' +
        'verbosity — <span class="mono">observability.verbosity = "verbose"</span> shows what the ' +
        'optimizer decided.</p>';
    if ($('log-follow').checked && !state.logRange) host.scrollTop = host.scrollHeight;
    revealLinkedLine();
  });
}

/* Scroll to the line a copied link named, once, when it first appears.
 *
 * Once: the log list re-renders on every poll, and re-scrolling to an old line each time
 * would fight a reader who has since scrolled somewhere else. */
function revealLinkedLine() {
  const seq = UI.readRoute().line;
  if (!seq || state.linkedShown === seq) return;
  const row = document.getElementById(`L${seq}`);
  if (!row) return;                 // not in the fetched window yet; try again next render
  state.linkedShown = seq;
  // Following the tail would immediately scroll away from the line just landed on.
  $('log-follow').checked = false;
  row.classList.add('is-linked');
  row.scrollIntoView({ block: 'center' });
  announce(`Jumped to the linked log line`);
}

/* Drag across the histogram to select a time window.
 *
 * A 3px threshold separates a drag from a click, so clicking a single bar still works. The
 * selection rectangle is one div positioned in percentages, and the window is derived from
 * the pixel positions inverted through the SVG's own width — the same coordinate the bars
 * were laid out on, so the selection lines up with what the reader sees. */
function installHistoDrag() {
  const svg = $('log-histo');
  if (!svg) return;
  const wrap = $('log-histo-wrap');
  let startX = null, band = null, geom = null;

  const frac = (clientX) => {
    const box = svg.getBoundingClientRect();
    return Math.max(0, Math.min(1, (clientX - box.left) / (box.width || 1)));
  };

  svg.addEventListener('pointerdown', (e) => {
    geom = state.logHistoGeom;
    if (!geom) return;
    startX = e.clientX;
    svg.setPointerCapture(e.pointerId);
    band = document.createElement('div');
    band.className = 'log-histo-band';
    wrap.appendChild(band);
  });
  svg.addEventListener('pointermove', (e) => {
    if (startX == null) return;
    const a = frac(startX), b = frac(e.clientX);
    band.style.left = `${Math.min(a, b) * 100}%`;
    band.style.width = `${Math.abs(b - a) * 100}%`;
  });
  svg.addEventListener('pointerup', (e) => {
    if (startX == null) return;
    const moved = Math.abs(e.clientX - startX);
    const a = frac(startX), b = frac(e.clientX);
    if (band) { band.remove(); band = null; }
    const wasStart = startX;
    startX = null;
    // Under the 3px threshold this was a click; let the bar's own handler take it.
    if (moved < 3 || !geom) return;
    const lo = geom.min + Math.min(a, b) * (geom.max - geom.min);
    const hi = geom.min + Math.max(a, b) * (geom.max - geom.min);
    if (hi > lo) pickLogRange(lo, hi);
  });
}

/** Narrow to a bucket picked off the histogram. */
/* Show the lines immediately around one line, ignoring the active filters. The filters that
 * surfaced a line are usually the wrong ones for understanding it — the neighbouring DEBUG
 * lines a level filter hid are often exactly what explains it. Rendered inline, dismissable. */
function showLogContext(seq) {
  const CONTEXT = 12;
  const around = state.logLines
    .filter((l) => l.seq != null && Math.abs(l.seq - seq) <= CONTEXT)
    .sort((a, b) => a.seq - b.seq);
  if (!around.length) return;
  const host = $('logs');
  const anchor = host.querySelector(`[data-seq="${seq}"]`);
  const panel = document.createElement('div');
  panel.className = 'log-context-panel';
  panel.innerHTML =
    `<div class="log-context-head">Lines around this one, ignoring filters` +
    `<button class="linkish" type="button" data-close-context>close</button></div>` +
    around.map((l) => VIEWS.logLine(l, l.seq, { anchored: false })).join('');
  panel.querySelector('[data-close-context]').addEventListener('click', () => panel.remove());
  // Mark the origin line inside the panel so the reader keeps their place.
  const marked = panel.querySelector(`[data-seq="${seq}"]`);
  if (marked) marked.classList.add('is-linked');
  if (anchor) anchor.after(panel); else host.appendChild(panel);
  marked?.scrollIntoView?.({ block: 'center' });
}

function pickLogRange(from, to) {
  state.logRange = [from, to];
  $('log-histo-clear').hidden = false;
  // Following the tail while pinned to a past window fights the reader.
  $('log-follow').checked = false;
  UI.setPref('logFollow', false);
  invalidate('logs');
  renderLogs();
  announce(`Filtered to ${UI.clock(from)} – ${UI.clock(to)}`);
}

function clearLogRange() {
  state.logRange = null;
  $('log-histo-clear').hidden = true;
  invalidate('logs');
  renderLogs();
}

/** Applied field filters render as removable chips, so an active filter is never invisible. */
function renderLogFilterChips() {
  const host = $('log-active-filters');
  if (!host) return;
  host.innerHTML = [...state.logFields].map(([k, v]) => (
    `<button class="status-chip is-on" type="button" data-drop-field="${UI.esc(k)}">` +
    `${UI.esc(k)}=${UI.esc(v)} ×</button>`)).join('');
  for (const b of host.querySelectorAll('[data-drop-field]')) {
    b.addEventListener('click', () => {
      state.logFields.delete(b.dataset.dropField);
      invalidate('logs');
      renderLogs();
    });
  }
}

/* ---------- tooltip ---------- */

function showTip(event, html) {
  const tip = $('tip');
  tip.innerHTML = html;
  tip.hidden = false;
  const x = Math.min(event.clientX + 16, window.innerWidth - tip.offsetWidth - 8);
  const y = Math.min(event.clientY + 16, window.innerHeight - tip.offsetHeight - 8);
  tip.style.left = `${Math.max(8, x)}px`;
  tip.style.top = `${Math.max(8, y)}px`;
}
const hideTip = () => { $('tip').hidden = true; };

/* ---------- navigation ---------- */

//: How deep each destination sits, so a transition can tell "drilled in" from "went back".
const VIEW_DEPTH = { pipelines: 0, live: 0, logs: 0, system: 0, learn: 0, pipeline: 1, run: 2 };

function switchView(view) {
  // Direction encodes the move: down slides in from the right, up from the left. The
  // animation then reports which way you went rather than only that something changed.
  const from = VIEW_DEPTH[state.view] ?? 0;
  const to = VIEW_DEPTH[view] ?? 0;
  document.body.classList.toggle('nav-down', to > from);
  document.body.classList.toggle('nav-up', to < from);
  state.view = view;
  // Only the three top-level destinations light a nav tab; pipeline and run are drill-downs
  // that the breadcrumb tracks instead, so the nav never claims you are somewhere you left.
  for (const t of document.querySelectorAll('.viewtab')) {
    const on = t.dataset.view === view;
    t.classList.toggle('is-active', on);
    t.setAttribute('aria-selected', String(on));
  }
  for (const s of document.querySelectorAll('.view')) s.classList.toggle('is-active', s.id === `view-${view}`);
  moveInk('.viewnav');
  // The reference is static content, so it is drawn on arrival rather than by the poll loop.
  if (view === 'learn') LEARN.renderLearn($('learn-body'), $('learn-search').value);
  // The live page is skipped by the poll loop while it is hidden, so arriving at it has to
  // draw once rather than wait up to five seconds for the next tick.
  if (view === 'live') { invalidate('live'); renderLive(); }
  UI.writeRoute({ view }, { replace: true });
  renderCrumbs();
  announce(`${view.replace('pipelines', 'all pipelines')} view`);
  // A new destination starts at its top. Without this, drilling from a lane you scrolled to
  // into its pipeline page left you halfway down that page, past the header and the plan.
  window.scrollTo(0, 0);
  // Move focus to the new page's heading. Without this a keyboard user stays parked on the
  // link they clicked and has to tab back through the chrome to reach the content.
  const heading = $(`view-${view}`)?.querySelector('h1, h2');
  if (heading) { heading.setAttribute('tabindex', '-1'); heading.focus({ preventScroll: true }); }
  if (view === 'run' && state.detail) requestAnimationFrame(() => DAG.fit());
  updateDocumentTitle();
}

/** The browser tab title, tracking where you are. Several runs open in tabs are otherwise
 *  indistinguishable, and a bookmark of "Batcher" tells you nothing. */
function updateDocumentTitle() {
  const parts = ['Batcher'];
  if (state.view === 'pipeline' && currentPipeline()) {
    parts.unshift(pipelineDisplayName(currentPipeline()));
  } else if (state.view === 'run' && state.detail) {
    parts.unshift(`${DAG.friendlyKind(state.detail.label)} · ${UI.ms(state.detail.total_ms)}`);
  } else if (state.view !== 'pipelines') {
    parts.unshift(state.view.charAt(0).toUpperCase() + state.view.slice(1));
  }
  document.title = parts.join(' — ');
}

/** Announce a navigation to assistive tech, which cannot see the transition. */
function announce(message) {
  const region = $('live-region');
  if (region) region.textContent = message;
}

/* A tab owns one or more panes. `Steps` owns whichever of the three step renderings is
 * selected; `Findings` and `Details` each own two panes that used to be separate tabs and
 * were never worth the navigation. */
const TAB_PANES = {
  steps: () => [`tab-${state.stepsView}`],
  query: () => [`tab-${state.queryView}`],
  insights: () => ['tab-insights', 'tab-adaptive', 'tab-decisions'],
  compare: () => ['tab-compare'],
  logs: () => ['tab-logs'],
  meta: () => ['tab-meta', 'tab-raw'],
};

//: Which rendering switch belongs above which tab. A switch shown over a tab it does not
//: control is a control that appears to do nothing, which is worse than an absent one.
const TAB_SWITCH = { steps: 'steps-switch', query: 'query-switch' };

const STEPS_VIEWS = ['plan', 'stages', 'flame', 'timeline', 'operators'];
const QUERY_VIEWS = ['explain', 'diff', 'ir'];

function switchTab(tab) {
  state.tab = TAB_PANES[tab] ? tab : 'steps';
  for (const t of document.querySelectorAll('.tab')) {
    const on = t.dataset.tab === state.tab;
    t.classList.toggle('is-active', on);
    t.setAttribute('aria-selected', String(on));
  }
  const shown = new Set(TAB_PANES[state.tab]());
  for (const p of document.querySelectorAll('.tabpane')) {
    p.classList.toggle('is-active', shown.has(p.id));
  }
  moveInk('.tabs');
  for (const [owner, id] of Object.entries(TAB_SWITCH)) {
    const sw = $(id);
    if (sw) sw.hidden = state.tab !== owner;
  }
  UI.writeRoute({ tab: state.tab }, { replace: true });
  if (state.tab === 'steps' && state.stepsView === 'plan') {
    requestAnimationFrame(() => DAG.fit());
  }
}

/* Put the travelling indicator under the active tab.
 *
 * Measured from the tab's own box rather than computed from an index, so it stays correct
 * when a tab's label changes width, the density switch fires, or a narrow window wraps the
 * row. Deferred to the next frame because a tab that has just become visible has no layout
 * yet and would measure as zero-width. */
function moveInk(container) {
  const nav = typeof container === 'string' ? document.querySelector(container) : container;
  const ink = nav?.querySelector('.viewnav-ink, .tabs-ink');
  if (!ink) return;
  requestAnimationFrame(() => {
    const active = nav.querySelector('.is-active');
    if (!active || !active.offsetWidth) { ink.style.opacity = '0'; return; }
    ink.style.transform = `translateX(${active.offsetLeft}px)`;
    ink.style.width = `${active.offsetWidth}px`;
    ink.style.opacity = '1';
  });
}

/** Keep both indicators in step with a resize, which changes every offset they read. */
function installInkBars() {
  const reposition = () => { moveInk('.viewnav'); moveInk('.tabs'); };
  window.addEventListener('resize', UI.debounce(reposition, 80));
  // A late web font changes every tab's width after the first paint.
  if (document.fonts?.ready) document.fonts.ready.then(reposition);
  reposition();
}

/** Mark one segmented control's buttons, so pressed state and ARIA never disagree. */
function markSwitch(group, attr, value) {
  for (const b of document.querySelectorAll(`#${group} .seg`)) {
    const on = b.dataset[attr] === value;
    b.classList.toggle('is-on', on);
    b.setAttribute('aria-pressed', String(on));
  }
}

/** Pick which rendering of the steps to show. The tab does not change — the subject is the
 *  same, only the view of it. */
function switchStepsView(view) {
  state.stepsView = STEPS_VIEWS.includes(view) ? view : 'plan';
  UI.setPref('stepsView', state.stepsView);
  markSwitch('steps-switch', 'steps', state.stepsView);
  switchTab('steps');
  // Stages and flame are drawn from the detail document rather than by the poll loop, so
  // arriving at one for the first time has to draw it.
  if (state.detail) renderStepsView(state.detail);
}

/** Pick which rendering of the plan *document* to show. */
function switchQueryView(view) {
  state.queryView = QUERY_VIEWS.includes(view) ? view : 'explain';
  UI.setPref('queryView', state.queryView);
  markSwitch('query-switch', 'query', state.queryView);
  switchTab('query');
  if (state.detail) renderQueryView(state.detail);
}

/** Render the shortcut sheet from the one registry, so it cannot drift from what the keys
 *  actually do. Chords like `g p` render as separate keycaps with a "then". */
function renderShortcuts() {
  const host = $('shortcuts-list');
  if (!host) return;
  const caps = (keys) => keys.split(' ')
    .map((k) => `<kbd>${UI.esc(k === 'space' ? 'space' : k)}</kbd>`)
    .join(keys.includes(' ') ? ' <span class="kbd-then">then</span> ' : ' ');
  host.innerHTML = ACTIONS.filter((a) => a.keys)
    .map((a) => `<div>${caps(a.keys)}<span>${UI.esc(a.label)}</span></div>`)
    .join('');
}

/* Row+column highlight on the run grid.
 *
 * One delegated listener and one CSS rule — deliberately no re-render. Moving the pointer
 * across a grid of hundreds of cells must not cost a repaint, so this toggles classes on the
 * matching cells directly rather than routing through the render path.
 *
 * `CSS.escape` because a step id or run id is data, and an id containing a quote would
 * otherwise turn a selector into a syntax error. */
function installGridCrosshair() {
  const host = $('p-grid');
  if (!host) return;
  let lastRun = null, lastStep = null;

  const clear = () => {
    for (const el of host.querySelectorAll('.xhair-row, .xhair-col')) {
      el.classList.remove('xhair-row', 'xhair-col');
    }
    lastRun = lastStep = null;
  };

  host.addEventListener('pointerover', (e) => {
    const cell = e.target.closest('[data-col][data-step]');
    if (!cell) return;
    const col = cell.dataset.col, step = cell.dataset.step;
    if (col === lastRun && step === lastStep) return;   // same cell, nothing moved
    clear();
    lastRun = col; lastStep = step;
    const esc = (v) => (window.CSS && CSS.escape ? CSS.escape(v) : v);
    // Match on the column index, which both the cells and the header carry — a query id is
    // on the cell but not consistently on the header, so keying on it lit only half the column.
    for (const el of host.querySelectorAll(`[data-col="${esc(col)}"]`)) el.classList.add('xhair-col');
    for (const el of host.querySelectorAll(`[data-step="${esc(step)}"]`)) el.classList.add('xhair-row');
  });
  host.addEventListener('pointerleave', clear);
  // Enter or Space on a focused cell opens its run, the same as a click.
  host.addEventListener('keydown', (e) => {
    if (e.key !== 'Enter' && e.key !== ' ') return;
    const cell = e.target.closest('.grid-cell[data-run]');
    if (!cell) return;
    e.preventDefault();
    cell.click();
  });
}

/* Arrow keys move within a segmented control, activating as they go — the toolbar/radiogroup
 * idiom. A screen-reader user expects left/right to switch options inside a `role="group"`,
 * not to have to Tab through each. */
function installSegmentedGroups() {
  for (const group of document.querySelectorAll('.segmented, .pane-switch')) {
    group.addEventListener('keydown', (e) => {
      if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End'].includes(e.key)) return;
      const segs = [...group.querySelectorAll('.seg:not(:disabled)')]
        .filter((b) => !b.classList.contains('is-label'));
      const at = segs.indexOf(document.activeElement);
      if (at < 0) return;
      e.preventDefault();
      let to = at;
      if (e.key === 'ArrowRight' || e.key === 'ArrowDown') to = (at + 1) % segs.length;
      else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') to = (at - 1 + segs.length) % segs.length;
      else if (e.key === 'Home') to = 0;
      else if (e.key === 'End') to = segs.length - 1;
      segs[to].focus();
      segs[to].click();
    });
  }
}

/* Anything monospaced is an identifier someone will want to paste — a query id, a plan
 * signature, a step detail. Rather than adding a copy button beside each, one delegated
 * handler makes them all copyable, and a `data-copy-value` attribute overrides the visible
 * text when the two differ (a truncated id must copy in full). */
function installCopyAnywhere() {
  const grab = (el) => {
    // The visible text may be truncated; `data-copy-value` carries the whole thing, so a
    // 12-character preview of a plan signature still copies the signature.
    const text = el.dataset.copyValue || el.textContent.trim().replace(/…$/, '');
    UI.copy(text, 'Copied');
  };
  document.addEventListener('click', (e) => {
    const el = e.target.closest('[data-copyable]');
    if (!el) return;
    // Never hijack a selection: a reader highlighting part of a value wants that, not a copy.
    if (String(window.getSelection?.() || '').length) return;
    grab(el);
  });
  // These carry `tabindex="0"` and so are in the focus order. Something focusable that only
  // responds to a mouse is a keyboard trap in miniature: you can reach it and not use it.
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Enter' && e.key !== ' ') return;
    const el = e.target.closest('[data-copyable]');
    if (!el) return;
    e.preventDefault();
    grab(el);
  });
}

/* ---------- the query viewer ---------- */

/* Everything the Explain / Diff / IR panes need, bound once by delegation. Their contents
 * are rebuilt on every navigation between runs, so a per-element binding would go stale on
 * the first redraw and the controls would silently stop working. */
function installQueryViewer() {
  on('explain-search', 'input', UI.debounce((e) => {
    state.explainNeedle = e.target.value;
    if (state.detail) renderExplain(state.detail);
  }, 90));
  on('explain-original', 'change', (e) => {
    state.explainOriginal = e.target.checked;
    // Collapsing is keyed by op_id, and the two plans number their operators differently.
    // Carrying the set across would fold an unrelated subtree.
    state.explainCollapsed.clear();
    if (state.detail) renderExplain(state.detail);
  });
  on('explain-expand', 'click', () => {
    const anyClosed = state.explainCollapsed.size > 0;
    state.explainCollapsed.clear();
    if (!anyClosed) {
      // Collapse to the roots: every operator that has something nested under it.
      const rows = (state.explainOriginal ? state.detail?.logical_explain : state.detail?.explain) || [];
      for (let i = 1; i < rows.length; i += 1) {
        if (rows[i].depth > rows[i - 1].depth && rows[i - 1].depth > 0) {
          state.explainCollapsed.add(rows[i - 1].op_id);
        }
      }
    }
    $('explain-expand').textContent = state.explainCollapsed.size ? 'Expand all' : 'Collapse all';
    if (state.detail) renderExplain(state.detail);
  });
  on('explain-copy', 'click', copyExplain);

  const explainHost = $('explain');
  if (explainHost) {
    explainHost.addEventListener('click', (e) => {
      const twisty = e.target.closest('[data-collapse]');
      if (!twisty) return;
      e.stopPropagation();
      const id = Number(twisty.dataset.collapse);
      if (state.explainCollapsed.has(id)) state.explainCollapsed.delete(id);
      else state.explainCollapsed.add(id);
      if (state.detail) renderExplain(state.detail);
    });
  }

  on('plan-diff', 'click', (e) => {
    if (!e.target.closest('[data-show-all]')) return;
    state.diffShowAll = true;
    if (state.detail) renderQueryView(state.detail);
  });

  for (const [id, original] of [['ir-optimized', false], ['ir-logical', true]]) {
    on(id, 'click', () => {
      state.irOriginal = original;
      $('ir-optimized').classList.toggle('is-on', !original);
      $('ir-logical').classList.toggle('is-on', original);
      $('ir-optimized').setAttribute('aria-pressed', String(!original));
      $('ir-logical').setAttribute('aria-pressed', String(original));
      if (state.detail) renderQueryView(state.detail);
    });
  }
  on('ir-copy', 'click', () => {
    const doc = state.irOriginal ? state.detail?.profile?.logical_ir : state.detail?.profile?.optimized_ir;
    if (!doc) return UI.toast('No plan document for this run', 'warn');
    UI.copy(JSON.stringify(doc, null, 2), 'Plan document copied');
  });
}

function copyExplain() {
  const rows = (state.explainOriginal ? state.detail?.logical_explain : state.detail?.explain) || [];
  if (!rows.length) return UI.toast('No plan to copy', 'warn');
  UI.copy(PLAN.explainText(rows), 'Plan copied as text');
}

/* Every chart raises `chart-hover` / `chart-leave` rather than reaching for the tooltip
 * itself, so the plots stay renderers and there is one tooltip on the page. */
function installChartTooltips() {
  document.addEventListener('chart-hover', (e) => showTip(e.detail.event, e.detail.html));
  document.addEventListener('chart-leave', hideTip);
}

/* ---------- command palette ---------- */
/* One keystroke to anything on the page. Power users stop hunting through views, and it
 * doubles as a discoverability surface: every action is listed with its shortcut. */

const ACTIONS = [
  { id: 'view-pipelines', label: 'All pipelines', keys: 'g p', run: () => switchView('pipelines') },
  { id: 'view-live', label: 'Live — what is running now', keys: 'g r', run: () => switchView('live') },
  { id: 'view-logs', label: 'Logs', keys: 'g l', run: () => switchView('logs') },
  { id: 'view-system', label: 'System', keys: 'g s', run: () => switchView('system') },
  { id: 'up', label: 'Up one level', keys: 'u', run: goUp },
  { id: 'prev-run', label: 'Previous run in this pipeline', keys: 'j', run: () => stepRun(-1) },
  { id: 'next-run', label: 'Next run in this pipeline', keys: 'k', run: () => stepRun(1) },
  { id: 'refresh', label: 'Refresh now', keys: 'r', run: () => { invalidateAll(); poll(); UI.toast('Refreshed'); } },
  { id: 'pause', label: 'Pause / resume auto-refresh', keys: 'space', run: togglePause },
  { id: 'theme', label: 'Toggle light / dark theme', keys: 't', run: toggleTheme },
  { id: 'density', label: 'Toggle compact density', keys: 'd', run: toggleDensity },
  { id: 'layout', label: 'Switch pipelines between cards and table', keys: 'v',
    run: () => setPipelineLayout(state.pipelineLayout === 'cards' ? 'table' : 'cards') },
  { id: 'focus-search', label: 'Focus the search box', keys: '/', run: focusActiveSearch },
  { id: 'shortcuts', label: 'Show keyboard shortcuts', keys: '?', run: () => showModal('shortcuts') },
  { id: 'help', label: 'Toggle inline explanations', keys: 'e', run: toggleHelp },
  { id: 'fit', label: 'Fit plan to view', keys: 'f', run: () => DAG.fit() },
  { id: 'steps-graph', label: 'Show steps as a graph', run: () => switchStepsView('plan') },
  { id: 'steps-stages', label: 'Show steps grouped into pipeline stages', run: () => switchStepsView('stages') },
  { id: 'steps-flame', label: 'Show steps as a flame graph', run: () => switchStepsView('flame') },
  { id: 'steps-timeline', label: 'Show steps ranked by duration', run: () => switchStepsView('timeline') },
  { id: 'steps-table', label: 'Show steps as a table', run: () => switchStepsView('operators') },
  { id: 'explain', label: 'Explain this query (the plan as text)', keys: 'x',
    run: () => switchQueryView('explain') },
  { id: 'plan-diff', label: 'Show what the optimizer changed', run: () => switchQueryView('diff') },
  { id: 'plan-ir', label: 'Show the raw plan document', run: () => switchQueryView('ir') },
  { id: 'copy-explain', label: 'Copy the plan as text', run: copyExplain },
  { id: 'critical', label: 'Toggle critical path', keys: 'c', run: () => {
      const box = $('dag-critical'); box.checked = !box.checked; DAG.setCritical(box.checked); } },
  { id: 'export-run', label: 'Download this run as JSON', run: exportRun },
  { id: 'export-pipelines', label: 'Download the pipeline list as CSV', run: exportPipelines },
  { id: 'export-ops', label: 'Download the operator table as CSV', run: exportOperators },
  { id: 'copy-link', label: 'Copy a link to this view', run: () => UI.copy(location.href, 'Link copied') },
  { id: 'learn', label: 'Open the reference (terms, plan steps, how-tos)', run: () => switchView('learn') },
  { id: 'tour', label: 'Take the guided tour', run: () => { switchView('pipelines'); setTimeout(LEARN.startTour, 60); } },
  { id: 'glossary', label: 'Look up a term in the glossary', run: () => switchView('learn') },
];

/** How well `needle` matches `text`: 4 exact, 3 prefix, 2 word-start, 1 subsequence, 0 none.
 *  A subsequence match that is merely short is not a better match than a prefix. */
function matchScore(needle, text) {
  const n = needle.toLowerCase(), t = text.toLowerCase();
  if (!n) return 1;
  if (t === n) return 4;
  if (t.startsWith(n)) return 3;
  if (t.split(/[\s—·-]+/).some((w) => w.startsWith(n))) return 2;
  return UI.fuzzy(n, t) ? 1 : 0;
}

function paletteItems(needle) {
  const items = ACTIONS.map((a) => ({ ...a, group: 'Action' }));
  for (const p of state.pipelines) {
    items.push({ id: `pipe-${p.signature}`, group: 'Pipeline',
                 label: `${pipelineDisplayName(p)} — ${p.runs} runs, ${UI.ms(p.median_ms)} typical`,
                 run: () => { rememberPaletteChoice(`pipe-${p.signature}`); openPipeline(p.signature); } });
  }
  for (const q of state.queries.slice(0, 40)) {
    items.push({ id: `run-${q.query_id}`, group: 'Run',
                 label: `${DAG.friendlyKind(q.label)} — ${UI.ago(q.started_wall)} · ${UI.ms(q.total_ms)}`,
                 run: () => { rememberPaletteChoice(`run-${q.query_id}`); openRun(q.query_id); } });
  }

  if (!needle) {
    // Empty query: surface what was used recently, most recent first, then everything else.
    const recent = UI.getPref('paletteRecent') || [];
    const rank = new Map(recent.map((id, i) => [id, recent.length - i]));
    return [...items]
      .sort((a, b) => (rank.get(b.id) || 0) - (rank.get(a.id) || 0))
      .slice(0, 40);
  }

  return items
    .map((i) => ({ i, score: matchScore(needle, `${i.group} ${i.label}`) }))
    .filter((x) => x.score > 0)
    .sort((a, b) => b.score - a.score || a.i.label.length - b.i.label.length)
    .map((x) => x.i)
    .slice(0, 40);
}

/** Record a palette choice so an empty query surfaces it next time. Bounded to 8. */
function rememberPaletteChoice(id) {
  const recent = (UI.getPref('paletteRecent') || []).filter((r) => r !== id);
  recent.unshift(id);
  UI.setPref('paletteRecent', recent.slice(0, 8));
}

let paletteIndex = 0;
/** Run a palette item, recording actions too (pipelines/runs record themselves in `run`). */
function runPaletteItem(item) {
  if (item.group === 'Action') rememberPaletteChoice(item.id);
  item.run();
  closeModal();
}
function renderPalette() {
  const items = paletteItems($('palette-input').value.trim());
  paletteIndex = Math.min(paletteIndex, Math.max(items.length - 1, 0));
  $('palette-list').innerHTML = items.length ? items.map((i, n) => (
    `<li class="pal-item${n === paletteIndex ? ' is-active' : ''}" data-n="${n}">` +
    `<span class="pal-group">${UI.esc(i.group)}</span>` +
    `<span class="pal-label">${UI.esc(i.label)}</span>` +
    (i.keys ? `<kbd>${UI.esc(i.keys)}</kbd>` : '') + `</li>`)).join('')
    : '<li class="pal-empty">Nothing matches.</li>';
  for (const li of $('palette-list').querySelectorAll('[data-n]')) {
    li.addEventListener('click', () => { runPaletteItem(items[Number(li.dataset.n)]); });
  }
  return items;
}

/* ---------- modals ---------- */

let openModalId = null;
/* Everything a keyboard can land on. Queried fresh each Tab because a modal's contents are
 * re-rendered (the palette rebuilds its list on every keystroke), so a list captured at open
 * time goes stale immediately. */
const FOCUSABLE = 'a[href], button:not([disabled]), input:not([disabled]), select, textarea, ' +
                  '[tabindex]:not([tabindex="-1"])';

let returnFocusTo = null;

/* Keep Tab inside the open dialog.
 *
 * `aria-modal` tells a screen reader the rest of the page is inert, but it does not stop the
 * browser moving focus there. Without this, Tab walks out of the dialog and into a page the
 * reader has been told is unavailable — they are then typing into something they cannot see. */
function trapFocus(e) {
  if (e.key !== 'Tab' || !openModalId) return;
  const modal = $(`modal-${openModalId}`);
  const items = [...modal.querySelectorAll(FOCUSABLE)].filter((el) => !el.hidden);
  if (!items.length) { e.preventDefault(); return; }
  const first = items[0], last = items[items.length - 1];
  const active = document.activeElement;
  if (e.shiftKey && (active === first || !modal.contains(active))) {
    e.preventDefault(); last.focus();
  } else if (!e.shiftKey && (active === last || !modal.contains(active))) {
    e.preventDefault(); first.focus();
  }
}

function showModal(id) {
  closeModal();
  openModalId = id;
  // Remember where focus came from so closing returns it, rather than dropping the reader at
  // the top of the document with no idea where they were.
  returnFocusTo = document.activeElement;
  const modal = $(`modal-${id}`);
  modal.hidden = false;
  $('scrim').hidden = false;
  document.addEventListener('keydown', trapFocus, true);
  if (id === 'palette') {
    $('palette-input').value = ''; paletteIndex = 0; renderPalette(); $('palette-input').focus();
  } else {
    // Focus the dialog itself, so the next Tab starts inside it and a screen reader
    // announces the dialog rather than continuing from wherever it was.
    modal.setAttribute('tabindex', '-1');
    modal.focus({ preventScroll: true });
  }
}

function closeModal() {
  if (!openModalId) return;
  $(`modal-${openModalId}`).hidden = true;
  $('scrim').hidden = true;
  openModalId = null;
  document.removeEventListener('keydown', trapFocus, true);
  if (returnFocusTo && document.contains(returnFocusTo)) returnFocusTo.focus();
  returnFocusTo = null;
}

/* ---------- actions ---------- */

function invalidateAll() { hashes.clear(); state.detailId = null; }

function togglePause() {
  state.paused = !state.paused;
  $('pause').setAttribute('aria-pressed', String(state.paused));
  $('pause').textContent = state.paused ? '▶' : '❚❚';
  setConnected(true);
  UI.toast(state.paused ? 'Auto-refresh paused' : 'Auto-refresh resumed');
  if (!state.paused) poll();
}

function applyTheme(theme) {
  const preferLight = window.matchMedia('(prefers-color-scheme: light)').matches;
  const resolved = theme === 'auto' ? (preferLight ? 'light' : 'dark') : theme;
  document.documentElement.dataset.theme = resolved;
  const button = $('theme');
  if (button) {
    button.title = `Theme: ${theme}${theme === 'auto' ? ` (currently ${resolved})` : ''}`;
    // Show the state, not a generic glyph: a toggle whose icon never changes gives no
    // feedback that it did anything.
    button.textContent = theme === 'dark' ? '☾' : theme === 'light' ? '☀' : '◐';
    button.setAttribute('aria-label', `Theme: ${theme}. Click to change.`);
  }
}

/** Cycle dark → light → auto. Dark leads because that is the default and the common case. */
function toggleTheme() {
  const order = ['dark', 'light', 'auto'];
  const next = order[(order.indexOf(UI.getPref('theme')) + 1) % order.length];
  UI.setPref('theme', next);
  applyTheme(next);
  UI.toast(`Theme: ${next}`);
}
/** Switch the pipeline list between cards and a dense table. */
function setPipelineLayout(layout) {
  state.pipelineLayout = layout;
  UI.setPref('pipelineLayout', layout);
  invalidate('pipelines');
  renderPipelineList();
}

/** Focus the search box on whichever view is showing one, so "/" always lands somewhere. */
function focusActiveSearch() {
  const box = { pipelines: 'pipeline-filter', logs: 'log-filter', learn: 'learn-search' }[state.view];
  const el = box && $(box);
  if (el) { el.focus(); el.select?.(); }
}

function toggleDensity() {
  const next = UI.getPref('density') === 'compact' ? 'comfortable' : 'compact';
  UI.setPref('density', next);
  document.body.classList.toggle('is-compact', next === 'compact');
  // Density changes every tab's box, so the travelling indicator has to be re-measured.
  moveInk('.viewnav'); moveInk('.tabs');
  UI.toast(`Density: ${next}`);
}
function toggleHelp() {
  const next = !UI.getPref('help');
  UI.setPref('help', next);
  document.body.classList.toggle('help-on', next);
  $('help').setAttribute('aria-pressed', String(next));
  UI.toast(next ? 'Explanations on' : 'Explanations off');
}

function exportRun() {
  if (!state.detail) return UI.toast('No run selected', 'warn');
  UI.download(`batcher-run-${state.detail.query_id}.json`, JSON.stringify(state.detail, null, 2));
}
function exportOperators() {
  const nodes = state.detail?.dag?.nodes || [];
  if (!nodes.length) return UI.toast('No operators to export', 'warn');
  UI.download(`batcher-operators-${state.detail.query_id}.csv`,
              UI.tableCSV(VIEWS.OPERATOR_COLUMNS, nodes), 'text/csv');
}
function exportPipelines() {
  const list = state.pipelines || [];
  if (!list.length) return UI.toast('No pipelines to export', 'warn');
  UI.download('batcher-pipelines.csv',
    UI.toCSV(['pipeline', 'signature', 'runs', 'failed', 'p50_ms', 'p95_ms', 'last_ms'],
             list.map((p) => [p.label, p.signature, p.runs, p.n_failed || 0,
                              p.p50_ms || '', p.p95_ms || '', p.last_ms || ''])),
    'text/csv');
}

function exportLogs() {
  const lines = filteredLogs();
  if (!lines.length) return UI.toast('No log lines to export', 'warn');
  UI.download('batcher-logs.csv',
    UI.toCSV(['time', 'level', 'logger', 'message', 'fields'],
             lines.map((l) => [UI.stamp(l.wall), l.level, l.logger, l.message, JSON.stringify(l.fields || {})])),
    'text/csv');
}

/* ---------- wiring ---------- */

function boot() {
  UI.loadPrefs();
  applyTheme(UI.getPref('theme'));
  document.body.classList.toggle('is-compact', UI.getPref('density') === 'compact');
  document.body.classList.toggle('help-on', UI.getPref('help'));
  state.stepsView = STEPS_VIEWS.includes(UI.getPref('stepsView')) ? UI.getPref('stepsView') : 'plan';
  state.queryView = QUERY_VIEWS.includes(UI.getPref('queryView')) ? UI.getPref('queryView') : 'explain';
  markSwitch('steps-switch', 'steps', state.stepsView);
  markSwitch('query-switch', 'query', state.queryView);
  $('help').setAttribute('aria-pressed', String(UI.getPref('help')));
  $('log-level').value = UI.getPref('logLevel');
  $('log-follow').checked = UI.getPref('logFollow');
  $('log-regex').checked = UI.getPref('logRegex');
  $('dag-minimap').checked = UI.getPref('showMinimap');

  state.pipelineLayout = UI.getPref('pipelineLayout') || 'cards';
  const route = UI.readRoute();
  if (route.pipeline) state.pipeline = route.pipeline;
  if (route.run) state.selected = route.run;
  if (route.cmp) state.compareWith = route.cmp;
  // Routes saved before the tab consolidation still name a retired pane; map them onto the
  // section that absorbed them rather than dropping the reader on an empty page.
  if (route.tab) {
    const LEGACY = { plan: 'steps', timeline: 'steps', operators: 'steps', stages: 'steps',
                     flame: 'steps', decisions: 'insights', raw: 'meta',
                     explain: 'query', diff: 'query', ir: 'query' };
    if (STEPS_VIEWS.includes(route.tab)) state.stepsView = route.tab;
    if (QUERY_VIEWS.includes(route.tab)) state.queryView = route.tab;
    switchTab(LEGACY[route.tab] || route.tab);
  }
  switchView(route.view || 'pipelines');

  for (const t of document.querySelectorAll('.viewtab')) {
    t.addEventListener('click', () => switchView(t.dataset.view));
  }
  for (const t of document.querySelectorAll('.tab')) {
    t.addEventListener('click', () => switchTab(t.dataset.tab));
  }
  for (const b of document.querySelectorAll('#steps-switch .seg')) {
    b.addEventListener('click', () => switchStepsView(b.dataset.steps));
  }
  for (const b of document.querySelectorAll('#query-switch .seg')) {
    b.addEventListener('click', () => switchQueryView(b.dataset.query));
  }
  installQueryViewer();
  for (const c of document.querySelectorAll('.status-chip')) {
    c.addEventListener('click', () => {
      c.classList.toggle('is-on');
      c.setAttribute('aria-pressed', String(c.classList.contains('is-on')));
      invalidate('pipeline-page');
      renderPipelinePage();
    });
  }

  on('pipeline-filter', 'input', UI.debounce((e) => {
    state.pipelineFilter = e.target.value.trim();
    $('clear-filters').hidden = !state.pipelineFilter;
    invalidate('pipelines');
    renderPipelineList();
  }));
  on('pipeline-sort', 'change', (e) => {
    state.pipelineSort = e.target.value; invalidate('pipelines'); renderPipelineList();
  });
  for (const [id, layout] of [['lay-cards', 'cards'], ['lay-table', 'table']]) {
    $(id).addEventListener('click', () => setPipelineLayout(layout));
  }
  on('p-pin', 'click', () => {
    if (!state.pipeline) return;
    UI.togglePin(state.pipeline);
    invalidate('pipelines', 'pipeline-page');
    renderPipelineList(); renderPipelinePage();
    UI.toast(UI.isPinned(state.pipeline) ? 'Pipeline pinned' : 'Pin removed');
  });
  on('run-prev', 'click', () => stepRun(-1));
  on('run-next', 'click', () => stepRun(1));
  on('compare-pick', 'change', (e) => {
    state.compareWith = e.target.value || null;
    state.compare = null;
    UI.writeRoute({ cmp: state.compareWith || '' }, { replace: true });
    if (state.selected) loadDetail(state.selected);
  });

  on('log-level', 'change', (e) => { UI.setPref('logLevel', e.target.value); invalidate('logs'); renderLogs(); });
  on('log-filter', 'input', UI.debounce(() => { invalidate('logs'); renderLogs(); }));
  on('log-follow', 'change', (e) => UI.setPref('logFollow', e.target.checked));
  on('log-regex', 'change', (e) => { UI.setPref('logRegex', e.target.checked); invalidate('logs'); renderLogs(); });
  on('log-export', 'click', exportLogs);
  on('log-clear', 'click', () => {
    // The engine keeps its own log; this only clears the browser's copy. Say so rather than
    // letting someone think they destroyed something.
    const kept = state.logLines.length;
    state.logLines = [];
    invalidate('logs');
    renderLogs();
    UI.toast(`Cleared ${kept} lines from this view — the engine's log is untouched`);
  });

  on('dag-fit', 'click', () => DAG.fit());
  on('p-dag-fit', 'click', () => PIPE_DAG.fit());
  on('p-dag-critical', 'change', (e) => PIPE_DAG.setCritical(e.target.checked));
  on('dag-reset', 'click', () => { DAG.reset(); UI.toast('Layout reset'); });
  on('dag-search', 'input', UI.debounce((e) => DAG.setSearch(e.target.value), 80));
  on('dag-focus', 'change', (e) => DAG.setFocus(e.target.checked));
  on('dag-critical', 'change', (e) => { UI.setPref('showCritical', e.target.checked); DAG.setCritical(e.target.checked); });
  on('dag-minimap', 'change', (e) => {
    UI.setPref('showMinimap', e.target.checked);
    $('minimap').hidden = !e.target.checked;
    if (state.detail) renderPlan(state.detail);
  });
  on('dag-full', 'click', () => {
    document.body.classList.toggle('dag-full');
    requestAnimationFrame(() => DAG.fit());
  });

  on('theme', 'click', toggleTheme);
  on('help', 'click', toggleHelp);
  renderShortcuts();
  installGridCrosshair();
  installCopyAnywhere();
  LEARN.install();
  installSegmentedGroups();
  // Log affordances are re-rendered on every poll, so they bind by delegation.
  const logHost = $('logs');
  if (logHost) {
    logHost.addEventListener('click', (e) => {
      const field = e.target.closest('[data-field]');
      if (field) {
        state.logFields.set(field.dataset.field, field.dataset.value);
        invalidate('logs');
        renderLogs();
        announce(`Filtered to ${field.dataset.field}=${field.dataset.value}`);
        return;
      }
      const ctxBtn = e.target.closest('[data-context]');
      if (ctxBtn) {
        e.stopPropagation();
        showLogContext(Number(ctxBtn.dataset.context));
        return;
      }
      const perma = e.target.closest('[data-permalink]');
      if (perma) {
        // A route parameter, not a bare `#L123` fragment. The whole page is addressed by
        // the hash, so a fragment written into it wiped the view, the pipeline, and the run
        // — the copied link reopened the dashboard on the overview with no log line in
        // sight, which is the opposite of what "copy a link to this line" promises.
        for (const row of logHost.querySelectorAll('.logline')) row.classList.remove('is-linked');
        perma.closest('.logline')?.classList.add('is-linked');
        const seq = perma.dataset.permalink;
        UI.copy(`${location.href.split('#')[0]}#view=logs&line=${encodeURIComponent(seq)}`,
                'Link to line copied');
      }
    });
  }
  on('log-histo-clear', 'click', clearLogRange);
  installHistoDrag();
  on('log-in-results', 'input', UI.debounce(() => { invalidate('logs'); renderLogs(); }, 120));
  on('log-new', 'click', () => {
    state.logPending = 0;
    $('log-new').hidden = true;
    $('log-follow').checked = true;
    UI.setPref('logFollow', true);
    invalidate('logs');
    renderLogs();
    $('logs').scrollTop = $('logs').scrollHeight;
  });
  // Empty-state affordances are re-rendered on every poll, so they are bound by delegation
  // rather than per-element — a direct binding would go stale on the first repaint.
  document.addEventListener('click', (e) => {
    const copyBtn = e.target.closest('[data-copy]');
    if (copyBtn) { UI.copy(copyBtn.getAttribute('data-copy'), 'Snippet copied'); return; }
    const action = e.target.closest('[data-empty-action]');
    if (!action) return;
    const id = action.getAttribute('data-empty-action');
    if (id === 'tour') LEARN.startTour();
    if (id === 'learn') switchView('learn');
  });
  on('learn-search', 'input', UI.debounce(() => LEARN.renderLearn($('learn-body'), $('learn-search').value), 120));
  on('learn-tour', 'click', () => { switchView('pipelines'); setTimeout(LEARN.startTour, 60); });
  on('pause', 'click', togglePause);
  on('refresh', 'click', () => { invalidateAll(); poll(); UI.toast('Refreshed'); });
  on('palette-open', 'click', () => showModal('palette'));
  on('scrim', 'click', closeModal);
  on('retry', 'click', () => { invalidateAll(); poll(); });
  for (const b of document.querySelectorAll('[data-close-modal]')) b.addEventListener('click', closeModal);
  on('export-run', 'click', exportRun);
  on('export-ops', 'click', exportOperators);
  on('copy-id', 'click', () => state.detail && UI.copy(state.detail.query_id, 'Run id copied'));
  on('copy-link', 'click', () => UI.copy(location.href, 'Link copied'));
  on('palette-input', 'input', () => { paletteIndex = 0; renderPalette(); });

  // Back/forward moves between levels, so the browser's own history is the undo stack.
  window.addEventListener('hashchange', () => {
    const r = UI.readRoute();
    if (r.pipeline !== state.pipeline) { state.pipeline = r.pipeline || null; invalidate('pipeline-page'); }
    if ((r.run || null) !== state.selected) {
      state.selected = r.run || null;
      invalidate('detail');
      // Back/forward to a run loads it now rather than after the next poll (up to 5 s idle).
      if (state.selected) loadDetail(state.selected);
    }
    if (r.view && r.view !== state.view) switchView(r.view);
    // Sync the run-detail tab and sub-view too, so a shared link to a run's Query or Findings
    // tab opens there — not only on a fresh load but on any hash change, which is what the
    // browser's back/forward buttons produce.
    if (r.tab) {
      if (STEPS_VIEWS.includes(r.tab)) switchStepsView(r.tab);
      else if (QUERY_VIEWS.includes(r.tab)) switchQueryView(r.tab);
      else if (r.tab !== state.tab) switchTab(r.tab);
    }
  });

  // KPIs summarise something; make them the way to it.
  on(up($('k-failed'), '.kpi'), 'click', () => {
    if (!state.summary.n_failed) return;
    switchView('pipelines');
    $('failures')?.scrollIntoView({ behavior: 'smooth', block: 'center' });
  });
  on(up($('k-spill'), '.kpi'), 'click', () => {
    const spilled = state.queries.find((q) => (q.n_stages || 0) > 0 && q.status === 'ok');
    if (spilled) openRun(spilled.query_id);
  });
  // Every summary KPI is a way *to* the thing it counts: pipelines and runs jump to the
  // list, running work jumps to the Live page. A tile that leads somewhere says so (the
  // cursor and hover come from `is-clickable`, set on render for these).
  for (const [id, run] of [
    ['k-pipelines', () => { switchView('pipelines'); $('pipeline-cards')?.scrollIntoView({ behavior: 'smooth', block: 'start' }); }],
    ['k-queries', () => { switchView('pipelines'); $('pipeline-cards')?.scrollIntoView({ behavior: 'smooth', block: 'start' }); }],
    ['k-running', () => switchView('live')],
  ]) {
    const kpi = up($(id), '.kpi');
    if (kpi) {
      kpi.classList.add('is-clickable');
      kpi.setAttribute('tabindex', '0');
      kpi.setAttribute('role', 'button');
      on(kpi, 'click', run);
    }
  }
  // Enter or Space on a focused, actionable KPI does what a click does.
  for (const el of ['k-failed', 'k-spill', 'k-pipelines', 'k-queries', 'k-running']) {
    on(up($(el), '.kpi'), 'keydown', (e) => {
      if ((e.key === 'Enter' || e.key === ' ') && e.currentTarget.classList.contains('is-clickable')) {
        e.preventDefault();
        e.currentTarget.click();
      }
    });
  }
  on('clear-filters', 'click', () => {
    $('pipeline-filter').value = '';
    state.pipelineFilter = '';
    for (const c of document.querySelectorAll('.status-chip.is-on')) {
      c.classList.remove('is-on');
      c.setAttribute('aria-pressed', 'false');
    }
    invalidate('pipelines', 'pipeline-page');
    renderPipelineList();
    renderPipelinePage();
    UI.toast('Filters cleared');
  });
  for (const panel of document.querySelectorAll('[data-panel]')) UI.collapsible(panel);

  installShortcuts();
  installCrossReferences();
  installChartTooltips();
  installInkBars();
  installBackToTop();
  renderRecentRail();
  poll();
}

/* The back-to-top button: shown only once there is enough page above to want it, and it
 * takes focus to the top so a keyboard user is not dropped back where they were. */
function installBackToTop() {
  const btn = $('to-top');
  if (!btn) return;
  const update = () => { btn.hidden = window.scrollY < 600; };
  window.addEventListener('scroll', update, { passive: true });
  btn.addEventListener('click', () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
    const heading = document.querySelector('.view.is-active h1, .view.is-active h2');
    if (heading) { heading.setAttribute('tabindex', '-1'); heading.focus({ preventScroll: true }); }
  });
  update();
}

let chord = null;
function installShortcuts() {
  document.addEventListener('keydown', (e) => {
    const typing = e.target.matches('input, select, textarea');
    // Cmd/Ctrl-K opens the palette from anywhere, including a text field.
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
      e.preventDefault(); showModal('palette'); return;
    }
    if (openModalId === 'palette') {
      const items = renderPalette();
      if (e.key === 'ArrowDown') { e.preventDefault(); paletteIndex = Math.min(paletteIndex + 1, items.length - 1); renderPalette(); }
      else if (e.key === 'ArrowUp') { e.preventDefault(); paletteIndex = Math.max(paletteIndex - 1, 0); renderPalette(); }
      else if (e.key === 'Enter') { e.preventDefault(); if (items[paletteIndex]) runPaletteItem(items[paletteIndex]); }
      else if (e.key === 'Escape') closeModal();
      return;
    }
    if (e.key === 'Escape') { closeModal(); hideTip(); DAG.select(null); return; }
    if (typing) return;

    // `g` then a letter jumps between views, the convention every developer tool uses.
    if (chord === 'g') {
      chord = null;
      const target = { p: 'pipelines', r: 'live', l: 'logs', s: 'system', h: 'learn' }[e.key];
      if (target) { e.preventDefault(); switchView(target); }
      return;
    }
    if (e.key === 'g') { chord = 'g'; setTimeout(() => { chord = null; }, 1200); return; }

    if (e.key === '?') { e.preventDefault(); showModal('shortcuts'); }
    else if (e.key === '/') { e.preventDefault(); focusActiveSearch(); }
    else if (e.key === 'e') toggleHelp();
    else if (e.key === 'v') setPipelineLayout(state.pipelineLayout === 'cards' ? 'table' : 'cards');
    else if (e.key === 'u') goUp();
    else if (e.key === 'Home') { e.preventDefault(); window.scrollTo({ top: 0, behavior: 'smooth' }); }
    else if (e.key === 'End') { e.preventDefault(); window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' }); }
    else if (e.key === 'x' && state.view === 'run') switchQueryView('explain');
    else if (e.key === 'j') stepRun(-1);
    else if (e.key === 'k') stepRun(1);
    else if (e.key === 'r') { invalidateAll(); poll(); UI.toast('Refreshed'); }
    else if (e.key === 't') toggleTheme();
    else if (e.key === 'd') toggleDensity();
    else if (e.key === ' ') { e.preventDefault(); togglePause(); }
    else if (state.view === 'run' && state.tab === 'steps' && state.stepsView === 'plan') {
      if (e.key === 'ArrowDown') { e.preventDefault(); DAG.step(1); }
      else if (e.key === 'ArrowUp') { e.preventDefault(); DAG.step(-1); }
      else if (e.key === 'ArrowRight') { e.preventDefault(); DAG.stepAcross(1); }
      else if (e.key === 'ArrowLeft') { e.preventDefault(); DAG.stepAcross(-1); }
      else if (e.key === 'f') DAG.fit();
      else if (e.key === 'c') { const b = $('dag-critical'); b.checked = !b.checked; DAG.setCritical(b.checked); }
    }
  });
}

boot();
