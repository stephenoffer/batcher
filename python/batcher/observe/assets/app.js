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
  tab: 'steps', stepsView: 'plan', compareWith: null,
  logCursor: 0, logLines: [], lastSystemAt: 0,
  logRange: null, logFields: new Map(), logPending: 0, logHistoGeom: null,
  detail: null, detailId: null, compare: null,
  queries: [], pipelines: [], summary: {}, system: {}, operators: [],
  pipelineSort: 'time', pipelineFilter: '', pipelineLayout: 'cards',
  paused: false, lastError: null, loaded: false, report: null, reportFor: null,
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

function setConnected(ok, message) {
  const el = $('conn');
  el.textContent = ok ? (state.paused ? 'paused' : 'live') : 'disconnected';
  el.className = `pill ${ok ? (state.paused ? '' : 'is-live') : 'is-down'}`;
  $('error-banner').hidden = ok;
  if (!ok && message) $('error-text').textContent = message;
}

async function poll() {
  if (state.paused) { setTimeout(poll, POLL_IDLE_MS); return; }
  let running = 0;
  try {
    const wants = [getJSON('/api/summary'), getJSON('/api/pipelines'), getJSON('/api/queries'),
                   getJSON(`/api/logs?since=${state.logCursor}`), getJSON('/api/health'),
                   getJSON('/api/timeseries'), getJSON('/api/operators'),
                   getJSON('/api/failures')];
    if (Date.now() - state.lastSystemAt > SYSTEM_EVERY_MS) wants.push(getJSON('/api/system'));
    const [summary, pipelines, queries, logs, health, series, operators, failures, system] =
      await Promise.all(wants);
    running = summary.n_running || 0;
    Object.assign(state, { summary, queries: queries.queries, pipelines: pipelines.pipelines,
                           operators: operators.operators, loaded: true });

    paint('health', health, () => VIEWS.health(health));
    paint('kpis', summary, () => VIEWS.kpis(summary, flash));
    paint('series', series, () => VIEWS.throughput(series));
    paint('rollup', operators.operators, () => VIEWS.operatorRollup(operators.operators));
    paint('split', pipelines.pipelines, () => VIEWS.timeSplit(pipelines.pipelines));
    paint('failures', failures.groups, () => {
      $('failure-count').textContent = failures.groups.reduce((n, g) => n + g.count, 0);
      VIEWS.failures(failures.groups);
    });
    paint('attention', [queries.queries.filter((q) => q.status === 'error').map((q) => q.query_id),
                        state.detail?.insights],
          () => VIEWS.attention(queries.queries, state.detail?.insights));
    renderPipelineList();
    renderPipelinePage();
    ingestLogs(logs);
    if (system) { state.lastSystemAt = Date.now(); state.system = system; paint('system', system, () => VIEWS.system(system)); }

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
    });
    VIEWS.pipelineTable(shown || state.pipelines, openPipeline);
    const cards = state.pipelineLayout === 'cards';
    $('pipeline-cards').hidden = !cards;
    $('pipeline-table').hidden = cards;
    $('lay-cards').classList.toggle('is-on', cards);
    $('lay-table').classList.toggle('is-on', !cards);
    $('lay-cards').setAttribute('aria-pressed', String(cards));
    // Offered once, and only now: a tour of an empty dashboard points at nothing, so it
    // waits until there is real work on screen to point at.
    LEARN.maybeOfferTour(state.pipelines.length > 0);
    $('lay-table').setAttribute('aria-pressed', String(!cards));
  });
}

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
  if (p) noteVisit('pipeline', signature, DAG.friendlyKind(p.label));
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
  const slowest = Math.max(...siblings.map((r) => r.total_ms || 0), 1);
  host.innerHTML = `<span class="rail-label">This pipeline's runs</span>` + siblings.map((r) => {
    const h = Math.max(12, ((r.total_ms || 0) / slowest) * 100);
    const cls = `related-dot${r.query_id === d.query_id ? ' is-current' : ''}` +
                `${r.status === 'error' ? ' is-error' : ''}`;
    return `<button class="${cls}" type="button" data-run="${UI.esc(r.query_id)}" ` +
      `title="${UI.clock(r.started_wall)} · ${UI.ms(r.total_ms)}${r.status === 'error' ? ' · failed' : ''}">` +
      `<i style="height:${h}%"></i></button>`;
  }).join('');
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
      switchTab('plan');
      requestAnimationFrame(() => DAG.select(Number(el.dataset.op)));
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

function renderCrumbs() {
  const crumbs = $('crumbs');
  if (state.view === 'logs' || state.view === 'system' || state.view === 'pipelines') {
    crumbs.hidden = true;
    return;
  }
  const p = currentPipeline();
  const parts = [`<button class="crumb" data-go="pipelines" type="button">All pipelines</button>`];
  if (p) {
    parts.push(state.view === 'pipeline'
      ? `<span class="crumb is-current" aria-current="page">${UI.esc(DAG.friendlyKind(p.label))}</span>`
      : `<button class="crumb" data-go="pipeline" type="button">${UI.esc(DAG.friendlyKind(p.label))}</button>`);
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
  VIEWS.timeline(d.dag?.nodes || []);
  VIEWS.operators(d.dag?.nodes || [], (opId) => { switchTab('plan'); DAG.select(opId); });
  VIEWS.insights(d.insights || [], d.query_id);
  VIEWS.decisions(d.decisions || []);
  renderRunLogs(d);
  VIEWS.meta(d);
  $('raw-json').textContent = JSON.stringify(d, null, 2);
  VIEWS.comparison(state.compare, 'baseline', 'this run');
  renderRunPosition();
  renderRelated(d);
  renderCrumbs();
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
    ? `<p class="hint">Lines written while this run was executing.</p>` + lines.map(logLine).join('')
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
    host.innerHTML = lines.length
      ? lines.map((l, i) => VIEWS.logLine(l, i)).join('')
      : '<p class="empty">No lines match. Lower the level, clear the search, or raise the engine’s ' +
        'verbosity — <span class="mono">observability.verbosity = "verbose"</span> shows what the ' +
        'optimizer decided.</p>';
    if ($('log-follow').checked && !state.logRange) host.scrollTop = host.scrollHeight;
  });
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
    around.map((l) => VIEWS.logLine(l, l.seq)).join('');
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
const VIEW_DEPTH = { pipelines: 0, logs: 0, system: 0, pipeline: 1, run: 2 , learn: 0 };

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
  // The reference is static content, so it is drawn on arrival rather than by the poll loop.
  if (view === 'learn') LEARN.renderLearn($('learn-body'), $('learn-search').value);
  UI.writeRoute({ view }, { replace: true });
  renderCrumbs();
  announce(`${view.replace('pipelines', 'all pipelines')} view`);
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
    parts.unshift(DAG.friendlyKind(currentPipeline().label));
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
  insights: () => ['tab-insights', 'tab-decisions'],
  compare: () => ['tab-compare'],
  logs: () => ['tab-logs'],
  meta: () => ['tab-meta', 'tab-raw'],
};

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
  // The rendering switch belongs to Steps and would be meaningless above the others.
  const sw = $('steps-switch');
  if (sw) sw.hidden = state.tab !== 'steps';
  UI.writeRoute({ tab: state.tab }, { replace: true });
  if (state.tab === 'steps' && state.stepsView === 'plan') {
    requestAnimationFrame(() => DAG.fit());
  }
}

/** Pick which rendering of the steps to show. The tab does not change — the subject is the
 *  same, only the view of it. */
function switchStepsView(view) {
  state.stepsView = ['plan', 'timeline', 'operators'].includes(view) ? view : 'plan';
  UI.setPref('stepsView', state.stepsView);
  for (const b of document.querySelectorAll('#steps-switch .seg')) {
    const on = b.dataset.steps === state.stepsView;
    b.classList.toggle('is-on', on);
    b.setAttribute('aria-pressed', String(on));
  }
  switchTab('steps');
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
  document.addEventListener('click', (e) => {
    const el = e.target.closest('[data-copyable]');
    if (!el) return;
    // Never hijack a selection: a reader highlighting part of a value wants that, not a copy.
    if (String(window.getSelection?.() || '').length) return;
    UI.copy(el.dataset.copyValue || el.textContent.trim(), 'Copied');
  });
}

/* ---------- command palette ---------- */
/* One keystroke to anything on the page. Power users stop hunting through views, and it
 * doubles as a discoverability surface: every action is listed with its shortcut. */

const ACTIONS = [
  { id: 'view-pipelines', label: 'All pipelines', keys: 'g p', run: () => switchView('pipelines') },
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
  { id: 'steps-timeline', label: 'Show steps as a timeline', run: () => switchStepsView('timeline') },
  { id: 'steps-table', label: 'Show steps as a table', run: () => switchStepsView('operators') },
  { id: 'critical', label: 'Toggle critical path', keys: 'c', run: () => {
      const box = $('dag-critical'); box.checked = !box.checked; DAG.setCritical(box.checked); } },
  { id: 'export-run', label: 'Download this run as JSON', run: exportRun },
  { id: 'export-pipelines', label: 'Download the pipeline list as CSV', run: exportPipelines },
  { id: 'export-ops', label: 'Download the operator table as CSV', run: exportOperators },
  { id: 'copy-link', label: 'Copy a link to this view', run: () => UI.copy(location.href, 'Link copied') },
  { id: 'learn', label: 'Open the reference (terms, plan steps, how-tos)', run: () => setView('learn') },
  { id: 'tour', label: 'Take the guided tour', run: () => { setView('pipelines'); setTimeout(LEARN.startTour, 60); } },
  { id: 'glossary', label: 'Look up a term in the glossary', run: () => setView('learn') },
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
                 label: `${DAG.friendlyKind(p.label)} — ${p.runs} runs, ${UI.ms(p.median_ms)} typical`,
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
  state.stepsView = UI.getPref('stepsView') || 'plan';
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
    const LEGACY = { plan: 'steps', timeline: 'steps', operators: 'steps',
                     decisions: 'insights', raw: 'meta' };
    const stepView = ['plan', 'timeline', 'operators'].includes(route.tab) ? route.tab : null;
    if (stepView) state.stepsView = stepView;
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
        const id = `L${perma.dataset.permalink}`;
        for (const row of logHost.querySelectorAll('.logline')) row.classList.remove('is-linked');
        perma.closest('.logline')?.classList.add('is-linked');
        UI.copy(`${location.href.split('#')[0]}#${id}`, 'Link to line copied');
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
    if (id === 'learn') setView('learn');
  });
  on('learn-search', 'input', UI.debounce(() => LEARN.renderLearn($('learn-body'), $('learn-search').value), 120));
  on('learn-tour', 'click', () => { setView('pipelines'); setTimeout(LEARN.startTour, 60); });
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
    if ((r.run || null) !== state.selected) { state.selected = r.run || null; invalidate('detail'); }
    if (r.view && r.view !== state.view) switchView(r.view);
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
  // Enter or Space on a focused, actionable KPI does what a click does.
  for (const el of ['k-failed', 'k-spill']) {
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
  renderRecentRail();
  poll();
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
      const target = { p: 'pipelines', l: 'logs', s: 'system' }[e.key];
      if (target) { e.preventDefault(); switchView(target); }
      return;
    }
    if (e.key === 'g') { chord = 'g'; setTimeout(() => { chord = null; }, 1200); return; }

    if (e.key === '?') { e.preventDefault(); showModal('shortcuts'); }
    else if (e.key === '/') { e.preventDefault(); focusActiveSearch(); }
    else if (e.key === 'e') toggleHelp();
    else if (e.key === 'v') setPipelineLayout(state.pipelineLayout === 'cards' ? 'table' : 'cards');
    else if (e.key === 'u') goUp();
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
