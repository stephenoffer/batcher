/* The panel renderers — one function per thing the dashboard shows.
 *
 * Every renderer takes data and writes into a host element. None of them fetch, none hold
 * state beyond what they are given, and none know about polling. That separation is what
 * lets `app.js` decide *when* to draw (only on change) without each panel re-implementing
 * the check.
 *
 * Each panel leads with a plain-language verdict and puts the measured detail directly
 * beneath it: a newcomer reads the sentence and stops, a power user reads the table.
 */

'use strict';

const VIEWS = (() => {
  const { esc, count, ms, bytes, pct, clock, ago, duration, sparkline, histogram,
          table, median } = UI;
  const friendly = (k) => DAG.friendlyKind(k);
  const $ = (id) => document.getElementById(id);

  /* Terminology comes from the shared reference, so a word means the same thing here, in
   * the plan inspector, and on the Learn page — and is defined in exactly one file. */
  const term = (word, label) => LEARN.term(word, label);
  /* ═══════════ overview ═══════════ */

  function health(report) {
    const rank = { ok: 'All clear', warn: 'Worth a look', critical: 'Needs attention' };
    $('health-banner').className = `health is-${report.status}`;
    $('health-banner').innerHTML =
      `<div class="health-head"><span class="health-dot"></span>` +
      `<b>${esc(rank[report.status] || 'Unknown')}</b>` +
      `<span class="dim">engine up ${duration(report.uptime_s || 0)}</span></div>` +
      `<div class="health-checks">` + (report.checks || []).map((c) => (
        `<div class="check is-${esc(c.status)}"><span class="check-name">${esc(c.name)}</span>` +
        `<span class="check-detail">${esc(c.detail)}</span>` +
        (c.action ? `<span class="check-action">${esc(c.action)}</span>` : '') +
        // A verdict you cannot click through to is a dead end; link straight to its evidence.
        ((c.runs || []).length
          ? `<button class="xref" data-run="${esc(c.runs[0])}" type="button" ` +
            `aria-label="Show the ${c.runs.length === 1 ? 'run' : `${c.runs.length} runs`} behind this check">` +
            `see ${c.runs.length === 1 ? 'the run' : `the ${c.runs.length} runs`}</button>` : '') +
        `</div>`)).join('') + `</div>`;
  }

  /* Counted KPIs roll to their new value; formatted ones (durations, byte sizes) are set
   * directly, because interpolating "1.4 GiB" through intermediate units reads as noise. */
  function kpis(summary, onChange) {
    const p = summary.percentiles || {};
    const rolled = [
      ['k-pipelines', summary.n_pipelines, count],
      ['k-queries', summary.n_queries, count],
      ['k-running', summary.n_running, count],
      ['k-rate', summary.rows_per_sec || 0, (v) => (v ? count(v) : '—')],
      ['k-failed', summary.n_failed, count],
    ];
    for (const [id, value, format] of rolled) {
      const el = $(id);
      if (!el) continue;
      const before = el.textContent;
      UI.rollTo(el, value, format);
      if (before !== format(value)) onChange?.(el);
    }
    for (const [id, value] of [['k-median', p.p50 ? ms(p.p50) : '—'],
                               ['k-p95', p.p95 ? ms(p.p95) : '—'],
                               ['k-spill', summary.spill_bytes ? bytes(summary.spill_bytes) : 'none']]) {
      const el = $(id);
      if (el && el.textContent !== value) { el.textContent = value; onChange?.(el); }
    }
    $('k-running').classList.toggle('is-live', summary.n_running > 0);
    $('k-failed').classList.toggle('is-critical', summary.n_failed > 0);
    $('k-p95-note').textContent = p.reliable ? `over ${p.count} runs` : `only ${p.count || 0} runs`;
    // A KPI is a summary of something; make it the way *to* that something.
    setKpiActionable($('k-failed'), summary.n_failed > 0, 'See the failed runs');
    setKpiActionable($('k-spill'), Boolean(summary.spill_bytes), 'Open a run that spilled');
  }

  /** A KPI tile leads somewhere only sometimes; give it (and remove) keyboard reach in step
   *  with that, so focus order never lands on a tile that does nothing. */
  function setKpiActionable(valueEl, on, label) {
    const kpi = valueEl?.closest?.('.kpi');
    if (!kpi) return;
    kpi.classList.toggle('is-clickable', on);
    if (on) {
      kpi.setAttribute('tabindex', '0');
      kpi.setAttribute('role', 'button');
      kpi.setAttribute('aria-label', label);
    } else {
      kpi.removeAttribute('tabindex');
      kpi.removeAttribute('role');
      kpi.removeAttribute('aria-label');
    }
  }

  function throughput(series) {
    const buckets = series.buckets || [];
    if (!UI.panelState('throughput', {
      empty: !buckets.length,
      emptyState: {
        glyph: 'chart', title: 'No throughput yet',
        body: 'Rows processed per second across the session appears here as runs finish. ' +
              'It is what makes runs over different data comparable — a slower run over ten ' +
              'times the rows is not a regression.',
      },
    })) return;
    const host = $('throughput');
    const rates = buckets.map((b) => b.rows_per_sec);
    const runs = buckets.map((b) => b.runs);
    const peak = Math.max(...rates);
    host.innerHTML =
      `<div class="chart-head"><span>Rows per second across the session</span>` +
      `<span class="mono dim">peak ${count(peak)}/s</span></div>` +
      sparkline(rates, { width: 520, height: 64, label: 'rows per second', unit: '/s' }) +
      `<div class="chart-foot"><span>${clock(series.start)}</span>` +
      `<span class="dim">${runs.reduce((a, b) => a + b, 0)} runs · ` +
      `${duration(series.end - series.start)} span</span>` +
      `<span>${clock(series.end)}</span></div>`;
  }

  function operatorRollup(rows) {
    if (!rows.length) {
      $('op-rollup').innerHTML = UI.emptyState({
        glyph: 'chart', title: 'No profiled runs yet',
        body: 'Once queries have run, this ranks the <em>kinds</em> of work the session spent ' +
              'time in — joins against scans against sorts — so you can see which class of ' +
              'operation dominates before looking at any single query.',
      });
      return;
    }
    $('op-rollup').innerHTML =
      `<p class="hint">Where the engine spent its time this session, by step type — the ` +
      `workload's shape, which no single run can show.</p>` +
      rows.map((r) => (
        `<div class="rollup-row"><span class="rollup-name">${esc(friendly(r.kind))}</span>` +
        `<span class="rollup-track"><i style="width:${(r.share * 100).toFixed(1)}%"></i></span>` +
        `<span class="rollup-val mono">${ms(r.total_ms)}</span>` +
        `<span class="rollup-pct mono dim">${pct(r.share)}</span></div>` +
        `<div class="rollup-sub dim">${r.runs} step(s) · mean ${ms(r.mean_ms)} · slowest ${ms(r.max_ms)}` +
        `${r.spilled ? ` · ${r.spilled} spilled ${bytes(r.spill_bytes)}` : ''}` +
        (r.slowest_run ? `<button class="xref" data-run="${esc(r.slowest_run)}" type="button">slowest run</button>` : '') +
        `</div>`)).join('');
  }

  function timeSplit(pipelines) {
    const total = pipelines.reduce((s, p) => s + p.total_ms, 0);
    if (!total) {
      $('time-split').innerHTML = UI.emptyState({
        glyph: 'pipeline', title: 'No finished runs yet',
        body: 'This splits the session’s time across pipelines, so a single expensive query ' +
              'is distinguishable from many cheap ones. It fills in as runs complete.',
      });
      return;
    }
    $('time-split').innerHTML =
      `<div class="stack">` + pipelines.map((p, i) => (
        `<i class="stack-seg seq-${Math.min(i + 1, 5)}" style="width:${(p.total_ms / total) * 100}%" ` +
        `title="${esc(friendly(p.label))} · ${ms(p.total_ms)}"></i>`)).join('') + `</div>` +
      pipelines.slice(0, 6).map((p, i) => (
        `<div class="split-row" data-pipe="${esc(p.signature)}" role="button" tabindex="0">` +
        `<i class="swatch seq-${Math.min(i + 1, 5)}"></i>` +
        `<span class="split-name">${esc(friendly(p.label))}</span>` +
        `<span class="split-bar-val mono">${ms(p.total_ms)}</span>` +
        `<span class="split-pct mono">${((p.total_ms / total) * 100).toFixed(0)}%</span></div>`)).join('');
  }

  function attention(queries, insights) {
    const failed = queries.filter((q) => q.status === 'error').slice(0, 4);
    const cards = failed.map((q) => (
      `<div class="insight sev-critical" data-run="${esc(q.query_id)}"><div class="insight-head">` +
      `<span class="insight-sev">failed</span>` +
      `<span class="insight-title">${esc(friendly(q.label))} did not finish</span>` +
      `<span class="insight-rule dim">${ago(q.started_wall)}</span></div>` +
      `<div class="insight-evidence">${esc(q.error || 'The run ended with an error.')}</div>` +
      `<div class="insight-links"><button class="xref" data-run="${esc(q.query_id)}" type="button">` +
      `open this run</button></div></div>`));
    for (const i of (insights || []).filter((x) => x.severity !== 'info').slice(0, 4)) cards.push(card(i));
    $('attention-count').textContent = cards.length;
    $('attention').innerHTML = cards.length ? cards.join('')
      : UI.emptyState({
          glyph: 'check', title: 'Nothing needs attention',
          body: 'No failures, spills, or badly-estimated steps in recent runs. When one ' +
                'appears, it shows here with what happened and what to do about it.',
        });
  }

  function recentTable(queries, onOpen) {
    const columns = [
      { label: 'Query', value: (q) => friendly(q.label),
        render: (q) => `<span class="dot dot-${esc(q.status)}"></span> ${esc(friendly(q.label))}` },
      { label: 'When', value: (q) => q.started_wall, render: (q) => `<span class="dim">${ago(q.started_wall)}</span>` },
      { label: 'Duration', num: true, value: (q) => q.total_ms,
        render: (q) => (q.status === 'running' ? '—' : ms(q.total_ms)) },
      { label: 'Rows', num: true, value: (q) => q.rows, render: (q) => count(q.status === 'running' ? q.rows_seen : q.rows) },
      { label: 'Steps', num: true, value: (q) => q.n_stages || 0 },
      { label: 'Status', value: (q) => q.status,
        render: (q) => (q.status === 'error' ? '<span class="is-critical">failed</span>' : esc(q.status)) },
    ];
    return table('recent-table', columns, queries.slice(0, 25), {
      caption: 'Recent runs',
      emptyText: 'No runs yet — they appear here as queries complete.',
      defaultSort: 1, rowAttrs: (q) => `data-id="${esc(q.query_id)}"`, onRowClick: onOpen,
    });
  }

  /** Failures grouped by cause. Twenty red rows are usually one bug. */
  function failures(groups) {
    const host = $('failures');
    if (!host) return;
    if (!groups.length) {
      host.innerHTML = UI.emptyState({
        glyph: 'check', title: 'No failures',
        body: 'Every run this session completed. Failures group by error message here, so ' +
              'twenty runs hitting one bug read as one cause rather than twenty rows.',
      });
      return;
    }
    host.innerHTML = groups.map((g) => (
      `<div class="insight sev-critical"><div class="insight-head">` +
      `<span class="insight-sev">${g.count}x</span>` +
      `<span class="insight-title mono">${esc(g.error)}</span>` +
      `<span class="insight-rule dim">${ago(g.last_wall)}</span></div>` +
      `<div class="insight-links">` + g.runs.slice(0, 6).map((id, i) => (
        `<button class="xref" data-run="${esc(id)}" type="button">run ${i + 1}</button>`)).join('') +
      `</div></div>`)).join('');
  }

  /** What is true of a pipeline across all its runs, not just the newest. */
  function pipelineReport(report) {
    const host = $('p-report');
    if (!host) return;
    if (!report || !report.runs) { host.innerHTML = ''; return; }
    const always = report.steps.filter((s) => s.critical_share >= 0.999);
    host.innerHTML =
      (report.recurring.length
        ? `<h3 class="sub-head">Recurring findings</h3>` + report.recurring.map((r) => (
            `<div class="insight sev-${esc(r.severity)}"><div class="insight-head">` +
            `<span class="insight-sev">${r.count} of ${report.runs} runs</span>` +
            `<span class="insight-title">${esc(r.title)}</span>` +
            (r.chronic ? `<span class="chip is-warn">chronic</span>` : `<span class="chip">occasional</span>`) +
            `</div><div class="insight-action"><b>What to do:</b> ${esc(r.action)}</div></div>`)).join('')
        : `<p class="hint">No finding has recurred across these runs.</p>`) +
      (always.length
        ? `<h3 class="sub-head">Always on the critical path</h3>` +
          `<p class="hint">These steps set the floor on this pipeline's time in every run, so ` +
          `tuning them pays every time — unlike a step that only sometimes matters.</p>` +
          always.slice(0, 5).map((s) => (
            `<div class="split-line"><span class="split-name">${esc(friendly(s.kind))}` +
            `<span class="mono dim"> ${esc(s.detail || '')}</span></span>` +
            `<span class="split-bar-val mono">${ms(s.mean_ms)} mean</span>` +
            `<span class="split-pct mono">${ms(s.max_ms)} max</span></div>`)).join('')
        : '');
  }

  /* The runs x steps matrix — the pattern Airflow's grid view established, and for its
   * reason: a graph shows one run's structure but cannot show one step across twenty runs.
   * Cells are shaded by ratio to that step's own median, so a slow cell stands out whether
   * the step takes microseconds or minutes. */
  function runGrid(grid, { onOpenRun, onSelectStep }) {
    const host = $('p-grid');
    if (!host) return;
    if (!grid || !grid.steps.length) {
      host.innerHTML = UI.emptyState({
        glyph: 'chart', title: 'No step history yet',
        body: 'Once this pipeline has run more than once, every step appears here as a row ' +
              'and every run as a column — so a step that got slower is a stripe you can see.',
      });
      return;
    }
    const cell = new Map(grid.cells.map((c) => [`${c.op_id}:${c.run}`, c]));
    const header = grid.runs.map((r, i) => (
      `<th class="grid-run" data-col="${esc(String(i))}" data-run="${esc(r.query_id)}" title="${UI.clock(r.started_wall)} · ${ms(r.total_ms)}">` +
      `<span class="dot dot-${esc(r.status)}"></span>${i + 1}</th>`)).join('');
    const body = grid.steps.map((step) => {
      const cells = grid.runs.map((r, i) => {
        const c = cell.get(`${step.op_id}:${i}`);
        if (!c || c.elapsed_ms == null) {
          return `<td class="grid-cell is-absent" title="not measured in this run"></td>`;
        }
        // Five buckets rather than a continuous ramp: adjacent cells must be tellable apart.
        const ratio = c.ratio ?? 1;
        const level = ratio >= 2 ? 5 : ratio >= 1.35 ? 4 : ratio >= 0.85 ? 3 : ratio >= 0.5 ? 2 : 1;
        // A grid cell is a real button target: clickable to open the run, and reachable by
        // keyboard (a `<td>` is not focusable on its own). `data-col` drives the crosshair;
        // `data-run` carries the query id the click opens.
        return `<td class="grid-cell lvl-${level}${c.spilled ? ' is-spilled' : ''}" ` +
          `data-col="${esc(String(c.run))}" data-step="${esc(String(c.op_id))}" ` +
          `data-run="${esc(r.query_id)}" tabindex="0" role="button" ` +
          `aria-label="${esc(friendly(step.kind))}, run ${i + 1}, ${ms(c.elapsed_ms)}, ` +
          `${ratio.toFixed(2)} times its median${c.spilled ? ', spilled' : ''}" ` +
          `title="${esc(friendly(step.kind))} · run ${i + 1} · ${ms(c.elapsed_ms)} · ` +
          `${ratio.toFixed(2)}x its median${c.spilled ? ' · spilled' : ''}"></td>`;
      }).join('');
      return `<tr><th class="grid-step" data-op="${step.op_id}" title="median ${ms(step.median_ms)}">` +
        `${esc(friendly(step.kind))}<span class="dim mono"> ${ms(step.median_ms)}</span></th>${cells}</tr>`;
    }).join('');
    host.innerHTML =
      `<p class="hint">Each column is a run (oldest left), each row a step. Cells are shaded by ` +
      `how the step compared to its own median that run, so a slow step is a visible stripe.</p>` +
      `<div class="scroll-x"><table class="grid"><thead><tr><th></th>${header}</tr></thead>` +
      `<tbody>${body}</tbody></table></div>` +
      `<div class="legend"><span><i class="swatch lvl-1"></i>much faster</span>` +
      `<span><i class="swatch lvl-3"></i>typical</span>` +
      `<span><i class="swatch lvl-5"></i>2x slower or worse</span>` +
      `<span><i class="swatch is-spilled-key"></i>spilled</span></div>`;
    for (const el of host.querySelectorAll('[data-run]')) {
      el.addEventListener('click', () => onOpenRun(el.dataset.run));
    }
    for (const el of host.querySelectorAll('[data-op]')) {
      el.addEventListener('click', () => onSelectStep(Number(el.dataset.op)));
    }
  }

  /* ═══════════ pipelines ═══════════ */

  function pipelines(list, { onOpen, onPin, sort, needle }) {
    let shown = [...list];
    if (needle) shown = shown.filter((p) => UI.fuzzy(needle, friendly(p.label) + p.signature));
    const sorters = {
      time: (a, b) => b.total_ms - a.total_ms,
      runs: (a, b) => b.runs - a.runs,
      slowest: (a, b) => b.median_ms - a.median_ms,
      recent: (a, b) => b.last_wall - a.last_wall,
      name: (a, b) => friendly(a.label).localeCompare(friendly(b.label)),
    };
    shown.sort(sorters[sort] || sorters.time);
    // Pinned first, order otherwise preserved — a pin is a bookmark, not a re-ranking.
    shown.sort((a, b) => Number(UI.isPinned(b.signature)) - Number(UI.isPinned(a.signature)));

    $('pipelines-empty').hidden = shown.length > 0;
    if (!shown.length && list.length) {
      // Filtered to nothing is a different problem from having nothing, and needs a
      // different way out.
      $('pipelines-empty').hidden = false;
      $('pipelines-empty').className = '';
      $('pipelines-empty').innerHTML = UI.emptyState({
        glyph: 'search', title: 'No pipelines match',
        body: `Nothing matches <b>${esc(needle)}</b>. Clear the filter to see all ` +
              `${list.length} pipelines.`,
      });
    }
    $('pipeline-cards').innerHTML = shown.map((p) => {
      const pc = p.percentiles || {};
      const recent = p.recent_ms || [];
      const drift = recent.length > 2 ? recent[recent.length - 1] / (median(recent.slice(0, -1)) || 1) : 1;
      const driftChip = recent.length > 2
        ? (drift > 1.3 ? `<span class="chip is-warn">${drift.toFixed(1)}x slower than usual</span>`
          : drift < 0.77 ? `<span class="chip is-good">${(1 / drift).toFixed(1)}x faster</span>`
          : `<span class="chip">steady</span>`) : '';
      const health = p.n_failed ? 'critical' : (drift > 1.3 ? 'warn' : 'ok');
      return `<article class="pcard has-ribbon is-${health}" data-sig="${esc(p.signature)}" ` +
        `tabindex="0" role="button" aria-label="Open pipeline ${esc(friendly(p.label))}">` +
        `<div class="pcard-head"><span class="dot dot-${esc(p.last_status)}"></span>` +
        `<span class="pcard-name">${esc(friendly(p.label))}</span>` +
        `<button class="pin${UI.isPinned(p.signature) ? ' is-on' : ''}" data-pin="${esc(p.signature)}" ` +
        `type="button" title="Pin this pipeline" aria-label="Pin ${esc(friendly(p.label))}">★</button></div>` +
        `<div class="pcard-metrics">` +
        `<span class="metric"><b>${p.runs}</b><span>runs</span></span>` +
        `<span class="metric"><b>${ms(pc.p50 ?? p.median_ms)}</b><span>${term('p95', 'p50')}</span></span>` +
        `<span class="metric"><b>${ms(pc.p95 ?? p.max_ms)}</b><span>p95</span></span>` +
        `<span class="metric${p.n_failed ? ' is-critical' : ''}"><b>${p.n_failed}</b><span>failed</span></span>` +
        `</div>` +
        (recent.length > 1
          ? `<div class="pcard-charts"><div class="chartlet"><span class="chartlet-label">trend</span>` +
            sparkline(recent, { width: 150, height: 34, label: 'recent durations', unit: 'ms' }) + `</div>` +
            (recent.length > 3 ? `<div class="chartlet"><span class="chartlet-label">spread</span>` +
              histogram(recent, { width: 150, height: 34 }) + `</div>` : '') + `</div>`
          : `<p class="hint">Run it again to see a trend.</p>`) +
        `<div class="pcard-foot">${driftChip}` +
        `<span class="dim">${count(p.total_rows)} rows · ${ago(p.last_wall)}</span></div>` +
        `<div class="pcard-actions"><span class="dim">Open pipeline →</span></div></article>`;
    }).join('');

    for (const b of $('pipeline-cards').querySelectorAll('[data-pin]')) {
      b.addEventListener('click', (e) => { e.stopPropagation(); onPin(b.dataset.pin); });
    }
    for (const card of $('pipeline-cards').querySelectorAll('[data-sig]')) {
      card.addEventListener('click', () => onOpen(card.dataset.sig));
      // A card that opens on click must open on Enter/Space too, or the footer's
      // "Open pipeline →" is a promise the keyboard cannot keep.
      card.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onOpen(card.dataset.sig); }
      });
    }
    $('pipeline-count').textContent = shown.length;
    return shown;
  }

  /** The same pipelines as a dense table, for someone scanning twenty of them. */
  function pipelineTable(list, onOpen) {
    const columns = [
      { label: 'Pipeline', value: (p) => friendly(p.label),
        render: (p) => `<span class="dot dot-${esc(p.last_status)}"></span> ${esc(friendly(p.label))}` +
                       (UI.isPinned(p.signature) ? ' <span class="pin is-on">★</span>' : '') },
      { label: 'Runs', num: true, value: (p) => p.runs },
      { label: 'p50', num: true, value: (p) => (p.percentiles || {}).p50 ?? p.median_ms,
        render: (p) => ms((p.percentiles || {}).p50 ?? p.median_ms) },
      { label: 'p95', num: true, value: (p) => (p.percentiles || {}).p95 ?? p.max_ms,
        render: (p) => ms((p.percentiles || {}).p95 ?? p.max_ms) },
      { label: 'Fastest', num: true, value: (p) => p.min_ms, render: (p) => ms(p.min_ms) },
      { label: 'Slowest', num: true, value: (p) => p.max_ms, render: (p) => ms(p.max_ms) },
      { label: 'Total time', num: true, value: (p) => p.total_ms, render: (p) => ms(p.total_ms) },
      { label: 'Rows', num: true, value: (p) => p.total_rows, render: (p) => count(p.total_rows) },
      { label: 'Failed', num: true, value: (p) => p.n_failed,
        cls: (p) => (p.n_failed ? 'is-critical' : '') },
      { label: 'Last run', value: (p) => p.last_wall, render: (p) => ago(p.last_wall) },
      { label: 'Trend', sortable: false, value: () => 0,
        render: (p) => (p.recent_ms.length > 1 ? sparkline(p.recent_ms, { width: 90, height: 22, fill: false, label: 'durations', unit: 'ms' }) : '') },
    ];
    table('pipeline-table', columns, list, {
      caption: 'All pipelines',
      emptyText: 'No pipelines yet. Every distinct query shape becomes one.',
      defaultSort: 6,
      rowAttrs: (p) => `data-id="${esc(p.signature)}"`,
      onRowClick: onOpen,
    });
  }

  /* ═══════════ one pipeline ═══════════ */

  /** The header numbers for a single pipeline — its behaviour, not the engine's. */
  function pipelineDetail(p, runs, { onOpenRun, onCompare }) {
    const pc = p.percentiles || {};
    $('p-title').textContent = friendly(p.label);
    $('p-sub').innerHTML =
      `${p.runs} run${p.runs === 1 ? '' : 's'} · last ${ago(p.last_wall)} · ` +
      `<span class="mono dim">${esc(p.signature.slice(0, 12))}</span>`;
    $('p-pin').classList.toggle('is-on', UI.isPinned(p.signature));
    $('p-pin').textContent = UI.isPinned(p.signature) ? '★ Pinned' : '★ Pin';

    const recent = p.recent_ms || [];
    const drift = recent.length > 2 ? recent[recent.length - 1] / (median(recent.slice(0, -1)) || 1) : null;
    const badge = $('p-verdict');
    if (drift == null) { badge.hidden = true; } else {
      badge.hidden = false;
      if (drift > 1.3) { badge.className = 'verdict is-warn'; badge.textContent = `getting slower — ${drift.toFixed(1)}x`; }
      else if (drift < 0.77) { badge.className = 'verdict is-good'; badge.textContent = `getting faster — ${(1 / drift).toFixed(1)}x`; }
      else { badge.className = 'verdict is-good'; badge.textContent = 'steady'; }
    }

    const cells = [
      ['Runs', p.runs, `${p.n_failed} failed`],
      ['Typical (p50)', ms(pc.p50 ?? p.median_ms), ''],
      ['p95', ms(pc.p95 ?? p.max_ms), pc.reliable ? `over ${pc.count} runs` : `only ${pc.count || 0} runs`],
      ['Fastest', ms(p.min_ms), ''],
      ['Slowest', ms(p.max_ms), ''],
      ['Total time', ms(p.total_ms), 'all runs'],
      ['Rows produced', count(p.total_rows), 'all runs'],
      ['Consistency', pc.p50 ? `${((pc.p95 ?? p.max_ms) / pc.p50).toFixed(1)}x` : '—', 'p95 ÷ p50'],
    ];
    $('p-strip').innerHTML = cells.map(([k, v, note]) => (
      `<div class="stat"><span class="stat-label">${esc(k)}</span>` +
      `<span class="stat-value">${esc(String(v))}</span><span class="stat-note">${esc(note)}</span></div>`)).join('');

    if (recent.length <= 1) {
      // A trend from one point is a dot, not a trend. Say so rather than drawing nothing.
      $('p-trend').innerHTML = UI.emptyState({
        glyph: 'clock', title: 'Not enough runs for a trend',
        body: 'Run this pipeline once more and its duration over time appears here.',
      });
    }
    $('p-trend').innerHTML = recent.length > 1
      ? `<div class="chart-head"><span>Each bar is one run</span>` +
        `<span class="mono dim">${ms(p.min_ms)} – ${ms(p.max_ms)}</span></div>` +
        sparkline(recent, { width: 420, height: 70, label: 'duration over time', unit: 'ms' }) +
        `<div class="chart-foot"><span>oldest</span><span>newest</span></div>`
      : '<p class="empty">Run it again to see a trend.</p>';

    if (recent.length <= 3) {
      // Percentiles over three samples are the max wearing a lab coat.
      $('p-spread').innerHTML = UI.emptyState({
        glyph: 'chart', title: 'Not enough runs for a spread',
        body: 'A distribution needs a handful of runs before it describes anything. ' +
              'This fills in at four.',
      });
    }
    $('p-spread').innerHTML = recent.length > 3
      ? `<div class="chart-head"><span>How often it lands where</span>` +
        `<span class="mono dim">${(pc.p95 / (pc.p50 || 1)).toFixed(1)}x spread</span></div>` +
        histogram(recent, { width: 420, height: 70 }) +
        `<div class="chart-foot"><span>${ms(p.min_ms)}</span><span>${ms(p.max_ms)}</span></div>`
      : '<p class="empty">A few more runs and the spread appears here.</p>';

    $('p-dag-note').innerHTML =
      `This pipeline always runs the same plan — that is what makes it one pipeline. Each ` +
      `step shows what it <b>typically</b> costs across its ${p.runs} run${p.runs === 1 ? '' : 's'}, ` +
      `not one run's figure.`;
    $('p-compare').disabled = !(p.fastest_id && p.slowest_id && p.fastest_id !== p.slowest_id);
    $('p-compare').onclick = () => onCompare(p.slowest_id, p.fastest_id);
    pipelineRuns(runs, onOpenRun);
  }

  /** Every run of one pipeline, as a table you can sort. */
  function pipelineRuns(runs, onOpen) {
    $('p-run-count').textContent = runs.length;
    if (!runs.length) {
      $('p-runs').innerHTML = UI.emptyState({
        glyph: 'search', title: 'No runs match this filter',
        body: 'Clear the status filter above to see every run of this pipeline.',
      });
      return;
    }
    const slowest = Math.max(...runs.map((r) => r.total_ms || 0), 1);
    const columns = [
      { label: 'When', value: (r) => r.started_wall,
        render: (r) => `<span class="dot dot-${esc(r.status)}"></span> ${UI.clock(r.started_wall)}` +
                       `<span class="dim"> · ${ago(r.started_wall)}</span>` },
      { label: 'Duration', num: true, value: (r) => r.total_ms,
        // The bar rides inside the cell rather than in a column of its own: a number and its
        // magnitude belong together, and a separate column made the table wider for nothing.
        render: (r) => (r.status === 'running' ? 'running'
          : `<span class="cell-bar"><i style="width:${Math.max(2, ((r.total_ms || 0) / slowest) * 100)}%"></i>` +
            `<b>${ms(r.total_ms)}</b></span>`) },
      { label: 'Rows', num: true, value: (r) => r.rows, render: (r) => count(r.status === 'running' ? r.rows_seen : r.rows) },
      { label: 'Steps', num: true, value: (r) => r.n_stages || 0 },
      { label: 'Status', value: (r) => r.status,
        render: (r) => (r.status === 'error' ? '<span class="is-critical">failed</span>' : esc(r.status)) },
    ];
    table('p-runs', columns, runs, {
      caption: 'Runs of this pipeline',
      emptyText: 'No runs of this pipeline yet.',
      defaultSort: 0, rowAttrs: (r) => `data-id="${esc(r.query_id)}"`, onRowClick: onOpen,
    });
  }

  /* ═══════════ run detail ═══════════ */

  /** Where the run's wall clock went: the steps versus the fixed cost around them. Only what
   *  the profile actually measures — no invented planning/queue/exec split. */
  function timeBar(d) {
    const host = $('d-timebar');
    if (!host) return;
    const nodes = (d.dag?.nodes || []).filter((n) => n.measured);
    const total = d.total_ms || 0;
    if (!nodes.length || total <= 0) { host.hidden = true; return; }
    // Operator time is summed across steps and can exceed wall clock under parallelism, so
    // clamp the executed share to the run's own duration — the bar shows time, not a sum.
    const executed = Math.min(total, nodes.reduce((sum, n) => sum + (n.elapsed_ms || 0), 0));
    const overhead = Math.max(0, total - executed);
    const execPct = (executed / total) * 100;
    host.hidden = false;
    host.innerHTML =
      `<div class="timebar-track" role="img" ` +
      `aria-label="Of ${ms(total)}, ${ms(executed)} in the steps and ${ms(overhead)} around them">` +
      `<span class="timebar-exec" style="width:${execPct.toFixed(1)}%"></span>` +
      `<span class="timebar-over" style="width:${(100 - execPct).toFixed(1)}%"></span></div>` +
      `<div class="timebar-key"><span><i class="sw-exec"></i>steps ${ms(executed)}</span>` +
      `<span><i class="sw-over"></i>planning & setup ${ms(overhead)}</span></div>`;
  }

  function verdict(d) {
    const el = $('d-verdict');
    const b = d.baseline;
    if (d.status === 'error') {
      el.hidden = false; el.className = 'verdict is-critical'; el.textContent = 'failed';
      el.title = 'This run did not complete.';
      return;
    }
    if (!b || !b.ratio) {
      el.hidden = false; el.className = 'verdict';
      // Not silence: "we cannot judge yet" is itself information. A first run has nothing to
      // compare against, and saying so beats an empty space the reader has to interpret.
      el.textContent = 'first run — no baseline yet';
      el.title = 'A verdict needs other runs of this same query shape to compare against.';
      return;
    }
    el.hidden = false;
    // The arrow carries direction; the class carries whether that direction is good. Slower
    // is bad, faster is good — but both are shown, because "5x faster" is worth seeing too.
    const median = b.median_ms != null ? ` (${ms(b.median_ms)} typical)` : '';
    if (b.ratio > 1.3) {
      el.className = 'verdict is-warn';
      el.innerHTML = `\u25b2 ${b.ratio.toFixed(1)}\u00d7 slower than usual`;
      el.title = `This run took ${ms(d.total_ms)}${median}.`;
    } else if (b.ratio < 0.77) {
      el.className = 'verdict is-good';
      el.innerHTML = `\u25bc ${(1 / b.ratio).toFixed(1)}\u00d7 faster than usual`;
      el.title = `This run took ${ms(d.total_ms)}${median}.`;
    } else {
      el.className = 'verdict is-good';
      el.textContent = 'typical for this pipeline';
      el.title = `This run took ${ms(d.total_ms)}${median} \u2014 within the usual range.`;
    }
  }

  function statStrip(d) {
    const nodes = (d.dag?.nodes || []).filter((n) => n.measured);
    const strip = $('d-strip');
    if (!nodes.length) { strip.hidden = true; return; }
    strip.hidden = false;
    const read = nodes.filter((n) => n.kind === 'scan').reduce((s, n) => s + n.rows_out, 0);
    const opTotal = nodes.reduce((s, n) => s + n.elapsed_ms, 0);
    const peak = Math.max(0, ...nodes.map((n) => n.peak_rss_bytes || 0));
    const spill = nodes.reduce((s, n) => s + (n.spill_bytes || 0), 0);
    // Averaged over measured operators only; a tier that does not sample the CPU clock
    // contributes nothing rather than dragging the mean toward 1/threads.
    const cpuNodes = nodes.filter(UI.cpuMeasured);
    const cpuTotal = cpuNodes.reduce((s, n) => s + n.elapsed_ms, 0);
    const cpu = cpuTotal > 0
      ? cpuNodes.reduce((s, n) => s + n.cpu_util * n.elapsed_ms, 0) / cpuTotal : 0;
    const threads = Math.max(0, ...nodes.map((n) => n.threads || 0));
    const cells = [
      ['Duration', ms(d.total_ms), 'wall clock'],
      ['Rows read', count(read), 'from sources'],
      ['Rows out', count(d.rows), ''],
      ['Steps', nodes.length, `${(d.dag.critical_path || []).length} on the ${term('critical path', 'critical path')}`],
      ['Operator time', ms(opTotal), term('operator time', 'summed, concurrent')],
      ['CPU', pct(cpu), `${threads} threads`],
      ['Peak memory', bytes(peak), ''],
      ['Spilled', spill ? bytes(spill) : 'no', ''],
    ];
    // Each tile carries a fill showing where it sits in its own sensible range, so the strip
    // reads as a set of gauges rather than eight disconnected numerals.
    const fills = [null, read ? Math.min(1, read / 1e6) : 0, d.rows ? Math.min(1, d.rows / read) : 0,
                   nodes.length / 12, null, cpu, peak ? Math.min(1, peak / 2 ** 30) : 0,
                   spill ? Math.min(1, spill / 2 ** 30) : 0];
    strip.innerHTML = cells.map(([k, v, note], i) => (
      `<div class="stat"><span class="stat-label">${esc(k)}</span>` +
      `<span class="stat-value">${esc(String(v))}</span><span class="stat-note">${note}</span>` +
      (fills[i] ? `<span class="stat-fill"><i style="width:${Math.min(100, fills[i] * 100).toFixed(0)}%"></i></span>` : '') +
      `</div>`)).join('');
  }

  function story(d) {
    if (d.status === 'error') {
      return `<b class="is-critical">This run failed</b> after ${ms(d.total_ms)}. ` +
             `<span class="mono">${esc(d.error || 'No error message was recorded.')}</span>`;
    }
    const nodes = (d.dag?.nodes || []).filter((n) => n.measured);
    if (!nodes.length) {
      return `Finished in <b>${ms(d.total_ms)}</b>, producing <b>${count(d.rows)} rows</b>. ` +
             `No per-step detail: streamed queries report live rows but no profile, and a query ` +
             `answered from metadata never reaches the optimizer.`;
    }
    const read = nodes.filter((n) => n.kind === 'scan').reduce((s, n) => s + n.rows_out, 0);
    const opTotal = nodes.reduce((s, n) => s + n.elapsed_ms, 0);
    const slowest = nodes.reduce((a, b) => (a.elapsed_ms > b.elapsed_ms ? a : b), nodes[0]);
    const share = opTotal > 0 ? slowest.elapsed_ms / opTotal : 0;
    const spilled = nodes.filter((n) => n.spilled);
    let text = `Read <b>${count(read)} rows</b>, returned <b>${count(d.rows)}</b>, in ` +
               `<b>${ms(d.total_ms)}</b> across ${nodes.length} steps. `;
    text += share > 0.4
      ? `Most of the work was <b>${esc(friendly(slowest.kind))}</b> (${pct(share)} of ${term('operator time')}).`
      : `No single step dominated — time was spread across the plan.`;
    if (spilled.length) {
      text += ` <span class="is-serious">${spilled.length} step(s) ${term('spill', 'spilled to disk')}</span>.`;
    }
    if (d.baseline?.ratio > 1.3) {
      text += ` This run took <b>${d.baseline.ratio.toFixed(1)}x</b> the usual ${ms(d.baseline.median_ms)} ` +
              `for this ${term('pipeline')} (${term('baseline', `${d.baseline.runs} other runs`)}).`;
    }
    return text;
  }

  function timeline(all) {
    const nodes = all.filter((n) => n.measured);
    if (!nodes.length) {
      $('timeline').innerHTML = UI.emptyState({
        glyph: 'clock', title: 'No step timings for this run',
        body: 'The timeline shows each step as a bar, so a step that ran long is a stripe you ' +
              'can see. It fills in once a run records per-step timing.',
      });
      return;
    }
    const max = Math.max(...nodes.map((n) => n.elapsed_ms), 1);
    const total = nodes.reduce((s, n) => s + n.elapsed_ms, 0) || 1;
    $('timeline').innerHTML =
      `<p class="hint">Steps run concurrently across cores, so these sum to more than the run's ` +
      `wall-clock time. Bars share one scale — the longest is the step to tune.</p>` +
      [...nodes].sort((a, b) => b.elapsed_ms - a.elapsed_ms).map((n) => (
        `<div class="row" data-op="${n.op_id}" role="button" tabindex="0"><div class="row-name" title="${esc(n.kind)}">` +
        `${n.on_critical_path ? '<i class="crit-dot" title="on the critical path"></i>' : ''}` +
        `${esc(friendly(n.kind))}</div>` +
        `<div class="track"><div class="bar${n.spilled ? ' is-spilled' : ''}" ` +
        `style="width:${Math.max(2, (n.elapsed_ms / max) * 100)}%"></div></div>` +
        `<div class="bar-value">${ms(n.elapsed_ms)} · ${pct(n.elapsed_ms / total)}</div></div>`)).join('');
  }

  const OPERATOR_COLUMNS = [
    { label: 'Step', value: (n) => friendly(n.kind),
      render: (n) => `${n.on_critical_path ? '<i class="crit-dot"></i>' : ''}${esc(friendly(n.kind))}` },
    { label: 'Detail', optional: true, value: (n) => n.detail || '', render: (n) => `<span class="mono dim">${esc(n.detail || '—')}</span>` },
    { label: 'Rows in', optional: true, num: true, value: (n) => n.rows_in || 0, render: (n) => (n.rows_in ? n.rows_in.toLocaleString() : '—') },
    { label: 'Rows out', num: true, value: (n) => n.rows_out || 0, render: (n) => (n.rows_out || 0).toLocaleString() },
    { label: 'Selectivity', num: true, value: (n) => n.selectivity ?? -1, render: (n) => pct(n.selectivity),
      help: 'Rows out ÷ rows in' },
    { label: 'Expected', num: true, value: (n) => n.est_rows ?? -1,
      render: (n) => (n.est_rows == null ? '—' : Math.round(n.est_rows).toLocaleString()) },
    { label: 'vs exp.', num: true, value: (n) => n.est_error ?? 1,
      render: (n) => (n.est_error == null ? '—' : `${n.est_error.toFixed(1)}x`),
      cls: (n) => (n.est_error != null && (n.est_error > 10 || n.est_error < 0.1) ? 'is-warn' : ''),
      help: 'Actual rows ÷ the planner’s estimate. Over 10x means it costed the wrong query.' },
    { label: 'Time', num: true, value: (n) => n.elapsed_ms || 0, render: (n) => (n.measured ? ms(n.elapsed_ms) : '—') },
    { label: 'CPU', optional: true, num: true, value: (n) => (UI.cpuMeasured(n) ? n.cpu_util : 0),
      render: (n) => (UI.cpuMeasured(n) ? pct(n.cpu_util) : '—') },
    { label: 'Threads', optional: true, num: true, value: (n) => n.threads || 0, render: (n) => n.threads || '—' },
    { label: 'Output', optional: true, num: true, value: (n) => n.result_bytes || 0, render: (n) => bytes(n.result_bytes) },
    { label: 'Spilled', value: (n) => (n.spilled ? n.spill_bytes : 0), render: (n) => (n.spilled ? bytes(n.spill_bytes) : 'no') },
  ];

  function operators(nodes, onSelect) {
    if (!nodes.length) {
      $('operators').innerHTML = UI.emptyState({
        glyph: 'chart', title: 'No steps recorded',
        body: 'The step table lists every operator with its rows, time, and how close the ' +
              'planner\u2019s estimate was. It appears once a run has a measured plan.',
      });
      return null;
    }
    return table('operators', OPERATOR_COLUMNS, nodes, {
      caption: 'Steps in this run',
      emptyText: 'This run recorded no per-step detail.',
      defaultSort: 7,
      rowAttrs: (n) => `data-id="${n.op_id}"${n.on_critical_path ? ' class="is-crit-row"' : ''}`,
      onRowClick: (id) => onSelect(Number(id)),
    });
  }

  /* An insight names a step; the step lives in the plan. Linking the two is the difference
   * between advice you must go hunt for and advice you can act on. */
  function card(i, { runId } = {}) {
    const opId = i.detail && i.detail.op_id;
    return `<div class="insight sev-${esc(i.severity)}"><div class="insight-head">` +
      `<span class="insight-sev">${esc(i.severity)}</span>` +
      `<span class="insight-title">${esc(i.title)}</span>` +
      `<span class="insight-rule mono">${esc(i.rule)}</span></div>` +
      `<div class="insight-evidence">${LEARN.autolink(i.evidence)}</div>` +
      `<div class="insight-action"><b>What to do:</b> ${LEARN.autolink(i.action)}</div>` +
      ((opId != null || runId) ? `<div class="insight-links">` +
        (opId != null ? `<button class="xref" data-op="${opId}" type="button">show this step in the plan</button>` : '') +
        (runId ? `<button class="xref" data-run="${esc(runId)}" type="button">open the run</button>` : '') +
        `</div>` : '') +
      `</div>`;
  }

  function insights(list, runId) {
    const badge = $('insight-count');
    badge.hidden = list.length === 0;
    badge.textContent = list.length;
    badge.classList.toggle('is-warn', list.some((i) => i.severity !== 'info'));
    $('insights').innerHTML = list.length ? list.map((i) => card(i, { runId })).join('')
      : '<p class="empty">Nothing to flag — this run looks healthy. Insights appear when a step ' +
        'spills to disk, the planner badly misjudges row counts, one step dominates, the cores ' +
        'sit idle, or a scan reads far more than a filter keeps.</p>';
  }

  function decisions(list) {
    const badge = $('decision-count');
    badge.hidden = list.length === 0;
    badge.textContent = list.length;
    $('decisions').innerHTML = list.length
      ? `<p class="hint">Choices the optimizer and the resource manager made for this run.</p>` +
        list.map((d) => (
          `<div class="decision"><div class="decision-head">` +
          `<span class="tagpill">${esc(d.subsystem)} · ${esc(d.category)}</span>` +
          `<span>${esc(d.summary)}</span></div>` +
          (d.detail && Object.keys(d.detail).length
            ? `<div class="decision-detail mono">${esc(JSON.stringify(d.detail))}</div>` : '') + `</div>`)).join('')
      : UI.emptyState({
          glyph: 'pipeline', title: 'No optimizer decisions recorded',
          body: 'When the optimizer rewrites a query \u2014 pushing a filter down, choosing a ' +
                'join order \u2014 each change is listed here with the rule that made it.',
        });
  }

  function comparison(cmp, aLabel, bLabel) {
    const host = $('compare');
    if (!cmp) { host.innerHTML = '<p class="empty">Pick a run to compare against from the Pipelines view.</p>'; return; }
    if (!cmp.ok && !cmp.steps.length) {
      host.innerHTML = `<p class="empty">${esc(cmp.reason)}</p>` + totalsTable(cmp.totals, aLabel, bLabel);
      return;
    }
    host.innerHTML =
      `<p class="hint">Steps are matched by their position in the plan, so the same step is ` +
      `compared against itself. Sorted by the biggest change.</p>` +
      totalsTable(cmp.totals, aLabel, bLabel) +
      `<table class="dense"><thead><tr><th>Step</th><th class="num">${esc(aLabel)}</th>` +
      `<th class="num">${esc(bLabel)}</th><th class="num">Change</th><th class="num">Ratio</th></tr></thead><tbody>` +
      (function(){
        const biggest = Math.max(1, ...cmp.steps.map((s) => Math.abs(s.delta_ms || 0)));
        return cmp.steps.map((s) => {
          const delta = s.delta_ms || 0;
          const worse = delta > 0;
          // Arrow = direction of change; colour = whether that direction is good. Slower is
          // bad, faster is good, and both are shown so an improvement is as visible as a
          // regression.
          const arrow = s.delta_ms == null ? '' : worse ? '\u25b2' : '\u25bc';
          const cls = s.delta_ms == null ? '' : worse ? 'is-warn' : 'is-good';
          const barW = (Math.abs(delta) / biggest) * 100;
          return `<tr><td>${esc(friendly(s.kind))}<span class="mono dim"> ${esc(s.detail || '')}</span></td>` +
            `<td class="num">${ms(s.a_ms)}</td><td class="num">${ms(s.b_ms)}</td>` +
            `<td class="num ${cls}"><span class="cmp-bar"><i class="${cls}" style="width:${barW.toFixed(0)}%"></i>` +
            `<b>${s.delta_ms == null ? '\u2014' : `${arrow} ${Math.abs(delta).toFixed(1)}ms`}</b></span></td>` +
            `<td class="num">${s.ratio == null ? '\u2014' : `${s.ratio.toFixed(2)}x`}</td></tr>`;
        }).join('');
      })() + `</tbody></table>`;
  }

  function totalsTable(totals, aLabel, bLabel) {
    if (!totals?.length) return '';
    return `<table class="dense totals"><thead><tr><th>Metric</th><th class="num">${esc(aLabel)}</th>` +
      `<th class="num">${esc(bLabel)}</th><th class="num">Change</th></tr></thead><tbody>` +
      totals.map((t) => {
        const fmt = t.unit === 'ms' ? ms : count;
        // Only duration has a "worse" direction; a change in rows returned is a different
        // result, not a regression, so it is shown without a good/bad colour.
        const judged = t.unit === 'ms' && t.delta;
        const worse = (t.delta || 0) > 0;
        const arrow = !t.delta ? '' : worse ? '\u25b2' : '\u25bc';
        const cls = judged ? (worse ? 'is-warn' : 'is-good') : '';
        return `<tr><td>${esc(t.label)}</td><td class="num">${fmt(t.a)}</td><td class="num">${fmt(t.b)}</td>` +
          `<td class="num ${cls}">${arrow} ${t.ratio == null ? '\u2014' : `${t.ratio.toFixed(2)}x`}</td></tr>`;
      }).join('') + `</tbody></table>`;
  }

  function meta(d) {
    const p = d.profile || {};
    const b = d.baseline;
    const groups = [
      ['This run', [['run id', d.query_id], ['pipeline', (d.signature || '—').slice(0, 12)],
                    ['started', UI.stamp(d.started_wall)], ['duration', ms(d.total_ms)],
                    ['rows returned', count(d.rows)], ['status', d.status]]],
      ['Compared to this pipeline', b
        ? [['other runs', b.runs], ['typical (p50)', ms(b.median_ms)], ['fastest', ms(b.fastest_ms)],
           ['slowest', ms(b.slowest_ms)], ['this run', b.ratio ? `${b.ratio.toFixed(2)}x typical` : '—']]
        : [['history', 'this is the only run of this shape']]],
      ['Execution', [['steps', (d.dag?.nodes || []).length || '—'],
                     ['critical path', `${(d.dag?.critical_path || []).length} steps`],
                     ['ran distributed', p.distributed == null ? '—' : (p.distributed ? 'yes' : 'no')],
                     ['spilled to disk', (d.dag?.nodes || []).some((n) => n.spilled) ? 'yes' : 'no'],
                     ['peak memory', bytes(Math.max(0, ...(p.ops || []).map((o) => o.peak_rss_bytes || 0)))],
                     ['memory budget', bytes(p.memory_budget_bytes)]]],
    ];
    $('meta').innerHTML = groups.map(([title, rows]) => (
      `<div class="meta-group"><h3>${esc(title)}</h3><dl>` +
      rows.map(([k, v]) => `<div class="meta-row"><dt>${esc(k)}</dt><dd>${esc(String(v ?? '—'))}</dd></div>`).join('') +
      `</dl></div>`)).join('');
  }

  /* ═══════════ system ═══════════ */

  function system(sys) {
    const host = sys.host || {}, eng = sys.engine || {}, cluster = sys.cluster || {}, cfg = sys.config || {};
    const gpus = host.gpus || [];
    const usedPct = host.memory_total_bytes && host.memory_available_bytes != null
      ? 1 - host.memory_available_bytes / host.memory_total_bytes : null;
    const groups = [
      ['Machine', [['hostname', host.hostname], ['operating system', `${host.platform} · ${host.arch}`],
                   ['cores usable', host.cpus], ['cores physical', host.cpus_physical ?? 'unknown'],
                   ['memory total', bytes(host.memory_total_bytes)],
                   ['memory free', bytes(host.memory_available_bytes)]],
       usedPct == null ? null : { label: 'memory in use', pct: usedPct }],
      ['Accelerators', gpus.length
        ? gpus.map((g) => [`gpu ${g.index}`, `${g.name} · ${bytes(g.memory_bytes)}`])
        : [['gpus', 'none detected'], ['execution', 'CPU only']]],
      ['Engine', [['batcher', eng.version ?? 'source tree'], ['native engine', eng.native ?? 'not loaded'],
                  ['python', (sys.runtime || {}).python], ['process id', (sys.runtime || {}).pid]]],
      ['Cluster', cluster.attached
        ? [['nodes', cluster.nodes], ['cpus', cluster.cpus], ['gpus', cluster.gpus ?? 0],
           ['cpus free', cluster.cpus_available], ['memory', bytes(cluster.memory_bytes)]]
        : [['ray', 'not attached'], ['mode', 'single node'],
           ['to distribute', 'collect(distributed=True)']]],
      ['How work is sized', [['worker threads', cfg.parallelism === 0 ? 'all cores' : cfg.parallelism, 'parallelism'],
                             ['rows per batch', count(cfg.morsel_rows), 'morsel'],
                             ['bytes per batch', bytes(cfg.morsel_bytes), 'batch'],
                             ['file split size', bytes(cfg.split_bytes), 'partition'],
                             ['adaptive batch sizing', cfg.adaptive_morsel_sizing ? 'on' : 'off', 'adaptive execution']]],
      ['Memory and spilling', [['memory limit', cfg.max_memory_bytes ? bytes(cfg.max_memory_bytes) : 'unbounded', 'memory budget'],
                               ['spilling', cfg.spill_enabled ? 'enabled' : 'disabled (fully in memory)', 'spill'],
                               ['spill compression', cfg.spill_compression],
                               ['throttle / spill at', `${cfg.soft_limit} / ${cfg.hard_limit}`],
                               ['log verbosity', cfg.verbosity]]],
    ];
    $('system-cards').innerHTML = groups.map(([title, rows, gauge]) => (
      `<div class="pcard is-static"><div class="pcard-head"><span class="pcard-name">${esc(title)}</span></div>` +
      (gauge ? `<div class="gauge"><div class="gauge-track"><i style="width:${(gauge.pct * 100).toFixed(0)}%"></i></div>` +
               `<span class="gauge-label">${esc(gauge.label)} · ${pct(gauge.pct)}</span></div>` : '') +
      `<dl>` + rows.map(([k, v, termKey]) => (
        `<div class="meta-row"><dt>${esc(k)}${termKey ? LEARN.hint(termKey) : ''}</dt>` +
        `<dd>${esc(String(v ?? '—'))}</dd></div>`)).join('') +
      `</dl></div>`)).join('');
  }

  /* The ranked cost table — the primary triage path.
   *
   * Every engine's own troubleshooting guide says the same thing: find the step with the
   * largest share of time, then look at that step. A graph shows structure; a sorted list
   * answers the question directly, so it sits beside the graph rather than behind it.
   *
   * Share is of *summed operator time*, not wall-clock: steps run concurrently, so shares
   * against wall-clock sum to well over 100% and read as nonsense. */
  function costliest(nodes, onSelect, selectedId) {
    const host = $('costliest');
    if (!host) return;
    const measured = (nodes || []).filter((n) => n.measured && n.elapsed_ms != null);
    if (!measured.length) {
      host.innerHTML = UI.emptyState({
        glyph: 'clock', title: 'No step timings',
        body: 'This run recorded no per-step timing, so there is nothing to rank. ' +
              'Timings appear once a run completes.',
      });
      return;
    }
    const total = measured.reduce((sum, n) => sum + n.elapsed_ms, 0) || 1;
    const sorted = [...measured].sort((a, b) => b.elapsed_ms - a.elapsed_ms);
    // Self-truncating rather than a fixed top-N: a plan whose cost is spread evenly has no
    // "top 5" worth naming, and one dominated by a single step should not pad the list to
    // five. Anything under 1% of the time is noise by definition. Always keep at least one.
    const ranked = sorted.filter((n, i) => i === 0 || n.elapsed_ms / total >= 0.01);
    const hidden = sorted.length - ranked.length;
    const worst = ranked[0].elapsed_ms;

    // How well the planner predicted this run. Stated up front because a plan built on bad
    // estimates is a different problem from a plan that is simply expensive, and the two
    // have completely different fixes.
    const estimated = measured.filter((n) => n.est_error != null && n.est_error > 0);
    const wrong = estimated.filter((n) => n.est_error > 10 || n.est_error < 0.1);
    const accuracy = estimated.length
      ? `<p class="cost-accuracy${wrong.length ? ' is-warn' : ''}">` +
        (wrong.length
          ? `The planner misjudged <b>${wrong.length} of ${estimated.length}</b> steps by more ` +
            `than 10x. Plan choices rest on those estimates, so this plan may not be the one ` +
            `you want \u2014 re-running feeds the measured counts back in.`
          : `The planner predicted all ${estimated.length} measured steps within 10x, so this ` +
            `plan was chosen on sound estimates.`) + `</p>`
      : '';

    host.innerHTML = accuracy + ranked.map((n, i) => {
      const share = n.elapsed_ms / total;
      // Two bands rather than a continuous ramp: a gradient makes 12% and 19% look alike,
      // whereas banding says "this one, then that one" and can be read without a legend.
      const band = share > 0.3 ? ' is-dominant' : (share > 0.15 ? ' is-heavy' : '');
      const flags = [
        n.on_critical_path ? '<span class="chip is-crit">critical path</span>' : '',
        n.spilled ? `<span class="chip is-serious">spilled ${bytes(n.spill_bytes)}</span>` : '',
        n.est_error != null && (n.est_error > 10 || n.est_error < 0.1)
          ? `<span class="chip is-warn">${n.est_error.toFixed(0)}x off</span>` : '',
      ].filter(Boolean).join('');
      return `<button class="cost-row${band}${n.op_id === selectedId ? ' is-selected' : ''}" ` +
        `type="button" data-op="${n.op_id}" ` +
        `aria-label="${esc(friendly(n.kind))}, ${pct(share)} of operator time">` +
        `<span class="cost-rank">${i + 1}</span>` +
        `<span class="cost-glyph">${DAG.glyphMarkup(n.kind, 15)}</span>` +
        `<span class="cost-name"><b>${esc(friendly(n.kind))}</b>` +
        (n.detail ? `<span class="cost-detail">${esc(n.detail)}</span>` : '') + `</span>` +
        `<span class="cost-track"><i style="width:${(n.elapsed_ms / worst) * 100}%"></i></span>` +
        `<span class="cost-share">${pct(share)}</span>` +
        `<span class="cost-ms">${ms(n.elapsed_ms)}</span>` +
        `<span class="cost-rows">${count(n.rows_out)} rows` +
        (n.est_error != null && n.est_error > 0
          ? `<span class="cost-est" title="rows the planner predicted">` +
            `est ${count(n.rows_out / n.est_error)}</span>` : '') + `</span>` +
        (flags ? `<span class="cost-flags">${flags}</span>` : '') +
        `</button>`;
    }).join('') +
      (hidden
        ? `<p class="cost-rest">${hidden} more step${hidden === 1 ? '' : 's'} below 1% of ` +
          `operator time \u2014 too small to matter here.</p>`
        : '');

    for (const row of host.querySelectorAll('.cost-row')) {
      row.addEventListener('click', () => onSelect(Number(row.dataset.op)));
    }
  }

  /* ---------- log volume histogram ----------
   *
   * Stacked by level so a spike answers two questions at once: how much, and how bad. Bars
   * are SVG rects on a shared time domain; clicking one narrows to that bucket.
   *
   * Bucket count adapts to the span rather than being fixed: 60 buckets over 4 seconds is
   * noise, over 4 hours it is a shape. */
  const LOG_LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'];

  function logHistogram(lines, opts) {
    const svg = $('log-histo');
    if (!svg) return null;
    svg.innerHTML = '';
    if (!lines.length) {
      $('log-histo-from').textContent = '';
      $('log-histo-to').textContent = '';
      return null;
    }
    const times = lines.map((l) => l.wall).filter(Boolean);
    const min = Math.min(...times), max = Math.max(...times);
    const span = Math.max(max - min, 0.001);
    // One bucket per ~6px of width keeps bars finger-sized without inventing resolution.
    const buckets = Math.max(12, Math.min(120, Math.round(span > 0 ? 60 : 12)));
    const step = span / buckets;
    const grid = Array.from({ length: buckets }, () => ({ total: 0, byLevel: {} }));
    for (const l of lines) {
      if (!l.wall) continue;
      const i = Math.min(buckets - 1, Math.floor((l.wall - min) / step));
      const lvl = LOG_LEVELS.includes(l.level) ? l.level : 'INFO';
      grid[i].total += 1;
      grid[i].byLevel[lvl] = (grid[i].byLevel[lvl] || 0) + 1;
    }
    const peak = Math.max(1, ...grid.map((g) => g.total));
    const W = 1000, H = 44, gap = 1;
    const bw = W / buckets;
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.setAttribute('preserveAspectRatio', 'none');

    const parts = [];
    grid.forEach((g, i) => {
      if (!g.total) return;
      let y = H;
      // Worst level at the bottom so the eye reads severity along one baseline.
      for (const lvl of LOG_LEVELS) {
        const n = g.byLevel[lvl] || 0;
        if (!n) continue;
        const h = (n / peak) * (H - 2);
        y -= h;
        parts.push(`<rect class="lh-bar lh-${esc(lvl)}" x="${(i * bw).toFixed(2)}" ` +
          `y="${y.toFixed(2)}" width="${Math.max(0.5, bw - gap).toFixed(2)}" ` +
          `height="${h.toFixed(2)}"></rect>`);
      }
      // A transparent full-height target: a 2px-tall bar must still be clickable.
      parts.push(`<rect class="lh-hit" x="${(i * bw).toFixed(2)}" y="0" ` +
        `width="${Math.max(0.5, bw).toFixed(2)}" height="${H}" ` +
        `data-bucket="${i}"><title>${g.total} line${g.total === 1 ? '' : 's'}</title></rect>`);
    });
    svg.innerHTML = parts.join('');
    $('log-histo-from').textContent = clock(min);
    $('log-histo-to').textContent = clock(max);

    for (const hit of svg.querySelectorAll('.lh-hit')) {
      hit.addEventListener('click', () => {
        const i = Number(hit.dataset.bucket);
        opts.onPickRange(min + i * step, min + (i + 1) * step);
      });
    }
    return { min, max, step, buckets };
  }

  /* One log line. The level is a left stripe rather than a coloured row: a wall of red text
   * is unreadable, and the stripe survives being scanned at speed. Structured fields are
   * buttons, because the value you can see is the value you want to filter by. */
  function logLine(l, index) {
    const kv = Object.entries(l.fields || {}).map(([k, v]) => (
      `<button class="log-kv-item" type="button" data-field="${esc(k)}" ` +
      `data-value="${esc(String(v))}" title="Filter to ${esc(k)}=${esc(String(v))}">` +
      `<span class="log-kv-k">${esc(k)}</span>=<span class="log-kv-v">${esc(String(v))}</span>` +
      `</button>`)).join('');
    const seq = l.seq != null ? l.seq : index;
    return `<div class="logline lvl-${esc(l.level)}" data-line="${index}" data-seq="${seq}" id="L${index}">` +
      `<button class="log-time" type="button" data-permalink="${index}" ` +
      `aria-label="Copy a link to the line logged at ${esc(clock(l.wall))}">` +
      `${esc(clock(l.wall))}</button>` +
      `<span class="log-level">${esc(l.level)}</span>` +
      `<span class="log-src">${esc(l.logger || 'engine')}</span>` +
      `<span class="log-msg">${esc(l.message)}` +
      (kv ? `<span class="log-kv">${kv}</span>` : '') + `</span>` +
      `<button class="log-context" type="button" data-context="${seq}" ` +
      `aria-label="Show the lines around this one" title="Show surrounding lines">\u2195</button></div>`;
  }

  return { logHistogram, logLine, costliest, timeBar, health, kpis, throughput, operatorRollup, timeSplit, attention, recentTable,
           failures, pipelineReport, runGrid,
           pipelines, pipelineTable, pipelineDetail, pipelineRuns,
           verdict, statStrip, story, timeline, operators, insights, decisions,
           comparison, meta, system, card, term, OPERATOR_COLUMNS };
})();
