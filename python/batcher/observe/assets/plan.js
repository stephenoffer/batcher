/* The query viewer — the plan as text, as a diff, as raw IR, and as a flame graph.
 *
 * Every SQL engine ships an EXPLAIN and its users arrive looking for one. The graph in
 * `dag.js` shows a plan's *shape*; this file shows its *nesting*, its *rewrite*, and its
 * *cost distribution* — the three questions a shape cannot answer:
 *
 *   Text tree   what is nested inside what, searchable, copy-pasteable into an issue.
 *   Diff        what the optimizer changed between the plan you wrote and the plan that ran.
 *   IR          the exact document that crossed into Rust, for when the rendering is the
 *               thing under suspicion.
 *   Flame       where the time actually went, by depth, in one picture.
 *
 * ── One thing this file deliberately does NOT draw ──────────────────────────────
 * A Gantt chart with start offsets. The engine measures operators inside Rust and replays
 * them onto the event bus *after* the query finishes (`event_log.py::_publish_stages`), so
 * every operator's recorded start instant is the same one. Laying those out on a clock
 * would produce a chart that looks like concurrency measurement and is nothing of the kind.
 * What IS real is each operator's duration and which pipeline stage it belongs to, so that
 * is what the stage view shows, and it says so on the panel.
 */

'use strict';

const PLAN = (() => {
  const { esc, ms, count, pct } = UI;
  const friendly = (k) => DAG.friendlyKind(k);
  const $ = (id) => document.getElementById(id);

  /* ---------- the text tree ---------- */

  /* An EXPLAIN a person can work in, rather than a wall of preformatted text.
   *
   * The spine is pre-computed by the server (`observe/dag/explain.py`) so the client never
   * re-derives which branches are still open — the same rows drive this and the plain-text
   * copy, which is what stops the two from disagreeing about the tree.
   *
   * Each row carries a proportion bar for its own share of operator time. That is what
   * makes the text tree a *diagnostic* rather than a schematic: the expensive line is
   * visible without reading a single number. */
  function explain(host, rows, opts = {}) {
    const el = typeof host === 'string' ? $(host) : host;
    if (!el) return;
    if (!rows || !rows.length) {
      el.innerHTML = UI.emptyState({
        glyph: 'pipeline', title: 'No plan recorded for this run',
        body: 'A query answered from metadata never reaches the optimizer, so it has no plan ' +
              'to explain. Anything that executes does.',
      });
      return;
    }
    const measured = rows.filter((r) => r.measured);
    const total = measured.reduce((s, r) => s + r.elapsed_ms, 0) || 1;
    const needle = (opts.needle || '').trim().toLowerCase();
    const collapsed = opts.collapsed || new Set();

    // A collapsed row hides everything nested beneath it, which is every following row of
    // greater depth up to the next sibling. Computed here rather than stored, so collapsing
    // is a set of op_ids and never a second copy of the tree that can go stale.
    const hidden = new Set();
    for (let i = 0; i < rows.length; i += 1) {
      if (!collapsed.has(rows[i].op_id)) continue;
      for (let j = i + 1; j < rows.length && rows[j].depth > rows[i].depth; j += 1) {
        hidden.add(rows[j].op_id);
      }
    }

    const hasKids = new Set();
    for (let i = 1; i < rows.length; i += 1) {
      if (rows[i].depth > rows[i - 1].depth) hasKids.add(rows[i - 1].op_id);
    }

    el.innerHTML = rows.filter((r) => !hidden.has(r.op_id)).map((row) => {
      const share = row.measured ? row.elapsed_ms / total : 0;
      const hit = needle && (`${row.kind} ${row.detail}`.toLowerCase().includes(needle));
      const spine = row.ancestors.map((open) => (
        `<i class="pv-spine${open ? ' is-open' : ''}"></i>`)).join('') +
        (row.depth ? `<i class="pv-branch${row.last ? ' is-last' : ''}"></i>` : '');
      const twisty = hasKids.has(row.op_id)
        ? `<button class="pv-twisty${collapsed.has(row.op_id) ? ' is-closed' : ''}" type="button" ` +
          `data-collapse="${row.op_id}" aria-expanded="${!collapsed.has(row.op_id)}" ` +
          `aria-label="${collapsed.has(row.op_id) ? 'Expand' : 'Collapse'} ${esc(friendly(row.kind))}">` +
          `<span aria-hidden="true">▸</span></button>`
        : `<span class="pv-twisty is-leaf" aria-hidden="true"></span>`;
      const badges = [
        row.spilled ? `<span class="chip is-serious">spilled</span>` : '',
        row.algorithm ? `<span class="chip is-quiet">${esc(row.algorithm)}</span>` : '',
      ].filter(Boolean).join('');
      return `<div class="pv-row${hit ? ' is-hit' : ''}${row.op_id === opts.selected ? ' is-selected' : ''}` +
        `${collapsed.has(row.op_id) ? ' is-collapsed' : ''}" data-op="${row.op_id}" ` +
        `role="button" tabindex="0" aria-label="${esc(friendly(row.kind))}${row.detail ? `, ${esc(row.detail)}` : ''}">` +
        `<span class="pv-tree">${spine}${twisty}` +
        `<span class="pv-glyph">${DAG.glyphMarkup(row.kind, 13)}</span>` +
        `<span class="pv-kind">${esc(friendly(row.kind))}</span>` +
        (row.detail ? `<span class="pv-detail">${esc(row.detail)}</span>` : '') +
        (collapsed.has(row.op_id) ? `<span class="pv-folded">…</span>` : '') +
        badges + `</span>` +
        `<span class="pv-metrics">` +
        (row.measured
          ? `<span class="pv-share" title="${esc(pct(share))} of operator time">` +
            `<i style="width:${(share * 100).toFixed(1)}%"></i></span>` +
            `<span class="pv-ms">${esc(ms(row.elapsed_ms))}</span>` +
            `<span class="pv-rows">${esc(count(row.rows_out))}</span>`
          : `<span class="pv-unmeasured">not measured</span>`) +
        `<span class="pv-est">${row.est_rows == null ? '—' : `est ${esc(count(row.est_rows))}`}</span>` +
        `</span></div>`;
    }).join('');
  }

  /** The same tree as plain text, for the clipboard. Built from the same rows the DOM used. */
  function explainText(rows) {
    return (rows || []).map((row) => {
      const spine = row.ancestors.map((open) => (open ? '│  ' : '   ')).join('');
      const branch = row.depth ? (row.last ? '└─ ' : '├─ ') : '';
      const detail = row.detail ? `  [${row.detail}]` : '';
      const stats = row.measured
        ? `  (${ms(row.elapsed_ms)}, ${count(row.rows_out)} rows)` : '';
      return `${spine}${branch}${row.kind}${detail}${stats}`;
    }).join('\n');
  }

  /* ---------- the optimizer diff ---------- */

  /* What Kyber did to the query, which until now the dashboard never showed.
   *
   * Three states, and they must stay distinguishable: the optimizer changed things, the
   * optimizer changed nothing, and *we did not record the plan you wrote*. The third one
   * looks like the second in every naive rendering, which would silently credit the
   * optimizer for leaving alone a plan it never saw. */
  function diff(host, payload, opts = {}) {
    const el = typeof host === 'string' ? $(host) : host;
    if (!el) return;
    if (!payload || !payload.available) {
      el.innerHTML = UI.emptyState({
        glyph: 'pipeline', title: 'The original plan was not recorded',
        body: 'This comparison needs the plan as written <em>and</em> the plan that ran. ' +
              'This run carries only one of them, so there is nothing to compare — which is ' +
              'not the same as the optimizer having changed nothing.',
      });
      return;
    }
    const changes = payload.changes || [];
    const primary = changes.filter((c) => c.primary);
    const knock = changes.filter((c) => !c.primary);
    const showAll = Boolean(opts.showAll);

    el.innerHTML =
      `<p class="pv-diff-summary${payload.identical ? ' is-quiet' : ''}">${esc(payload.summary || '')}</p>` +
      (payload.identical
        ? `<p class="hint">Every step you wrote is a step that ran, in the order you wrote it. ` +
          `That is a normal outcome for a plan with nothing to push down or fold away — not ` +
          `a sign the optimizer was skipped.</p>`
        : `<div class="pv-changes">` +
          primary.map((c) => changeCard(c, true)).join('') +
          (knock.length
            ? (showAll
                ? knock.map((c) => changeCard(c, false)).join('')
                : `<button class="pv-more" type="button" data-show-all>` +
                  `${knock.length} knock-on change${knock.length === 1 ? '' : 's'} — every step ` +
                  `the rewrite above moved past. Show them.</button>`)
            : '') +
          `</div>`) +
      countTable(payload.counts || [], payload);
  }

  function changeCard(change, isPrimary) {
    const verb = { moved: 'reordered', added: 'added', removed: 'removed' }[change.change];
    const tone = { moved: 'accent', added: 'good', removed: 'warn' }[change.change];
    return `<div class="pv-change is-${tone}${isPrimary ? ' is-primary' : ' is-secondary'}"` +
      (change.op_id != null ? ` data-op="${change.op_id}" role="button" tabindex="0"` : '') + `>` +
      `<span class="pv-change-verb">${esc(verb)}</span>` +
      `<span class="pv-change-what">${DAG.glyphMarkup(change.kind, 13)}` +
      `<b>${esc(friendly(change.kind))}</b>` +
      (change.detail ? `<span class="mono dim"> ${esc(change.detail)}</span>` : '') + `</span>` +
      `<span class="pv-change-note">${esc(change.note || '')}</span>` +
      (change.op_id != null ? `<span class="pv-change-go">show in the plan →</span>` : '') +
      `</div>`;
  }

  /* Per-kind tallies. A table, because that is what it is: a lookup of operator kind
   * against a count on each side, which prose would only make longer. */
  function countTable(counts, payload) {
    if (!counts.length) return '';
    return `<h3 class="sub-head">Steps, before and after</h3>` +
      `<table class="dense pv-counts"><thead><tr><th scope="col">Step</th>` +
      `<th scope="col" class="num">As written</th><th scope="col" class="num">As run</th>` +
      `<th scope="col" class="num">Change</th></tr></thead><tbody>` +
      counts.map((row) => {
        const cls = row.delta === 0 ? '' : row.delta < 0 ? 'is-good' : 'is-warn';
        const arrow = row.delta === 0 ? '—' : `${row.delta > 0 ? '+' : ''}${row.delta}`;
        return `<tr><td>${DAG.glyphMarkup(row.kind, 13)} ${esc(friendly(row.kind))}</td>` +
          `<td class="num">${row.before}</td><td class="num">${row.after}</td>` +
          `<td class="num ${cls}">${esc(arrow)}</td></tr>`;
      }).join('') +
      `<tr class="pv-counts-total"><td>total</td><td class="num">${payload.before_ops}</td>` +
      `<td class="num">${payload.after_ops}</td>` +
      `<td class="num">${payload.after_ops - payload.before_ops || '—'}</td></tr>` +
      `</tbody></table>`;
  }

  /* ---------- the raw IR ---------- */

  /* The exact JSON that crossed into Rust, as a foldable tree.
   *
   * Present because when a plan renders wrongly, the first question is whether the plan is
   * wrong or the rendering is — and nothing else on the page can answer that. Rendered as
   * a DOM tree rather than `JSON.stringify` into a `<pre>` so a 400-line plan is navigable
   * and a key is searchable. */
  function ir(host, doc, opts = {}) {
    const el = typeof host === 'string' ? $(host) : host;
    if (!el) return;
    if (!doc) {
      el.innerHTML = `<p class="empty">No plan document for this run.</p>`;
      return;
    }
    el.innerHTML = `<div class="pv-ir mono">${node(doc, opts.open ?? 3, 0, '')}</div>`;
  }

  /* One JSON value. Objects and arrays past `openTo` render closed, because a plan's leaf
   * expressions are deep and nobody opens a viewer wanting every literal expanded. */
  function node(value, openTo, depth, key) {
    const label = key ? `<span class="pv-key">${esc(key)}</span><span class="pv-colon">:</span> ` : '';
    if (value === null) return `<div class="pv-line">${label}<span class="pv-null">null</span></div>`;
    if (Array.isArray(value)) {
      if (!value.length) return `<div class="pv-line">${label}<span class="pv-punct">[]</span></div>`;
      return group(label, `[${value.length}]`, depth < openTo,
        value.map((v, i) => node(v, openTo, depth + 1, String(i))).join(''));
    }
    if (typeof value === 'object') {
      const keys = Object.keys(value);
      if (!keys.length) return `<div class="pv-line">${label}<span class="pv-punct">{}</span></div>`;
      // Lead with the operator tag when there is one: `op` is the thing a reader is
      // scanning for, and alphabetical order buries it among a dozen sibling keys.
      const tag = value.op || value.e;
      const ordered = ['op', 'e', ...keys.filter((k) => k !== 'op' && k !== 'e')]
        .filter((k) => k in value);
      return group(label, tag ? `${esc(String(tag))}` : `{${keys.length}}`, depth < openTo,
        ordered.map((k) => node(value[k], openTo, depth + 1, k)).join(''));
    }
    const cls = typeof value === 'number' ? 'pv-num'
      : typeof value === 'boolean' ? 'pv-bool' : 'pv-str';
    const shown = typeof value === 'string' ? `"${value}"` : String(value);
    return `<div class="pv-line">${label}<span class="${cls}">${esc(shown)}</span></div>`;
  }

  function group(label, summary, open, inner) {
    return `<details class="pv-node"${open ? ' open' : ''}>` +
      `<summary class="pv-line">${label}<span class="pv-tag">${summary}</span></summary>` +
      `<div class="pv-children">${inner}</div></details>`;
  }

  /* ---------- flame ---------- */

  function flame(host, nodes, onSelect) {
    const el = typeof host === 'string' ? $(host) : host;
    if (!el) return;
    const body = CHARTS.flame(nodes || [], { name: (n) => friendly(n.kind), label: 'time by plan depth' });
    if (!body) {
      el.innerHTML = UI.emptyState({
        glyph: 'chart', title: 'No measured steps to lay out',
        body: 'The flame view sizes each step by the time it took. It appears once a run ' +
              'records per-step timing.',
      });
      return;
    }
    el.innerHTML =
      `<p class="hint">Each row is one level of the plan; each block is a step, sized by its ` +
      `own measured time. A parent is <b>not</b> the sum of its children — steps run ` +
      `concurrently, and stacking them would invent a total larger than the query took.</p>` +
      body;
    if (onSelect) CHARTS.onPick(el, (id) => onSelect(Number(id)));
  }

  /* ---------- pipeline stages ---------- */

  /* Which steps stream together, and what each of those groups cost.
   *
   * The honest form of the "timeline" every engine UI shows. Everything between two
   * pipeline breakers runs at the same time, so it belongs to one group; the breaker above
   * it has to wait for all of it. The bars are *durations* laid side by side — not
   * positions on a clock, because the engine does not record when each operator started
   * (it replays them all at once when the query ends). The panel says so, rather than
   * letting the shape imply a measurement that was never taken. */
  function stages(host, dag, onSelect) {
    const el = typeof host === 'string' ? $(host) : host;
    if (!el) return;
    const nodes = (dag?.nodes || []).filter((n) => n.measured);
    if (!nodes.length) {
      el.innerHTML = UI.emptyState({
        glyph: 'clock', title: 'No stage timings yet',
        body: 'Steps between two pipeline breakers stream into each other and run together. ' +
              'This groups them, so the run reads as a handful of phases rather than twenty ' +
              'unrelated steps.',
      });
      return;
    }
    const groups = new Map();
    for (const n of nodes) {
      const key = n.stage ?? 0;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(n);
    }
    const ordered = [...groups.entries()].sort((a, b) => a[0] - b[0]);
    const worst = Math.max(...ordered.map(([, g]) => Math.max(...g.map((n) => n.elapsed_ms))), 1);

    el.innerHTML =
      `<p class="hint">A <b>stage</b> is everything that streams together between two ` +
      `pipeline breakers — a sort, an aggregate, or a join build, each of which must see all ` +
      `its input before it emits a row. Bars are each step's measured duration, drawn on one ` +
      `scale. They are <b>not</b> placed on a clock: the engine records how long each step ` +
      `took, not when it began.</p>` +
      ordered.map(([stage, group]) => {
        const slowest = group.reduce((a, b) => (a.elapsed_ms > b.elapsed_ms ? a : b));
        const sum = group.reduce((s, n) => s + n.elapsed_ms, 0);
        return `<section class="pv-stage">` +
          `<header class="pv-stage-head"><span class="pv-stage-num">stage ${stage + 1}</span>` +
          `<span class="pv-stage-what">${group.length} step${group.length === 1 ? '' : 's'} ` +
          `streaming together</span>` +
          `<span class="pv-stage-cost mono">${esc(ms(sum))} of step time</span>` +
          `<span class="pv-stage-lead">slowest: ${esc(friendly(slowest.kind))}</span></header>` +
          [...group].sort((a, b) => b.elapsed_ms - a.elapsed_ms).map((n) => (
            `<button class="pv-stage-row${n.on_critical_path ? ' is-crit' : ''}" type="button" ` +
            `data-pick="${n.op_id}" aria-label="${esc(friendly(n.kind))}, ${esc(ms(n.elapsed_ms))}">` +
            `<span class="pv-stage-name">${DAG.glyphMarkup(n.kind, 13)}${esc(friendly(n.kind))}` +
            (n.detail ? `<span class="mono dim"> ${esc(n.detail)}</span>` : '') + `</span>` +
            `<span class="pv-stage-track"><i style="width:${Math.max(1.5, (n.elapsed_ms / worst) * 100).toFixed(1)}%"` +
            `${n.spilled ? ' class="is-spilled"' : ''}></i></span>` +
            `<span class="pv-stage-ms mono">${esc(ms(n.elapsed_ms))}</span>` +
            `<span class="pv-stage-rows mono dim">${esc(count(n.rows_out))} rows</span>` +
            `</button>`)).join('') +
          `</section>`;
      }).join('');
    if (onSelect) CHARTS.onPick(el, (id) => onSelect(Number(id)));
  }

  return { explain, explainText, diff, ir, flame, stages };
})();
