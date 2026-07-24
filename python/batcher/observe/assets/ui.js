/* Shared UI machinery — formatting, preferences, routing, tables, exports, notifications.
 *
 * Everything here is view-agnostic. `views.js` renders panels with it; `app.js` wires the
 * shell around it. Splitting it out keeps the renderers about *what* to show rather than
 * re-deriving how to format a byte count or persist a checkbox for the fifth time.
 */

'use strict';

const UI = (() => {
  /** The one rendering of "no value": never `null`, `NaN`, `undefined`, or an empty cell. */
  function blankToDash(v) {
    if (v == null) return '\u2014';
    if (typeof v === 'number' && !Number.isFinite(v)) return '\u2014';
    const s = String(v);
    return s.trim() === '' ? '\u2014' : s;
  }

  /* ---------- measurement validity ---------- */

  /* Is this operator's CPU reading an actual measurement?
   *
   * The streaming tier does not sample the OS clock per morsel — it reports `cpu_ns` as the
   * operator's wall time (bc-interp/src/stream/meter.rs). `cpu_util` is then
   * `elapsed / (elapsed * threads)`, i.e. exactly `1/threads`, for every operator of every
   * query. That is a constant wearing a measurement's clothes, and rendering it as a
   * percentage invites the reader to draw conclusions from a number that carries no
   * information about their query.
   *
   * So: treat `cpu_util * threads == 1` as the unmeasured sentinel and show an em dash.
   * A genuinely measured operator that happens to land on exactly one busy core is
   * indistinguishable here and will also read as unmeasured — that is the safe direction to
   * be wrong in, and it costs one operator's cell rather than a wrong conclusion. */
  function cpuMeasured(node) {
    const util = node && node.cpu_util;
    const threads = node && node.threads;
    if (!util || !threads) return false;
    return Math.abs(util * threads - 1) > 1e-6;
  }

  /* ---------- formatting ---------- */

  function count(n) {
    if (n == null || Number.isNaN(n)) return '—';
    const abs = Math.abs(n);
    if (abs >= 1e12) return `${(n / 1e12).toFixed(1)}T`;
    if (abs >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
    if (abs >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
    if (abs >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
    return `${Math.round(n)}`;
  }

  /* Sub-millisecond work is real work. Rounding 0.009 ms to "0.0ms" made the fastest steps
   * in a plan look unmeasured, which is the opposite of the truth — so drop to microseconds
   * rather than to zero. This is the ONLY duration formatter on the page; `DAG.fmtMs`
   * delegates here, because two copies of this function had already drifted once. */
  function ms(v) {
    if (v == null) return '—';
    if (v === 0) return '0';
    if (v < 1) return `${Math.max(1, Math.round(v * 1000))}\u00b5s`;
    if (v < 1000) return `${v.toFixed(v < 10 ? 1 : 0)}ms`;
    if (v < 60000) return `${(v / 1000).toFixed(2)}s`;
    return `${Math.floor(v / 60000)}m${((v % 60000) / 1000).toFixed(0).padStart(2, '0')}s`;
  }

  function bytes(n) {
    if (n == null || n === 0) return '—';
    let size = n;
    for (const unit of ['B', 'KiB', 'MiB', 'GiB', 'TiB']) {
      if (size < 1024 || unit === 'TiB') return `${unit === 'B' ? size : size.toFixed(1)} ${unit}`;
      size /= 1024;
    }
    return `${n}`;
  }

  /* A share that is small but present reads as "<1%", never as "0%" — the second says
   * "nothing here", which is a different and wrong claim. */
  function pct(fraction, digits = 0) {
    if (fraction == null || Number.isNaN(fraction)) return '—';
    if (!fraction) return '0%';
    const value = fraction * 100;
    if (value > 0 && value < 1) return '<1%';
    return `${value.toFixed(digits)}%`;
  }
  const clock = (wall) => new Date(wall * 1000).toLocaleTimeString([], { hour12: false });
  const stamp = (wall) => new Date(wall * 1000).toLocaleString([], { hour12: false });

  function ago(wall) {
    const secs = Math.max(0, Date.now() / 1000 - wall);
    if (secs < 10) return 'just now';
    if (secs < 60) return `${Math.round(secs)}s ago`;
    if (secs < 3600) return `${Math.round(secs / 60)} min ago`;
    if (secs < 86400) return `${Math.round(secs / 3600)} hr ago`;
    return `${Math.round(secs / 86400)} d ago`;
  }

  function duration(secs) {
    // A clock skew or a not-yet-started run can hand this a negative or a NaN, and both
    // reach the page as a duration a reader will try to interpret ("-3s left").
    if (secs == null || !Number.isFinite(secs) || secs < 0) return '—';
    if (secs < 1) return '<1s';
    if (secs < 60) return `${Math.round(secs)}s`;
    if (secs < 3600) return `${Math.floor(secs / 60)}m ${Math.round(secs % 60)}s`;
    return `${Math.floor(secs / 3600)}h ${Math.floor((secs % 3600) / 60)}m`;
  }

  /* Every value reaching the DOM passes through here. Query labels, log messages, and error
   * strings carry user data; building markup from them with innerHTML would be an injection
   * into the operator's own browser. */
  function esc(text) {
    return String(text ?? '').replace(/[&<>"']/g, (c) => (
      { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }

  /* ---------- preferences ---------- */
  /* Persisted so the dashboard a person configured is the one they get back. Reads are
   * guarded: a browser with storage disabled must degrade to defaults, not a blank page. */

  const PREF_KEY = 'batcher-ui-prefs';
  // Dark by default, not "auto". An engine dashboard is read for long stretches beside a
  // terminal, and every comparable tool (Spark, Ray, Grafana, Datadog) ships dark-first —
  // inheriting a light OS preference put people somewhere they had not asked to be. "auto"
  // is still reachable through the theme cycle for anyone who wants it.
  const DEFAULTS = {
    theme: 'dark', density: 'comfortable', help: false, autoRefresh: true,
    logLevel: 'INFO', logFollow: true, logRegex: false, pinned: [],
    showMinimap: true, showCritical: true, tableDensity: 'normal',
  };
  let prefs = { ...DEFAULTS };

  function loadPrefs() {
    try {
      prefs = { ...DEFAULTS, ...JSON.parse(localStorage.getItem(PREF_KEY) || '{}') };
    } catch { prefs = { ...DEFAULTS }; }
    return prefs;
  }
  function setPref(key, value) {
    prefs[key] = value;
    try { localStorage.setItem(PREF_KEY, JSON.stringify(prefs)); } catch { /* storage off */ }
    return value;
  }
  const getPref = (key) => prefs[key];

  function togglePin(signature) {
    const pinned = new Set(prefs.pinned || []);
    if (pinned.has(signature)) pinned.delete(signature); else pinned.add(signature);
    return setPref('pinned', [...pinned]);
  }
  const isPinned = (signature) => (prefs.pinned || []).includes(signature);

  /* ---------- routing ---------- */
  /* The URL is the source of truth for "where am I", so a view is linkable, the browser's
   * back button works, and a reload lands where you were rather than on the overview. */

  function readRoute() {
    const raw = location.hash.replace(/^#/, '');
    const out = {};
    for (const part of raw.split('&')) {
      const [k, v] = part.split('=');
      if (k) out[k] = decodeURIComponent(v || '');
    }
    return out;
  }

  function writeRoute(next, { replace = false } = {}) {
    const merged = { ...readRoute(), ...next };
    for (const k of Object.keys(merged)) if (!merged[k]) delete merged[k];
    const hash = `#${Object.entries(merged).map(([k, v]) => `${k}=${encodeURIComponent(v)}`).join('&')}`;
    if (hash === location.hash) return;
    if (replace) history.replaceState(null, '', hash); else location.hash = hash;
  }

  /* ---------- notifications ---------- */

  /* A toast is usually a receipt ("Copied") and should evaporate. Occasionally it is an
   * offer ("take the tour?"), which must wait for an answer and carry buttons. Both shapes
   * live here so callers never hand-roll the second one: `toast(msg)`, `toast(msg, 'good')`,
   * or `toast(msg, { action, dismiss, sticky })`. */
  function toast(message, opts = 'info') {
    const host = document.getElementById('toasts');
    if (!host) return null;
    const { kind = 'info', action = null, dismiss = null, sticky = false } =
      typeof opts === 'string' ? { kind: opts } : opts;

    const el = document.createElement('div');
    el.className = `toast is-${kind}${action || dismiss ? ' has-actions' : ''}`;
    // A receipt is a passive status; an offer demands a response and must interrupt.
    el.setAttribute('role', action || dismiss ? 'alertdialog' : 'status');

    const text = document.createElement('span');
    text.className = 'toast-msg';
    text.textContent = message;
    el.appendChild(text);

    const close = () => {
      el.classList.add('is-leaving');
      setTimeout(() => el.remove(), 300);
    };
    for (const [spec, cls] of [[action, 'is-primary'], [dismiss, 'is-quiet']]) {
      if (!spec) continue;
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = `toast-btn ${cls}`;
      btn.textContent = spec.label;
      btn.addEventListener('click', () => { close(); if (spec.run) spec.run(); });
      el.appendChild(btn);
    }

    host.appendChild(el);
    if (!sticky) setTimeout(close, 2600);
    return close;
  }

  async function copy(text, label = 'Copied') {
    try {
      await navigator.clipboard.writeText(text);
      toast(label, 'good');
    } catch {
      toast('Could not copy — your browser blocked clipboard access', 'warn');
    }
  }

  function download(filename, content, type = 'application/json') {
    const blob = new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
    toast(`Downloaded ${filename}`, 'good');
  }

  /** Rows to CSV, quoting only what needs it, so the file opens cleanly in a spreadsheet. */
  function toCSV(headers, rows) {
    const cell = (v) => {
      const text = v == null ? '' : String(v);
      return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
    };
    return [headers.map(cell).join(','), ...rows.map((r) => r.map(cell).join(','))].join('\n');
  }

  /* ---------- sortable tables ---------- */
  /* One implementation, because six panels want the same behaviour and six copies would
   * drift. `columns` describe the data; the caller never writes a <th> by hand. */

  const sortState = new Map();

  function hiddenCols(id) {
    return new Set((getPref('hiddenCols') || {})[id] || []);
  }
  function toggleCol(id, label) {
    const all = getPref('hiddenCols') || {};
    const set = new Set(all[id] || []);
    if (set.has(label)) set.delete(label); else set.add(label);
    all[id] = [...set];
    setPref('hiddenCols', all);
  }

  function table(id, columns, rows, opts = {}) {
    // A column marked `optional` can be hidden by the reader; a column without the flag is
    // load-bearing and always shown, so the table can never be reduced to nothing.
    const hidden = hiddenCols(id);
    const visible = columns.filter((c) => !(c.optional && hidden.has(c.label)));
    const hasOptional = columns.some((c) => c.optional);
    const key = sortState.get(id) || { col: opts.defaultSort ?? null, dir: opts.defaultDir || 'desc' };
    let ordered = [...rows];
    if (key.col != null && columns[key.col]) {
      const get = columns[key.col].sortValue || columns[key.col].value;
      ordered.sort((a, b) => {
        const x = get(a), y = get(b);
        const cmp = typeof x === 'string' || typeof y === 'string'
          ? String(x ?? '').localeCompare(String(y ?? ''))
          : (Number(x ?? 0) - Number(y ?? 0));
        return key.dir === 'asc' ? cmp : -cmp;
      });
    }
    const head = visible.map((c) => {
      const i = columns.indexOf(c);
      const active = key.col === i;
      const arrow = active ? (key.dir === 'asc' ? '\u2191' : '\u2193') : '';
      const sortable = c.sortable !== false;
      // `aria-sort` belongs on the cell; the control belongs inside it. A `<th>` with a click
      // handler is invisible to a keyboard and announces nothing about what clicking does.
      const ariaSort = active ? (key.dir === 'asc' ? 'ascending' : 'descending') : 'none';
      const inner = sortable
        ? `<button class="th-sort" type="button" data-col="${i}" ` +
          `aria-label="${esc(c.label)}, sort ${active && key.dir === 'desc' ? 'ascending' : 'descending'}">` +
          `${esc(c.label)}<span class="sort-arrow" aria-hidden="true">${arrow}</span></button>`
        : esc(c.label);
      return `<th scope="col" class="${c.num ? 'num' : ''}${sortable ? ' sortable' : ''}` +
             `${active ? ' is-sorted' : ''}"${sortable ? ` aria-sort="${ariaSort}"` : ''}` +
             `${c.help ? ` title="${esc(c.help)}"` : ''}>${inner}</th>`;
    }).join('');
    // Cap the rendered rows so a session with thousands of runs does not build thousands of
    // <tr>s. The cap is generous — content-visibility handles the rest — but stated when hit,
    // because a silently truncated table claims to be complete.
    const ROW_CAP = 500;
    const capped = ordered.length > ROW_CAP;
    const drawn = capped ? ordered.slice(0, ROW_CAP) : ordered;
    const body = drawn.map((row, r) => {
      const attrs = opts.rowAttrs ? opts.rowAttrs(row) : '';
      return `<tr ${attrs}>` + visible.map((c) => {
        const raw = c.render ? c.render(row, r) : esc(blankToDash(c.value(row)));
        return `<td class="${c.num ? 'num' : ''}${c.cls ? ` ${c.cls(row)}` : ''}">${raw}</td>`;
      }).join('') + '</tr>';
    }).join('');
    const host = document.getElementById(id);
    if (!host) return;
    // A caption a screen reader can hear; sighted readers get the count in the panel head.
    const caption = `<caption class="visually-hidden">${esc(opts.caption || 'Table')}, ` +
      `${ordered.length} row${ordered.length === 1 ? '' : 's'}</caption>`;
    const empty = `<tr><td class="table-empty" colspan="${columns.length}">` +
      `${esc(opts.emptyText || 'Nothing to show yet.')}</td></tr>`;
    const colMenu = hasOptional
      ? `<details class="col-menu"><summary>Columns</summary><div class="col-menu-pop">` +
        columns.filter((c) => c.optional).map((c) => (
          `<label><input type="checkbox" data-col-toggle="${esc(c.label)}"` +
          `${hidden.has(c.label) ? '' : ' checked'}> ${esc(c.label)}</label>`)).join('') +
        `</div></details>`
      : '';
    const cappedNote = capped
      ? `<p class="table-capped">Showing the first ${ROW_CAP.toLocaleString()} of ` +
        `${ordered.length.toLocaleString()} rows. Filter or sort to bring what you need into view.</p>`
      : '';
    host.innerHTML = colMenu +
      `<table class="dense">${caption}<thead><tr>${head}</tr></thead>` +
      `<tbody>${body || empty}</tbody></table>` + cappedNote;
    for (const box of host.querySelectorAll('[data-col-toggle]')) {
      box.addEventListener('change', () => { toggleCol(id, box.dataset.colToggle); table(id, columns, rows, opts); });
    }
    for (const th of host.querySelectorAll('.th-sort')) {
      th.addEventListener('click', () => {
        const col = Number(th.dataset.col);
        const next = key.col === col && key.dir === 'desc' ? 'asc' : 'desc';
        sortState.set(id, { col, dir: next });
        table(id, columns, rows, opts);
      });
    }
    if (opts.onRowClick) {
      for (const tr of host.querySelectorAll('tbody tr[data-id]')) {
        tr.addEventListener('click', () => opts.onRowClick(tr.dataset.id));
      }
    }
    return { columns, rows: ordered };
  }

  /** The CSV for whatever a table is currently showing, in its current sort order. */
  function tableCSV(columns, rows) {
    return toCSV(columns.map((c) => c.label),
                 rows.map((r) => columns.map((c) => (c.csv ? c.csv(r) : c.value(r)))));
  }

  /* ---------- axis ticks ---------- */

  /* "Nice" tick values: 1, 2 or 5 times a power of ten, chosen so ticks land at least
   * `minPx` apart. Every charting library implements this and it is ten lines; without it
   * you get axes labelled 0, 0.333, 0.667 and the reader has to do arithmetic to place a
   * value. */
  function niceTicks(min, max, widthPx, minPx = 64) {
    if (!(max > min) || !(widthPx > 0)) return [];
    const target = Math.max(1, Math.floor(widthPx / minPx));
    const raw = (max - min) / target;
    const mag = Math.pow(10, Math.ceil(Math.log10(raw)));
    let step = mag;
    if (step / 5 >= raw) step /= 5;
    else if (step / 2 >= raw) step /= 2;
    const out = [];
    for (let t = Math.ceil(min / step) * step; t <= max + step * 1e-9; t += step) {
      // Re-round each tick: repeated addition of a fractional step drifts (0.1+0.2…).
      out.push(Number((Math.round(t / step) * step).toPrecision(12)));
    }
    return out;
  }

  /* ---------- tiny charts ---------- */
  /* Inline SVG rather than a charting library: these are three shapes, and a dependency
   * would be larger than the dashboard it decorates. */

  function sparkline(values, { width = 140, height = 30, fill = true, label = 'trend', unit = '' } = {}) {
    if (!values.length) return '';
    const max = Math.max(...values, 1);
    const step = width / Math.max(values.length - 1, 1);
    const y = (v) => height - (v / max) * (height - 2) - 1;
    const pts = values.map((v, i) => `${i * step},${y(v)}`);
    const line = `M ${pts.join(' L ')}`;
    const area = `${line} L ${width},${height} L 0,${height} Z`;
    // A one-line description of the shape, so the trend is legible without seeing the line.
    const peak = Math.max(...values), last = values[values.length - 1], first = values[0];
    const dir = last > first * 1.1 ? 'rising' : last < first * 0.9 ? 'falling' : 'steady';
    const u = unit ? ` ${unit}` : '';
    const desc = `${label}: ${dir}, now ${count(last)}${u}, peak ${count(peak)}${u}`;
    // Invisible hit targets so a mouse can read any point; the value goes in a title.
    const hits = values.map((v, i) =>
      `<rect class="spark-hit" x="${i * step - step / 2}" y="0" width="${step}" height="${height}">` +
      `<title>${count(v)}${u}</title></rect>`).join('');
    return `<svg class="spark" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" ` +
      `role="img" aria-label="${esc(desc)}">` +
      (fill ? `<path class="spark-area" d="${area}"/>` : '') +
      `<path class="spark-line" d="${line}"/>` +
      `<circle class="spark-end" cx="${(values.length - 1) * step}" cy="${y(last)}" r="2.5"/>` +
      hits + `</svg>`;
  }

  /** A duration histogram — the shape of a pipeline's runtime, not just its median. */
  function histogram(values, { bins = 12, width = 220, height = 46 } = {}) {
    if (values.length < 2) return '';
    const min = Math.min(...values), max = Math.max(...values);
    const span = max - min || 1;
    const counts = new Array(bins).fill(0);
    for (const v of values) counts[Math.min(bins - 1, Math.floor(((v - min) / span) * bins))] += 1;
    const peak = Math.max(...counts, 1);
    const bw = width / bins;
    return `<svg class="histo" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" aria-hidden="true">` +
      counts.map((c, i) => {
        const h = (c / peak) * (height - 4);
        return `<rect x="${i * bw + 0.5}" y="${height - h}" width="${bw - 1}" height="${h}" rx="1.5"/>`;
      }).join('') + `</svg>`;
  }

  /* ---------- icons ---------- */
  /* A tiny inline set rather than an icon font or an SVG sprite request: eight glyphs, each
   * a single path, currentColor-filled so they inherit whatever text colour they sit in.
   * Emoji were the alternative and they render differently on every platform, which makes
   * a dense instrument panel look assembled from spare parts. */

  const ICONS = {
    pipeline: 'M3 7h4l2 5 3-9 2 7h5',
    run: 'M5 3l10 6-10 6z',
    clock: 'M9 4v5l3 2M9 1a8 8 0 100 16A8 8 0 009 1z',
    warn: 'M9 2l7 13H2L9 2zm0 5v4m0 2v.5',
    check: 'M3 9l4 4 8-9',
    spill: 'M4 3h10v5a5 5 0 01-10 0zM6 14h6',
    search: 'M8 3a5 5 0 100 10A5 5 0 008 3zm4 9l4 4',
    chart: 'M3 15V8m4 7V4m4 11v-5m4 5V6',
    up: 'M9 15V4m0 0L4 9m5-5l5 5',
    down: 'M9 3v11m0 0l5-5m-5 5l-5-5',
    close: 'M4 4l10 10M14 4L4 14',
    copy: 'M6 6h8v8H6zM4 12V2h8',
  };

  /** An inline SVG icon, sized to the current font. */
  function icon(name, { size = 14, cls = '' } = {}) {
    const path = ICONS[name];
    if (!path) return '';
    return `<svg class="icon ${cls}" width="${size}" height="${size}" viewBox="0 0 18 18" ` +
      `fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" ` +
      `stroke-linejoin="round" aria-hidden="true"><path d="${path}"/></svg>`;
  }

  /* ---------- empty & loading states ---------- */

  /** A first-run state that teaches rather than an apology that the box is empty.
   *
   * `code` renders a copyable snippet and `actions` render buttons, because the most useful
   * empty state is one the reader can act on without leaving it — "no data yet" plus the
   * exact line that produces some beats "no data yet" on its own. */
  function emptyState({ glyph = 'chart', title, body, hint = '', code = '', actions = [] }) {
    return `<div class="empty-state">${icon(glyph, { size: 30 })}` +
      `<h3>${esc(title)}</h3><p>${body}</p>` +
      (code
        ? `<div class="empty-code"><pre><code>${esc(code)}</code></pre>` +
          `<button class="ghost empty-copy" type="button" data-copy="${esc(code)}" ` +
          `aria-label="Copy this snippet">${icon('copy')} Copy</button></div>`
        : '') +
      (actions.length
        ? `<div class="empty-actions">` + actions.map((a) => (
            `<button class="ghost${a.primary ? ' is-primary' : ''}" type="button" ` +
            `data-empty-action="${esc(a.id)}">${esc(a.label)}</button>`)).join('') + `</div>`
        : '') +
      (hint ? `<p class="dim">${hint}</p>` : '') + `</div>`;
  }

  /** Shimmer placeholders sized like the content they stand in for. */
  /** Placeholder lines, shaped roughly like the content they stand in for. */
  const skeleton = (lines = 3) =>
    `<div class="sk-block" aria-hidden="true">${Array.from({ length: lines },
      (_, i) => `<div class="skeleton sk-line" style="width:${[92, 78, 85, 64][i % 4]}%"></div>`
    ).join('')}</div>`;

  /* A panel's whole lifecycle in one call: loading, failed, empty, or content.
   *
   * These four are a state machine, and conflating any two of them is the commonest lie a
   * dashboard tells. "Nothing here" and "we could not reach the engine" look identical if you
   * only ever render an empty list, and a reader who cannot tell them apart will either chase
   * a phantom problem or ignore a real one. */
  function panelState(host, { loading, error, empty, onRetry, emptyState: emptyOpts, render }) {
    const el = typeof host === 'string' ? document.getElementById(host) : host;
    if (!el) return false;
    if (loading) {
      el.setAttribute('aria-busy', 'true');
      el.innerHTML = skeleton(3);
      return false;
    }
    el.removeAttribute('aria-busy');
    if (error) {
      el.innerHTML = `<div class="panel-error" role="alert">` +
        `${icon('warn', { size: 22 })}<p class="panel-error-msg">${esc(String(error))}</p>` +
        (onRetry ? `<button class="ghost" type="button" data-panel-retry>Try again</button>` : '') +
        `</div>`;
      const btn = el.querySelector('[data-panel-retry]');
      if (btn && onRetry) btn.addEventListener('click', onRetry);
      return false;
    }
    if (empty) {
      el.innerHTML = emptyState(emptyOpts || { title: 'Nothing here yet', body: '' });
      return false;
    }
    if (render) render(el);
    return true;
  }

  /* ---------- number animation ---------- */

  /* Roll a number to its new value rather than snapping. On a dashboard that repaints only
   * on change, the roll *is* the signal that something changed — and it reads as a
   * measurement moving, which snapping does not. Short enough not to lag the truth. */
  function rollTo(el, next, format) {
    const from = Number(el.dataset.raw || 0);
    const to = Number(next);
    el.dataset.raw = String(to);
    if (!Number.isFinite(from) || !Number.isFinite(to) || from === to) {
      el.textContent = format(to);
      return;
    }
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      el.textContent = format(to);
      return;
    }
    const start = performance.now();
    const step = (now) => {
      const t = Math.min(1, (now - start) / 420);
      const eased = 1 - (1 - t) ** 3;
      el.textContent = format(from + (to - from) * eased);
      if (t < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  /* ---------- collapsible panels ---------- */

  /** Make a panel head toggle its body, remembering the choice per panel id. */
  function collapsible(panel) {
    const head = panel.querySelector('.panel-head');
    const body = panel.querySelector('.panel-body, .scroll-x');
    const id = panel.dataset.panel;
    if (!head || !body || !id) return;
    const apply = (on) => {
      body.hidden = on;
      panel.classList.toggle('is-collapsed', on);
      head.setAttribute('aria-expanded', String(!on));
    };
    apply(Boolean((getPref('collapsed') || {})[id]));
    head.setAttribute('role', 'button');
    head.setAttribute('tabindex', '0');
    const toggle = () => {
      const collapsed = { ...(getPref('collapsed') || {}) };
      collapsed[id] = !collapsed[id];
      setPref('collapsed', collapsed);
      apply(collapsed[id]);
    };
    head.addEventListener('click', (e) => { if (!e.target.closest('input, select, button')) toggle(); });
    head.addEventListener('keydown', (e) => {
      if ((e.key === 'Enter' || e.key === ' ') && !e.target.closest('input, select, button')) {
        e.preventDefault(); toggle();
      }
    });
  }

  /* ---------- misc ---------- */

  const median = (xs) => {
    if (!xs.length) return 0;
    const s = [...xs].sort((a, b) => a - b);
    const m = s.length >> 1;
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
  };

  /** Subsequence match, so "hj" finds "hash_join" the way a command palette should. */
  function fuzzy(needle, haystack) {
    if (!needle) return true;
    const n = needle.toLowerCase(), h = haystack.toLowerCase();
    let i = 0;
    for (const ch of h) if (ch === n[i]) i += 1;
    return i === n.length;
  }

  const debounce = (fn, wait = 120) => {
    let timer = null;
    return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), wait); };
  };

  return {
    count, ms, bytes, pct, clock, stamp, ago, duration, esc, cpuMeasured, blankToDash,
    loadPrefs, setPref, getPref, togglePin, isPinned,
    readRoute, writeRoute, toast, copy, download, toCSV,
    table, tableCSV, sparkline, histogram, median, fuzzy, debounce, niceTicks,
    icon, ICONS, emptyState, skeleton, panelState, rollTo, collapsible,
  };
})();
