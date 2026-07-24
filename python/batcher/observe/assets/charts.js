/* The chart layer — axes, marks, and the hover behaviour every plot on the page shares.
 *
 * `ui.js` has three decorative shapes (a sparkline, a histogram, a tick generator). Those
 * are *glyphs*: they sit inside a table cell and carry a trend, not a reading. This file is
 * the other thing — plots a person reads a value off, which means axes, gridlines, a
 * crosshair, and a tooltip, and which therefore must be built once rather than seven times.
 *
 * ── The encoding rules it enforces ──────────────────────────────────────────────
 * These are the page's, stated in `app.css`, and every function here obeys them so a
 * caller cannot break them by accident:
 *
 *   magnitude -> the sequential ramp (--seq-1..5), one hue light to dark.
 *   state     -> the reserved status palette, ALWAYS with a text label beside it.
 *   identity  -> not colour-coded. Plan steps are stages of one query, not unrelated
 *                categories, so they are labelled. Where a stack genuinely needs its
 *                segments told apart, they are separated by a 2px surface gap and
 *                direct-labelled, not tinted through a ramp that means magnitude.
 *
 * ── The mark specs ─────────────────────────────────────────────────────────────
 *   thin marks; 2px lines; >=8px hit targets on every point; 4px rounded ends on bars,
 *   anchored square to the baseline; a 2px gap between adjacent fills; recessive grid.
 *
 * No dual axes, ever. Two measures at different scales are two charts.
 *
 * Everything is inline SVG built as a string. A charting library would be an order of
 * magnitude larger than the dashboard it decorated, and would still need this file to
 * teach it the rules above.
 */

'use strict';

const CHARTS = (() => {
  const { esc, ms, count, bytes, pct, niceTicks, clock } = UI;

  const REDUCED = typeof window !== 'undefined' && window.matchMedia
    ? window.matchMedia('(prefers-reduced-motion: reduce)').matches : false;

  /* A chart's usable box. Left gutter fits a y label; the bottom fits one row of ticks.
   * Stated once so every plot on the page shares a baseline and they visually align. */
  const PAD = { top: 10, right: 12, bottom: 22, left: 46 };

  /* ---------- formatting ---------- */

  /* Which formatter a value axis should use, named rather than sniffed. A chart that
   * guessed would eventually label a byte count in milliseconds. */
  const FORMATS = {
    ms, count, bytes, pct,
    rate: (v) => `${count(v)}/s`,
    plain: (v) => (v == null ? '—' : String(Math.round(v * 100) / 100)),
    clock,
  };
  const fmt = (name) => FORMATS[name] || FORMATS.plain;

  /* ---------- scales ---------- */

  /* A linear scale as a plain closure. `nice` extends the domain to the tick step so the
   * top gridline is the top of the plot rather than floating below it. */
  function linear(domain, range) {
    const [d0, d1] = domain;
    const [r0, r1] = range;
    const span = (d1 - d0) || 1;
    const scale = (v) => r0 + ((v - d0) / span) * (r1 - r0);
    scale.invert = (px) => d0 + ((px - r0) / ((r1 - r0) || 1)) * span;
    scale.domain = domain;
    scale.range = range;
    return scale;
  }

  /* ---------- frame ---------- */

  /* The axes, gridlines, and labels every plot shares.
   *
   * Y gridlines only. A vertical grid behind a time series adds ink that helps nobody read
   * a value — the x ticks under the axis already say where you are — and it competes with
   * the data line, which is the one thing that should be salient. */
  function frame({ w, h, x, y, xTicks, yTicks, yFormat = 'plain', xFormat = 'plain' }) {
    const fy = fmt(yFormat), fx = fmt(xFormat);
    const grid = yTicks.map((t) => (
      `<line class="ch-grid" x1="${PAD.left}" y1="${y(t).toFixed(1)}" ` +
      `x2="${w - PAD.right}" y2="${y(t).toFixed(1)}"/>`)).join('');
    const yLabels = yTicks.map((t) => (
      `<text class="ch-tick ch-tick-y" x="${PAD.left - 6}" y="${(y(t) + 3.5).toFixed(1)}" ` +
      `text-anchor="end">${esc(fy(t))}</text>`)).join('');
    const xLabels = (xTicks || []).map((t) => (
      `<text class="ch-tick" x="${x(t).toFixed(1)}" y="${h - PAD.bottom + 14}" ` +
      `text-anchor="middle">${esc(fx(t))}</text>`)).join('');
    // The baseline is drawn; the y spine is not. The gridlines already establish the left
    // edge, and a second vertical rule there is a box for the sake of a box.
    const axis = `<line class="ch-axis" x1="${PAD.left}" y1="${h - PAD.bottom}" ` +
      `x2="${w - PAD.right}" y2="${h - PAD.bottom}"/>`;
    return grid + axis + yLabels + xLabels;
  }

  /** Open an SVG element sized to `w`x`h`, scaling to its container's width. */
  function open(w, h, { cls = '', label = '' } = {}) {
    return `<svg class="chart ${cls}" viewBox="0 0 ${w} ${h}" ` +
      `preserveAspectRatio="none" role="img" aria-label="${esc(label)}">`;
  }

  /* ---------- area / line ---------- */

  /* A time series with a real value axis and a crosshair.
   *
   * `series` is `[{key, label, values: [{x, y}], format}]`. Two series get a legend; one
   * does not, because the panel heading already names it and a legend box for a single
   * line is furniture.
   *
   * Deliberately NOT a dual-axis chart. Callers wanting rows/s beside latency get two
   * calls and two plots — a shared x, separate y, which is readable, against one plot with
   * two y scales, which is not.
   */
  function timeSeries(host, series, opts = {}) {
    const el = typeof host === 'string' ? document.getElementById(host) : host;
    if (!el) return null;
    const live = series.filter((s) => s.values && s.values.length);
    if (!live.length) { el.innerHTML = ''; return null; }

    const w = opts.width || 640, h = opts.height || 180;
    const xs = live.flatMap((s) => s.values.map((p) => p.x));
    const ys = live.flatMap((s) => s.values.map((p) => p.y));
    const xMin = Math.min(...xs), xMax = Math.max(...xs);
    // The value axis starts at zero. A truncated baseline makes a 3% wobble look like a
    // collapse, which is the single most effective way for a chart to mislead.
    const yMax = Math.max(...ys, opts.minTop || 0) || 1;
    const x = linear([xMin, xMax], [PAD.left, w - PAD.right]);
    const y = linear([0, yMax * 1.08], [h - PAD.bottom, PAD.top]);
    const yTicks = niceTicks(0, yMax * 1.08, h - PAD.top - PAD.bottom, 34);
    const xTicks = opts.xFormat === 'clock'
      ? [xMin, (xMin + xMax) / 2, xMax] : niceTicks(xMin, xMax, w - PAD.left - PAD.right, 90);

    const paths = live.map((s, i) => {
      const pts = s.values.map((p) => `${x(p.x).toFixed(1)},${y(p.y).toFixed(1)}`);
      const line = `M ${pts.join(' L ')}`;
      const area = `${line} L ${x(s.values[s.values.length - 1].x).toFixed(1)},${y(0)} ` +
                   `L ${x(s.values[0].x).toFixed(1)},${y(0)} Z`;
      const cls = `ch-s${Math.min(i + 1, 5)}`;
      return (opts.fill === false ? '' : `<path class="ch-area ${cls}" d="${area}"/>`) +
        `<path class="ch-line ${cls}" d="${line}"/>`;
    }).join('');

    // Markers only when the series is sparse enough that each one is a real reading. Past
    // that they merge into a caterpillar and the line alone is clearer.
    const dots = live.length === 1 && live[0].values.length <= 40
      ? live[0].values.map((p) => (
          `<circle class="ch-dot ch-s1" cx="${x(p.x).toFixed(1)}" cy="${y(p.y).toFixed(1)}" r="2.5"/>`
        )).join('')
      : '';

    el.innerHTML = open(w, h, { cls: 'ch-ts', label: opts.label || 'time series' }) +
      frame({ w, h, x, y, xTicks, yTicks, yFormat: opts.format, xFormat: opts.xFormat }) +
      paths + dots +
      `<g class="ch-cross" hidden><line class="ch-cross-line" y1="${PAD.top}" y2="${h - PAD.bottom}"/></g>` +
      `<rect class="ch-capture" x="${PAD.left}" y="${PAD.top}" ` +
      `width="${w - PAD.left - PAD.right}" height="${h - PAD.top - PAD.bottom}"/>` +
      `</svg>` +
      (live.length > 1 ? legend(live) : '');

    installCrosshair(el, { series: live, x, y, w, h, format: opts.format,
                           xFormat: opts.xFormat || 'plain' });
    return { x, y };
  }

  /** A legend. Present whenever more than one series is drawn — identity is never colour
   *  alone, so the swatch is always paired with its name. */
  function legend(series) {
    return `<div class="ch-legend">` + series.map((s, i) => (
      `<span class="ch-key"><i class="ch-swatch ch-s${Math.min(i + 1, 5)}"></i>${esc(s.label)}</span>`
    )).join('') + `</div>`;
  }

  /* The crosshair and tooltip.
   *
   * Bound once per render by delegation on the SVG rather than per point: a 600-point
   * series would otherwise install 600 listeners, and the capture rect gives a hit target
   * the full height of the plot instead of an 8px circle a mouse has to find. */
  function installCrosshair(el, ctx) {
    const svg = el.querySelector('svg');
    const capture = el.querySelector('.ch-capture');
    const cross = el.querySelector('.ch-cross');
    if (!svg || !capture || !cross) return;
    const line = cross.querySelector('.ch-cross-line');
    const f = fmt(ctx.format), fx = fmt(ctx.xFormat);

    const at = (clientX) => {
      const box = svg.getBoundingClientRect();
      const px = ((clientX - box.left) / (box.width || 1)) * ctx.w;
      return ctx.x.invert(px);
    };
    const nearest = (values, target) => values.reduce(
      (best, p) => (Math.abs(p.x - target) < Math.abs(best.x - target) ? p : best), values[0]);

    const move = (event) => {
      const target = at(event.clientX);
      const rows = ctx.series.map((s, i) => {
        const p = nearest(s.values, target);
        return { s, p, i };
      });
      const anchor = rows[0].p;
      cross.hidden = false;
      line.setAttribute('x1', ctx.x(anchor.x).toFixed(1));
      line.setAttribute('x2', ctx.x(anchor.x).toFixed(1));
      const body = rows.map(({ s, p, i }) => (
        `<div class="t-row"><span><i class="ch-swatch ch-s${Math.min(i + 1, 5)}"></i>` +
        `${esc(s.label)}</span><span>${esc(f(p.y))}</span></div>`)).join('');
      el.dispatchEvent(new CustomEvent('chart-hover', {
        bubbles: true,
        detail: { event, html: `<b>${esc(fx(anchor.x))}</b>${body}` },
      }));
    };
    const leave = () => {
      cross.hidden = true;
      el.dispatchEvent(new CustomEvent('chart-leave', { bubbles: true }));
    };
    capture.addEventListener('pointermove', move);
    capture.addEventListener('pointerleave', leave);
  }

  /* ---------- bars ---------- */

  /* A ranked horizontal bar list. Rows, not an SVG: a bar chart of named things is a table
   * with one graphical column, and building it as HTML means the labels wrap, the values
   * stay selectable, and the whole thing is readable with styles off.
   *
   * `rows` is `[{label, value, note, tone, id, sub}]`. `tone` names a *state*
   * (good/warn/serious/critical) and always renders its own text, never colour alone. */
  function bars(rows, opts = {}) {
    if (!rows.length) return '';
    const max = Math.max(...rows.map((r) => Math.abs(r.value)), 1);
    const f = fmt(opts.format);
    const total = rows.reduce((s, r) => s + Math.abs(r.value), 0) || 1;
    return `<div class="ch-bars${opts.dense ? ' is-dense' : ''}">` + rows.map((r, i) => {
      const width = Math.max(1.5, (Math.abs(r.value) / max) * 100);
      const tone = r.tone ? ` is-${r.tone}` : '';
      const tag = opts.onPick ? 'button' : 'div';
      const attrs = opts.onPick ? ` type="button" data-pick="${esc(String(r.id ?? i))}"` : '';
      // The bars stagger in, 24ms apart, so a repaint reads as the list *arriving* rather
      // than as a flicker. Under reduced motion they are simply there at full width.
      const grow = REDUCED ? '' : ' ch-grow';
      const delay = REDUCED ? '' : `;animation-delay:${Math.min(i, 12) * 24}ms`;
      return `<${tag} class="ch-bar-row${tone}"${attrs}>` +
        `<span class="ch-bar-label">${esc(r.label)}` +
        (r.sub ? `<span class="ch-bar-sub">${esc(r.sub)}</span>` : '') + `</span>` +
        `<span class="ch-bar-track"><i class="ch-bar-fill${grow}" ` +
        `style="width:${width.toFixed(1)}%${delay}"></i></span>` +
        `<span class="ch-bar-value">${esc(f(r.value))}</span>` +
        `<span class="ch-bar-share">${esc(pct(Math.abs(r.value) / total))}</span>` +
        (r.note ? `<span class="ch-bar-note">${r.note}</span>` : '') +
        `</${tag}>`;
    }).join('') + `</div>`;
  }

  /* ---------- stacked proportion bar ---------- */

  /* One bar split into named parts.
   *
   * Segments are sorted largest first and tinted down the sequential ramp, which keeps the
   * page's encoding rule intact: the ramp still means magnitude, because position in it and
   * size of the segment are the same ordering. It is not being borrowed as a set of
   * category colours.
   *
   * The 2px gap between segments is load-bearing. Without it two adjacent steps of one hue
   * read as a single segment and the reader silently undercounts. Everything past the top
   * few folds into one "others" rather than into slivers nobody can hit or label. */
  function proportion(parts, opts = {}) {
    const total = parts.reduce((s, p) => s + p.value, 0);
    if (!total) return '';
    const f = fmt(opts.format);
    const ordered = [...parts].sort((a, b) => b.value - a.value);
    const shown = ordered.slice(0, opts.max || 6);
    const rest = ordered.slice(opts.max || 6);
    const restTotal = rest.reduce((s, p) => s + p.value, 0);
    const all = restTotal
      ? [...shown, { label: `${rest.length} others`, value: restTotal, id: null }]
      : shown;
    // `link` names the data attribute a segment carries, so a caller can emit the same
    // `data-pipe` / `data-run` the rest of the page navigates by and reuse the one
    // delegated cross-reference listener instead of wiring its own.
    const attr = opts.link || 'pick';
    // The interactive attributes are written out inline rather than folded into a helper.
    // The dead-CSS check reads the markup immediately around a class to decide whether a
    // `:focus-visible` rule on it can ever fire, and `role`/`tabindex` hidden behind a
    // helper call are invisible to it — the rule then reads as dead when it is not.
    return `<div class="ch-prop" role="img" aria-label="${esc(opts.label || 'proportions')}">` +
      all.map((p, i) => (
        // The "others" bucket has no id, so it takes no role and no place in the focus
        // order: something focusable that does nothing when you activate it is a dead stop.
        `<span class="ch-prop-seg ch-p${Math.min(i + 1, 6)}"` +
        (p.id == null ? '' : ` role="button" tabindex="0" data-${attr}="${esc(String(p.id))}"`) +
        ` style="flex-basis:${((p.value / total) * 100).toFixed(2)}%" ` +
        `title="${esc(p.label)} · ${esc(f(p.value))} · ${esc(pct(p.value / total))}"></span>`
      )).join('') + `</div>` +
      `<div class="ch-prop-keys">` + all.map((p, i) => (
        `<span class="ch-prop-key"` +
        (p.id == null ? '' : ` role="button" tabindex="0" data-${attr}="${esc(String(p.id))}"`) +
        `>` +
        `<i class="ch-swatch ch-p${Math.min(i + 1, 6)}"></i>` +
        `<span class="ch-prop-name">${esc(p.label)}</span>` +
        `<span class="ch-prop-val">${esc(f(p.value))}</span>` +
        `<span class="ch-prop-pct">${esc(pct(p.value / total))}</span></span>`)).join('') +
      `</div>`;
  }

  /* ---------- flame ---------- */

  /* The plan as a flame graph: width is time, depth is nesting.
   *
   * The form profilers settled on, for a reason that applies exactly here — it shows where
   * time went *and* the call structure it went through in one picture, which a ranked list
   * (structure lost) and a graph (time lost) each give up half of.
   *
   * Width is the operator's own measured time, so a parent is not the sum of its children:
   * these steps run concurrently, and drawing a parent as the sum would invent a total that
   * exceeds the wall clock. Each frame is drawn at its own width and left-aligned under its
   * parent, which is honest about that and still reads as a flame.
   */
  function flame(nodes, opts = {}) {
    const measured = nodes.filter((n) => n.measured && n.elapsed_ms > 0);
    if (!measured.length) return '';
    const total = measured.reduce((s, n) => s + n.elapsed_ms, 0) || 1;
    const maxDepth = Math.max(...nodes.map((n) => n.depth));
    // Rows are drawn root-first, matching the plan tree and the EXPLAIN text. A flame graph
    // is conventionally inverted (leaves up), but the reader has just been looking at a
    // plan drawn root-at-top and flipping it costs more than the convention buys.
    const byDepth = new Map();
    for (const n of measured) {
      if (!byDepth.has(n.depth)) byDepth.set(n.depth, []);
      byDepth.get(n.depth).push(n);
    }
    return `<div class="ch-flame" role="img" aria-label="${esc(opts.label || 'time by plan depth')}">` +
      Array.from({ length: maxDepth + 1 }, (_, d) => {
        const row = (byDepth.get(d) || []).sort((a, b) => a.op_id - b.op_id);
        if (!row.length) return '';
        return `<div class="ch-flame-row">` + row.map((n) => {
          const share = n.elapsed_ms / total;
          // Magnitude, so: the sequential ramp, five steps, light to dark.
          const step = share > 0.35 ? 5 : share > 0.2 ? 4 : share > 0.1 ? 3 : share > 0.04 ? 2 : 1;
          return `<button class="ch-flame-cell ch-seq-${step}${n.on_critical_path ? ' is-crit' : ''}" ` +
            `type="button" data-pick="${n.op_id}" ` +
            `style="flex-grow:${Math.max(0.02, share).toFixed(4)}" ` +
            `title="${esc(opts.name ? opts.name(n) : n.kind)} · ${esc(ms(n.elapsed_ms))} · ${esc(pct(share))}" ` +
            `aria-label="${esc(opts.name ? opts.name(n) : n.kind)}, ${esc(pct(share))} of operator time">` +
            `<span class="ch-flame-text">${esc(opts.name ? opts.name(n) : n.kind)}</span></button>`;
        }).join('') + `</div>`;
      }).join('') + `</div>`;
  }

  /* ---------- heatmap ---------- */

  /* A value grid. Five discrete steps rather than a continuous ramp, because adjacent cells
   * must be tellable apart at a glance — a smooth gradient turns a grid into a wash, and
   * the reader can no longer say which cell is the bad one, which is the only question a
   * heatmap exists to answer. */
  function heat(value, mid = 1) {
    const ratio = mid ? value / mid : 1;
    return ratio >= 2 ? 5 : ratio >= 1.35 ? 4 : ratio >= 0.85 ? 3 : ratio >= 0.5 ? 2 : 1;
  }

  /* ---------- gauge ---------- */

  /* A bounded reading against its limit — GPU utilization, memory against budget. A gauge
   * only makes sense when the maximum is real and known; anything unbounded gets a number
   * and a sparkline instead, because a gauge implies a ceiling that does not exist.
   *
   * `bands` mark where the reading *should* sit. Without them a gauge says "72%" without
   * saying whether 72% is good, which for GPU utilization is the whole question. */
  function gauge({ value, max = 1, label, format = 'pct', bands = [], tone = '' }) {
    const f = fmt(format);
    const share = max > 0 ? Math.min(1, Math.max(0, value / max)) : 0;
    return `<div class="ch-gauge${tone ? ` is-${tone}` : ''}">` +
      `<div class="ch-gauge-head"><span class="ch-gauge-label">${esc(label)}</span>` +
      `<span class="ch-gauge-value">${esc(f(value))}</span></div>` +
      `<div class="ch-gauge-track" role="meter" aria-valuenow="${value}" aria-valuemin="0" ` +
      `aria-valuemax="${max}" aria-label="${esc(label)}">` +
      bands.map((b) => (
        `<i class="ch-gauge-band is-${esc(b.tone)}" style="left:${(b.from * 100).toFixed(1)}%;` +
        `width:${((b.to - b.from) * 100).toFixed(1)}%"></i>`)).join('') +
      `<i class="ch-gauge-fill" style="width:${(share * 100).toFixed(1)}%"></i></div></div>`;
  }

  /* ---------- box plot ---------- */

  /* A distribution's shape in one row: the p5-p95 whisker, the p25-p75 box, the median.
   * The honest alternative to reporting a mean for a latency distribution, which is almost
   * always skewed and for which the mean is a number no run ever took. */
  function box({ p5, p25, p50, p75, p95, min, max, format = 'ms' }) {
    const lo = min ?? p5, hi = max ?? p95;
    const span = (hi - lo) || 1;
    const at = (v) => ((v - lo) / span) * 100;
    const f = fmt(format);
    const span_ = (a, b) => Math.max(0.6, at(b) - at(a)).toFixed(2);
    return `<div class="ch-box" role="img" ` +
      `aria-label="median ${esc(f(p50))}, half of runs between ${esc(f(p25))} and ${esc(f(p75))}, ` +
      `range ${esc(f(lo))} to ${esc(f(hi))}">` +
      `<i class="ch-box-whisker" style="left:${at(p5).toFixed(2)}%;width:${span_(p5, p95)}%"></i>` +
      `<i class="ch-box-iqr" style="left:${at(p25).toFixed(2)}%;width:${span_(p25, p75)}%"></i>` +
      `<i class="ch-box-median" style="left:${at(p50).toFixed(2)}%"></i>` +
      `</div><div class="ch-box-scale"><span>${esc(f(lo))}</span>` +
      `<span class="ch-box-mid">p50 ${esc(f(p50))}</span><span>${esc(f(hi))}</span></div>`;
  }

  /* ---------- delta bar ---------- */

  /* A signed change against a zero centre line. Direction is the arrow; whether that
   * direction is good is the colour — slower is bad and faster is good, and both are shown,
   * because an improvement deserves to be as visible as a regression. */
  function delta(value, worst, { format = 'ms', goodWhenNegative = true } = {}) {
    const f = fmt(format);
    const share = Math.min(1, Math.abs(value) / (worst || 1));
    const good = goodWhenNegative ? value < 0 : value > 0;
    if (!value) return `<span class="ch-delta is-flat">no change</span>`;
    return `<span class="ch-delta ${good ? 'is-good' : 'is-warn'}">` +
      `<i class="ch-delta-track"><b class="${value < 0 ? 'to-left' : 'to-right'}" ` +
      `style="width:${(share * 50).toFixed(1)}%"></i></i>` +
      `<b class="ch-delta-value">${value > 0 ? '▲' : '▼'} ${esc(f(Math.abs(value)))}</b></span>`;
  }

  /* ---------- axis-free micro shapes ---------- */

  /** A stepped bar strip — one bar per run, for a header that has no room for an axis. */
  function strip(values, { max = null, height = 26, format = 'ms', onPick = null } = {}) {
    if (!values.length) return '';
    const top = max || Math.max(...values.map((v) => v.value), 1);
    const f = fmt(format);
    return `<div class="ch-strip" style="height:${height}px">` + values.map((v, i) => (
      `<${onPick ? 'button' : 'span'} class="ch-strip-bar${v.tone ? ` is-${v.tone}` : ''}` +
      `${v.current ? ' is-current' : ''}" ${onPick ? `type="button" data-pick="${esc(String(v.id ?? i))}"` : ''} ` +
      `style="height:${Math.max(6, (v.value / top) * 100).toFixed(1)}%" ` +
      `title="${esc(v.label || '')} ${esc(f(v.value))}"></${onPick ? 'button' : 'span'}>`
    )).join('') + `</div>`;
  }

  /** Wire `data-pick` buttons inside `host` to a handler, once per render. */
  function onPick(host, handler) {
    const el = typeof host === 'string' ? document.getElementById(host) : host;
    if (!el) return;
    el.addEventListener('click', (event) => {
      const hit = event.target.closest('[data-pick]');
      if (hit && el.contains(hit)) handler(hit.dataset.pick, event);
    });
    el.addEventListener('keydown', (event) => {
      if (event.key !== 'Enter' && event.key !== ' ') return;
      const hit = event.target.closest('[data-pick]');
      if (!hit || !el.contains(hit) || hit.tagName === 'BUTTON') return;
      event.preventDefault();
      handler(hit.dataset.pick, event);
    });
  }

  return { timeSeries, bars, proportion, flame, gauge, box, delta, strip, heat, onPick };
})();
