/* The plan explorer — a navigable, inspectable graph of the executed query.
 *
 * The server computes the layout and the critical path next to the IR they derive from;
 * this owns drawing and interaction. A plan wider than the panel is the normal case, so the
 * canvas is navigable rather than merely scrollable.
 *
 * WHAT MAKES IT AN EXPLORER RATHER THAN A PICTURE:
 *   • select a node -> a full inspector, every measured field, no hover required
 *   • critical path -> the chain that actually sets the runtime, highlighted on demand
 *   • focus        -> dim everything that does not feed, or is not fed by, the selection
 *   • search       -> match by operator name or detail, non-matches recede
 *   • minimap      -> position and extent when the graph is larger than the viewport
 *   • drag/zoom/fit + keyboard (arrows to walk the plan, f to fit, c for critical path)
 *
 * Encoding: fill is share of operator time on a single-hue ramp (time is a magnitude).
 * State — spilled, on the critical path, selected — is carried by border, opacity, and a
 * text label, never by hue, so a fast spilled operator cannot read as a cool one.
 */

'use strict';

/* A *factory*, not a singleton. The run page and the pipeline page each own a graph, and
 * they must keep independent selections, pan, zoom, and dragged-node positions — sharing one
 * instance made navigating between them reset the other's view every time. */
function createDagExplorer() {
  /* The viewer's motion preference, read once. The bar sweep and node rise are decorative;
   * a reader who has asked for no motion should get the graph drawn, not animated in. */
  const REDUCED_MOTION = typeof window !== 'undefined' && window.matchMedia
    ? window.matchMedia('(prefers-reduced-motion: reduce)').matches
    : false;
  const NODE_W = 236;
  const NODE_H = 96;
  const HEAD_H = 28;

  /* One mark per operator kind, so a plan is scannable by shape before it is read. Drawn on
   * an 18x18 grid to match the rest of the icon set. */
  const GLYPHS = {
    hash_join: 'M3 5h5v8H3zM10 5h5v8h-5zM8 9h2',
    aggregate: 'M3 4h12L9 10v4l-2 1v-5z',
    scan: 'M3 5c0-1.1 2.7-2 6-2s6 .9 6 2v8c0 1.1-2.7 2-6 2s-6-.9-6-2zM3 9c0 1.1 2.7 2 6 2s6-.9 6-2',
    filter: 'M3 4h12l-5 5v5l-2 1V9z',
    project: 'M3 4h12M3 9h12M3 14h7',
    sort: 'M5 3v12m0 0l-3-3m3 3l3-3M13 15V3m0 0l3 3m-3-3l-3 3',
    limit: 'M3 9h12M6 5l-3 4 3 4',
    distinct: 'M6 4h8v8H6zM3 7h3v8h8',
    union: 'M4 4v5a5 5 0 0010 0V4',
    window: 'M3 4h12v10H3zM3 7h12M7 7v7',
    _default: 'M3 4h12v10H3z',
  };
  const GAP_X = 40;
  const GAP_Y = 56;
  const PAD = 28;
  const SVG_NS = 'http://www.w3.org/2000/svg';
  const MIN_SCALE = 0.25;
  const MAX_SCALE = 2.6;

  const moved = new Map();          // queryId -> { op_id: {x,y} } for user-dragged boxes
  let view = { x: 0, y: 0, k: 1 };
  let ctx = null;                   // everything about the currently rendered graph
  let sizeObserver = null;          // fits the graph once its container first has a size

  const el = (name, attrs = {}) => {
    const node = document.createElementNS(SVG_NS, name);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
    return node;
  };

  /* Time share -> one of five ramp steps. Discrete because five steps are distinguishable
   * side by side, where a continuous gradient across separated boxes is not. */
  const rampStep = (share) =>
    share >= 0.5 ? 5 : share >= 0.25 ? 4 : share >= 0.1 ? 3 : share >= 0.02 ? 2 : 1;

  const basePos = (n) => ({ x: PAD + n.column * (NODE_W + GAP_X), y: PAD + n.row * (NODE_H + GAP_Y) });
  const posOf = (n, qid) => ({ ...(moved.get(qid)?.[n.op_id] || basePos(n)) });

  /* Below this scale the per-node text is smaller than ~7px and unreadable, so it is hidden
   * rather than painted. Structure stays legible, and a large plan gets cheaper to draw at
   * exactly the zoom where it has the most nodes on screen. */
  const LOD_SCALE = 0.55;

  function applyView() {
    if (!ctx) return;
    ctx.root.setAttribute('transform', `translate(${view.x} ${view.y}) scale(${view.k})`);
    ctx.svg.classList.toggle('is-far', view.k < LOD_SCALE);
    ctx.onZoom?.(view.k);
    drawMinimap();
  }

  function bounds() {
    const pts = ctx.placed.map((n) => ctx.boxes.get(n.op_id));
    return {
      minX: Math.min(...pts.map((p) => p.x)) - PAD,
      minY: Math.min(...pts.map((p) => p.y)) - PAD,
      maxX: Math.max(...pts.map((p) => p.x)) + NODE_W + PAD,
      maxY: Math.max(...pts.map((p) => p.y)) + NODE_H + PAD,
    };
  }

  function fit() {
    if (!ctx || !ctx.placed.length) return;
    const b = bounds();
    const box = ctx.svg.getBoundingClientRect();
    // A zero-width box means the graph's pane is not laid out yet — hidden behind another
    // tab, or revealed a frame later than this call. Leave `fitted` false so the size
    // observer re-fits the moment the box gains a width, instead of stranding the graph at
    // scale 1 in the top-left corner, which is what read as "the plan graph is broken".
    if (!box.width) return;
    const k = Math.max(MIN_SCALE, Math.min(1.15, Math.min(box.width / (b.maxX - b.minX),
                                                          box.height / (b.maxY - b.minY)) * 0.94));
    view = { k, x: (box.width - (b.maxX - b.minX) * k) / 2 - b.minX * k,
                y: (box.height - (b.maxY - b.minY) * k) / 2 - b.minY * k };
    applyView();
    ctx.fitted = true;
  }

  /* Fit the graph once its container first has a size.
   *
   * `render` schedules a fit on the next frame, but a graph that renders into a hidden pane
   * (another tab is active, or the view is revealed a beat later) has no width then, so that
   * fit no-ops and nothing ever re-fits it. A ResizeObserver closes the gap: the first time
   * the box transitions to a real width it fits, once, and then stops auto-fitting so a zoom
   * the reader has set is never yanked back. */
  function observeSize(svg) {
    if (typeof ResizeObserver === 'undefined') return;
    if (sizeObserver) sizeObserver.disconnect();
    sizeObserver = new ResizeObserver(() => { if (ctx && !ctx.fitted) fit(); });
    sizeObserver.observe(svg);
  }

  /* Pan a specific step to the middle of the viewport without changing zoom.
   *
   * Zoom is left alone deliberately: a reader who has zoomed in to read one region has
   * expressed an intent, and re-fitting the whole graph to reveal one node throws that away.
   * Only the position moves. */
  function reveal(id) {
    if (!ctx) return;
    // The node's drawn position lives in `ctx.boxes`, keyed by op_id — `placed` holds the
    // node data, not its coordinates. Reading `.x`/`.y` off a placed node (or `.node.op_id`)
    // silently found nothing, so clicking a cost row never panned to its step.
    const pos = ctx.boxes.get(id);
    if (!pos) return;
    const box = ctx.svg.getBoundingClientRect();
    if (!box.width) return;
    view = {
      k: view.k,
      x: box.width / 2 - (pos.x + NODE_W / 2) * view.k,
      y: box.height / 2 - (pos.y + NODE_H / 2) * view.k,
    };
    applyView();
  }

  /** Select `id` outright rather than toggling — for callers driving selection from
   *  elsewhere, where "click the row that is already selected" must not deselect it. */
  function selectOnly(id) {
    if (!ctx) return;
    ctx.selected = id;
    applyEmphasis();
    ctx.opts.onSelect?.(id == null ? null : ctx.byId.get(id), ctx.stats);
  }

  function reset() {
    if (!ctx) return;
    moved.delete(ctx.queryId);
    render(ctx.svg, ctx.dag, ctx.opts, ctx.queryId);
  }

  /* --- selection, focus, search ------------------------------------------ */

  /** Every node reachable from `id` in either direction — its lineage in the plan. */
  function related(id) {
    const up = new Map();     // node -> its children
    const down = new Map();   // node -> its parents
    for (const e of ctx.dag.edges) {
      up.set(e.to, [...(up.get(e.to) || []), e.from]);
      down.set(e.from, [...(down.get(e.from) || []), e.to]);
    }
    const seen = new Set([id]);
    const walk = (start, table) => {
      const stack = [start];
      while (stack.length) {
        for (const next of table.get(stack.pop()) || []) {
          if (!seen.has(next)) { seen.add(next); stack.push(next); }
        }
      }
    };
    walk(id, up);
    walk(id, down);
    return seen;
  }

  function select(id) {
    if (!ctx) return;
    ctx.selected = ctx.selected === id ? null : id;
    applyEmphasis();
    ctx.opts.onSelect?.(ctx.selected == null ? null : ctx.byId.get(ctx.selected), ctx.stats);
  }

  /** One place decides which nodes are lit: selection, focus, search, critical path. */
  function applyEmphasis() {
    if (!ctx) return;
    const { selected, focus, needle, showCritical } = ctx;
    const lineage = focus && selected != null ? related(selected) : null;
    const critical = new Set(ctx.dag.critical_path || []);

    for (const [id, group] of ctx.groups) {
      const node = ctx.byId.get(id);
      const matches = !needle ||
        `${node.kind} ${node.detail}`.toLowerCase().includes(needle);
      const inLineage = !lineage || lineage.has(id);
      const dimmed = !matches || !inLineage;
      group.classList.toggle('is-dim', dimmed);
      group.classList.toggle('is-selected', id === selected);
      group.classList.toggle('is-critical', showCritical && critical.has(id));
    }
    for (const { path, from, to } of ctx.edgePaths) {
      const onCritical = showCritical && critical.has(from) && critical.has(to);
      const dimmed = lineage && !(lineage.has(from) && lineage.has(to));
      path.classList.toggle('is-critical', onCritical);
      path.classList.toggle('is-dim', Boolean(dimmed));
    }
  }

  const setSearch = (text) => { if (ctx) { ctx.needle = text.trim().toLowerCase(); applyEmphasis(); } };
  const setFocus = (on) => { if (ctx) { ctx.focus = on; applyEmphasis(); } };
  const setCritical = (on) => { if (ctx) { ctx.showCritical = on; applyEmphasis(); } };
  const selectedNode = () => (ctx && ctx.selected != null ? ctx.byId.get(ctx.selected) : null);

  /* --- minimap ------------------------------------------------------------ */

  function drawMinimap() {
    const mini = ctx?.opts.minimap;
    if (!mini || !ctx.placed.length) return;
    const b = bounds();
    const w = b.maxX - b.minX;
    const h = b.maxY - b.minY;
    const box = ctx.svg.getBoundingClientRect();
    const scale = Math.min(mini.clientWidth / w, mini.clientHeight / h);
    const dots = ctx.placed.map((n) => {
      const p = ctx.boxes.get(n.op_id);
      return `<i style="left:${(p.x - b.minX) * scale}px;top:${(p.y - b.minY) * scale}px;` +
             `width:${Math.max(3, NODE_W * scale)}px;height:${Math.max(2, NODE_H * scale)}px"></i>`;
    }).join('');
    // The viewport rectangle, in graph coordinates, projected onto the map.
    const vx = (-view.x / view.k - b.minX) * scale;
    const vy = (-view.y / view.k - b.minY) * scale;
    mini.innerHTML = dots +
      `<u style="left:${vx}px;top:${vy}px;width:${(box.width / view.k) * scale}px;` +
      `height:${(box.height / view.k) * scale}px"></u>`;
  }

  /* --- render ------------------------------------------------------------- */

  function render(svg, dag, opts = {}, queryId = '') {
    svg.textContent = '';
    const prev = ctx;
    ctx = { svg, dag, opts, queryId, placed: [], boxes: new Map(), groups: new Map(),
            byId: new Map(), edgePaths: [], selected: null, focus: false, needle: '',
            showCritical: prev && prev.queryId === queryId ? prev.showCritical : true,
            onZoom: opts.onZoom, stats: null, fitted: false };
    if (!dag || !dag.nodes.length) return [];

    // Rows are numbered bottom-up (0 = sources) but drawn top-down.
    const maxRow = Math.max(...dag.nodes.map((n) => n.row));
    const placed = dag.nodes.map((n) => ({ ...n, row: maxRow - n.row }));
    ctx.placed = placed;
    const opTotal = placed.reduce((s, n) => s + (n.elapsed_ms || 0), 0) || 1;
    ctx.stats = { opTotal, count: placed.length };
    for (const n of placed) { ctx.byId.set(n.op_id, n); ctx.boxes.set(n.op_id, posOf(n, queryId)); }

    const root = el('g');
    ctx.root = root;
    svg.appendChild(root);
    const edgeLayer = el('g');
    const nodeLayer = el('g');
    root.append(edgeLayer, nodeLayer);

    const redrawEdges = () => {
      for (const { path, label, from, to } of ctx.edgePaths) {
        const a = ctx.boxes.get(from);
        const b = ctx.boxes.get(to);
        if (!a || !b) continue;
        const x1 = a.x + NODE_W / 2, y1 = a.y;
        const x2 = b.x + NODE_W / 2, y2 = b.y + NODE_H;
        const mid = (y1 + y2) / 2;
        path.setAttribute('d', `M ${x1} ${y1} C ${x1} ${mid}, ${x2} ${mid}, ${x2} ${y2}`);
        if (label) {
          // Sat at the curve's midpoint. The label reads the *producing* node's output,
          // which is what actually flows along this edge.
          label.setAttribute('x', (x1 + x2) / 2);
          label.setAttribute('y', mid + 3);
        }
      }
    };

    for (const edge of dag.edges) {
      if (!ctx.boxes.has(edge.from) || !ctx.boxes.has(edge.to)) continue;
      const path = el('path', { class: 'dag-edge' });
      edgeLayer.appendChild(path);
      // The row count travelling this edge — the producing node's output. Drawn only when
      // measured, so an unprofiled plan stays clean rather than covered in em-dashes.
      const source = ctx.byId.get(edge.from);
      let label = null;
      if (source && source.measured && source.rows_out != null) {
        label = el('text', { class: 'edge-rows', 'text-anchor': 'middle' });
        label.textContent = fmtCount(source.rows_out);
        edgeLayer.appendChild(label);
      }
      ctx.edgePaths.push({ path, label, from: edge.from, to: edge.to });
    }
    redrawEdges();

    placed.forEach((node, i) => {
      const pos = ctx.boxes.get(node.op_id);
      const share = (node.elapsed_ms || 0) / opTotal;
      const step = rampStep(share);
      const group = el('g', { class: 'dag-node', transform: `translate(${pos.x} ${pos.y})`,
                              tabindex: '0', role: 'button',
                              'aria-label': `${friendlyKind(node.kind)} ${node.detail || ''}` });
      if (!REDUCED_MOTION) group.style.animationDelay = `${Math.min(i * 24, 280)}ms`;
      else group.style.animation = 'none';

      // Card body. A single flat rectangle read as a placeholder; this is a header band over
      // a body, which is how every other card on the page is built.
      group.appendChild(el('rect', { class: 'node-box', x: 0, y: 0, width: NODE_W, height: NODE_H,
        rx: 10, fill: 'var(--surface-1)',
        stroke: node.spilled ? 'var(--serious)' : 'var(--border-strong)',
        'stroke-width': node.spilled ? 2 : 1 }));
      group.appendChild(el('path', {
        class: 'node-band',
        d: `M 0 10 a 10 10 0 0 1 10 -10 h ${NODE_W - 20} a 10 10 0 0 1 10 10 v ${HEAD_H - 10} h -${NODE_W} z`,
        fill: `var(--seq-${step})`, 'fill-opacity': step >= 4 ? 0.34 : 0.18 }));

      // The operator's own mark, so a plan is scannable by shape before it is read.
      const glyph = el('path', { class: 'node-icon', transform: `translate(11 ${(HEAD_H - 14) / 2}) scale(0.78)`,
        d: GLYPHS[node.kind] || GLYPHS._default, fill: 'none', stroke: 'currentColor',
        'stroke-width': 2, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' });
      group.appendChild(glyph);

      const kind = el('text', { class: 'node-kind', x: 32, y: HEAD_H / 2 + 4 });
      kind.textContent = friendlyKind(node.kind);
      group.appendChild(kind);

      const badge = el('text', { class: 'node-op', x: NODE_W - 11, y: HEAD_H / 2 + 4, 'text-anchor': 'end' });
      badge.textContent = `op ${node.op_id}`;
      group.appendChild(badge);

      if (node.detail) {
        const d = el('text', { class: 'node-detail', x: 12, y: HEAD_H + 17 });
        d.textContent = node.detail.length > 30 ? `${node.detail.slice(0, 29)}\u2026` : node.detail;
        group.appendChild(d);
      }

      // Share of operator time as a bar, not only a number: the eye compares lengths across
      // a graph far faster than it compares percentages.
      const barY = HEAD_H + 26;
      const barW = NODE_W - 62;
      group.appendChild(el('rect', { class: 'node-bar-bg', x: 12, y: barY, width: barW, height: 5, rx: 2.5 }));
      if (share > 0) {
        group.appendChild(el('rect', { class: 'node-bar', x: 12, y: barY,
          width: Math.max(3, barW * Math.min(share, 1)), height: 5, rx: 2.5,
          fill: `var(--seq-${Math.max(step, 3)})` }));
      }
      const sharePct = el('text', { class: 'node-share', x: NODE_W - 12, y: barY + 5, 'text-anchor': 'end' });
      sharePct.textContent = fmtShare(share);
      group.appendChild(sharePct);

      // Metrics row: rows out and time, each labelled, plus an estimate-accuracy flag when
      // the planner was badly wrong — the one number that explains a wrong plan.
      const metricY = NODE_H - 11;
      const rows = el('text', { class: 'node-metric', x: 12, y: metricY });
      rows.textContent = node.measured ? `${fmtCount(node.rows_out)} rows` : 'not measured';
      group.appendChild(rows);

      const time = el('text', { class: 'node-metric is-strong', x: 12 + barW * 0.55, y: metricY });
      time.textContent = node.measured ? fmtMs(node.elapsed_ms) : '';
      group.appendChild(time);

      const off = node.est_error != null && (node.est_error > 10 || node.est_error < 0.1);
      if (off || node.spilled) {
        const flag = el('text', { class: `node-flag${off ? ' is-warn' : ' is-serious'}`,
                                  x: NODE_W - 12, y: metricY, 'text-anchor': 'end' });
        flag.textContent = node.spilled ? 'spilled' : `${node.est_error.toFixed(0)}x off`;
        group.appendChild(flag);
      }

      group.addEventListener('click', (e) => { e.stopPropagation(); select(node.op_id); });
      group.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); select(node.op_id); }
      });
      if (opts.onHover) {
        group.addEventListener('mouseenter', (e) => opts.onHover(e, node, share));
        group.addEventListener('mousemove', (e) => opts.onHover(e, node, share));
        group.addEventListener('mouseleave', opts.onLeave);
      }

      // --- drag one box ---
      group.addEventListener('pointerdown', (event) => {
        event.stopPropagation();
        group.setPointerCapture(event.pointerId);
        group.classList.add('is-dragging');
        const start = { x: event.clientX, y: event.clientY };
        const origin = { ...ctx.boxes.get(node.op_id) };
        let dragged = false;
        const onMove = (m) => {
          if (Math.hypot(m.clientX - start.x, m.clientY - start.y) > 3) dragged = true;
          const next = { x: origin.x + (m.clientX - start.x) / view.k,
                         y: origin.y + (m.clientY - start.y) / view.k };
          ctx.boxes.set(node.op_id, next);
          group.setAttribute('transform', `translate(${next.x} ${next.y})`);
          redrawEdges();
          drawMinimap();
        };
        const onUp = () => {
          group.classList.remove('is-dragging');
          group.removeEventListener('pointermove', onMove);
          group.removeEventListener('pointerup', onUp);
          if (dragged) {
            const store = moved.get(queryId) || {};
            store[node.op_id] = ctx.boxes.get(node.op_id);
            moved.set(queryId, store);
          }
        };
        group.addEventListener('pointermove', onMove);
        group.addEventListener('pointerup', onUp);
      });

      ctx.groups.set(node.op_id, group);
      nodeLayer.appendChild(group);
    });

    installGestures(svg);
    applyEmphasis();
    requestAnimationFrame(fit);
    // Belt to the rAF-fit's suspenders: if the pane has no width this frame, this fits the
    // instant it gains one, so a graph rendered off-screen is never left un-fitted.
    observeSize(svg);

    const legend = [{ swatch: 'ramp', label: 'share of operator time' }];
    if (dag.critical_path?.length) legend.push({ cls: 'crit', label: 'critical path — the chain that sets the runtime' });
    if (placed.some((n) => n.spilled)) legend.push({ color: 'var(--serious)', label: 'spilled to disk' });
    return legend;
  }

  /** Pan by dragging the background; zoom on wheel, anchored at the cursor. */
  function installGestures(svg) {
    if (svg.dataset.gestures === '1') return;
    svg.dataset.gestures = '1';

    svg.addEventListener('pointerdown', (event) => {
      svg.setPointerCapture(event.pointerId);
      svg.classList.add('is-panning');
      const start = { x: event.clientX, y: event.clientY, vx: view.x, vy: view.y };
      let dragged = false;
      const onMove = (m) => {
        if (Math.hypot(m.clientX - start.x, m.clientY - start.y) > 3) dragged = true;
        view.x = start.vx + (m.clientX - start.x);
        view.y = start.vy + (m.clientY - start.y);
        applyView();
      };
      const onUp = () => {
        svg.classList.remove('is-panning');
        svg.removeEventListener('pointermove', onMove);
        svg.removeEventListener('pointerup', onUp);
        if (!dragged && ctx?.selected != null) select(ctx.selected);   // click empty = deselect
      };
      svg.addEventListener('pointermove', onMove);
      svg.addEventListener('pointerup', onUp);
    });

    svg.addEventListener('wheel', (event) => {
      event.preventDefault();
      const box = svg.getBoundingClientRect();
      const px = event.clientX - box.left, py = event.clientY - box.top;
      const next = Math.max(MIN_SCALE, Math.min(MAX_SCALE, view.k * (event.deltaY < 0 ? 1.12 : 1 / 1.12)));
      view.x = px - ((px - view.x) / view.k) * next;   // keep the point under the cursor fixed
      view.y = py - ((py - view.y) / view.k) * next;
      view.k = next;
      applyView();
    }, { passive: false });
  }

  /* Walk the plan with the arrow keys: up/down move along the executed order, which is what
   * a person means by "the next step", and left/right move between siblings. */
  function step(direction) {
    if (!ctx || !ctx.placed.length) return;
    const order = [...ctx.placed].sort((a, b) => a.row - b.row || a.column - b.column);
    const at = order.findIndex((n) => n.op_id === ctx.selected);
    const next = at < 0 ? 0 : Math.max(0, Math.min(order.length - 1, at + direction));
    focusNode(order[next].op_id);
  }

  /* Move left or right among the nodes on the *same* row, so arrow keys navigate the graph
   * in the two dimensions it is drawn in rather than only up and down its flattening. */
  function stepAcross(direction) {
    if (!ctx || ctx.selected == null) { step(direction); return; }
    const here = ctx.placed.find((n) => n.op_id === ctx.selected);
    if (!here) { step(direction); return; }
    const sameRow = ctx.placed
      .filter((n) => n.row === here.row)
      .sort((a, b) => a.column - b.column);
    const at = sameRow.findIndex((n) => n.op_id === ctx.selected);
    const nextIdx = at + direction;
    if (nextIdx < 0 || nextIdx >= sameRow.length) return;   // stay put at the row's edge
    focusNode(sameRow[nextIdx].op_id);
  }

  function focusNode(opId) {
    ctx.selected = null;                       // select() toggles; force a fresh selection
    select(opId);
    ctx.groups.get(opId)?.focus?.();
  }

  /* Operator names as a person would say them. The IR tag is the engine's vocabulary; a
   * dashboard that only repeats it makes the reader learn the engine's words first. */
  const FRIENDLY = {
    hash_join: 'Join', aggregate: 'Group & aggregate', scan: 'Read source',
    filter: 'Filter rows', project: 'Select columns', sort: 'Sort', limit: 'Limit',
    distinct: 'Deduplicate', union: 'Union', window: 'Window', unnest: 'Unnest',
  };
  const friendlyKind = (kind) => FRIENDLY[kind] || kind;
  /* Exposed so the reference can be checked for coverage: every kind the graph can label
   * must be a kind the reference can explain. */
  const friendlyKinds = () => ({ ...FRIENDLY });

  /* The same mark the plan graph draws, as standalone markup, so an operator means the
   * same shape wherever it appears — graph, reference page, or inspector. One source. */
  function glyphMarkup(kind, size = 18) {
    return `<svg class="opglyph" width="${size}" height="${size}" viewBox="0 0 18 18" ` +
      `fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" ` +
      `stroke-linejoin="round" aria-hidden="true">` +
      `<path d="${GLYPHS[kind] || GLYPHS._default}"/></svg>`;
  }

  function fmtCount(n) {
    if (n == null || Number.isNaN(n)) return '—';
    const abs = Math.abs(n);
    if (abs >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
    if (abs >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
    if (abs >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
    return `${Math.round(n)}`;
  }

  /* Both delegate to the one formatter in `ui.js`. These wrappers exist only so the graph
   * code reads naturally; they must never grow their own rules. */
  const fmtMs = (v) => UI.ms(v);
  const fmtShare = (fraction) => UI.pct(fraction);

  /* A pipeline's plan as a tiny, static drawing — its visual fingerprint.
   *
   * Not the interactive explorer: no pan, zoom, selection, or measurements. Just the shape,
   * so a reader can tell one pipeline from another down a list the way you recognise a face
   * before you read the name. Laid out from the same column/row the server computed for the
   * full graph, so the thumbnail and the graph never disagree about where an operator sits.
   *
   * Nodes are neutral chips with the operator's glyph, not colour-coded — the page's rule is
   * that plan steps are stages of one query, labelled, not categories tinted by hue. Only the
   * critical path gets a faint accent, because that is state, not identity. */
  function thumbnail(shape, { width = 148, height = 92 } = {}) {
    const nodes = (shape && shape.nodes) || [];
    if (!nodes.length) {
      return `<div class="dag-thumb is-empty" role="img" aria-label="No plan recorded yet">` +
        `${glyphMarkup('_default', 22)}</div>`;
    }
    const cols = Math.max(1, shape.width || 1);
    const maxRow = Math.max(1, (shape.depth || 1) - 1);
    const pad = 14;
    const px = (c) => (cols === 1
      ? width / 2
      : pad + (c / (cols - 1)) * (width - 2 * pad));
    // row 0 is the sources (bottom); the highest row is the root (top).
    const py = (row) => pad + ((maxRow - row) / maxRow) * (height - 2 * pad);
    const byId = new Map(nodes.map((n) => [n.op_id, n]));
    const edges = (shape.edges || []).map((e) => {
      const a = byId.get(e.from), b = byId.get(e.to);
      if (!a || !b) return '';
      return `<line class="tn-edge" x1="${px(a.column).toFixed(1)}" y1="${py(a.row).toFixed(1)}" ` +
        `x2="${px(b.column).toFixed(1)}" y2="${py(b.row).toFixed(1)}"/>`;
    }).join('');
    const chips = nodes.map((n) => {
      const cx = px(n.column), cy = py(n.row);
      const glyph = GLYPHS[n.kind] || GLYPHS._default;
      return `<g transform="translate(${cx.toFixed(1)},${cy.toFixed(1)})" ` +
        `class="tn-node${n.on_critical_path ? ' is-crit' : ''}">` +
        `<circle r="11"/>` +
        `<path transform="translate(-7,-7) scale(0.78)" d="${glyph}" ` +
        `fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" ` +
        `stroke-linejoin="round"/>` +
        `<title>${UI.esc(friendlyKind(n.kind))}${n.detail ? ` — ${UI.esc(n.detail)}` : ''}</title>` +
        `</g>`;
    }).join('');
    const label = nodes.map((n) => friendlyKind(n.kind)).join(', ');
    return `<svg class="dag-thumb" viewBox="0 0 ${width} ${height}" ` +
      `role="img" aria-label="Plan: ${UI.esc(label)}">${edges}${chips}</svg>`;
  }

  /* A short, readable name for a pipeline, derived from its plan shape — the default when a
   * person has not named it themselves. The operators in execution order, collapsed and
   * capped, so a filter-join-group pipeline reads "Read → Filter → Join → Group" rather than
   * repeating the raw IR tag of whichever operator happened to be the terminal one. */
  const SHORT = {
    scan: 'Read', filter: 'Filter', hash_join: 'Join', sort_merge_join: 'Join',
    asof_join: 'Join', aggregate: 'Group', sort: 'Sort', limit: 'Limit',
    distinct: 'Distinct', project: 'Select', union: 'Union', window: 'Window', unnest: 'Unnest',
  };
  function pipelineName(shape, custom) {
    if (custom) return custom;
    const nodes = (shape && shape.nodes) || [];
    if (!nodes.length) return 'Unnamed pipeline';
    // Execution order: row 0 (sources) first, up toward the root; ties broken left to right.
    const order = [...nodes].sort((a, b) => a.row - b.row || a.column - b.column);
    const seq = [];
    for (const n of order) {
      const step = SHORT[n.kind] || friendlyKind(n.kind);
      if (seq[seq.length - 1] !== step) seq.push(step);  // collapse consecutive repeats
    }
    const parts = seq.length > 4 ? [seq[0], '…', seq[seq.length - 2], seq[seq.length - 1]] : seq;
    return parts.join(' → ');
  }

  return { render, fit, reset, select, step, stepAcross, setSearch, setFocus, setCritical,
           selectedNode, fmtCount, fmtMs, friendlyKind, friendlyKinds, glyphMarkup,
           reveal, selectOnly, fmtShare, thumbnail, pipelineName };
}

//: The run page's graph. Also the module's home for the shared formatters, which the rest
//: of the dashboard imports as `DAG.fmtMs` etc.
const DAG = createDagExplorer();

//: The pipeline page's graph — same code, its own selection and viewport.
const PIPE_DAG = createDagExplorer();
