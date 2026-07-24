/* The teaching layer — how the knowledge base in `reference.js` reaches the reader.
 *
 * Three surfaces, in increasing order of how much the reader asked for them:
 *
 *   1. Term popovers   — a dotted word anywhere on the page, explained where it stands, so
 *                        nobody has to leave the thing they were reading to find out what a
 *                        word meant. Replaces `title=`, which no keyboard can reach, no
 *                        stylesheet can touch, and which appears after an unhelpful delay.
 *   2. The guided tour — shown once, on a first visit, and re-runnable from the help menu.
 *   3. The Learn view  — the whole reference, browsable, for when someone wants to read
 *                        rather than to be interrupted.
 *
 * The popover is deliberately one shared element rather than one per term: hundreds of
 * hidden popovers is hundreds of nodes to lay out, and only one can ever be open.
 */

'use strict';

const LEARN = (() => {
  const esc = UI.esc;
  let pop = null;          // the single popover element
  let popTrigger = null;   // what opened it, so focus can be handed back
  let tourState = null;

  /* ---------- term markup ----------
   * Rendered as a real <button>: focusable, activatable by keyboard, and announced as
   * interactive. An <abbr title> is none of those things. */

  function term(word, label) {
    const key = String(word).toLowerCase();
    const entry = REFERENCE.lookup(key);
    if (!entry) return esc(label || word);
    return `<button class="term" type="button" data-term="${esc(key)}" ` +
           `aria-label="${esc(label || word)} — what does this mean?">${esc(label || word)}</button>`;
  }

  /* A standalone help affordance for places where underlining the word itself would be
   * noise — a panel heading, a table column. */
  function hint(word) {
    const entry = REFERENCE.lookup(String(word).toLowerCase());
    if (!entry) return '';
    return `<button class="helptip" type="button" data-term="${esc(String(word).toLowerCase())}" ` +
           `aria-label="What does ${esc(word)} mean?">` +
           `<span aria-hidden="true">?</span></button>`;
  }

  /* ---------- the popover ---------- */

  function ensurePop() {
    if (pop) return pop;
    pop = document.createElement('div');
    pop.className = 'popover';
    pop.setAttribute('role', 'dialog');
    pop.setAttribute('aria-label', 'Definition');
    pop.hidden = true;
    document.body.appendChild(pop);
    return pop;
  }

  function bodyFor(key) {
    const entry = REFERENCE.lookup(key);
    if (!entry) return '';
    const related = (entry.see || []).filter((w) => REFERENCE.lookup(w));
    return `<h4 class="pop-title">${esc(key)}</h4>` +
      `<p class="pop-what">${esc(entry.what)}</p>` +
      (entry.why ? `<p class="pop-why"><span class="pop-tag">Why it matters</span>${esc(entry.why)}</p>` : '') +
      (entry.fix ? `<p class="pop-fix"><span class="pop-tag">What to do</span>${esc(entry.fix)}</p>` : '') +
      (related.length
        ? `<p class="pop-see"><span class="pop-tag">See also</span>` +
          related.map((w) => `<button class="term" type="button" data-term="${esc(w)}">${esc(w)}</button>`).join(', ') +
          `</p>` : '');
  }

  /* Positioned below the trigger, flipped above when that would run off the bottom, and
   * clamped horizontally. Written against the viewport rather than an offset parent so it
   * behaves the same inside a scrolled panel as at the top of the page. */
  function place(trigger) {
    const r = trigger.getBoundingClientRect();
    const w = pop.offsetWidth, h = pop.offsetHeight;
    const margin = 8;
    let top = r.bottom + 6;
    let flipped = false;
    if (top + h > window.innerHeight - margin && r.top - h - 6 > margin) {
      top = r.top - h - 6;
      flipped = true;
    }
    let left = r.left + r.width / 2 - w / 2;
    left = Math.max(margin, Math.min(left, window.innerWidth - w - margin));
    pop.style.top = `${Math.round(top + window.scrollY)}px`;
    pop.style.left = `${Math.round(left + window.scrollX)}px`;
    pop.classList.toggle('is-flipped', flipped);
  }

  function openTerm(key, trigger) {
    const html = bodyFor(key);
    if (!html) return;
    ensurePop();
    pop.innerHTML = html +
      `<button class="pop-close" type="button" aria-label="Close definition">${UI.icon('close')}</button>`;
    pop.hidden = false;
    popTrigger = trigger || null;
    place(trigger);
    pop.classList.add('is-open');
    // Focus moves into the popover so a keyboard reader lands on the text they asked for,
    // and so Escape has somewhere to return from.
    const first = pop.querySelector('.pop-close');
    if (first) first.focus();
  }

  function closeTerm(restoreFocus) {
    if (!pop || pop.hidden) return;
    pop.hidden = true;
    pop.classList.remove('is-open');
    if (restoreFocus && popTrigger && document.contains(popTrigger)) popTrigger.focus();
    popTrigger = null;
  }

  const isOpen = () => !!pop && !pop.hidden;

  /* ---------- the guided tour ---------- */

  const TOUR = [
    { sel: '#view-pipelines', title: 'Everything starts here',
      body: 'Each card is one pipeline — one query shape. Running the same query again adds to its card rather than making a new one, which is what lets you see a trend.' },
    { sel: '.kpis', title: 'The engine at a glance',
      body: 'Totals across everything the engine has run since it started. Hover any tile for what it measures and what a healthy value looks like.' },
    { sel: '#health-banner', title: 'What needs attention',
      body: 'Automatic checks over recent runs. Each one links straight to the evidence behind it, so a verdict is never a dead end.' },
    { sel: '.pcard', title: 'Open a pipeline',
      body: 'A pipeline page shows every run of that query, its history, and the plan the engine chose. Click a card to go in.' },
    { sel: '[data-view="live"]', title: 'Work in flight',
      body: 'Everything else here is retrospective. Live is the page for a job that runs for minutes — partition progress, GPU load, and whether the pipeline is keeping the workers fed. A dot appears on this tab whenever something is running.' },
    { sel: '[data-view="logs"]', title: 'The log stream',
      body: 'Structured logs from the engine, filterable by level and source. Useful when a query failed rather than merely ran slowly.' },
    { sel: '#help', title: 'Help whenever you want it',
      body: 'Press ? for shortcuts, ⌘K to jump anywhere, and open Learn for the full reference. Any dotted word on the page can be clicked for a definition.' },
  ];

  function tourAvailable() {
    return TOUR.filter((step) => document.querySelector(step.sel));
  }

  function startTour() {
    const steps = tourAvailable();
    if (!steps.length) return;
    tourState = { steps, at: 0 };
    renderTour();
  }

  function renderTour() {
    if (!tourState) return;
    const step = tourState.steps[tourState.at];
    const target = document.querySelector(step.sel);
    if (!target) { endTour(); return; }

    let layer = document.getElementById('tour-layer');
    if (!layer) {
      layer = document.createElement('div');
      layer.id = 'tour-layer';
      layer.className = 'tour-layer';
      layer.setAttribute('role', 'dialog');
      layer.setAttribute('aria-modal', 'true');
      layer.setAttribute('aria-label', 'Guided tour');
      document.body.appendChild(layer);
    }

    target.scrollIntoView({ block: 'center', behavior: 'smooth' });
    const r = target.getBoundingClientRect();
    const pad = 6;
    const boxTop = r.top + window.scrollY - pad;
    const boxLeft = r.left + window.scrollX - pad;
    const last = tourState.at === tourState.steps.length - 1;

    layer.innerHTML =
      `<div class="tour-spot" style="top:${boxTop}px;left:${boxLeft}px;` +
      `width:${r.width + pad * 2}px;height:${r.height + pad * 2}px"></div>` +
      `<div class="tour-card" style="top:${boxTop + r.height + pad * 2 + 10}px;left:${boxLeft}px">` +
      `<div class="tour-step">Step ${tourState.at + 1} of ${tourState.steps.length}</div>` +
      `<h3>${esc(step.title)}</h3><p>${esc(step.body)}</p>` +
      `<div class="tour-actions">` +
      `<button class="linkish" type="button" data-tour="end">Skip the tour</button>` +
      `<span class="spacer"></span>` +
      (tourState.at > 0 ? `<button class="ghost" type="button" data-tour="prev">Back</button>` : '') +
      `<button class="ghost is-primary" type="button" data-tour="next">${last ? 'Done' : 'Next'}</button>` +
      `</div></div>`;

    const next = layer.querySelector('[data-tour="next"]');
    if (next) next.focus();
  }

  function stepTour(dir) {
    if (!tourState) return;
    const at = tourState.at + dir;
    if (at < 0 || at >= tourState.steps.length) { endTour(); return; }
    tourState.at = at;
    renderTour();
  }

  function endTour() {
    tourState = null;
    const layer = document.getElementById('tour-layer');
    if (layer) layer.remove();
    UI.setPref('tourSeen', true);
  }

  const tourRunning = () => !!tourState;

  /* Offered once, and only when there is something on screen worth pointing at — a tour of
   * an empty dashboard teaches nothing. */
  let tourOffered = false;
  function maybeOfferTour(hasData) {
    // Offered at most once per session, and never twice on screen: `renderPipelineList`
    // calls this on every repaint, and without the guard a stack of identical sticky toasts
    // grew over the content. A person who dismisses it, or has seen it before, is not asked
    // again.
    if (!hasData || tourOffered || UI.getPref('tourSeen')) return;
    tourOffered = true;
    UI.toast('New here? Take the 6-step tour.', {
      action: { label: 'Start tour', run: startTour },
      dismiss: { label: 'No thanks', run: () => UI.setPref('tourSeen', true) },
      sticky: true,
    });
  }

  /* ---------- the Learn view ---------- */

  function renderLearn(host, filter) {
    const q = String(filter || '').trim().toLowerCase();
    const match = (text) => !q || String(text).toLowerCase().includes(q);

    const recipes = REFERENCE.RECIPES.filter((r) => match(r.task) || r.steps.some(match));
    const metrics = Object.entries(REFERENCE.METRICS)
      .filter(([, m]) => match(m.label) || match(m.what) || match(m.good) || match(m.bad));
    // Search the whole entry, not only its name: someone typing "spill" is looking for the
    // steps that spill, and those words live in `slow` and `fix`.
    const operators = Object.entries(REFERENCE.OPERATORS)
      .filter(([kind, o]) => match(kind) || match(o.label) || match(o.what) ||
                             match(o.slow) || match(o.fix) || o.terms.some(match));
    const terms = REFERENCE.termKeys
      .map((k) => [k, REFERENCE.TERMS[k]])
      .filter(([k, t]) => match(k) || match(t.what) || match(t.why || ''));
    const comparisons = REFERENCE.COMPARISONS.filter((c) => (
      match(c.tool) || match(c.familiar) || c.rows.some(([a, b]) => match(a) || match(b))));

    if (!recipes.length && !operators.length && !metrics.length && !terms.length &&
        !comparisons.length) {
      host.innerHTML = UI.emptyState({
        glyph: 'search', title: `Nothing matches “${esc(filter)}”`,
        body: 'Try a shorter word, or clear the box to browse everything.',
      });
      return;
    }

    // At 50+ entries the page is long enough that landing at the top and scrolling is the
    // wrong default. A contents strip and per-section counts let a reader jump.
    const toc = [
      recipes.length ? ['how', `How do I\u2026 (${recipes.length})`] : null,
      comparisons.length ? ['coming', `Coming from another engine (${comparisons.length})`] : null,
      operators.length ? ['steps', `Plan steps (${operators.length})`] : null,
      metrics.length ? ['metrics', `Metrics (${metrics.length})`] : null,
      terms.length ? ['glossary', `Glossary (${terms.length})`] : null,
    ].filter(Boolean);

    host.innerHTML =
      (toc.length > 1
        ? `<nav class="learn-toc" aria-label="Sections on this page">` +
          toc.map(([id, label]) => `<a href="#learn-${id}">${esc(label)}</a>`).join('') +
          `</nav>`
        : '') +
      (recipes.length ? `<section class="learn-block" id="learn-how"><h3>How do I\u2026</h3>` +
        `<div class="recipes">` + recipes.map((r) => (
          `<article class="recipe"><h4>${esc(r.task)}</h4><ol>` +
          r.steps.map((s) => `<li>${esc(s)}</li>`).join('') + `</ol></article>`)).join('') +
        `</div></section>` : '') +

      // A map, not a scoreboard. Nobody arrives at a new engine's dashboard without having
      // read another one first, and naming the familiar panel is the fastest way to make
      // this one legible. Performance claims belong in the competitive scorecard, not here.
      (comparisons.length ? `<section class="learn-block" id="learn-coming">` +
        `<h3>Coming from another engine</h3>` +
        `<p class="lede">Where the panel you already know lives here. This is a map for ` +
        `finding your way around — it says nothing about which engine is faster.</p>` +
        comparisons.map((c) => (
          `<article class="compare-block"><h4>${esc(c.tool)}</h4>` +
          `<p class="compare-familiar">${esc(c.familiar)}</p>` +
          `<table class="dense compare-table"><thead><tr>` +
          `<th scope="col">There</th><th scope="col">Here</th></tr></thead><tbody>` +
          c.rows.map(([there, here]) => (
            `<tr><td class="compare-there">${esc(there)}</td><td>${esc(here)}</td></tr>`)).join('') +
          `</tbody></table></article>`)).join('') + `</section>` : '') +

      (operators.length ? `<section class="learn-block" id="learn-steps"><h3>Plan steps</h3>` +
        `<p class="lede">Every step the engine can put in a plan, what makes each one slow, and what to do about it.</p>` +
        `<div class="oprefs">` + operators.map(([kind, o]) => (
          `<article class="opref" id="opref-${esc(kind)}">` +
          `<div class="opref-head">${DAG.glyphMarkup(kind)}<h4>${esc(o.label)}</h4>` +
          `<code class="opref-kind">${esc(kind)}</code></div>` +
          `<p class="opref-what">${esc(o.what)}</p>` +
          `<p class="opref-slow"><span class="pop-tag">Slow when</span>${esc(o.slow)}</p>` +
          `<p class="opref-fix"><span class="pop-tag">What to do</span>${esc(o.fix)}</p>` +
          (o.terms.length ? `<p class="opref-terms">` +
            o.terms.map((t) => term(t)).join(' · ') + `</p>` : '') +
          `</article>`)).join('') + `</div></section>` : '') +

      (metrics.length ? `<section class="learn-block" id="learn-metrics"><h3>Metrics</h3>` +
        `<p class="lede">Every number the dashboard shows, what it means, and what a healthy ` +
        `value looks like \u2014 so &ldquo;is 340ms bad?&rdquo; has an answer.</p>` +
        `<dl class="glossary">` + metrics.map(([, m]) => (
          `<div class="gloss-row"><dt>${esc(m.label)}</dt><dd><p>${esc(m.what)}</p>` +
          `<p class="metric-good"><span class="pop-tag">Healthy</span>${esc(m.good)}</p>` +
          `<p class="metric-bad"><span class="pop-tag">Worth a look</span>${esc(m.bad)}</p>` +
          (m.term ? `<p class="gloss-see">See also: ${term(m.term)}</p>` : '') +
          `</dd></div>`)).join('') + `</dl></section>` : '') +

      (terms.length ? `<section class="learn-block" id="learn-glossary"><h3>Glossary</h3>` +
        `<dl class="glossary">` + terms.map(([k, t]) => (
          `<div class="gloss-row" id="gloss-${esc(k.replace(/\s+/g, '-'))}">` +
          `<dt>${esc(k)}</dt><dd><p>${esc(t.what)}</p>` +
          (t.why ? `<p class="dim">${esc(t.why)}</p>` : '') +
          (t.fix ? `<p class="gloss-fix">${esc(t.fix)}</p>` : '') +
          ((t.see || []).length ? `<p class="gloss-see">See also: ` +
            t.see.map((w) => term(w)).join(', ') + `</p>` : '') +
          `</dd></div>`)).join('') + `</dl></section>` : '');
  }

  /* ---------- operator explanation, for the plan inspector ---------- */

  function explainOperator(kind) {
    const o = REFERENCE.operator(kind);
    if (!o) return '';
    return `<div class="insp-explain">` +
      `<p class="insp-what">${esc(o.what)}</p>` +
      `<details class="insp-more"><summary>Why this step can be slow</summary>` +
      `<p>${esc(o.slow)}</p><p class="insp-fix"><b>What to do:</b> ${esc(o.fix)}</p></details>` +
      `</div>`;
  }

  /* One delegated listener for every term and helptip on the page, present or future —
   * panels re-render constantly, and per-element binding would leak or silently stop
   * working after the first redraw. */
  function install() {
    document.addEventListener('click', (e) => {
      const trigger = e.target.closest('[data-term]');
      if (trigger) {
        e.preventDefault();
        e.stopPropagation();
        const key = trigger.getAttribute('data-term');
        // Clicking the open term again closes it, which is what a toggle should do.
        if (isOpen() && popTrigger === trigger) closeTerm(true);
        else openTerm(key, trigger);
        return;
      }
      if (e.target.closest('.pop-close')) { closeTerm(true); return; }
      if (isOpen() && !e.target.closest('.popover')) closeTerm(false);

      const tourBtn = e.target.closest('[data-tour]');
      if (tourBtn) {
        const action = tourBtn.getAttribute('data-tour');
        if (action === 'end') endTour();
        else stepTour(action === 'prev' ? -1 : 1);
      }
    });

    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        if (isOpen()) { closeTerm(true); e.stopPropagation(); }
        else if (tourRunning()) { endTour(); e.stopPropagation(); }
        return;
      }
      if (!tourRunning()) return;
      if (e.key === 'ArrowRight' || e.key === 'Enter') { e.preventDefault(); stepTour(1); }
      if (e.key === 'ArrowLeft') { e.preventDefault(); stepTour(-1); }
    });

    // A popover pinned to a moved trigger points at nothing; close rather than chase.
    window.addEventListener('resize', () => { closeTerm(false); if (tourRunning()) renderTour(); });
    window.addEventListener('scroll', () => { if (isOpen()) closeTerm(false); }, { passive: true });
  }

  /* Turn known glossary terms inside a plain-text string into clickable definitions.
   *
   * The string is HTML-escaped first, then terms are matched on word boundaries. Longest
   * terms match first so "hash join" is not pre-empted by "join", and each distinct term is
   * linked only once — a sentence with "spill … spill … spill" gets one dotted word, not a
   * field of them. */
  function autolink(text) {
    let html = esc(text);
    const terms = REFERENCE.termKeys.slice().sort((a, b) => b.length - a.length);
    const linked = new Set();
    for (const t of terms) {
      if (linked.has(t)) continue;
      // Escape regex metachars in the term, and require word boundaries so "plan" does not
      // match inside "planner". `\b` works because terms are lower-case words.
      const safe = t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const re = new RegExp(`\\b(${safe})\\b`, 'i');
      if (re.test(html) && !new RegExp(`data-term="[^"]*"[^>]*>[^<]*${safe}`, 'i').test(html)) {
        html = html.replace(re, (m) => {
          linked.add(t);
          return `<button class="term" type="button" data-term="${esc(t)}">${m}</button>`;
        });
      }
    }
    return html;
  }

  return { term, hint, autolink, openTerm, closeTerm, isOpen, install,
           startTour, endTour, tourRunning, maybeOfferTour,
           renderLearn, explainOperator };
})();
