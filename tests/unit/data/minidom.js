// A small DOM good enough to execute rendering: element trees, attributes, classes,
// querySelector by tag/class/id, and getBoundingClientRect. Enough to assert what a
// renderer actually produced, rather than only that it did not throw.
function mkEl(tag, ns) {
  const el = {
    tagName: String(tag).toUpperCase(), ns, children: [], parent: null,
    attrs: {}, style: {}, dataset: {}, _text: '', _html: '',
    classList: {
      _s: new Set(),
      add(...c) { c.forEach((x) => this._s.add(x)); },
      remove(...c) { c.forEach((x) => this._s.delete(x)); },
      toggle(c, on) { const has = this._s.has(c); const want = on === undefined ? !has : !!on;
                      if (want) this._s.add(c); else this._s.delete(c); return want; },
      contains(c) { return this._s.has(c); },
      get value() { return [...this._s].join(' '); },
    },
    get id() { return this.attrs.id; },
    set id(v) { this.setAttribute('id', v); },
    get className() { return this.attrs.class || ''; },
    set className(v) { this.setAttribute('class', v); },
    setAttribute(k, v) {
      this.attrs[k] = String(v);
      // A browser keeps class and classList in sync; the harness must too, or every
      // querySelector('.x') silently finds nothing and a passing test proves nothing.
      if (k === 'class') { this.classList._s = new Set(String(v).split(/\s+/).filter(Boolean)); }
    },
    getAttribute(k) { return this.attrs[k] ?? null; },
    removeAttribute(k) { delete this.attrs[k]; },
    appendChild(c) { c.parent = this; this.children.push(c); return c; },
    append(...cs) { cs.forEach((c) => this.appendChild(c)); },
    addEventListener() {}, removeEventListener() {},
    setPointerCapture() {}, focus() {}, click() {}, scrollIntoView() {},
    getBoundingClientRect() { return { width: 900, height: 600, left: 0, top: 0 }; },
    matches(sel) { return matchSel(this, sel); },
    /* Walk up the parent chain for the nearest element matching `sel`. Returned `null` for
     * this element's whole life before, so any code path guarded by `closest('.thing')` was
     * silently dead in a test — the same vacuous-pass trap as the other harness stubs. */
    closest(sel) {
      let el = this;
      while (el) { if (el.matches && el.matches(sel)) return el; el = el.parent; }
      return null;
    },
    remove() { if (this.parent) { const k = this.parent.children.indexOf(this); if (k >= 0) this.parent.children.splice(k, 1); this.parent = null; } },
    get textContent() {
      // Text nodes contribute their own value; elements contribute their stripped html.
      if (this._html != null) return String(this._html).replace(/<[^>]+>/g, '');
      return this._text || '';
    },
    set textContent(v) { this._text = String(v); this._html = null; if (v === '') this.children = []; },
    get innerHTML() { return this._html || ''; },
    set innerHTML(v) { this._html = String(v); this._text = null; },
    querySelectorAll(sel) { return descendants(this).filter((e) => matchSel(e, sel)); },
    querySelector(sel) { return this.querySelectorAll(sel)[0] || null; },
    get clientWidth() { return 900; }, get clientHeight() { return 600; },
  };
  return el;
}
function descendants(root) {
  const out = [];
  (function walk(n) { for (const c of n.children) { out.push(c); walk(c); } })(root);
  return out;
}
function matchSel(el, sel) {
  return sel.split(',').map((s) => s.trim()).some((s) => {
    if (s.startsWith('.')) return el.classList.contains(s.slice(1));
    if (s.startsWith('#')) return el.attrs.id === s.slice(1);
    return el.tagName === s.toUpperCase();
  });
}
const registry = {};
var document = {
  documentElement: mkEl('html'),
  body: mkEl('body'),
  createElement: (t) => mkEl(t),
  createElementNS: (ns, t) => mkEl(t, ns),
  getElementById: (id) => (registry[id] ||= Object.assign(mkEl('div'), { attrs: { id } })),
  /* Search the body tree AND the id registry, deduped.
   *
   * This returned a hardcoded `[]` for a long time, which made every assertion that went
   * through `document.querySelectorAll` pass without testing anything — the same failure
   * mode as the classList gap above. The registry has to be included because
   * `getElementById` hands out detached elements that production code then mutates; a
   * body-only walk would miss exactly the elements a test just set up. */
  querySelectorAll(sel) {
    const seen = new Set();
    const out = [];
    const consider = (e) => {
      if (!e || seen.has(e)) return;
      seen.add(e);
      if (matchSel(e, sel)) out.push(e);
    };
    descendants(document.body).forEach(consider);
    Object.keys(registry).forEach((id) => {
      consider(registry[id]);
      descendants(registry[id]).forEach(consider);
    });
    return out;
  },
  querySelector(sel) { return this.querySelectorAll(sel)[0] || null; },
  addEventListener() {},
};
var window = { matchMedia: () => ({ matches: false }), innerWidth: 1400, innerHeight: 900,
               scrollY: 0, scrollTo() {}, addEventListener() {} };
var location = { hash: '', href: 'http://x/' };
var history = { replaceState() {} };
var localStorage = { getItem: () => null, setItem() {} };
var navigator = { clipboard: { writeText: () => Promise.resolve() } };
var performance = { now: () => 0 };
/* rAF runs the callback once, at the final timestamp. Calling it synchronously with t=0
 * makes an animation loop (rollTo, the number-roll) recurse forever, because each frame
 * schedules the next before it has advanced. Passing a large timestamp lets the callback
 * see the animation as already complete and stop, which is the state a test wants to
 * assert against anyway. */
var requestAnimationFrame = (fn) => { fn(1e6); return 0; };
var cancelAnimationFrame = () => {};
var setTimeout = () => 0;
var clearTimeout = () => {};
var URL = { createObjectURL: () => '', revokeObjectURL() {} };
var Blob = function () {};
