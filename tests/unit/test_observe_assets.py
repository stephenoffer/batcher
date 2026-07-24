"""Static checks over the dashboard's own front end.

The UI ships as plain files served off disk — no bundler, no type checker, no test runner
for the browser. That is the right trade for something that has to live inside a Python
wheel and start with `bt.start_ui()`, but it removes every guard that would normally catch
a renamed function or a control wired to an element that no longer exists. Those bugs are
invisible until a person clicks the thing.

So the guards live here, as checks a Python test suite can make over the source text. Each
one below corresponds to a class of failure that has actually shipped in this dashboard: a
palette action calling a function that was renamed (`setView`), a handler calling a helper
that only exists inside another file's closure (`logLine`), and controls bound by id to
markup that changed.

These are deliberately syntactic. They cannot prove the UI works; they prove it is not
obviously broken, which is the part that regresses silently.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ASSETS = Path(__file__).resolve().parents[2] / "python" / "batcher" / "observe" / "assets"

#: The load order in `index.html`. Order matters: a file may only use globals defined by
#: itself or by a file loaded before it.
SCRIPTS = [
    "ui.js",
    "reference.js",
    "charts.js",
    "dag.js",
    "learn.js",
    "plan.js",
    "live.js",
    "views.js",
    "app.js",
]

#: Identifiers a browser provides. Anything called that is neither defined in the bundle nor
#: on this list is a typo or a rename that was not carried through.
BROWSER_GLOBALS = frozenset(
    [
        "window",
        "document",
        "console",
        "location",
        "history",
        "navigator",
        "performance",
        "localStorage",
        "setTimeout",
        "clearTimeout",
        "setInterval",
        "clearInterval",
        "requestAnimationFrame",
        "cancelAnimationFrame",
        "fetch",
        "Blob",
        "URL",
        "CustomEvent",
        "Event",
        "Error",
        "TypeError",
        "RangeError",
        "Math",
        "JSON",
        "Object",
        "Array",
        "String",
        "Number",
        "Boolean",
        "Date",
        "Set",
        "Map",
        "WeakMap",
        "WeakSet",
        "Promise",
        "RegExp",
        "Symbol",
        "Intl",
        "Proxy",
        "Reflect",
        "BigInt",
        "parseInt",
        "parseFloat",
        "isNaN",
        "isFinite",
        "encodeURIComponent",
        "decodeURIComponent",
        "encodeURI",
        "decodeURI",
        "structuredClone",
        "queueMicrotask",
        "matchMedia",
        "getComputedStyle",
        "alert",
        "confirm",
        "prompt",
        "CSS",
        "Node",
        "Element",
        "SVGElement",
        "IntersectionObserver",
        "ResizeObserver",
        "MutationObserver",
        "AbortController",
        "TextEncoder",
        "TextDecoder",
        "Uint8Array",
        "Float64Array",
        "Int32Array",
        "DOMParser",
    ]
)

#: Keywords that look like calls but are not.
KEYWORDS = frozenset(
    [
        "if",
        "else",
        "for",
        "while",
        "switch",
        "catch",
        "return",
        "function",
        "typeof",
        "instanceof",
        "new",
        "delete",
        "void",
        "do",
        "try",
        "finally",
        "case",
        "default",
        "break",
        "continue",
        "const",
        "let",
        "var",
        "class",
        "extends",
        "super",
        "this",
        "yield",
        "await",
        "async",
        "of",
        "in",
    ]
)


def _read(name: str) -> str:
    return (ASSETS / name).read_text(encoding="utf-8")


#: Keywords after which a "/" opens a regex rather than dividing. Without these,
#: `return /[",\n]/.test(x)` reads as division and the quote inside the character class
#: opens a string that swallows the rest of the function.
_REGEX_PRECEDERS = frozenset(
    [
        "return",
        "typeof",
        "instanceof",
        "in",
        "of",
        "new",
        "delete",
        "void",
        "case",
        "do",
        "else",
        "yield",
        "await",
        "throw",
    ]
)


def _starts_regex(source: str, at: int, prev: str) -> bool:
    """Whether the "/" at `at` opens a regex literal rather than being a division sign."""
    if prev in ")]}":
        return False
    if not (prev.isalnum() or prev in "_$"):
        return True
    word = re.search(r"[A-Za-z_$][\w$]*\s*$", source[:at])
    return bool(word) and word.group().strip() in _REGEX_PRECEDERS


def strip_literals(source: str) -> str:
    """Blank comments, string bodies, and regex literals; keep code, including template holes.

    A regex over raw JavaScript cannot tell code from prose, and both checks below were
    useless without this: the call scanner reported every English word followed by "(" in a
    comment, and the bracket counter tripped over `/[",\n]/`.

    Template literals are blanked, but the code inside their `${…}` holes is **kept**. Most
    of this codebase's real calls live in exactly that position — `${esc(label)}` — so
    blanking holes would have hidden them from the undefined-call check, which is the whole
    reason this function exists. Character positions are preserved so a failure can still be
    traced to a line.
    """
    out: list[str] = []
    # Mode stack. Each entry is ["tpl"] for template text or ["hole", depth] for the code
    # inside a `${…}`, whose brace depth tells us where the hole ends.
    stack: list[list] = []
    i, n = 0, len(source)
    prev = ""  # last significant character of *code*, for the regex-vs-division call

    def blank(text: str) -> str:
        return "".join(c if c == "\n" else " " for c in text)

    while i < n:
        ch = source[i]
        mode = stack[-1][0] if stack else "code"

        if mode == "tpl":
            if ch == "\\":
                out.append("  ")
                i += 2
            elif ch == "`":
                out.append(" ")
                stack.pop()
                prev = "x"
                i += 1
            elif source[i : i + 2] == "${":
                out.append("  ")
                stack.append(["hole", 0])
                prev = ""
                i += 2
            else:
                out.append("\n" if ch == "\n" else " ")
                i += 1
            continue

        two = source[i : i + 2]
        if two == "//":
            end = source.find("\n", i)
            end = n if end < 0 else end
            out.append(" " * (end - i))
            i = end
        elif two == "/*":
            end = source.find("*/", i + 2)
            end = n if end < 0 else end + 2
            out.append(blank(source[i:end]))
            i = end
        elif ch in "\"'":
            quote, j = ch, i + 1
            while j < n:
                if source[j] == "\\":
                    j += 2
                    continue
                if source[j] == quote:
                    j += 1
                    break
                j += 1
            out.append(blank(source[i:j]))
            i = j
            prev = "x"  # a string is a value, so a following "/" is division
        elif ch == "`":
            out.append(" ")
            stack.append(["tpl"])
            i += 1
        elif ch == "}" and mode == "hole" and stack[-1][1] == 0:
            # The hole closes and the template text resumes.
            out.append(" ")
            stack.pop()
            i += 1
        elif ch == "/" and _starts_regex(source, i, prev):
            j, in_class = i + 1, False
            while j < n:
                c = source[j]
                if c == "\\":
                    j += 2
                    continue
                if c == "[":
                    in_class = True
                elif c == "]":
                    in_class = False
                elif c == "/" and not in_class:
                    j += 1
                    break
                elif c == "\n":
                    break
                j += 1
            while j < n and source[j].isalpha():  # flags
                j += 1
            out.append(" " * (j - i))
            i = j
            prev = "x"
        else:
            if mode == "hole" and ch in "{}":
                stack[-1][1] += 1 if ch == "{" else -1
            out.append(ch)
            if not ch.isspace():
                prev = ch
            i += 1
    return "".join(out)


@pytest.fixture(scope="module")
def html() -> str:
    return _read("index.html")


@pytest.fixture(scope="module")
def bundle() -> dict[str, str]:
    return {name: _read(name) for name in SCRIPTS}


# --- the shell ---------------------------------------------------------------


def test_every_script_the_page_loads_exists(html):
    referenced = re.findall(r'<script src="/([^"]+)"', html)
    assert referenced == SCRIPTS, "index.html's load order drifted from the one tested here"
    for name in referenced:
        assert (ASSETS / name).is_file(), f"index.html loads {name}, which is not in assets/"


def test_no_asset_is_orphaned(html):
    served = {p.name for p in ASSETS.glob("*.js")}
    loaded = set(re.findall(r'<script src="/([^"]+)"', html))
    assert served == loaded, f"unreferenced asset(s): {sorted(served - loaded)}"


def test_element_ids_are_unique(html):
    ids = re.findall(r'\bid="([^"]+)"', html)
    duplicates = {i for i in ids if ids.count(i) > 1}
    # A duplicate id makes `getElementById` return whichever came first, so half the
    # controls bound to it silently stop working.
    assert not duplicates, f"duplicate element ids: {sorted(duplicates)}"


def test_every_id_the_scripts_reach_for_exists(html, bundle):
    ids = set(re.findall(r'\bid="([^"]+)"', html))
    # Ids created at runtime by a renderer rather than present in the shell.
    runtime = {"tour-layer"}
    pattern = re.compile(r"""(?:\$|getElementById|\bon)\(\s*['"]([a-z][a-z0-9-]*)['"]""")
    missing: dict[str, set[str]] = {}
    for name, source in bundle.items():
        found = {i for i in pattern.findall(source) if i not in ids and i not in runtime}
        if found:
            missing[name] = found
    assert not missing, f"scripts reference ids that are not in index.html: {missing}"


def test_every_nav_destination_has_a_section(html):
    for view in re.findall(r'data-view="([^"]+)"', html):
        assert f'id="view-{view}"' in html, f"nav offers {view!r} with no matching section"


def test_every_rendering_switch_has_a_pane(html):
    for attr in ("steps", "query"):
        for value in re.findall(rf'data-{attr}="([^"]+)"', html):
            assert f'id="tab-{value}"' in html, f"switch {attr}={value!r} has no pane to show"


# --- the scripts -------------------------------------------------------------


def _defined_globals(source: str) -> set[str]:
    """Top-level names a file introduces. Deliberately generous — this hunts typos."""
    names: set[str] = set()
    names |= set(re.findall(r"^(?:async\s+)?function\s+([A-Za-z_$][\w$]*)", source, re.M))
    names |= set(re.findall(r"^(?:const|let|var)\s+([A-Za-z_$][\w$]*)", source, re.M))
    # Locals too: this check is about *calls to things that exist nowhere*, so a helper
    # defined inside a closure still counts as defined.
    names |= set(re.findall(r"\b(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(", source))
    names |= set(re.findall(r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=", source))
    # Second and later declarators of one statement: `const f = fmt(a), fx = fmt(b);`
    names |= set(re.findall(r",\s*([A-Za-z_$][\w$]*)\s*=(?!=)", source))
    arrow = r"\b([A-Za-z_$][\w$]*)\s*(?:=|:)\s*(?:async\s*)?\([^)]*\)\s*=>"
    names |= set(re.findall(arrow, source))
    names |= set(re.findall(r"\b([A-Za-z_$][\w$]*)\s*(?:=|:)\s*(?:async\s+)?function", source))
    # Destructured imports from a module object: `const { esc, ms } = UI;`
    for block in re.findall(r"(?:const|let|var)\s*\{([^}]*)\}\s*=", source):
        names |= _idents(block)
    # Parameters. A callback passed in as `onOpen` is defined for the body that calls it,
    # and without these the check reports every one of them as undefined.
    for params in re.findall(r"function\s*[A-Za-z_$][\w$]*\s*\(([^)]*)\)", source):
        names |= _idents(params)
    for params in re.findall(r"function\s*\(([^)]*)\)", source):
        names |= _idents(params)
    for params in re.findall(r"\(([^()]*)\)\s*=>", source):
        names |= _idents(params)
    names |= set(re.findall(r"(?<![.\w$])([A-Za-z_$][\w$]*)\s*=>", source))
    return names


def _idents(block: str) -> set[str]:
    """Every identifier in a parameter or destructuring list, ignoring defaults and keys."""
    out: set[str] = set()
    for part in re.split(r"[,{}\[\]]", block):
        alias = part.split("=")[0].split(":")[-1].strip().lstrip(".")
        if re.fullmatch(r"[A-Za-z_$][\w$]*", alias):
            out.add(alias)
    return out


def test_no_script_calls_a_function_that_exists_nowhere(bundle):
    """The check that would have caught `setView` and `logLine`.

    Both shipped: five palette actions called a function that had been renamed to
    `switchView`, and the run-log renderer called `logLine`, which lives inside `views.js`'s
    closure and is not a global. Each threw at the moment a person clicked, and nothing in
    the build or the test suite could see it.
    """
    code = {name: strip_literals(source) for name, source in bundle.items()}
    defined: set[str] = set()
    for source in code.values():
        defined |= _defined_globals(source)
    problems: dict[str, set[str]] = {}
    for name, source in code.items():
        # A bare `foo(` — not `.foo(`, not `?.foo(`, not a keyword.
        called = set(re.findall(r"(?<![.\w$])([A-Za-z_$][\w$]*)\s*\(", source))
        unknown = called - defined - BROWSER_GLOBALS - KEYWORDS
        if unknown:
            problems[name] = unknown
    assert not problems, f"calls to undefined functions: {problems}"


@pytest.mark.parametrize("name", SCRIPTS)
def test_brackets_balance(name, bundle):
    """A crude parse. A JS file with unbalanced brackets fails to load *entirely*, taking
    every later script with it, so the page goes blank rather than degrading."""
    stripped = strip_literals(bundle[name])
    for open_ch, close_ch in [("{", "}"), ("(", ")"), ("[", "]")]:
        assert stripped.count(open_ch) == stripped.count(close_ch), (
            f"{name}: {open_ch}{close_ch} do not balance"
        )


@pytest.mark.parametrize("name", SCRIPTS)
def test_every_script_is_strict_mode(name, bundle):
    """Sloppy mode turns a typo'd assignment into a new global instead of an error."""
    assert "'use strict';" in bundle[name]


def test_no_script_imports_the_compiled_engine_directly(bundle):
    """The browser can only see the JSON API. A fetch to anything else is a mistake."""
    for name, source in bundle.items():
        for url in re.findall(r"""fetch\(\s*[`'"]([^`'"$]*)""", source):
            assert url.startswith("/api") or url.startswith("/metrics"), (
                f"{name} fetches {url!r}, which is not part of the read-only API"
            )


# --- encoding discipline -----------------------------------------------------


def test_no_raw_colour_outside_the_token_block():
    """Every colour is a token, or a theme swap only restyles half the page.

    The token block and the two theme overrides define the palette; anything below them
    referencing a literal hex is a colour that will not follow the theme.
    """
    css = _read("app.css")
    body = css.split("/* ═══════════════ base ═══════════════ */", 1)[-1]
    # Print is the one exempt context: paper has no theme to follow, and the print rules
    # deliberately force black on white rather than rendering a dark surface as ink.
    body = re.sub(r"@media print\s*\{.*?\n\}", "", body, flags=re.S)
    literals = [m for m in re.findall(r"#[0-9a-fA-F]{3,8}\b", body) if m not in {"#fff"}]
    assert not literals, f"raw colours below the token block: {sorted(set(literals))}"


def test_charts_never_hard_code_a_colour(bundle):
    """`charts.js` emits classes, never fills. A caller must not be able to smuggle a
    colour past the page's encoding rules."""
    assert not re.findall(r"#[0-9a-fA-F]{3,6}\b", bundle["charts.js"])
    assert 'fill="rgb' not in bundle["charts.js"]


# --- SVG animation safety ----------------------------------------------------


def _keyframes(css: str) -> dict[str, str]:
    """Every ``@keyframes`` block, name -> body text."""
    out: dict[str, str] = {}
    for m in re.finditer(r"@keyframes\s+([\w-]+)\s*\{", css):
        i = m.end()
        depth = 1
        while i < len(css) and depth:
            depth += css[i] == "{"
            depth -= css[i] == "}"
            i += 1
        out[m.group(1)] = css[m.end() : i - 1]
    return out


def test_svg_positioned_elements_never_animate_transform():
    """A CSS ``transform`` overrides an SVG ``transform`` *attribute* — so animating the
    transform of an element positioned by that attribute collapses every one of them onto
    the origin. This is exactly how the plan graph broke: each node is placed with
    ``transform="translate(x y)"`` and a ``rise`` (translateY) entrance wiped that out,
    stacking all nodes at one point while the edges still fanned out to empty space.

    Every element the DAG positions with the transform attribute must therefore take a
    transform-free (opacity-only) entrance. This test reads the animation each such class
    uses and fails if its keyframes touch ``transform``.
    """
    css = _read("app.css")
    frames = _keyframes(css)
    # The classes whose position comes from an SVG `transform` attribute set in dag.js.
    svg_positioned = ["dag-node"]
    problems: dict[str, str] = {}
    for cls in svg_positioned:
        m = re.search(rf"\.{cls}\s*\{{[^}}]*animation:\s*([\w-]+)", css)
        assert m, f".{cls} has no animation rule to check"
        keyframe = m.group(1)
        body = frames.get(keyframe, "")
        if "transform" in body:
            problems[cls] = f"animates `{keyframe}`, which sets transform"
    assert not problems, (
        f"SVG-positioned elements animating transform (collapses them to the origin): {problems}"
    )


def test_the_hidden_attribute_can_always_hide_an_element():
    """`[hidden]` must win over any component's `display`.

    Components set `display: flex/grid/block`, which has higher specificity than the UA
    `[hidden] { display: none }` — so toggling the `hidden` attribute silently stops working
    and the element is pinned open. That is what kept the "Lost contact" banner and the
    breadcrumb visible on every page. A single global guard fixes the whole class; this test
    makes sure it stays.
    """
    css = _read("app.css")
    assert re.search(r"\[hidden\]\s*\{[^}]*display:\s*none\s*!important", css), (
        "missing the global `[hidden] { display: none !important }` guard"
    )
