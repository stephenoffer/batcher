"""What the engine believes a terminal can do, and the conventions it promises to honor.

Getting a convention nearly right is worse than not implementing it: a user who sets
`NO_COLOR` or `FORCE_COLOR` and does not get what the variable promises has no way to tell
a bug from a policy. Both were nearly right here — `NO_COLOR=` (empty) disabled color where
the spec says it must not, and `FORCE_COLOR=0` *enabled* it, turning the one variable people
use to suppress color into one that forces it.
"""

from __future__ import annotations

import io

import pytest

from batcher.observe.theme import _color_depth as depth
from batcher.observe.theme import detect

pytestmark = pytest.mark.unit


class _Utf8(io.StringIO):
    encoding = "utf-8"


# --- the two specs -----------------------------------------------------------


def test_no_color_disables_color_when_set_to_a_non_empty_value():
    assert depth({"TERM": "xterm-256color", "NO_COLOR": "1"}) == 0
    assert depth({"TERM": "xterm-256color", "NO_COLOR": "anything"}) == 0


def test_an_empty_no_color_does_not_disable_color():
    """no-color.org: "when present and not an empty string".

    The empty form is what lets a wrapper script clear the variable for a child process by
    exporting it empty, and treating it as "off" takes that away.
    """
    assert depth({"TERM": "xterm-256color", "NO_COLOR": ""}) == 8


@pytest.mark.parametrize(
    ("value", "expected"),
    [("0", 0), ("false", 0), ("1", 4), ("2", 8), ("3", 24), ("true", 24), ("", 24)],
)
def test_force_color_levels_follow_the_convention_including_zero(value, expected):
    """`FORCE_COLOR=0` disables. Treating any value as "on" is the common mistake."""
    assert depth({"FORCE_COLOR": value}) == expected


def test_force_color_beats_a_terminal_that_advertises_less():
    assert depth({"TERM": "xterm", "FORCE_COLOR": "3"}) == 24


def test_no_color_beats_force_color():
    """Both are deliberate statements; the one that suppresses output has to win."""
    assert depth({"NO_COLOR": "1", "FORCE_COLOR": "3"}) == 0


# --- what terminals advertise ------------------------------------------------


def test_a_dumb_terminal_gets_nothing_whatever_else_is_set():
    assert depth({"TERM": "dumb", "COLORTERM": "truecolor"}) == 0


def test_colorterm_is_taken_at_its_word():
    assert depth({"TERM": "xterm", "COLORTERM": "truecolor"}) == 24
    assert depth({"TERM": "xterm", "COLORTERM": "24bit"}) == 24


@pytest.mark.parametrize(
    "env",
    [
        {"WT_SESSION": "abc"},  # Windows Terminal
        {"ConEmuANSI": "ON"},  # ConEmu
        {"TERM_PROGRAM": "iTerm.app"},
        {"TERM_PROGRAM": "WezTerm"},
        {"TERM_PROGRAM": "vscode"},
        {"TERM": "xterm-kitty"},
        {"TERM": "alacritty"},
    ],
)
def test_terminals_that_advertise_nothing_but_render_truecolor(env):
    """Windows Terminal and ConEmu set no `TERM` at all under cmd.exe, so without these
    the whole Windows story degraded to monochrome."""
    assert depth(env) == 24


def test_apple_terminal_gets_256_colors_which_is_what_it_has():
    assert depth({"TERM": "xterm-256color", "TERM_PROGRAM": "Apple_Terminal"}) == 8


def test_a_plain_terminal_gets_sixteen_colors_and_no_terminal_gets_none():
    assert depth({"TERM": "xterm"}) == 4
    assert depth({}) == 0


# --- unicode -----------------------------------------------------------------


def test_unicode_is_read_off_the_stream_rather_than_guessed(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    _, glyphs = detect(_Utf8())
    assert glyphs.unicode is True
    assert glyphs.ok == "✔"

    class _Ascii(io.StringIO):
        encoding = "ascii"

    _, ascii_glyphs = detect(_Ascii())
    assert ascii_glyphs.unicode is False
    assert ascii_glyphs.ok == "OK"


def test_a_dumb_terminal_gets_ascii_glyphs_even_on_a_utf8_stream(monkeypatch):
    monkeypatch.setenv("TERM", "dumb")
    _, glyphs = detect(_Utf8())
    assert glyphs.unicode is False


# --- the palette degrades by design ------------------------------------------


@pytest.mark.parametrize("bits", [0, 4, 8, 24])
def test_every_role_exists_at_every_depth(bits):
    """The renderer asks for a role; dropping to 16 colors changes this table and nothing
    else. A missing role at one depth would be an AttributeError only on that terminal."""
    from batcher.observe.theme import Palette

    palette = Palette(bits)
    for role in ("reset", "dim", "bold", "muted", "accent", "good", "warn", "critical"):
        assert isinstance(getattr(palette, role), str)
    assert isinstance(palette.ramp(0.5), str)


def test_a_monochrome_palette_emits_no_escape_codes_at_all():
    from batcher.observe.theme import Palette

    palette = Palette(0)
    for role in ("reset", "dim", "bold", "muted", "accent", "good", "warn", "critical"):
        assert getattr(palette, role) == ""
    assert palette.ramp(0.5) == ""


def test_the_ramp_is_a_gradient_where_there_are_colors_to_interpolate():
    from batcher.observe.theme import Palette

    for bits in (8, 24):
        palette = Palette(bits)
        assert palette.ramp(0.0) != palette.ramp(1.0), f"{bits}-bit ramp is flat"
    # At 16 colors there is nothing to interpolate between, so it collapses by design.
    assert Palette(4).ramp(0.0) == Palette(4).ramp(1.0)
