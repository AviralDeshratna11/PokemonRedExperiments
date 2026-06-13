"""
scripted_helpers.py -- deterministic button macros for menus/dialogue/healing.

The RL agent makes the strategic/exploration decisions, but purely mechanical menu
and dialogue steps should never hard-block progress. This module provides a small,
honest toolkit of deterministic helpers that operate directly on PyBoy:

  - ``tap`` / ``hold`` low-level button presses (mirrors the env's press/tick cadence)
  - ``mash_a`` / ``advance_dialogue`` to push through text boxes
  - ``heal_at_counter`` : best-effort "talk to the NPC in front and confirm yes"
                          sequence usable at a Poke Center counter / shop / Nurse
  - ``buy_confirm`` / ``use_confirm`` : confirm-style menu macros

These are intentionally conservative. Navigation *to* the nurse/clerk is left to the
RL policy (or a future scripted navigation skill in ``skills.py``); the helpers here
handle the deterministic "press A through the prompt, choose yes" tail that otherwise
wastes thousands of exploration steps.

``is_text_or_menu_open`` is a heuristic hook: Pokemon Red text/menu state detection
is version-specific, so the default implementation is approximate and clearly marked.
Override or refine it as you validate addresses for your ROM.
"""

from __future__ import annotations

from pyboy.utils import WindowEvent

from curriculum.ram_map import GameState

# name -> (press_event, release_event)
BUTTONS = {
    "a": (WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A),
    "b": (WindowEvent.PRESS_BUTTON_B, WindowEvent.RELEASE_BUTTON_B),
    "start": (WindowEvent.PRESS_BUTTON_START, WindowEvent.RELEASE_BUTTON_START),
    "select": (WindowEvent.PRESS_BUTTON_SELECT, WindowEvent.RELEASE_BUTTON_SELECT),
    "up": (WindowEvent.PRESS_ARROW_UP, WindowEvent.RELEASE_ARROW_UP),
    "down": (WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN),
    "left": (WindowEvent.PRESS_ARROW_LEFT, WindowEvent.RELEASE_ARROW_LEFT),
    "right": (WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT),
}

# wd730 holds assorted text/menu control bits; wTextBoxID lives near 0xCC. These
# are approximate "is the game showing a text box / menu" hints for Red. Kept here
# so they are easy to audit and tweak.
ADDR_WD730 = 0xD730
ADDR_TEXT_DELAY_FRAMES = 0xCC25  # nonzero while text is actively scrolling


class ScriptedHelpers:
    """Deterministic emulator macros. One instance per env (shares its PyBoy)."""

    def __init__(self, pyboy, act_freq: int = 24, headless: bool = True,
                 save_video: bool = False):
        self.pyboy = pyboy
        self.act_freq = act_freq
        self.headless = headless
        self.save_video = save_video
        self.gs = GameState(pyboy)

    # ----- low level ------------------------------------------------------- #
    def _render(self) -> bool:
        return self.save_video or not self.headless

    def tap(self, name: str, press_frames: int = 8) -> None:
        """Press a button then release, using the same cadence as the env step."""
        press, release = BUTTONS[name]
        render = self._render()
        self.pyboy.send_input(press)
        self.pyboy.tick(press_frames, render)
        self.pyboy.send_input(release)
        self.pyboy.tick(max(self.act_freq - press_frames - 1, 1), render)
        self.pyboy.tick(1, True)

    def mash_a(self, taps: int = 1) -> None:
        for _ in range(taps):
            self.tap("a")

    # ----- heuristics ------------------------------------------------------ #
    def is_in_battle(self) -> bool:
        return self.gs.in_battle()

    def is_text_or_menu_open(self) -> bool:
        """Approximate: a text box / menu appears active.

        Heuristic and ROM-version-sensitive -- treat as a hint, not ground truth.
        Refine with validated addresses for your ROM if you rely on it heavily.
        """
        try:
            return self.pyboy.memory[ADDR_TEXT_DELAY_FRAMES] != 0
        except Exception:
            return False

    # ----- composite macros ------------------------------------------------ #
    def advance_dialogue(self, max_taps: int = 8) -> int:
        """Tap A up to ``max_taps`` times to push through a dialogue box.

        Stops early once no text box seems open. Returns the number of taps used.
        Safe to call when stuck; it does nothing harmful in the overworld beyond a
        few A presses (which the agent might issue anyway).
        """
        used = 0
        for _ in range(max_taps):
            if not self.is_text_or_menu_open():
                break
            self.tap("a")
            used += 1
        return used

    def heal_at_counter(self, dialogue_taps: int = 16) -> None:
        """Best-effort 'talk to NPC in front and confirm yes'.

        Intended for use once the policy has positioned the player facing a Poke
        Center nurse (or a shop clerk). It taps A to start dialogue, nudges the
        cursor up to the first ("YES"/top) option, confirms, then mashes A through
        the remaining text. Navigation to the counter is the policy's job.
        """
        self.tap("a")                  # initiate dialogue
        self.tap("up")                 # ensure cursor on the top menu option
        self.tap("a")                  # confirm (e.g. "Yes, heal")
        self.mash_a(dialogue_taps)     # flush the rest of the dialogue

    def buy_confirm(self, quantity_taps: int = 0, dialogue_taps: int = 12) -> None:
        """Confirm a purchase: optionally bump quantity up, then accept."""
        self.tap("a")                  # select item / open buy prompt
        for _ in range(quantity_taps):
            self.tap("up")             # increase quantity
        self.tap("a")                  # confirm quantity
        self.tap("a")                  # confirm "Yes"
        self.mash_a(dialogue_taps)

    def use_confirm(self, dialogue_taps: int = 8) -> None:
        """Confirm using/teaching an item (e.g. an HM) through its prompts."""
        self.tap("a")
        self.tap("a")
        self.mash_a(dialogue_taps)
