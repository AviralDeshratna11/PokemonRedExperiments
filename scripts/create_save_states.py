"""
create_save_states.py
---------------------
Interactive script to create the curriculum save states for Phase 1.

You play the game manually through PyBoy's SDL2 window.
At each milestone, press F1 to save the current state.

Controls:
  Arrow keys  — movement
  Z           — A button
  X           — B button
  Enter       — Start
  Backspace   — Select
  F1          — Save state at current checkpoint
  F2          — Print current RAM info
  ESC         — Quit

Run from the project root:
    python scripts/create_save_states.py

IMPORTANT: Run from the baselines/ directory or adjust rom_path below.
"""

import os
import sys
import json
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyboy import PyBoy
from pyboy.utils import WindowEvent
from env.pokemon_ram import PokemonRAM, MAP_NAMES

import sdl2


# ─── Config ──────────────────────────────────────────────────────────────────

ROM_PATH        = "PokemonRed.gb"            # adjust if needed
STATE_DIR       = Path("checkpoints")
EXISTING_STATE  = "checkpoints/after_brock.state"                      # set to a .state path to start from

# Ordered list of checkpoints to create.
# You'll be prompted to press F1 when you reach each one.
CHECKPOINTS = [
    {
        "name":        "has_starter",
        "description": "Received starter Pokemon from Oak. Press F1 HERE.",
        "filename":    "has_starter.state",
    },
    {
        "name":        "delivered_parcel",
        "description": "Delivered Oak's Parcel in Viridian City. Press F1 HERE.",
        "filename":    "delivered_parcel.state",
    },
    {
        "name":        "before_brock",
        "description": "Party at roughly Lv10+, standing at Pewter Gym entrance. Press F1 HERE.",
        "filename":    "before_brock.state",
    },
    {
        "name":        "after_brock",
        "description": "Just received Boulder Badge. Press F1 HERE.",
        "filename":    "after_brock.state",
    },
    {
        "name":        "mt_moon_entrance",
        "description": "Entered Mt. Moon (first floor). Press F1 HERE.",
        "filename":    "mt_moon_entrance.state",
    },
    {
        "name":        "mt_moon_exit",
        "description": "Exited Mt. Moon into Route 4. Press F1 HERE.",
        "filename":    "mt_moon_exit.state",
    },
    {
        "name":        "cerulean_entrance",
        "description": "Arrived in Cerulean City. Press F1 HERE.",
        "filename":    "cerulean_entrance.state",
    },
    {
        "name":        "bills_quest_done",
        "description": "Completed Bill's quest (received SS Ticket). Press F1 HERE.",
        "filename":    "bills_quest_done.state",
    },
    {
        "name":        "after_misty",
        "description": "Just received Cascade Badge. Press F1 HERE.",
        "filename":    "after_misty.state",
    },
    {
        "name":        "vermilion_entrance",
        "description": "Arrived in Vermilion City. Press F1 HERE.",
        "filename":    "vermilion_entrance.state",
    },
    {
        "name":        "after_surge",
        "description": "Just received Thunder Badge. Press F1 HERE.",
        "filename":    "after_surge.state",
    },
]


def main():
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Pokemon Red — Curriculum Save State Creator")
    print("=" * 60)
    print(f"\nSave states will be written to: {STATE_DIR.resolve()}")
    print("\nControls:")
    print("  Arrow keys  = movement")
    print("  Z           = A button")
    print("  X           = B button")
    print("  Enter       = Start")
    print("  Backspace   = Select")
    print("  F1          = Save state at current milestone")
    print("  F2          = Print RAM info")
    print("  ESC         = Quit\n")

    # Launch emulator
    pyboy = PyBoy(ROM_PATH, window="SDL2")
    pyboy.set_emulation_speed(2)
    ram   = PokemonRAM(pyboy)

    # Optionally start from an existing save state
    if EXISTING_STATE and os.path.exists(EXISTING_STATE):
        with open(EXISTING_STATE, "rb") as f:
            pyboy.load_state(f)
        print(f"Loaded existing state: {EXISTING_STATE}")

    checkpoint_idx = 4

    def print_ram_info():
        pos    = ram.read_position()
        badges = ram.read_badges()
        party  = ram.read_party()
        money  = ram.read_money()
        events = ram.read_events()

        print("\n" + "─" * 50)
        print(f"  Map:    {pos['map_name']} (ID: {pos['map_id']:02X})")
        print(f"  Pos:    ({pos['x']}, {pos['y']})")
        print(f"  Badges: {badges['count']}/8  {badges['flags']}")
        print(f"  Money:  ${money:,}")
        print(f"  Party ({party['count']} pokemon):")
        for m in party["mons"]:
            print(f"    {m['name']:12s} Lv{m['level']:3d}  HP {m['current_hp']}/{m['max_hp']}")
        print(f"  Events: {events}")
        print("─" * 50 + "\n")

    def save_checkpoint(idx):
        cp   = CHECKPOINTS[idx]
        path = STATE_DIR / cp["filename"]
        with open(path, "wb") as f:
            pyboy.save_state(f)
        print(f"\n✅  SAVED: {cp['name']} → {path}")
        print(f"   Description: {cp['description']}\n")
        return idx + 1

    # Game loop
    print(f"\n▶  GOAL: {CHECKPOINTS[checkpoint_idx]['description']}\n")

    running = True
    f1_held = False
    f2_held = False

    try:
        while running:
            pyboy.tick()

            # Check for QUIT event from PyBoy (window close / ESC)
            for event in pyboy.events:
                if event == WindowEvent.QUIT:
                    running = False

            # Use SDL2 keyboard state for F1/F2 detection
            keystates = sdl2.SDL_GetKeyboardState(None)

            # F1 — save state (trigger on press, not hold)
            if keystates[sdl2.SDL_SCANCODE_F1]:
                if not f1_held:
                    f1_held = True
                    if checkpoint_idx < len(CHECKPOINTS):
                        checkpoint_idx = save_checkpoint(checkpoint_idx)
                        if checkpoint_idx < len(CHECKPOINTS):
                            print(f"▶  NEXT GOAL: {CHECKPOINTS[checkpoint_idx]['description']}\n")
                        else:
                            print("\n🎉  All checkpoints saved! You can now quit (ESC).\n")
                    else:
                        print("All checkpoints already saved.")
            else:
                f1_held = False

            # F2 — print RAM info (trigger on press, not hold)
            if keystates[sdl2.SDL_SCANCODE_F2]:
                if not f2_held:
                    f2_held = True
                    print_ram_info()
            else:
                f2_held = False

    except KeyboardInterrupt:
        pass

    # Write checkpoint manifest
    manifest_path = STATE_DIR / "manifest.json"
    manifest = []
    for cp in CHECKPOINTS:
        path = STATE_DIR / cp["filename"]
        manifest.append({
            "name":        cp["name"],
            "description": cp["description"],
            "filename":    cp["filename"],
            "exists":      path.exists(),
        })
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {manifest_path}")

    pyboy.stop()
    print("Done.")


if __name__ == "__main__":
    main()
