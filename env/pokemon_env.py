"""
pokemon_env.py
--------------
A Gymnasium-compatible Pokemon Red environment built on top of PyBoy.

Key improvements over the PWhiddy baseline:
  1. Dual observation: screen pixels  +  structured RAM feature vector
  2. Anti-exploit reward function with diminishing exploration returns
  3. Event-flag based milestone rewards (given exactly once)
  4. Curriculum support: load from any save-state checkpoint
  5. Episode statistics tracking for TensorBoard / WandB

Usage:
    env = PokemonRedEnv(PokemonRedConfig())
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
"""

import os
import math
import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from pyboy import PyBoy
from pyboy.utils import WindowEvent

from .pokemon_ram import PokemonRAM, MAP_NAMES, get_map_size


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PokemonRedConfig:
    rom_path:          str  = "../PokemonRed.gb"
    init_state_path:   str  = "checkpoints/has_starter.state"  # starting save state
    headless:          bool = True
    action_freq:       int  = 24          # game frames per agent action
    max_steps:         int  = 20_000      # episode length
    save_video:        bool = False
    video_dir:         str  = "videos"
    log_dir:           str  = "runs"
    instance_id:       int  = 0

    # Reward weights
    reward_explore:    float = 0.01       # per new tile discovered
    reward_badge:      List[float] = field(default_factory=lambda: [3,3,4,4,4,5,5,8])
    reward_level_up:   float = 0.1        # per level gained
    reward_event:      float = 1.0        # base for story events
    penalty_faint:     float = -0.5       # per pokemon fainting
    penalty_step:      float = -0.0001   # time pressure

    # Exploration cap
    max_explore_reward_per_map: float = 5.0   # hard cap per map


# ─────────────────────────────────────────────────────────────────────────────
#  ACTION MAP
# ─────────────────────────────────────────────────────────────────────────────

ACTIONS = [
    WindowEvent.PRESS_ARROW_DOWN,
    WindowEvent.PRESS_ARROW_LEFT,
    WindowEvent.PRESS_ARROW_RIGHT,
    WindowEvent.PRESS_ARROW_UP,
    WindowEvent.PRESS_BUTTON_A,
    WindowEvent.PRESS_BUTTON_B,
    WindowEvent.PRESS_BUTTON_START,
    WindowEvent.PRESS_BUTTON_SELECT,
]

RELEASE_ACTIONS = [
    WindowEvent.RELEASE_ARROW_DOWN,
    WindowEvent.RELEASE_ARROW_LEFT,
    WindowEvent.RELEASE_ARROW_RIGHT,
    WindowEvent.RELEASE_ARROW_UP,
    WindowEvent.RELEASE_BUTTON_A,
    WindowEvent.RELEASE_BUTTON_B,
    WindowEvent.RELEASE_BUTTON_START,
    WindowEvent.RELEASE_BUTTON_SELECT,
]

N_ACTIONS = len(ACTIONS)


# ─────────────────────────────────────────────────────────────────────────────
#  STORY EVENTS WITH BASE REWARDS
# ─────────────────────────────────────────────────────────────────────────────

STORY_EVENTS = {
    "got_starter":       2.0,
    "got_pokedex":       1.0,
    "delivered_parcel":  1.5,
    "bills_quest_done":  2.5,
    "ss_anne_left":      1.5,
    "silph_complete":    4.0,
    "got_lapras":        2.0,
}


# ─────────────────────────────────────────────────────────────────────────────
#  ENVIRONMENT CLASS
# ─────────────────────────────────────────────────────────────────────────────

class PokemonRedEnv(gym.Env):
    """
    Gymnasium environment for Pokemon Red.

    Observation space:
      - 'screen': (3, 144, 160) uint8 RGB image
      - 'ram':    (128,)       float32 feature vector

    Action space: Discrete(8)  [Down, Left, Right, Up, A, B, Start, Select]
    """

    metadata = {"render_modes": ["rgb_array", "human"]}

    # ─── Init ──────────────────────────────────────────────────────────────

    def __init__(self, config: Optional[PokemonRedConfig] = None, render_mode=None):
        super().__init__()
        self.config       = config or PokemonRedConfig()
        self.render_mode  = render_mode
        self.instance_id  = self.config.instance_id

        # Emulator
        window = "null" if self.config.headless else "SDL2"
        self.pyboy = PyBoy(
            self.config.rom_path,
            window=window,
            cgb_mode=False,
        )
        if not self.config.headless:
            self.pyboy.set_emulation_speed(6)

        self.ram = PokemonRAM(self.pyboy)

        # Spaces
        self.observation_space = spaces.Dict({
            "screen": spaces.Box(0, 255, (3, 144, 160), dtype=np.uint8),
            "ram":    spaces.Box(-1.0, 2.0, (128,), dtype=np.float32),
        })
        self.action_space = spaces.Discrete(N_ACTIONS)

        # Load initial save state
        self._load_state(self.config.init_state_path)

        # Episode state
        self._reset_episode_state()

        # Video writer
        self._video_writer = None
        if self.config.save_video:
            Path(self.config.video_dir).mkdir(parents=True, exist_ok=True)

    # ─── State management ─────────────────────────────────────────────────

    def _load_state(self, state_path: str):
        """Load a .state file into the emulator."""
        with open(state_path, "rb") as f:
            self.pyboy.load_state(f)

    def set_start_state(self, state_path: str):
        """Switch the starting checkpoint (called by curriculum manager)."""
        self.config.init_state_path = state_path

    # ─── Episode bookkeeping ──────────────────────────────────────────────

    def _reset_episode_state(self):
        self.step_count          = 0
        self.total_reward        = 0.0
        self.visited_coords: Dict[int, set] = {}   # map_id -> set of (x,y)
        self.explore_reward_acc: Dict[int, float] = {}  # accumulated per map

        # Previous game state snapshot for delta-based rewards
        self._prev_state = None

        # Event flags seen this episode (to reward each exactly once)
        self._seen_events:  Dict[str, bool] = {}
        self._seen_badges:  int             = 0   # badge bitmask at episode start
        self._seen_levels:  List[int]       = [0] * 6
        self._faint_counts: List[int]       = [0] * 6

        # Stats logged at end of episode
        self.episode_stats = {
            "total_reward":    0.0,
            "steps":           0,
            "badges":          0,
            "tiles_explored":  0,
            "max_party_level": 0,
            "events_triggered":[],
            "maps_visited":    set(),
        }

    # ─── Reset ────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._load_state(self.config.init_state_path)
        self._reset_episode_state()

        # Take a first observation
        full_state   = self.ram.read_all()
        self._prev_state = full_state

        # Initialise seen events from save-state baseline
        self._seen_events = {k: v for k, v in full_state["events"].items()}
        self._seen_badges  = full_state["badges"]["raw"]
        self._seen_levels  = [
            full_state["party"]["mons"][i]["level"] if i < full_state["party"]["count"] else 0
            for i in range(6)
        ]

        obs  = self._get_obs()
        info = self._get_info(full_state)
        return obs, info

    # ─── Step ─────────────────────────────────────────────────────────────

    def step(self, action: int):
        # Execute action on emulator
        self._run_action(action)
        self.step_count += 1

        # Read new state
        curr_state = self.ram.read_all()

        # Compute reward
        reward, reward_breakdown = self._compute_reward(curr_state, self._prev_state)

        # Update prev state
        self._prev_state = curr_state

        # Episode stats
        self.total_reward           += reward
        self.episode_stats["badges"] = curr_state["badges"]["count"]
        self.episode_stats["max_party_level"] = max(
            self.episode_stats["max_party_level"],
            curr_state["party"]["total_level"],
        )
        self.episode_stats["maps_visited"].add(curr_state["map_id"])

        # Termination
        terminated = curr_state["party"]["all_fainted"]
        truncated  = self.step_count >= self.config.max_steps

        if terminated or truncated:
            self.episode_stats["total_reward"] = self.total_reward
            self.episode_stats["steps"]        = self.step_count
            self.episode_stats["tiles_explored"] = sum(
                len(s) for s in self.visited_coords.values()
            )

        obs  = self._get_obs()
        info = self._get_info(curr_state)
        info["reward_breakdown"] = reward_breakdown

        return obs, reward, terminated, truncated, info

    # ─── Action execution ─────────────────────────────────────────────────

    def _run_action(self, action: int):
        """Press and hold button for action_freq frames, then release."""
        press   = ACTIONS[action]
        release = RELEASE_ACTIONS[action]

        self.pyboy.send_input(press)
        for _ in range(self.config.action_freq - 1):
            self.pyboy.tick(1, False)
        self.pyboy.send_input(release)
        self.pyboy.tick(1, True)   # final tick renders frame

    # ─── Observation ──────────────────────────────────────────────────────

    def _get_obs(self):
        screen = self.pyboy.screen.image  # PIL Image (160, 144, RGBA)
        import numpy as np
        screen_np = np.array(screen, dtype=np.uint8)[:, :, :3]  # drop alpha
        # Gymnasium convention: (C, H, W)
        screen_np = screen_np.transpose(2, 0, 1)  # (3, 144, 160)

        ram_vec = self.ram.to_feature_vector()

        return {"screen": screen_np, "ram": ram_vec}

    # ─── Info dict ────────────────────────────────────────────────────────

    def _get_info(self, state):
        return {
            "map_id":       state["map_id"],
            "map_name":     state.get("map_name", "Unknown"),
            "position":     (state["x"], state["y"]),
            "badges":       state["badges"]["count"],
            "party_levels": [m["level"] for m in state["party"]["mons"]],
            "money":        state["money"],
            "total_reward": self.total_reward,
            "step":         self.step_count,
        }

    # ─── Reward calculation ───────────────────────────────────────────────

    def _compute_reward(self, curr, prev) -> Tuple[float, dict]:
        """
        Compute the reward for a single transition.
        Returns (total_reward, breakdown_dict).
        """
        r      = 0.0
        breakdown = {}

        # ── 1. Exploration reward (coordinate-based, diminishing) ──────────
        explore_r = self._exploration_reward(curr)
        r        += explore_r
        breakdown["explore"] = explore_r

        # ── 2. Badge rewards ───────────────────────────────────────────────
        badge_r = self._badge_reward(curr["badges"]["raw"])
        r      += badge_r
        breakdown["badge"] = badge_r

        # ── 3. Level-up rewards ────────────────────────────────────────────
        level_r = self._level_reward(curr["party"])
        r      += level_r
        breakdown["level"] = level_r

        # ── 4. Story event rewards ─────────────────────────────────────────
        event_r = self._event_reward(curr["events"])
        r      += event_r
        breakdown["event"] = event_r

        # ── 5. Faint penalty ──────────────────────────────────────────────
        faint_p = self._faint_penalty(curr["party"], prev["party"])
        r      += faint_p
        breakdown["faint"] = faint_p

        # ── 6. Step penalty (time pressure) ───────────────────────────────
        step_p = self.config.penalty_step
        r     += step_p
        breakdown["step_penalty"] = step_p

        breakdown["total"] = r
        return r, breakdown

    def _exploration_reward(self, curr) -> float:
        map_id = curr["map_id"]
        pos    = (curr["x"], curr["y"])

        if map_id not in self.visited_coords:
            self.visited_coords[map_id]      = set()
            self.explore_reward_acc[map_id]  = 0.0

        if pos in self.visited_coords[map_id]:
            return 0.0  # already visited

        self.visited_coords[map_id].add(pos)
        acc = self.explore_reward_acc.get(map_id, 0.0)

        # Hard cap per map prevents endless farming
        if acc >= self.config.max_explore_reward_per_map:
            return 0.0

        # Diminishing returns: reward decays as coverage fraction grows
        n_tiles  = len(self.visited_coords[map_id])
        map_size = max(get_map_size(map_id), 1)
        frac     = n_tiles / map_size
        bonus    = self.config.reward_explore * (1.0 - frac * 0.8)
        bonus    = max(bonus, self.config.reward_explore * 0.1)  # floor

        # Clip to remaining cap
        remaining = self.config.max_explore_reward_per_map - acc
        bonus     = min(bonus, remaining)
        self.explore_reward_acc[map_id] += bonus
        return bonus

    def _badge_reward(self, current_badge_mask: int) -> float:
        reward     = 0.0
        new_badges = current_badge_mask & ~self._seen_badges
        if new_badges:
            for i in range(8):
                if (new_badges >> i) & 1:
                    reward += self.config.reward_badge[i]
            self._seen_badges = current_badge_mask
        return reward

    def _level_reward(self, party) -> float:
        reward = 0.0
        for i, mon in enumerate(party["mons"]):
            curr_level = mon["level"]
            prev_level = self._seen_levels[i]
            gain = curr_level - prev_level
            # Sanity check: ignore impossible jumps (save state artifacts)
            if 0 < gain <= 10:
                reward += gain * self.config.reward_level_up
            if gain > 0:
                self._seen_levels[i] = curr_level
        return reward

    def _event_reward(self, events: dict) -> float:
        reward = 0.0
        for event_name, base_r in STORY_EVENTS.items():
            if events.get(event_name) and not self._seen_events.get(event_name):
                reward += base_r * self.config.reward_event
                self._seen_events[event_name] = True
                self.episode_stats["events_triggered"].append(event_name)
        return reward

    def _faint_penalty(self, curr_party, prev_party) -> float:
        penalty = 0.0
        for i in range(min(curr_party["count"], prev_party["count"])):
            curr_mon = curr_party["mons"][i] if i < len(curr_party["mons"]) else None
            prev_mon = prev_party["mons"][i] if i < len(prev_party["mons"]) else None
            if curr_mon and prev_mon:
                if curr_mon["fainted"] and not prev_mon["fainted"]:
                    penalty += self.config.penalty_faint
        return penalty

    # ─── Render ───────────────────────────────────────────────────────────

    def render(self):
        import numpy as np
        screen = self.pyboy.screen.image
        arr    = np.array(screen, dtype=np.uint8)[:, :, :3]
        if self.render_mode == "human":
            import matplotlib.pyplot as plt
            plt.imshow(arr)
            plt.axis("off")
            plt.show()
        return arr

    # ─── Close ────────────────────────────────────────────────────────────

    def close(self):
        self.pyboy.stop()
