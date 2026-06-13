"""
curriculum_env.py -- CurriculumRedGymEnv: the hierarchical/curriculum environment.

This subclasses the existing v2 :class:`RedGymEnv` and wires together the curriculum
package (structured observations, milestone manager, configurable rewards, anti-loop
penalties, scripted helpers, and the per-milestone state store) **without modifying**
the base environment. ``baseline_fast_v2.py`` and ``red_gym_env_v2.py`` keep working
exactly as before.

What it changes relative to the base env:

* Observation: a primary structured RAM/event vector (``obs["structured"]``), with the
  pixel frame-stack kept as an optional auxiliary stream (``obs["screens"]``).
* Reward: computed by :class:`curriculum.rewards.RewardManager` (configurable, logged,
  one-time/max-based, with anti-loop/anti-farm penalties) instead of the base reward.
* Episodes: can start from saved success snapshots of a previous milestone and can end
  when a target milestone is reached or on a blackout.
* On milestone completion, the exact emulator state is snapshotted to the state store
  so later curriculum stages can warm-start from it.

The class degrades gracefully: with ``use_structured_obs=False`` and ``use_screens=True``
and no ``target_milestone`` it behaves much like the base env but with the new reward.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
from gymnasium import spaces

from red_gym_env_v2 import RedGymEnv

from curriculum.ram_map import GameState
from curriculum.milestones import MilestoneManager, build_default_milestones
from curriculum.rewards import RewardConfig, RewardManager
from curriculum.structured_obs import StructuredObservationBuilder
from curriculum.scripted_helpers import ScriptedHelpers
from curriculum.state_store import StateStore


class CurriculumRedGymEnv(RedGymEnv):
    def __init__(self, config=None):
        config = dict(config or {})

        # --- pull curriculum extras (with defaults) before base init ---
        self.use_structured_obs = config.get("use_structured_obs", True)
        self.use_screens = config.get("use_screens", True)
        self.history_len = int(config.get("history_len", 8))
        self.emulation_speed = int(config.get("emulation_speed", 0))
        self.target_milestone = config.get("target_milestone", None)
        self.start_state_strategy = config.get("start_state_strategy", "random")
        self.save_success_states = config.get("save_success_states", True)
        self.success_state_min_reward = float(config.get("success_state_min_reward", 0.0))
        self.end_on_target = config.get("end_on_target", False)
        self.end_on_blackout = config.get("end_on_blackout", False)
        self.use_scripted_helpers = config.get("use_scripted_helpers", False)
        self.auto_advance_dialogue = config.get("auto_advance_dialogue", False)
        self.dialogue_taps = int(config.get("dialogue_taps", 8))
        state_store_root = config.get("state_store_root", "curriculum_states")
        reward_config: RewardConfig = config.get("reward_config", None) or RewardConfig()

        # --- base env init (creates pyboy, base observation_space) ---
        super().__init__(config)

        # --- emulation speed (base sets 6 when windowed; honor our config) ---
        if not self.headless and self.emulation_speed > 0:
            self.pyboy.set_emulation_speed(self.emulation_speed)

        # --- curriculum components (persist across resets) ---
        self.gs = GameState(self.pyboy)
        self.milestones = build_default_milestones()
        self.milestone_mgr = MilestoneManager(self.milestones, target_key=self.target_milestone)
        self.reward_mgr = RewardManager(reward_config)
        self.state_store = StateStore(state_store_root)
        self.helpers = ScriptedHelpers(
            self.pyboy, act_freq=self.act_freq, headless=self.headless,
            save_video=self.save_video,
        ) if self.use_scripted_helpers else None

        self.num_actions = len(self.valid_actions)
        self.obs_builder = StructuredObservationBuilder(
            num_actions=self.num_actions,
            history_len=self.history_len,
            num_milestones=len(self.milestones),
        )

        # histories for the structured observation (most-recent-first)
        self._action_hist = deque(maxlen=self.history_len)
        self._map_hist = deque(maxlen=self.history_len)
        self.episode_reward = 0.0

        # --- rebuild observation space with our keys ---
        self.observation_space = self._build_observation_space()

    # ------------------------------------------------------------------ #
    def _build_observation_space(self) -> spaces.Dict:
        space = {}
        if self.use_structured_obs:
            space["structured"] = self.obs_builder.space()
        if self.use_screens or not self.use_structured_obs:
            # keep the pixel frame-stack (always keep at least one stream)
            space["screens"] = spaces.Box(
                low=0, high=255, shape=self.output_shape, dtype=np.uint8)
        return spaces.Dict(space)

    # ------------------------------------------------------------------ #
    def _choose_start_state(self) -> Optional[str]:
        """Pick a warm-start snapshot for the current target milestone, if any."""
        if not self.target_milestone:
            return None
        ms = self.milestone_mgr.by_key.get(self.target_milestone)
        if ms is None or not ms.allowed_start_states:
            return None
        return self.state_store.select_from_any(
            ms.allowed_start_states, strategy=self.start_state_strategy)

    # ------------------------------------------------------------------ #
    def reset(self, seed=None, options=None):
        options = options or {}
        chosen = self._choose_start_state()
        original_init = self.init_state
        if chosen is not None:
            self.init_state = chosen

        # base reset loads init_state and resets base bookkeeping
        _, info = super().reset(seed=seed, options=options)

        self.init_state = original_init

        # (re)initialize curriculum trackers against the freshly loaded state
        self._action_hist.clear()
        self._map_hist.clear()
        self.episode_reward = 0.0
        self.reward_mgr.reset()
        self.reward_mgr.baseline(self.gs)
        self.milestone_mgr.reset()
        # latch already-satisfied milestones silently (no reward for them)
        self.milestone_mgr.update(self.gs)

        return self._get_obs(), info

    # ------------------------------------------------------------------ #
    def _push_history(self, action: int) -> None:
        self._action_hist.appendleft(int(action))
        self._map_hist.appendleft(int(self.gs.map_id()))

    def _get_obs(self):
        obs = {}
        if self.use_structured_obs:
            sig = self.reward_mgr.episode_signals(
                self.milestone_mgr,
                list(self._action_hist),
                list(self._map_hist),
                len(self.milestones),
            )
            obs["structured"] = self.obs_builder.build(self.gs, sig)
        if self.use_screens or not self.use_structured_obs:
            screen = self.render()
            self.update_recent_screens(screen)
            obs["screens"] = self.recent_screens
        return obs

    # ------------------------------------------------------------------ #
    def step(self, action):
        if self.save_video and self.step_count == 0:
            self.start_video()

        # advance emulator + base bookkeeping we still want for logging/visuals
        self.run_action_on_emulator(action)
        self.update_recent_actions(action)
        self.update_seen_coords()
        self.update_explore_map()
        self.update_map_progress()
        self.party_size = self.read_m(0xD163)

        # optional deterministic dialogue advancement so menus don't hard-block
        if self.helpers is not None and self.auto_advance_dialogue:
            self.helpers.advance_dialogue(self.dialogue_taps)

        # curriculum: milestones first (so reward sees newly_completed), then reward
        newly = self.milestone_mgr.update(self.gs)
        comp = self.reward_mgr.step(self.gs, action, self.milestone_mgr)
        step_reward = self.reward_mgr.scaled_total(comp)
        self.episode_reward += step_reward

        # keep base stat fields meaningful for the existing TensorboardCallback
        self.total_healing_rew = self.reward_mgr.heal_total
        self.died_count = self.reward_mgr.blackout_count

        # snapshot successful milestone states for future curriculum stages
        if self.save_success_states and newly:
            self._maybe_snapshot(newly)

        # histories used by the next observation
        self._push_history(action)

        # base agent_stats (augmented) for logging
        self.append_agent_stats(action)

        # termination / truncation
        truncated = self.check_if_done()
        terminated = False
        if self.end_on_target and self.milestone_mgr.reached_target():
            terminated = True
        if self.end_on_blackout and self.gs.all_fainted():
            terminated = True

        obs = self._get_obs()
        self.step_count += 1

        info = {
            "milestones_completed": self.milestone_mgr.completion_count(),
            "milestone_progress": self.milestone_mgr.progress_fraction(),
            "reached_target": self.milestone_mgr.reached_target(),
            "episode_reward": self.episode_reward,
            "reward_components": dict(comp),
        }
        return obs, step_reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    def _maybe_snapshot(self, newly_completed) -> None:
        for key in newly_completed:
            ms = self.milestone_mgr.by_key.get(key)
            if ms is None or not ms.save_state:
                continue
            if self.episode_reward < self.success_state_min_reward:
                continue
            try:
                self.state_store.save(
                    key, self.pyboy, reward=self.episode_reward, step=self.step_count)
            except Exception as e:  # never let snapshotting break training
                print(f"[CurriculumEnv] snapshot failed for {key}: {e}")

    # ------------------------------------------------------------------ #
    def append_agent_stats(self, action):
        """Override base stats with curriculum-aware fields for TensorBoard.

        TensorboardCallback averages numeric fields of ``agent_stats[-1]`` across
        envs, so exposing milestone/reward counters here surfaces them in TB.
        """
        x_pos, y_pos, map_n = self.get_game_coords()
        levels = self.gs.party_levels()
        self.agent_stats.append({
            "step": self.step_count,
            "x": x_pos,
            "y": y_pos,
            "map": map_n,
            "max_map_progress": self.max_map_progress,
            "last_action": action,
            "pcount": self.gs.party_count(),
            "levels_sum": sum(levels),
            "hp": self.gs.total_hp_fraction(),
            "coord_count": len(self.seen_coords),
            "deaths": self.reward_mgr.blackout_count,
            "badge": self.gs.badge_count(),
            "event_flags": self.gs.event_flags_sum(),
            "milestones": self.milestone_mgr.completion_count(),
            "milestone_progress": self.milestone_mgr.progress_fraction(),
            "money": self.gs.money(),
            "pokedex_owned": self.gs.pokedex_owned_count(),
            "episode_reward": self.episode_reward,
            "healr": self.total_healing_rew,
        })
