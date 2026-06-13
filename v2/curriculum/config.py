"""
config.py -- typed configuration for the curriculum env + training driver.

Loadable from YAML or JSON. A single :class:`CurriculumConfig` bundles:

  - emulator/env settings (ROM path, init state, headless, action_freq, ...)
  - observation settings (structured vector on/off, keep pixel screens, history len)
  - the :class:`curriculum.rewards.RewardConfig` (all reward weights)
  - curriculum/staging settings (state-store location, start-state strategy, ...)
  - scripted-helper toggles

The env consumes a plain dict (for backwards compatibility with ``RedGymEnv``), so
:meth:`CurriculumConfig.to_env_config` flattens everything into the dict the env
reads in ``__init__``.

YAML support requires PyYAML (``pip install pyyaml``); JSON works with the stdlib.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Optional

from curriculum.rewards import RewardConfig


@dataclass
class CurriculumConfig:
    # --- emulator / env ---
    gb_path: str = "../PokemonRed.gb"
    init_state: str = "../init.state"
    headless: bool = True
    action_freq: int = 24
    max_steps: int = 2048 * 16
    save_video: bool = False
    fast_video: bool = True
    save_final_state: bool = False
    print_rewards: bool = False
    session_path: str = "runs_curriculum"
    emulation_speed: int = 0          # 0 = unlimited (headless); >0 throttles visual

    # --- observation ---
    use_structured_obs: bool = True
    use_screens: bool = True          # keep base pixel stream as auxiliary input
    history_len: int = 8

    # --- curriculum / staging ---
    state_store_root: str = "curriculum_states"
    target_milestone: Optional[str] = None   # which milestone this stage trains
    start_state_strategy: str = "random"      # random | latest | best
    save_success_states: bool = True
    success_state_min_reward: float = 0.0     # only snapshot if episode reward >= this
    end_on_target: bool = False               # end episode when target milestone done
    end_on_blackout: bool = False             # end episode on a blackout

    # --- scripted helpers ---
    use_scripted_helpers: bool = False
    auto_advance_dialogue: bool = False
    dialogue_taps: int = 8

    # --- reward weights (nested) ---
    reward: RewardConfig = field(default_factory=RewardConfig)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, d: dict) -> "CurriculumConfig":
        d = dict(d or {})
        reward_d = d.pop("reward", {}) or {}
        known = {f.name for f in fields(cls) if f.name != "reward"}
        kwargs = {k: v for k, v in d.items() if k in known}
        unknown = set(d) - known
        if unknown:
            print(f"[CurriculumConfig] ignoring unknown keys: {sorted(unknown)}")
        reward_known = {f.name for f in fields(RewardConfig)}
        reward_kwargs = {k: v for k, v in reward_d.items() if k in reward_known}
        return cls(reward=RewardConfig(**reward_kwargs), **kwargs)

    @classmethod
    def from_file(cls, path: str | Path) -> "CurriculumConfig":
        path = Path(path)
        text = path.read_text()
        if path.suffix.lower() in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError as e:
                raise ImportError(
                    "Loading a .yaml config requires PyYAML: pip install pyyaml "
                    "(or provide a .json config instead)."
                ) from e
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        return cls.from_dict(data)

    def to_dict(self) -> dict:
        return asdict(self)

    # ------------------------------------------------------------------ #
    def to_env_config(self) -> dict:
        """Flatten into the dict ``CurriculumRedGymEnv`` / ``RedGymEnv`` expect.

        Keeps every key the base ``RedGymEnv`` reads, plus the curriculum extras.
        """
        return {
            # base RedGymEnv keys
            "session_path": Path(self.session_path),
            "gb_path": self.gb_path,
            "init_state": self.init_state,
            "headless": self.headless,
            "action_freq": self.action_freq,
            "max_steps": self.max_steps,
            "save_video": self.save_video,
            "fast_video": self.fast_video,
            "save_final_state": self.save_final_state,
            "print_rewards": self.print_rewards,
            "reward_scale": self.reward.reward_scale,
            "explore_weight": 1.0,   # exploration handled by RewardManager instead
            "early_stop": False,
            "debug": False,
            # curriculum extras
            "emulation_speed": self.emulation_speed,
            "use_structured_obs": self.use_structured_obs,
            "use_screens": self.use_screens,
            "history_len": self.history_len,
            "state_store_root": self.state_store_root,
            "target_milestone": self.target_milestone,
            "start_state_strategy": self.start_state_strategy,
            "save_success_states": self.save_success_states,
            "success_state_min_reward": self.success_state_min_reward,
            "end_on_target": self.end_on_target,
            "end_on_blackout": self.end_on_blackout,
            "use_scripted_helpers": self.use_scripted_helpers,
            "auto_advance_dialogue": self.auto_advance_dialogue,
            "dialogue_taps": self.dialogue_taps,
            "reward_config": self.reward,
        }
