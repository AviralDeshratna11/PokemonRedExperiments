"""
train_full_game_curriculum.py -- staged / curriculum PPO training driver.

Builds ``CurriculumRedGymEnv`` (structured obs + milestone rewards + anti-loop),
vectorizes it, and trains a Stable-Baselines3 PPO policy. Supports two execution
modes and two training shapes.

Execution modes
---------------
* ``--mode headless`` (default): no emulator window, ``SubprocVecEnv``, many parallel
  envs at unlimited speed -- the fast path for real training.
* ``--mode visual``: a single SDL2 window via ``DummyVecEnv`` so you can watch the
  agent. ``--speed`` throttles the emulator (0 = unlimited).

Training shapes
---------------
* Default (no ``--staged``): one shared PPO policy trained on the *whole* curriculum
  reward at once. Every milestone still grants its one-time bonus, so this is already
  curriculum-by-reward-shaping and a good baseline.
* ``--staged``: walk an ordered list of milestones. Each stage warm-starts episodes
  from the previous milestone's saved success states, trains the shared policy until a
  per-stage timestep budget (or success-rate threshold) is hit, saves the best model
  per milestone (``<session>/poke_<key>.zip``), and advances. Successful emulator
  snapshots are collected automatically by the env into ``state_store_root``.

Examples
--------
    # fast headless training, 16 parallel envs, unlimited speed (whole-game reward)
    python v2/train_full_game_curriculum.py --mode headless --num-envs 16 --speed 0

    # watch a single agent
    python v2/train_full_game_curriculum.py --mode visual --speed 3

    # explicit staged curriculum across all milestones
    python v2/train_full_game_curriculum.py --staged --num-envs 16

    # train one specific milestone, warm-starting from earlier success states
    python v2/train_full_game_curriculum.py --target-milestone beat_brock --num-envs 8

ROM note: this script never includes or downloads a ROM. Place your legally obtained
``PokemonRed.gb`` at the repository root (one level above ``v2/``).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Run with cwd = v2/ so the base env's relative opens (events.json, ../init.state,
# ../PokemonRed.gb) resolve regardless of where the script was launched from.
_HERE = Path(__file__).resolve().parent
os.chdir(_HERE)
sys.path.insert(0, str(_HERE))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, CallbackList,
)

from curriculum.config import CurriculumConfig
from curriculum.milestones import build_default_milestones
from curriculum.tb_callback import CurriculumTensorboardCallback
from curriculum_env import CurriculumRedGymEnv


# --------------------------------------------------------------------------- #
# Env factory                                                                  #
# --------------------------------------------------------------------------- #
def make_env(rank: int, env_config: dict, seed: int = 0):
    def _init():
        env = CurriculumRedGymEnv(env_config)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


def build_vec_env(env_config: dict, num_envs: int, mode: str):
    if mode == "visual":
        # single windowed env to actually watch the agent
        return DummyVecEnv([make_env(0, env_config)])
    if num_envs <= 1:
        return DummyVecEnv([make_env(0, env_config)])
    return SubprocVecEnv([make_env(i, env_config) for i in range(num_envs)])


# --------------------------------------------------------------------------- #
# Callback: log milestone progress + optional stage early-stop                 #
# --------------------------------------------------------------------------- #
class MilestoneCallback(BaseCallback):
    """Logs milestone progress and (optionally) signals stage completion.

    Reads the ``info`` dicts emitted by ``CurriculumRedGymEnv.step``. When training a
    specific ``target_milestone``, tracks the rolling fraction of finished episodes
    that reached it and stops the stage once it clears ``success_threshold``.
    """

    def __init__(self, target_milestone=None, success_threshold: float = 0.8,
                 window: int = 50, verbose: int = 0):
        super().__init__(verbose)
        self.target = target_milestone
        self.success_threshold = success_threshold
        self.window = window
        self.recent_success = []
        self.stage_done = False

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for info, done in zip(infos, dones):
            if not done:
                continue
            self.logger.record("curriculum/milestones_completed",
                               info.get("milestones_completed", 0))
            self.logger.record("curriculum/milestone_progress",
                               info.get("milestone_progress", 0.0))
            if self.target is not None:
                reached = 1.0 if info.get("reached_target", False) else 0.0
                self.recent_success.append(reached)
                self.recent_success = self.recent_success[-self.window:]
                rate = sum(self.recent_success) / max(len(self.recent_success), 1)
                self.logger.record("curriculum/target_success_rate", rate)
                if (len(self.recent_success) >= self.window
                        and rate >= self.success_threshold):
                    self.stage_done = True
        # returning False stops training -> ends this stage early
        return not self.stage_done


# --------------------------------------------------------------------------- #
# Model helpers                                                                #
# --------------------------------------------------------------------------- #
def make_or_load_model(vec_env, sess_path: Path, n_steps: int, resume: str = "",
                       seed: int = 0):
    if resume and Path(resume + ".zip").exists():
        print(f"[train] loading model from {resume}.zip")
        model = PPO.load(resume, env=vec_env)
        model.n_steps = n_steps
        model.n_envs = vec_env.num_envs
        model.rollout_buffer.buffer_size = n_steps
        model.rollout_buffer.n_envs = vec_env.num_envs
        model.rollout_buffer.reset()
        return model
    return PPO(
        "MultiInputPolicy", vec_env, verbose=1,
        n_steps=n_steps, batch_size=512, n_epochs=3,
        gamma=0.998, gae_lambda=0.95, ent_coef=0.01,
        tensorboard_log=str(sess_path), seed=seed,
    )


def train_one_stage(cfg: CurriculumConfig, args, target_milestone, resume_path,
                    timesteps, sess_path: Path):
    """Build env, train (optionally to a success threshold), return best model path."""
    cfg.target_milestone = target_milestone
    if target_milestone is not None:
        cfg.end_on_target = True  # focus episodes on hitting the target

    env_config = cfg.to_env_config()
    label = target_milestone or "fullgame"
    print(f"\n=== Stage: {label} | timesteps={timesteps} | envs={args.num_envs} "
          f"| mode={args.mode} ===")
    print(env_config)

    vec_env = build_vec_env(env_config, args.num_envs, args.mode)
    n_steps = args.n_steps  # per-env steps per update (SB3 multiplies by n_envs)

    model = make_or_load_model(vec_env, sess_path, n_steps, resume=resume_path,
                               seed=args.seed)
    print(model.policy)

    ckpt = CheckpointCallback(save_freq=max(timesteps // (4 * vec_env.num_envs), 1),
                              save_path=str(sess_path),
                              name_prefix=f"poke_{label}")
    milestone_cb = MilestoneCallback(target_milestone=target_milestone,
                                     success_threshold=args.success_threshold,
                                     window=args.success_window)
    callbacks = CallbackList([ckpt, CurriculumTensorboardCallback(sess_path), milestone_cb])

    model.learn(total_timesteps=timesteps, callback=callbacks,
                tb_log_name=f"ppo_{label}", reset_num_timesteps=False)

    best_path = sess_path / f"poke_{label}_best"
    model.save(str(best_path))
    print(f"[train] saved stage model -> {best_path}.zip")
    vec_env.close()
    return str(best_path)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Curriculum PPO trainer for Pokemon Red")
    p.add_argument("--config", default="configs/curriculum.yaml",
                   help="path to YAML/JSON config (default: configs/curriculum.yaml)")
    p.add_argument("--mode", choices=["headless", "visual"], default="headless")
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--speed", type=int, default=None,
                   help="emulation speed for visual mode (0 = unlimited)")
    p.add_argument("--timesteps", type=int, default=20_000_000,
                   help="total timesteps for the (single) run")
    p.add_argument("--steps-per-stage", type=int, default=2_000_000,
                   help="timesteps budget per milestone in --staged mode")
    p.add_argument("--n-steps", type=int, default=2048,
                   help="PPO per-env rollout length")
    p.add_argument("--staged", action="store_true",
                   help="walk all milestones one stage at a time (shared policy)")
    p.add_argument("--target-milestone", default=None,
                   help="train a single milestone (warm-start from earlier states)")
    p.add_argument("--milestones", default=None,
                   help="comma-separated subset of milestone keys for --staged")
    p.add_argument("--resume", default="",
                   help="path (without .zip) to a checkpoint to resume from")
    p.add_argument("--session", default=None, help="override session_path")
    p.add_argument("--success-threshold", type=float, default=0.8)
    p.add_argument("--success-window", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()

    cfg = (CurriculumConfig.from_file(args.config)
           if Path(args.config).exists() else CurriculumConfig())

    # apply CLI overrides
    cfg.headless = (args.mode == "headless")
    if args.speed is not None:
        cfg.emulation_speed = args.speed
    if args.session:
        cfg.session_path = args.session
    sess_path = Path(cfg.session_path)
    sess_path.mkdir(parents=True, exist_ok=True)

    if not Path(cfg.gb_path).exists():
        print(f"!! ROM not found at '{cfg.gb_path}'. Place your legally obtained "
              f"PokemonRed.gb there (it is never bundled). Continuing will fail.")

    if args.staged:
        # ordered milestone keys
        all_keys = [m.key for m in build_default_milestones()]
        if args.milestones:
            wanted = [k.strip() for k in args.milestones.split(",") if k.strip()]
            keys = [k for k in all_keys if k in wanted]
        else:
            keys = all_keys
        resume = args.resume
        for key in keys:
            resume = train_one_stage(cfg, args, key, resume,
                                     args.steps_per_stage, sess_path)
        print("\n[train] staged curriculum complete.")
    else:
        # single run: either a focused target or the whole-game shared policy
        train_one_stage(cfg, args, args.target_milestone, args.resume,
                        args.timesteps, sess_path)


if __name__ == "__main__":
    main()
