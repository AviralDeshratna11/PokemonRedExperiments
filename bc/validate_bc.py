"""
validate_bc.py
--------------
Load a trained BC policy checkpoint and run it in the Pokemon Red environment.
Visualises the agent's behaviour and prints key stats.

Usage:
    python bc/validate_bc.py \
        --checkpoint checkpoints/bc_best.pt \
        --state      checkpoints/has_starter.state \
        --episodes   5 \
        --render              # open SDL2 window
"""

import sys
import argparse
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from env.pokemon_env   import PokemonRedEnv, PokemonRedConfig
from bc.behavioral_cloning import BCPolicy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--state",      default="checkpoints/has_starter.state")
    p.add_argument("--rom",        default="../PokemonRed.gb")
    p.add_argument("--episodes",   type=int,   default=5)
    p.add_argument("--max_steps",  type=int,   default=10_000)
    p.add_argument("--render",     action="store_true")
    p.add_argument("--device",     default="cpu")
    return p.parse_args()


ACTION_NAMES = ["Down", "Left", "Right", "Up", "A", "B", "Start", "Select"]


def run_episode(env, policy, device, max_steps, render=False):
    obs, info = env.reset()
    lstm_state = policy.init_lstm_state(1, device)

    total_r     = 0.0
    steps       = 0
    action_hist = [0] * 8

    while steps < max_steps:
        # Prepare tensors with sequence dim = 1
        screen_t = torch.from_numpy(obs["screen"]).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
        ram_t    = torch.from_numpy(obs["ram"]).float().unsqueeze(0).unsqueeze(0).to(device)

        action, lstm_state = policy.get_action(screen_t, ram_t, lstm_state, deterministic=False)
        action_idx = action.item()

        obs, reward, terminated, truncated, info = env.step(action_idx)
        total_r        += reward
        steps          += 1
        action_hist[action_idx] += 1

        if render:
            env.render()

        if terminated or truncated:
            break

    return {
        "steps":        steps,
        "total_reward": total_r,
        "badges":       info["badges"],
        "maps_visited": len(env.episode_stats["maps_visited"]),
        "events":       env.episode_stats["events_triggered"],
        "action_hist":  action_hist,
    }


def main():
    args   = parse_args()
    device = args.device

    # Load policy
    checkpoint = torch.load(args.checkpoint, map_location=device)
    policy     = BCPolicy(n_actions=8, lstm_hidden=512)
    policy.load_state_dict(checkpoint["model_state"])
    policy.eval()
    print(f"Loaded BC policy from {args.checkpoint}")
    if "val_acc" in checkpoint:
        print(f"  Val accuracy during training: {checkpoint['val_acc']:.3f}")

    # Environment
    config = PokemonRedConfig(
        rom_path        = args.rom,
        init_state_path = args.state,
        headless        = not args.render,
        max_steps       = args.max_steps,
    )
    env = PokemonRedEnv(config, render_mode="human" if args.render else None)

    # Run episodes
    all_stats = []
    for ep in range(args.episodes):
        print(f"\n── Episode {ep+1}/{args.episodes} ──")
        stats = run_episode(env, policy, device, args.max_steps, args.render)
        all_stats.append(stats)

        print(f"  Steps:        {stats['steps']:,}")
        print(f"  Total reward: {stats['total_reward']:.2f}")
        print(f"  Badges:       {stats['badges']}/8")
        print(f"  Maps visited: {stats['maps_visited']}")
        print(f"  Events:       {stats['events']}")
        print(f"  Actions:      {dict(zip(ACTION_NAMES, stats['action_hist']))}")

    env.close()

    # Summary
    print("\n" + "=" * 50)
    print("  SUMMARY")
    print("=" * 50)
    print(f"  Avg reward:  {np.mean([s['total_reward'] for s in all_stats]):.2f}")
    print(f"  Avg badges:  {np.mean([s['badges'] for s in all_stats]):.1f}")
    print(f"  Avg steps:   {np.mean([s['steps'] for s in all_stats]):.0f}")
    print(f"  Avg maps:    {np.mean([s['maps_visited'] for s in all_stats]):.1f}")


if __name__ == "__main__":
    main()
