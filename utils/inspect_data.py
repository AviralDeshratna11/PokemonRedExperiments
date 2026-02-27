"""
inspect_data.py
---------------
Visualise and audit a gameplay recording HDF5 file.

Usage:
    python utils/inspect_data.py --file data/gameplay_session.h5
    python utils/inspect_data.py --file data/gameplay_session.h5 --episode 0 --frames 200
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent.parent))


ACTION_NAMES = ["Down", "Left", "Right", "Up", "A", "B", "Start", "Select"]
BADGE_NAMES  = ["Boulder", "Cascade", "Thunder", "Rainbow",
                "Soul", "Marsh", "Volcano", "Earth"]
EVENT_NAMES  = [
    "got_starter", "got_pokedex", "delivered_parcel", "bills_quest_done",
    "ss_anne_left", "silph_complete", "got_lapras",
    "beat_brock", "beat_misty", "beat_surge", "beat_erika",
    "beat_koga", "beat_sabrina", "beat_blaine", "beat_giovanni",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--file",    required=True)
    p.add_argument("--episode", type=int, default=0, help="Episode index to inspect")
    p.add_argument("--frames",  type=int, default=0,
                   help="Number of frames to show in grid (0=summary only)")
    p.add_argument("--save",    default=None, help="Save figure to this path")
    return p.parse_args()


def print_file_summary(h5f: h5py.File):
    episodes = sorted(k for k in h5f.keys() if k.startswith("episode_"))
    total    = h5f.attrs.get("total_steps", 0)

    print(f"\n{'='*60}")
    print(f"  HDF5 File Summary")
    print(f"{'='*60}")
    print(f"  Episodes:    {len(episodes)}")
    print(f"  Total steps: {total:,}")
    print()

    for ep_key in episodes[:20]:  # show first 20
        grp    = h5f[ep_key]
        n      = grp.attrs.get("n_steps", 0)
        total_r= grp.attrs.get("total_reward", 0)
        max_rtg= grp.attrs.get("max_rtg", 0)
        print(f"  {ep_key}: {n:6,} steps | reward: {total_r:8.2f} | max_rtg: {max_rtg:8.2f}")

    if len(episodes) > 20:
        print(f"  ... and {len(episodes)-20} more episodes.")


def inspect_episode(h5f: h5py.File, ep_idx: int, n_frames: int = 50, save: str = None):
    ep_key = f"episode_{ep_idx:04d}"
    if ep_key not in h5f:
        print(f"Episode {ep_key} not found.")
        return

    grp     = h5f[ep_key]
    n       = grp.attrs["n_steps"]
    screens = grp["screens"][:]    # (N, 3, H, W)
    rams    = grp["rams"][:]        # (N, 128)
    actions = grp["actions"][:]    # (N,)
    rewards = grp["rewards"][:]    # (N,)

    print(f"\nEpisode {ep_idx}: {n:,} steps")
    print(f"  Total reward: {rewards.sum():.2f}")
    print(f"  Action distribution:")
    for i, name in enumerate(ACTION_NAMES):
        count = (actions == i).sum()
        bar   = "█" * int(count / len(actions) * 40)
        print(f"    {name:6s} ({i}): {count:5d} ({100*count/len(actions):.1f}%) {bar}")

    # Event detection from RAM
    print(f"\n  Event flags triggered:")
    prev_events = rams[0, 37:52] > 0.5
    for t in range(1, n):
        curr_events = rams[t, 37:52] > 0.5
        for e in range(15):
            if curr_events[e] and not prev_events[e]:
                if e < len(EVENT_NAMES):
                    print(f"    Step {t:5d}: {EVENT_NAMES[e]}")
        prev_events = curr_events

    # Badge progression
    print(f"\n  Badge progression:")
    prev_badge = 0
    for t in range(n):
        badge_count = round(rams[t, 3] * 8)
        if badge_count > prev_badge:
            badge_mask = int(sum(int(rams[t, 4+b]) << b for b in range(8)))
            for b in range(8):
                if (badge_mask >> b) & 1 and not (prev_badge >> b) & 1 if False else False:
                    pass
            print(f"    Step {t:5d}: {badge_count} badges")
            prev_badge = badge_count

    # Frame grid
    if n_frames > 0:
        step_size = max(1, n // n_frames)
        frame_idxs = list(range(0, n, step_size))[:n_frames]

        cols = min(10, n_frames)
        rows = (n_frames + cols - 1) // cols

        fig = plt.figure(figsize=(cols * 2, rows * 2.2))
        fig.suptitle(f"Episode {ep_idx} — {n} steps | Total reward: {rewards.sum():.1f}",
                     fontsize=12)

        for i, t in enumerate(frame_idxs):
            ax = fig.add_subplot(rows, cols, i + 1)
            # Convert (3, H, W) to (H, W, 3)
            img = screens[t].transpose(1, 2, 0)
            ax.imshow(img)
            ax.set_title(f"t={t}\n{ACTION_NAMES[actions[t]]}\nr={rewards[t]:.2f}",
                         fontsize=7)
            ax.axis("off")

        plt.tight_layout()

        if save:
            plt.savefig(save, dpi=100, bbox_inches="tight")
            print(f"\nSaved figure to {save}")
        else:
            plt.show()

    # Reward over time
    fig2, axes = plt.subplots(3, 1, figsize=(14, 8))

    t_range = np.arange(n)

    axes[0].plot(t_range, rewards, linewidth=0.5, alpha=0.7)
    axes[0].set_title("Reward per step")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Reward")
    axes[0].axhline(0, color="red", linewidth=0.5, linestyle="--")

    cum_reward = np.cumsum(rewards)
    axes[1].plot(t_range, cum_reward, linewidth=1)
    axes[1].set_title("Cumulative reward")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Cumulative reward")

    axes[2].bar(ACTION_NAMES, [(actions == i).sum() for i in range(8)], color="steelblue")
    axes[2].set_title("Action frequency")
    axes[2].set_ylabel("Count")

    plt.tight_layout()
    if save:
        reward_save = save.replace(".png", "_rewards.png")
        plt.savefig(reward_save, dpi=100, bbox_inches="tight")
    else:
        plt.show()


def main():
    args = parse_args()

    with h5py.File(args.file, "r") as h5f:
        print_file_summary(h5f)
        inspect_episode(h5f, args.episode, n_frames=args.frames, save=args.save)


if __name__ == "__main__":
    main()
