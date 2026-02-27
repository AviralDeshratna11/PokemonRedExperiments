"""
compute_rewards.py
------------------
Post-processing step after gameplay recording.

Takes a raw HDF5 file (screens + rams + actions) and computes the reward
signal for each step by replaying RAM features through the reward calculator.
Also computes return-to-go (discounted future reward) for each step, which
is useful for weighted behavioral cloning.

Usage:
    python bc/compute_rewards.py --file data/gameplay_session.h5

Output: adds 'rewards' and 'returns_to_go' datasets to each episode group.
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import h5py
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from env.pokemon_ram import Addr


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--file",   required=True, help="Path to HDF5 recording file")
    p.add_argument("--gamma",  type=float, default=0.999, help="Discount factor")
    return p.parse_args()


# ─── Reward computation from RAM feature vectors ─────────────────────────────

# RAM feature vector layout (matches PokemonRAM.to_feature_vector):
# [0]   map_id / 247
# [1]   x / 255
# [2]   y / 255
# [3]   badge_count / 8
# [4-11] individual badge flags
# [12-17] party HP fractions
# [18-23] party level / 100
# [24]  total_level / 600
# [25]  alive_count / 6
# [26]  in_battle
# [27]  is_trainer
# [28]  money (log-norm)
# [29-36] item flags (cut,surf,fly,flash,strength,bike,silph_scope,flute)
# [37-51] event flags (15 events)

BADGE_REWARDS    = [3.0, 3.0, 4.0, 4.0, 4.0, 5.0, 5.0, 8.0]
LEVEL_REWARD     = 0.1
EXPLORE_REWARD   = 0.005
FAINT_PENALTY    = -0.5
STEP_PENALTY     = -0.0001

EVENT_INDICES    = list(range(37, 52))  # event flags in RAM vector
EVENT_REWARDS    = [2.0, 1.0, 1.5, 2.5, 1.5, 4.0, 2.0,   # story events
                    3.0, 3.0, 4.0, 4.0, 4.0, 5.0, 5.0, 8.0]  # badges (alt)


def compute_step_rewards(rams: np.ndarray) -> np.ndarray:
    """
    Given an (N, 128) array of RAM feature vectors,
    compute an (N,) reward array by looking at deltas.
    """
    N       = len(rams)
    rewards = np.zeros(N, dtype=np.float32)

    # Previous episode state
    seen_badges  = 0
    seen_events  = np.zeros(15, dtype=bool)
    seen_levels  = rams[0, 18:24].copy()  # level/100 for each slot
    seen_alive   = rams[0, 25]            # alive fraction
    visited: set = set()

    for i in range(N):
        r = STEP_PENALTY
        v = rams[i]

        # ── Exploration ────────────────────────────────────────────────────
        map_id = int(v[0] * 247)
        x      = int(v[1] * 255)
        y      = int(v[2] * 255)
        coord  = (map_id, x, y)
        if coord not in visited:
            visited.add(coord)
            r += EXPLORE_REWARD

        # ── Badge rewards ──────────────────────────────────────────────────
        badge_count = round(v[3] * 8)
        badge_flags = v[4:12]
        cur_badge_mask = int(sum(int(badge_flags[b]) << b for b in range(8)))
        new_badges = cur_badge_mask & ~seen_badges
        for b in range(8):
            if (new_badges >> b) & 1:
                r += BADGE_REWARDS[b]
        seen_badges = cur_badge_mask

        # ── Level rewards ──────────────────────────────────────────────────
        cur_levels = v[18:24]
        for slot in range(6):
            gain = cur_levels[slot] - seen_levels[slot]
            if 0 < gain <= 0.1:   # 10 levels / 100 = 0.1 normalised
                r += gain * 100 * LEVEL_REWARD
        seen_levels = cur_levels.copy()

        # ── Faint penalty ──────────────────────────────────────────────────
        cur_alive = v[25]
        if cur_alive < seen_alive:
            r += FAINT_PENALTY
        seen_alive = cur_alive

        # ── Event rewards ──────────────────────────────────────────────────
        cur_events = v[37:52] > 0.5
        new_events = cur_events & ~seen_events
        for e in range(min(len(new_events), len(EVENT_REWARDS))):
            if new_events[e]:
                r += EVENT_REWARDS[e] * 1.0
        seen_events = cur_events

        rewards[i] = r

    return rewards


def compute_returns_to_go(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """Compute discounted returns-to-go (reverse cumsum)."""
    N   = len(rewards)
    rtg = np.zeros(N, dtype=np.float32)
    rtg[-1] = rewards[-1]
    for i in range(N - 2, -1, -1):
        rtg[i] = rewards[i] + gamma * rtg[i + 1]
    return rtg


def main():
    args = parse_args()

    print(f"Processing: {args.file}")

    with h5py.File(args.file, "a") as h5f:
        episodes = sorted(k for k in h5f.keys() if k.startswith("episode_"))
        print(f"Found {len(episodes)} episodes.")

        total_steps = 0
        total_reward = 0.0

        for ep_key in tqdm(episodes, desc="Computing rewards"):
            grp  = h5f[ep_key]
            rams = grp["rams"][:]    # (N, 128) float32

            rewards = compute_step_rewards(rams)
            rtg     = compute_returns_to_go(rewards, args.gamma)

            # Overwrite existing datasets
            if "rewards" in grp:
                del grp["rewards"]
            if "returns_to_go" in grp:
                del grp["returns_to_go"]

            grp.create_dataset("rewards",       data=rewards)
            grp.create_dataset("returns_to_go", data=rtg)

            grp.attrs["total_reward"]  = float(rewards.sum())
            grp.attrs["max_rtg"]       = float(rtg.max())

            total_steps  += len(rewards)
            total_reward += float(rewards.sum())

        h5f.attrs["total_reward"] = total_reward
        print(f"\nDone. {total_steps:,} steps | Total reward: {total_reward:.2f}")
        print(f"Avg reward per step: {total_reward / max(total_steps, 1):.4f}")


if __name__ == "__main__":
    main()
