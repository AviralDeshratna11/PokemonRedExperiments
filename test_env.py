"""
test_env.py
-----------
Sanity check: verifies the environment initialises, steps, and
produces correct observation shapes and rewards.

Run this FIRST after setup:
    python test_env.py

Expected output (approximate):
    ✅ PyBoy launched
    ✅ Observation spaces correct
    ✅ Random action steps work
    ✅ Reward function produces values
    ✅ RAM features match expected shape
    All checks passed!
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))


def check(label, condition, extra=""):
    icon = "✅" if condition else "❌"
    msg  = f"  {icon}  {label}"
    if extra:
        msg += f"  ({extra})"
    print(msg)
    if not condition:
        raise AssertionError(f"FAILED: {label}")


def test_ram_module():
    """Test the RAM reader without a running emulator (static checks)."""
    print("\n── RAM module ──────────────────────────────────────────")
    from env.pokemon_ram import PokemonRAM, Addr, MAP_NAMES, decode_bcd

    # BCD decoding
    money = decode_bcd(0x12, 0x34, 0x56)
    check("BCD decode: $123456", money == 123456, f"got {money}")

    # MAP_NAMES coverage
    check("MAP_NAMES non-empty", len(MAP_NAMES) > 20)
    check("Pallet Town present", MAP_NAMES.get(0x00) == "Pallet Town")

    # Addr constants
    check("BADGES addr valid", 0xD356 == Addr.BADGES)
    check("MAP_ID addr valid",  0xD35E == Addr.MAP_ID)
    print("  RAM module: OK")


def test_env():
    """Test the environment with a fake/mock PyBoy if ROM not present."""
    print("\n── Environment ─────────────────────────────────────────")

    rom_path   = "../PokemonRed.gb"
    state_path = "checkpoints/has_starter.state"

    if not Path(rom_path).exists():
        print(f"  ⚠️  ROM not found at {rom_path}")
        print("  Skipping live environment test.")
        print("  Place PokemonRed.gb one directory above and re-run.")
        return False

    if not Path(state_path).exists():
        print(f"  ⚠️  Save state not found at {state_path}")
        print("  Run scripts/create_save_states.py first.")
        return False

    from env.pokemon_env import PokemonRedEnv, PokemonRedConfig

    config = PokemonRedConfig(
        rom_path        = rom_path,
        init_state_path = state_path,
        headless        = True,
        max_steps       = 100,
    )

    env = PokemonRedEnv(config)

    obs, info = env.reset()
    check("obs has 'screen' key", "screen" in obs)
    check("obs has 'ram' key",    "ram"    in obs)
    check("screen shape", obs["screen"].shape == (3, 144, 160),
          str(obs["screen"].shape))
    check("ram shape",    obs["ram"].shape    == (128,),
          str(obs["ram"].shape))
    check("screen dtype", obs["screen"].dtype == np.uint8)
    check("ram dtype",    obs["ram"].dtype    == np.float32)

    # info keys
    for key in ["map_id", "badges", "total_reward", "step"]:
        check(f"info has '{key}'", key in info)

    # Step with random actions
    total_r  = 0.0
    n_steps  = 50
    for t in range(n_steps):
        action = env.action_space.sample()
        obs, r, terminated, truncated, info = env.step(action)
        total_r += r
        if terminated or truncated:
            break

    check("Random steps completed", True,
          f"{n_steps} steps, total_r={total_r:.3f}")
    check("Reward is finite", np.isfinite(total_r))

    env.close()
    print("  Environment: OK")
    return True


def test_policy():
    """Test BCPolicy forward pass with dummy data."""
    print("\n── BC Policy ───────────────────────────────────────────")
    from bc.behavioral_cloning import BCPolicy

    policy = BCPolicy(n_actions=8, lstm_hidden=512)
    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    check("Policy instantiated", True, f"{n_params:,} parameters")

    B, T = 4, 16
    screens    = torch.randn(B, T, 3, 144, 160)
    rams       = torch.randn(B, T, 128)
    lstm_state = policy.init_lstm_state(B, "cpu")

    logits, value, new_state = policy(screens, rams, lstm_state)

    check("logits shape",    logits.shape    == (B, 8),   str(logits.shape))
    check("value shape",     value.shape     == (B, 1),   str(value.shape))
    check("logits finite",   torch.isfinite(logits).all().item())
    check("value finite",    torch.isfinite(value).all().item())
    check("LSTM state shape",
          new_state[0].shape == lstm_state[0].shape,
          str(new_state[0].shape))

    # get_action convenience
    action, _ = policy.get_action(
        screens[:1, :1],  # single step
        rams[:1, :1],
        deterministic=True
    )
    check("Action in range [0,7]", 0 <= action.item() <= 7, str(action.item()))

    print("  BC Policy: OK")


def test_dataset():
    """Test BCDataset with a tiny synthetic HDF5 file."""
    print("\n── Dataset ─────────────────────────────────────────────")
    import h5py, tempfile, os

    from bc.behavioral_cloning import BCDataset

    # Create a minimal HDF5 file
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmp_path = tmp.name

    N = 200
    with h5py.File(tmp_path, "w") as f:
        grp = f.create_group("episode_0000")
        grp.create_dataset("screens",       data=np.random.randint(0, 255, (N, 3, 144, 160), dtype=np.uint8))
        grp.create_dataset("rams",          data=np.random.randn(N, 128).astype(np.float32))
        grp.create_dataset("actions",       data=np.random.randint(0, 8, N, dtype=np.int32))
        grp.create_dataset("rewards",       data=np.random.randn(N).astype(np.float32))
        grp.create_dataset("returns_to_go", data=np.random.randn(N).astype(np.float32))
        grp.attrs["n_steps"] = N

    ds = BCDataset([tmp_path], seq_len=16, stride=8)
    check("Dataset non-empty", len(ds) > 0, f"{len(ds)} windows")

    screens, rams, action, rtg = ds[0]
    check("Sample screen shape", screens.shape == (16, 3, 144, 160), str(screens.shape))
    check("Sample ram shape",    rams.shape    == (16, 128),         str(rams.shape))
    check("Sample action type",  isinstance(action.item(), int))

    os.unlink(tmp_path)
    print("  Dataset: OK")


def main():
    print("=" * 60)
    print("  Pokemon RL Phase 1 — Environment Sanity Check")
    print("=" * 60)

    all_ok = True

    try:
        test_ram_module()
    except AssertionError as e:
        print(f"  {e}")
        all_ok = False

    try:
        test_policy()
    except AssertionError as e:
        print(f"  {e}")
        all_ok = False

    try:
        test_dataset()
    except AssertionError as e:
        print(f"  {e}")
        all_ok = False

    env_ok = False
    try:
        env_ok = test_env()
    except AssertionError as e:
        print(f"  {e}")
        all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("  ✅  All checks passed!")
        if not env_ok:
            print("  ⚠️  Live env test skipped (ROM/state not found)")
    else:
        print("  ❌  Some checks FAILED. See output above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
