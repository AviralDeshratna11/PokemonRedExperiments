# Curriculum / Hierarchical RL Extension (v2)

A practical hierarchical-RL / curriculum-learning system layered **on top of** the
existing `v2/` Pokémon Red environment. It splits the full game into staged
milestones, adds a structured RAM/event observation, a configurable & logged reward
system with anti-loop/anti-farming penalties, saved emulator success-states for
curriculum warm-starts, scripted menu/dialogue helpers, and an option/skill scaffold
for future hierarchical control — while keeping the original V2 scripts working
unchanged.

Nothing here includes or downloads a ROM. You must legally provide
`PokemonRed.gb` at the **repository root** (one level above `v2/`).

---

## 1. Why this exists

Pokémon Red is a long-horizon, sparse-reward game with brutal credit assignment. A
single monolithic agent struggles to reach the credits. This extension follows the
report's recommended structure:

- **Structured observation** (RAM/events first, pixels optional) → sample-efficient.
- **Milestone curriculum** → the 25-hour quest becomes ~33 small, detectable goals.
- **Dense, shaped, configurable rewards** with explicit anti-reward-hacking penalties.
- **Saved success-states** so later stages warm-start from where earlier stages ended.
- **Skill/option scaffolding** so you can grow from one PPO policy to true HRL later.

## 2. Backwards compatibility

This extension is **purely additive**. It does not modify `red_gym_env_v2.py`,
`baseline_fast_v2.py`, or any other existing file. `CurriculumRedGymEnv` *subclasses*
`RedGymEnv`. Your existing training (`python baseline_fast_v2.py`) is unaffected.

## 3. Layout

```
v2/
  curriculum/
    ram_map.py          # GameState: single source of truth for reading RAM
    structured_obs.py   # flat normalized observation vector + gym space
    milestones.py       # the ~33-milestone curriculum graph + MilestoneManager
    rewards.py          # configurable, logged reward system + anti-loop tracker
    scripted_helpers.py # deterministic dialogue/heal/shop/use macros
    skills.py           # option-policy / skill scaffold + high-level controller stub
    state_store.py      # save/load successful emulator snapshots per milestone
    config.py           # CurriculumConfig dataclasses + YAML/JSON loader
  curriculum_env.py     # CurriculumRedGymEnv (subclass of RedGymEnv) — the glue
  configs/curriculum.yaml
  train_full_game_curriculum.py   # staged PPO training driver (headless/visual)
  run_curriculum_interactive.py   # watch/evaluate (random or trained policy)
  requirements-curriculum.txt     # adds PyYAML (for YAML configs)
```

## 4. Setup

```bash
cd v2
pip install -r requirements.txt
pip install -r requirements-curriculum.txt   # PyYAML, for YAML configs
# place your legally-obtained ROM at the repo root:
#   PokemonRedExperiments/PokemonRed.gb
```

Linux/macOS are the priority; the code is OS-agnostic Python and also runs on Windows.

## 5. Running

### Headless training (default, fast)
No emulator window, `SubprocVecEnv`, many parallel envs, unlimited speed:

```bash
python train_full_game_curriculum.py --mode headless --num-envs 16 --speed 0
```

This trains a single shared PPO policy on the **whole-game** curriculum reward (every
milestone grants its one-time bonus). It's the simplest working entry point.

### Visual mode (watch one agent)
```bash
python run_curriculum_interactive.py --mode visual --speed 3
# or evaluate a trained model:
python run_curriculum_interactive.py --mode visual --model runs_curriculum/poke_fullgame_best
```

### Explicit staged curriculum
Walk milestones one stage at a time, warm-starting each stage from the previous
milestone's saved success states, saving the best model per milestone:

```bash
python train_full_game_curriculum.py --staged --num-envs 16 --steps-per-stage 2000000
# subset of stages:
python train_full_game_curriculum.py --staged --milestones get_starter,reach_pewter,beat_brock
```

### Train one milestone
```bash
python train_full_game_curriculum.py --target-milestone beat_brock --num-envs 8
```

All flags: see `python train_full_game_curriculum.py --help`.

## 6. Observation space

`CurriculumRedGymEnv` exposes a `Dict` observation:

- `structured` *(primary)* — a normalized `float32` vector (~162 dims) built by
  `StructuredObservationBuilder` from `GameState` + per-episode signals: map id,
  prev map, x/y, badges (+count), event-flag sum, party species/levels/HP/status,
  party size, total HP, battle state, enemy species/level/HP, money, poké-ball count,
  Cut/Surf/Strength ownership, HM/TM counts, key items (parcel, SS ticket, town map,
  bike), Pokédex owned/has-dex, heal/blackout/poké-center signals, stuck/loop/
  inactivity/wall-bump scores, recent action & map history, and milestone bits.
- `screens` *(optional auxiliary)* — the base 72×80×3 grayscale frame-stack. Disable
  with `use_screens: false` to train RAM-only (faster).

The builder is schema-driven and asserts its output length matches the declared
space, so the two can never silently drift.

## 7. Milestones

`curriculum/milestones.py` defines an ordered list (Start Game → Get Starter → … →
Beat Champion). Each milestone has:

- a **detector** `(GameState, ctx) -> bool` using event flags (`events.json`),
  badge state, and/or map ids (`map_data.json`),
- a one-time **reward**, a **timeout**, **allowed start states** (previous milestone's
  snapshots), a **next** pointer, and a **save_state** flag.

`MilestoneManager` latches completion (once done, stays done), maintains a visited-map
context, and reports the active "frontier" for staging.

> **Detector accuracy note.** Most detectors use precise named event flags or badge
> bits. A few dungeon-completion milestones (`complete_rocket_hideout`,
> `complete_silph_co`) use "reached the deepest known floor" as a robust proxy and are
> commented as such. Refine these with exact event flags if you want stricter gating —
> the system is built so swapping a detector is a one-line change.

## 8. Reward system & anti-reward-hacking

`curriculum/rewards.py` (`RewardConfig` + `RewardManager`) computes **incremental**
per-step rewards and keeps cumulative totals for logging. Positive: milestones
(one-time), new maps, new coordinates, story events (max-based), badges (max-based),
level growth (diminishing returns), Pokédex, healing. Penalties: blackouts, standing
on over-visited tiles (stuck), wall bumps, menu spam, region loops, inactivity, and
repeated healing without progress (PC-farming).

Key anti-farming properties:
- Milestone / badge / event / level / Pokédex rewards are **one-time or max-based** —
  they cannot be farmed by revisiting.
- Warm-started episodes call `RewardManager.baseline(gs)` so already-satisfied progress
  isn't re-rewarded on step 1.
- The `AntiLoopTracker` produces the stuck/loop/inactivity signals that feed both the
  penalties and the observation.

Every reward component is surfaced via `info["reward_components"]` and the env's
`agent_stats`, so the existing `TensorboardCallback` and `MilestoneCallback` log them.

## 9. Curriculum state store

`curriculum/state_store.py` saves PyBoy `.state` snapshots (never ROMs) under
`curriculum_states/<milestone_key>/` whenever a milestone with `save_state=True`
completes. Reward is encoded in the filename, so selection (`random`/`latest`/`best`)
is a lock-free directory scan — safe across `SubprocVecEnv` workers. Later stages
warm-start from these via `allowed_start_states`.

## 10. Scripted helpers & skills (hybrid RL)

`scripted_helpers.py` provides deterministic button macros (advance dialogue, heal at a
counter, confirm a purchase, use/teach an item) so menus never hard-block exploration.
Enable with `use_scripted_helpers: true` / `auto_advance_dialogue: true`. The RL agent
still makes the strategic/exploration decisions.

`skills.py` is forward-looking scaffolding: a `Skill` option interface,
`ScriptedSkill`/`PolicySkill` implementations, a `SkillRegistry`, and a rule-based
`HighLevelController` stub. You can start with one PPO policy today and grow into true
hierarchical control (or an LLM-guided manager) without reworking the env.

## 11. Configuration

Everything is configurable via `configs/curriculum.yaml` (or a `.json` equivalent),
loaded by `CurriculumConfig`. CLI flags override the file. See the YAML comments for
each field. To disable a reward component, set its weight to `0`.

## 12. Suggested workflow toward full-game completion

1. Start with whole-game shared-policy headless training to get an agent making
   early-game progress and **collecting success snapshots** automatically.
2. Switch to `--staged` to harden later milestones using those snapshots as warm
   starts; save the best model per milestone.
3. Tighten reward weights / detectors as you observe reward-hacking (the expected
   iterative loop: train → observe exploit → patch reward → repeat).
4. When ready, add concrete skills in `skills.py` and a learned/LLM manager for HRL.
