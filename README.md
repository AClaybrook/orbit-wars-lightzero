# Orbit Wars LightZero

A complete reinforcement learning, MCTS, and exact-physics engineering stack for **Kaggle Orbit Wars** (`kaggle_environments` `orbit_wars`), integrated with [OpenDILab LightZero](https://github.com/opendilab/LightZero) (`AlphaZero`, `MuZero`, and `Gumbel MuZero`).

> **Galcon** (previously under `galcon/`) has moved to its own repo: [`AClaybrook/galcon`](https://github.com/AClaybrook/galcon).

## Architecture Overview

| Module | Purpose |
| --- | --- |
| `orbit_wars_bot/physics.py` | Exact fleet speed scaling, continuous swept-pair collision, sun avoidance, orbital & comet trajectory prediction (with the exact `current_step + dt - 1` phase offset), intercept solver, fleet arrival predictor, and planet future-state simulator. |
| `orbit_wars_bot/fast_sim.py` | Fast, cheaply cloneable Orbit Wars engine (`FastOrbitSim`) with 100% tick-for-tick parity against `kaggle_environments.make('orbit_wars')`, including deterministic comet spawning, `from_observation()`, `clone()`, and compact serialization. |
| `orbit_wars_bot/features.py` | Player-canonicalized `(12, 16, 16)` spatial tensor encoder (180° rotational symmetry for Player 1) and 2P/4P FFA-aware scalar state evaluator in `[-1, 1]`. |
| `orbit_wars_bot/macro_actions.py` | 16 discrete tactical macro-actions & full-turn strategic stances with per-opponent 4P tracking, static/orbiting durability scoring, unified neutral/enemy ROI allocation, atomic defense, and short-hop consolidation. |
| `orbit_wars_bot/heuristic_bot.py` | Strong 2P & 4P FFA rule-based baseline bot (atomic threat defense, last-tick doomed evacuation, neutral & enemy-vs-enemy vulture sniping, unified ROI expansion/attacks, and anti-ping-pong consolidation). |
| `orbit_wars_bot/lightzero_env.py` | DI-engine `BaseEnv` registered as `@ENV_REGISTRY.register('orbit_wars')` supporting both `play_with_bot_mode` and `self_play_mode` (simultaneous moves cleanly sequentialized into half-turns). |
| `orbit_wars_bot/lightzero_compat.py` | Automatic `external/LightZero` path wiring, Windows pure-Python ctree import guards, and `ptree_az.MCTS.get_next_actions_batch` compatibility patch. |
| `orbit_wars_bot/configs/` | Ready-to-run config factories for AlphaZero (`ptree`), MuZero (`ptree`), and Gumbel MuZero (`ctree`). |
| `orbit_wars_bot/mcts_agent.py` & `submission.py` | Official Kaggle 2P & 4P agent loading trained AlphaZero/MuZero checkpoints with active-opponent PUCT MCTS under the 500ms/turn limit. |
| `scripts/pretrain_alphazero.py` | Fast CPU warm-start trainer generating 2P & 4P `FastOrbitSim` games to train an official LightZero `AlphaZeroModel` checkpoint (`checkpoints/ckpt_best.pth.tar`). |
| `scripts/train_lightzero.py` | CLI entry point for AlphaZero, MuZero, and Gumbel MuZero RL training (+ `--smoke-test`, `--model-path`). |
| `scripts/fetch_opponents.py` | Downloads open-source community benchmark bots into gitignored `external/opponents/`. |
| `scripts/evaluate.py` | Slot-balanced 2P & 4P FFA head-to-head benchmark harness with win rates, average rank, and turn latency percentiles. |
| `scripts/export_replay.py` | Runs official 2P or 4P Kaggle matches and exports interactive `.html` or `.json` replays. |

## Setup

### 1. Python Environment (Python 3.11+ required by `kaggle-environments==1.33.3`)

```bash
python -m venv .venv
# Linux / WSL2 / macOS:
source .venv/bin/activate
# Windows PowerShell:
# .\.venv\Scripts\Activate.ps1

pip install --upgrade pip wheel setuptools cython
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
# On Windows, if kaggle-environments pulls orbax-checkpoint and hits MAX_PATH:
# pip install --no-deps kaggle-environments==1.33.3
```

### 2. Clone & Wire LightZero

```bash
git clone --depth 1 https://github.com/opendilab/LightZero.git external/LightZero
```

`orbit_wars_bot` automatically adds `external/LightZero` to `sys.path` via `setup_lightzero()`. Optionally add a `.pth` file pointing to `external/LightZero` in your venv's `site-packages`.

On **Linux / WSL2** (recommended for Gumbel MuZero and fast C++ MCTS), compile LightZero's Cython extensions:

```bash
cd external/LightZero
python -c "import setuptools, sys; sys.argv=['setup.py', 'build_ext', '--inplace']; exec(open('setup.py').read(), {'__file__': 'setup.py', '__name__': '__main__'})"
```

On **Windows**, `AlphaZero` and `MuZero` run natively with pure-Python MCTS (`--ptree`, the default).

## Usage

### Run Test Suite (including exact Kaggle tick-for-tick parity & LightZero smoke tests)

```bash
pytest -v
```

### Train with LightZero

```bash
# Fast CPU AlphaZero warm-start on 2P & 4P FastOrbitSim games -> checkpoints/ckpt_best.pth.tar
python scripts/pretrain_alphazero.py --games-2p 60 --games-4p 30 --epochs 18

# Fast CPU smoke test of full LightZero RL loop
python scripts/train_lightzero.py --algo alphazero --smoke-test

# AlphaZero self-play (pure-Python ptree — works on Windows & Linux)
python scripts/train_lightzero.py --algo alphazero --mode self_play --model-path checkpoints/ckpt_best.pth.tar --max-env-step 50000

# MuZero vs heuristic baseline
python scripts/train_lightzero.py --algo muzero --mode bot --bot heuristic --max-env-step 50000

# Gumbel MuZero (Linux/WSL2 with compiled C++ ctree)
python scripts/train_lightzero.py --algo gumbel_muzero --mode bot --ctree --max-env-step 50000
```

### Benchmark & Export Replays

```bash
# Download open-source community benchmark bots (into gitignored external/opponents/)
python scripts/fetch_opponents.py

# Benchmark heuristic and online MCTS agents (2P 1v1 or 4P FFA)
python scripts/evaluate.py --candidate heuristic --opponents random starter kvatsa5 wd_heuristic wd_bestv6 --players 2 --games 8
python scripts/evaluate.py --candidate heuristic --opponents starter kvatsa5 wd_bestv6 --players 4 --games 8
python scripts/evaluate.py --candidate mcts --opponents starter heuristic wd_bestv6 --players 2 --games 6 --sims 16

# Benchmark with a trained LightZero checkpoint
python scripts/evaluate.py --candidate mcts --checkpoint checkpoints/ckpt_best.pth.tar --opponents heuristic wd_bestv6 --games 6

# Export an interactive HTML replay (2P or 4P FFA)
python scripts/export_replay.py --p0 heuristic --p1 wd_bestv6 --p2 wd_bestv6 --p3 wd_bestv6 --out replays/match_4p.html
```

### Kaggle Submission

`submission.py` exposes `agent(obs, config)`. Place a trained LightZero checkpoint at `checkpoints/ckpt_best.pth.tar` (or set `ORBIT_WARS_CHECKPOINT=/path/to/ckpt.pth.tar`); if no checkpoint is present, `OnlineMCTSAgent` runs active-opponent PUCT MCTS over `FastOrbitSim` within a strict 220ms safety budget (well inside Kaggle's 500ms turn limit).
