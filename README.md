# Orbit Wars LightZero

A complete reinforcement learning, MCTS, and exact-physics engineering stack for **Kaggle Orbit Wars** (`kaggle_environments` `orbit_wars`), integrated with [OpenDILab LightZero](https://github.com/opendilab/LightZero) (`AlphaZero`, `MuZero`, and `Gumbel MuZero`).

## Architecture Overview

| Module | Purpose |
| --- | --- |
| `orbit_wars_bot/physics.py` | Exact fleet speed scaling, continuous swept-pair collision, sun avoidance, orbital & comet trajectory prediction (with the exact `current_step + dt - 1` phase offset), intercept solver, fleet arrival predictor, and planet future-state simulator. |
| `orbit_wars_bot/fast_sim.py` | Fast, cheaply cloneable Orbit Wars engine (`FastOrbitSim`) with 100% tick-for-tick parity against `kaggle_environments.make('orbit_wars')`, including deterministic comet spawning, `from_observation()`, `clone()`, and compact serialization. |
| `orbit_wars_bot/features.py` | Player-canonicalized `(12, 16, 16)` spatial tensor encoder (180° rotational symmetry for Player 1) and fast scalar heuristic state evaluator in `[-1, 1]`. |
| `orbit_wars_bot/macro_actions.py` | 16 Galcon/Orbit Wars strategic directives + legal `action_mask` mapped to verified, sun-safe launch orders. |
| `orbit_wars_bot/heuristic_bot.py` | Strong rule-based baseline bot (threat defense, doomed-planet evacuation, vulture sniping, ROI neutral/orbital/comet expansion, coordinated strikes, and frontline feeding). |
| `orbit_wars_bot/lightzero_env.py` | DI-engine `BaseEnv` registered as `@ENV_REGISTRY.register('orbit_wars')` supporting both `play_with_bot_mode` and `self_play_mode` (simultaneous moves cleanly sequentialized into half-turns). |
| `orbit_wars_bot/lightzero_compat.py` | Automatic `external/LightZero` path wiring, Windows pure-Python ctree import guards, and `ptree_az.MCTS.get_next_actions_batch` compatibility patch. |
| `orbit_wars_bot/configs/` | Ready-to-run config factories for AlphaZero (`ptree`), MuZero (`ptree`), and Gumbel MuZero (`ctree`). |
| `orbit_wars_bot/mcts_agent.py` & `submission.py` | Official Kaggle agent loading trained AlphaZero/MuZero checkpoints with time-budgeted online PUCT MCTS under the 500ms/turn limit. |
| `scripts/train_lightzero.py` | CLI entry point for AlphaZero, MuZero, and Gumbel MuZero training (+ `--smoke-test`). |
| `scripts/evaluate.py` | Side-balanced head-to-head benchmark harness with win rates, score margins, and turn latency percentiles. |
| `scripts/export_replay.py` | Runs official Kaggle matches and exports interactive `.html` or `.json` replays. |

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
# Fast CPU smoke test
python scripts/train_lightzero.py --algo alphazero --smoke-test

# AlphaZero self-play (pure-Python ptree — works on Windows & Linux)
python scripts/train_lightzero.py --algo alphazero --mode self_play --max-env-step 50000

# MuZero vs heuristic baseline
python scripts/train_lightzero.py --algo muzero --mode bot --bot heuristic --max-env-step 50000

# Gumbel MuZero (Linux/WSL2 with compiled C++ ctree)
python scripts/train_lightzero.py --algo gumbel_muzero --mode bot --ctree --max-env-step 50000
```

### Benchmark & Export Replays

```bash
# Benchmark heuristic and online MCTS agents against random and starter
python scripts/evaluate.py --candidate heuristic --opponents random starter --games 10
python scripts/evaluate.py --candidate mcts --opponents random starter heuristic --games 6 --sims 16

# Benchmark with a trained LightZero checkpoint
python scripts/evaluate.py --candidate mcts --checkpoint path/to/ckpt_best.pth.tar --opponents starter heuristic --games 6

# Export an interactive HTML replay
python scripts/export_replay.py --p0 mcts --p1 starter --out replays/match.html
```

### Kaggle Submission

`submission.py` exposes `agent(obs, config)`. Place a trained LightZero checkpoint at `checkpoints/ckpt_best.pth.tar` (or set `ORBIT_WARS_CHECKPOINT=/path/to/ckpt.pth.tar`); if no checkpoint is present, `OnlineMCTSAgent` runs heuristic-guided PUCT MCTS over `FastOrbitSim` within a strict 350ms safety budget (well inside Kaggle's 500ms turn limit).
