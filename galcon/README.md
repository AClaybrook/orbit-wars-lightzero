# Galcon AI Gen 2 — PlanetTimelineTransformer + Self-Play League RL + `gbots` Adapter

A competitive, self-contained Galcon AI stack inside `AClaybrook/orbit-wars-lightzero` that cross-pollinates our Orbit Wars **`PlanetTimelineTransformer`** (`orbit_wars_bot/planet_transformer.py`) directly to classic Galcon and Phil Hassey's official **`gbots`** protocol — pushed far beyond heuristic ceilings via **AlphaZero Search Self-Play Distillation + 608-Episode Teacher-Free League PPO**.

## Gen 2 Highlights
- **Scaled `1.26M` Entity-Transformer (`GalconPlanetTransformer`)**: Wraps `PlanetTimelineTransformer` (`d_model=160, n_heads=4, n_layers=5, dim_ff=384`) with:
  - Enriched D4-invariant rush-defense & exact-capture features (`22` planet dims, `20×8` exact arrival timelines, `12` edge dims)
  - Auxiliary **`t+10` future-state world-model head** (`97.3%` future planet ownership accuracy; zero inference overhead)
  - **Native 8× D4-invariant mid-flight `/REDIR` head** trained on rollout-verified rescues of doomed attacks, over-reinforcements, and emergency defenses
- **Multi-Style Sparring Suite**: `GalconRusherBot` (`all_in`, `blitz`, `adaptive`), `GalconClassicBot`, `GalconTacticalBot`, and `GalconMacroBot`.
- **Teacher-Free Self-Play RL Flywheel**:
  1. Multi-style winner-filtered Warm-Start (with balanced win/loss value + `t+10` world model + `/REDIR` supervision)
  2. Two rounds of **AlphaZero Policy-Proposed Search Distillation** (`95` search games, Boltzmann soft targets, batched leaf evals — `+135.8 Elo` over v1)
  3. **608 episodes of Teacher-Free League PPO** (growing Historical Snapshot Pool + Moving EMA Teacher KL + GAE + joint target & continuous send-fraction RL + `-0.35` stalemate draw penalty — **`+187.1 Elo` more** with zero teacher!)
- **Fast Batched Lookahead Search (`GalconSearchTransformerAgent`)**: Proposes 8–10 policy candidates (threshold sweep, fraction scales, top-3 entropy flips, rush-guard hold, `/REDIR` variants) and evaluates all rollout leaves in ONE batched forward pass (`67.9 ms/tick`).
- **Full `gbots pipe` Compatibility**: Verified live emission of `/SEND`, native `/REDIR`, and `/TOCK`.

## Verified Bradley-Terry Elo Ladder (Anchored at `GalconClassicBot = 1200.0`)

Computed from 300+ held-out multi-map games (`8–22` planets, `240×150` to `400×240`, balanced P1/P2 sides):

| Rank | Agent | Bradley-Terry Elo | Δ vs. v1 Pure | Mean Latency | Highlight |
|---:|---|---:|---:|---:|---|
| 1 | **`GalconPT-Gen2 + Search` (1.26M, 8× D4)** | **`1643.8`** | `+338.7` | `67.9 ms/tick` | **`47–0–1` (`97.9%` WR, 0 losses)** vs. Classic / Rushers / Macro |
| 2 | **`GalconPT-Gen2 Pure` (1.26M, 8× D4 + `/REDIR`)** | **`1628.0`** | **`+322.9`** | **`12.3 ms/tick`** | **`102–9–9` (`85.0%` WR / `88.8%` score)**; beats Search `5–3–4` H2H! |
| 3 | `GalconPT-Gen2 EI-R2` (Search Distill, No PPO) | `1440.9` | `+135.8` | `12.3 ms/tick` | Pre-RL checkpoint (`85/120` = `70.8%`) |
| 4 | `GalconRusherBot (All-In)` | `1320.5` | `+15.4` | `<2 ms/tick` | Beats v1 Pure `9–3–0` (`75%`); crushed `19–1–0` (`95%`) by Gen 2 Pure |
| 5 | `GalconPT-v1 Pure` (0.59M, 8× D4) | `1305.1` | `0.0` | `10.1–17.1 ms` | v1 baseline (60 BC + 25 DAgger + 64 PPO eps) |
| 6 | `GalconRusherBot (Adaptive)` | `1280.9` | `-24.2` | `<2 ms/tick` | Crushed `16–1–3` by Gen 2 Pure, `12–0–0` by Gen 2 + Search |
| 7 | `GalconTacticalBot` (Analytical Solver) | `1221.9` | `-83.2` | `<2 ms/tick` | Gen 2 Pure wins `15–2–3` (`75.0%` WR / `82.5%` score) |
| 8 | `GalconClassicBot` (Anchor) | `1200.0` | `-105.1` | `<1 ms/tick` | Gen 2 Pure `18–1–1` (`90%`), Gen 2 + Search `12–0–0` (`100%`) |
| 9 | `GalconMacroBot` (Boomer) | `1098.3` | `-206.8` | `<2 ms/tick` | Gen 2 Pure `19–0–1` (`95%`), Gen 2 + Search `11–0–1` |

### Held-Out Matchup Matrix (8–22 Planets, `240×150` to `400×240`)

| Agent | vs. `Tactical` | vs. `Classic` | vs. `Rusher (All-In)` | vs. `Rusher (Adapt)` | vs. `Macro` | vs. `v1 Pure` | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`GalconPT-Gen2 Pure` (12.3 ms)** | **`15–2–3`** (`75%`) | **`18–1–1`** (`90%`) | **`19–1–0`** (`95%`) | **`16–1–3`** (`80%`) | **`19–0–1`** (`95%`) | **`15–4–1`** (`75%`) | **`102–9–9` (`85.0%` WR / `88.8%` score)** |
| **`GalconPT-Gen2 + Search` (67.9 ms)** | `8–3–1` (`67%`) | **`12–0–0`** (`100%`) | **`12–0–0`** (`100%`) | **`12–0–0`** (`100%`) | **`11–0–1`** (`92%`) | `6–3–3` vs `v1+S` | **`61–6–5` (`84.7%` WR / `88.2%` score)** |
| `GalconPT-Gen2 EI-R2` (Pre-PPO) | `14–3–3` (`70%`) | `17–3–0` (`85%`) | `12–6–2` (`60%`) | `11–5–4` (`55%`) | `19–0–1` (`95%`) | `12–7–1` (`60%`) | `85–24–11` (`70.8%` WR) |
| `GalconPT-v1 Pure` (0.59M) | `4–7–1` (`33%`) | `8–4–0` (`67%`) | `3–9–0` (`25%`) | `8–4–0` (`67%`) | `9–1–2` (`75%`) | — | `32–25–3` (`53.3%` WR) |

> **Search Inflection Point Reached:** In direct head-to-head play, **`GalconPT-Gen2 Pure` (`12.3 ms/tick`) beat `GalconPT-Gen2 + Search` (`67.9 ms/tick`) `5–3–4` (`58.3%` score)** — demonstrating that 608 episodes of full-game League PPO internalized longer-horizon strategic play than 12-tick idle-opponent rollouts, at 5.5× lower latency.

## Quickstart

### 1. Train & Evaluate Locally (Modular Gen 2 Flywheel)
```bash
# Full Gen 2 flywheel (Warm-Start -> AlphaZero Search EI -> League PPO -> Gauntlet)
python3 -m galcon.train --mode gen2 --bc-games 60 --search-games 45 --search-rounds 2 --ppo-iters 26 --ppo-eps 16

# Or resume modular stages individually:
python3 -m galcon.train --mode gen2 --stage ei --resume checkpoints/galcon_gen2_ei_r1.pt --search-games 50 --out checkpoints/galcon_gen2_ei_r2.pt
python3 -m galcon.train --mode gen2 --stage ppo --resume checkpoints/galcon_gen2_ei_r2.pt --ppo-iters 14 --ppo-eps 16 --out checkpoints/galcon_gen2_ppo.pt
python3 -m galcon.train --mode gen2 --stage eval --out checkpoints/galcon_transformer_best.pt

# Record showcase replays & Elo viewer:
python3 -m galcon.record_replays --checkpoint checkpoints/galcon_transformer_best.pt --out-html docs/galcon_viewer.html

# Run unit tests:
pytest -v tests/test_galcon.py
```

### 2. Plug Straight Into Phil Hassey's Official `gbots`
```bash
# Run our 12.3 ms Gen 2 PlanetTimelineTransformer (with native /REDIR) inside gbots:
./gbots pipe -name OurGalconGen2 -key YOUR_KEY \
  -exec "python3 -m galcon.gbots_client --bot transformer --checkpoint checkpoints/galcon_transformer_best.pt"

# Or run with batched AlphaZero lookahead search (~68 ms/tick):
./gbots pipe -name OurGalconGen2Search -key YOUR_KEY \
  -exec "python3 -m galcon.gbots_client --bot search --checkpoint checkpoints/galcon_transformer_best.pt"

# Zero-weights sparring bots (tactical / classic / rusher / macro):
./gbots pipe -name OurGalconTactical -key YOUR_KEY \
  -exec "python3 -m galcon.gbots_client --bot tactical"
```

Interactive HTML5 Canvas Replay Viewer & Elo Ladder: [`docs/galcon_viewer.html`](../docs/galcon_viewer.html).
