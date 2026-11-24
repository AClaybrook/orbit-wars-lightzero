# Galcon AI Gen 3 — PlanetTimelineTransformer + Reactive 2-Ply Search + Mirror League PPO + `gbots` Adapter

A competitive, self-contained Galcon AI stack inside `AClaybrook/orbit-wars-lightzero` that cross-pollinates our Orbit Wars **`PlanetTimelineTransformer`** (`orbit_wars_bot/planet_transformer.py`) directly to classic Galcon, Phil Hassey's official **`gbots`** protocol, and head-to-head benchmarking against unmodified Galcon 2 Lua tournament bots (including Evan Sparano's `esparano/GalconModVault` **`Sparky v1.6`** and **`EsparanBot v0.2`** via [`galcon/lua_bridge.py`](lua_bridge.py)).

## Gen 3 Highlights
- **2-Ply Reactive Neural Search + Multi-Target Fan-Out (`GalconSearchTransformerAgent`)**:
  - Policy-proposed candidate search (~10 neural plans, completely avoiding raw MCTS branching explosions) upgraded with a **batched mid-rollout reactive reply** for both players, calibrated neural value weighting (`0.85`), prior-trust margin (`+0.025`), and **neural-guided multi-target fan-out candidates** (`k=3` conservative & `k=5` aggressive) for explosive simultaneous neutral openings.
  - **Ties `esparano Sparky v1.6` (`1835.2 Elo`) dead even `6–6–8` (`50.0%` score)** across 20 held-out multi-map games (**`1831.8 Elo`** — within **`3.4 Elo`** of the Lua Grandmaster!), beats `GalconPT-Gen3 Pure` `7–3–2` (`66.7%` score), goes undefeated `8–0–4` vs. Gen 2 Pure, and jumped **5× in wins** on Large `22–28` planet arenas vs. Sparky (`1–11–0` → `5–7–0`).
- **Large-Arena & Asymmetric `gbots` Generalization**:
  - Trained across compact (`8–12pl 240×150`), classic (`14–22pl 400×240`), and large (`22–28pl 520×320–600×360`, symmetric + asymmetric) arenas.
- **Reactive-Search Expert Iteration + Paired-Seed Mirror League PPO**:
  - AlphaZero Reactive-Search EI distills winning/drawing 2-ply search plans into the `1.26M` pure network while calibrating the value & `t+10` world-model heads on all outcomes.
  - Paired-Seed Mirror League PPO plays both P1 and P2 on every exact map with GRPO-style split-pair advantage centering, Historical Snapshot Pool (anchored to Gen 2 Champion), Moving EMA Teacher KL, and native `/REDIR` replay.
  - **`GalconPT-Gen3 Pure` (`12.3–22.6 ms/tick`, 8× D4 + native `/REDIR`)** reached **`1746.7 Elo`** (surpassing Gen 2 + Search while running 3–5× faster!), went **`104–5–11` (`86.7%` WR / `91.3%` score — only 5 losses in 120 games)** across our held-out gauntlet (including a **`20–0–0` sweep** vs. `ClassicBot`), tripled wins vs. `Sparky v1.6` (`2–14–4` → `6–10–4`), beat Gen 2 Pure `9–7–4` std & **`8–4–0` (`66.7%`)** on Large Arenas, and went **undefeated `10–0–2`** vs. `esparano EsparanBot v0.2`.
- **Clean External Lua Benchmark Bridge (`galcon/lua_bridge.py`)**:
  - Persistent `lua5.3` `g2` API bridge allowing fair, zero-copy head-to-head evaluation against external Galcon 2 Lua bots without committing any third-party bot code to this repository.
- **Full `gbots pipe` Compatibility**: Verified live emission of `/SEND`, native `/REDIR`, and `/TOCK`.

## Unified Bradley-Terry Elo Ladder (780 Held-Out Games • Anchor: `GalconClassicBot = 1200.0`)

| Rank | Agent | Bradley-Terry Elo | Δ vs. v1 Pure | Mean Latency | Highlight |
|---:|---|---:|---:|---:|---|
| 1 | `esparano Sparky v1.6` (Lua Grandmaster) | `1835.2` | `+440.8` | `<2 ms/tick` | Unmodified Lua bot via `lua_bridge.py`; `41–1–6` vs. heuristics |
| 2 | **`GalconPT-Gen3 + ReactiveSearch` (1.26M, 2-Ply)** | **`1831.8`** | **`+437.4`** | `129.4 ms/tick` | **`6–6–8` (`50.0%` EVEN) vs. Sparky v1.6**; `8–0–4` vs. Gen 2; `7–3–2` vs. Gen 3 Pure |
| 3 | **`GalconPT-Gen3 Pure` (1.26M, 8× D4 + `/REDIR`)** | **`1746.7`** | **`+352.3`** | **`22.6 ms/tick`** | **`104–5–11` (`86.7%` WR / `91.3%` score)**; `20–0–0` Classic; `8–4` Large Arena vs. Gen 2 |
| 4 | `GalconPT-Gen2 + Search` (1-Ply v2) | `1731.8` | `+337.4` | `67.9 ms/tick` | Surpassed by Gen 3 Pure (`22.6 ms`) at 3× lower latency |
| 5 | `GalconPT-Gen2 Pure` (1.26M, 8× D4) | `1691.5` | `+297.1` | `12–22 ms/tick` | Gen 2 champion (`102–9–9`); defeated `9–7–4` std & `8–4–0` large by Gen 3 Pure |
| 6 | `GalconPT-Gen2 EI-R2` (Pre-PPO) | `1536.2` | `+141.8` | `12–22 ms/tick` | Pre-RL search-distilled Gen 2 checkpoint |
| 7 | `GalconTacticalBot` (Analytical Solver) | `1443.9` | `+49.5` | `<2 ms/tick` | Gen 3 Pure wins `16–1–3` (`80.0%` WR / `87.5%` score) |
| 8 | `GalconRusherBot (All-In)` | `1419.9` | `+25.5` | `<2 ms/tick` | Gen 3 Pure `16–1–3`; Gen 3 + Search `8–0–4` (0 losses) |
| 9 | `GalconPT-v1 Pure` (0.59M, 8× D4) | `1394.4` | `0.0` | `10–17 ms/tick` | v1 baseline; Gen 3 Pure wins `17–2–1` (`85.0%` WR) |
| 10 | `GalconRusherBot (Adaptive)` | `1352.3` | `-42.1` | `<2 ms/tick` | Gen 3 Pure wins `16–1–3` (`80.0%` WR / `87.5%` score) |
| 11 | `esparano EsparanBot v0.2` (Lua) | `1334.9` | `-59.5` | `<2 ms/tick` | Beats Classic `10–2–0`; **Gen 3 Pure & Search undefeated `10–0–2`** |
| 12 | `GalconClassicBot` (Anchor) | `1200.0` | `-194.4` | `<1 ms/tick` | Gen 3 Pure **`20–0–0` (`100%`)**, Gen 3 + Search **`12–0–0` (`100%`)** |
| 13 | `GalconMacroBot` (Boomer) | `1101.6` | `-292.8` | `<2 ms/tick` | Gen 3 Pure **`19–0–1` (`95%` WR, 0 losses, 31.9t avg)** |

### Head-to-Head Matrix vs. `esparano` Bots & Large `22–28` Planet Arenas

| Agent | vs. `Sparky v1.6` (Std 20g) | vs. `Sparky v1.6` (Large 22–28pl) | vs. `EsparanBot v0.2` | vs. `Gen 2 Pure` (Std / Large) | 120g Held-Out Gauntlet |
|---|---:|---:|---:|---:|---:|
| **`GalconPT-Gen3 + ReactiveSearch`** | **`6–6–8` (`50.0%` sc)** | **`5–7–0` (`41.7%` WR)** | **`10–0–2` (`91.7%` sc)** | **`8–0–4`** (0 loss) | `29–2–5` (36g: Tac/Cls/Rush) |
| **`GalconPT-Gen3 Pure` (22.6 ms)** | `6–10–4` (`40.0%` sc) | `1–11–0` (`8.3%` WR) | **`10–0–2` (`91.7%` sc)** | **`9–7–4`** / **`8–4–0` (`66.7%`)** | **`104–5–11` (`86.7%` WR / `91.3%` sc)** |
| `GalconPT-Gen2 Pure` (12–22 ms) | `2–14–4` (`20.0%` sc) | `4–8–0` (`33.3%` WR) | `10–0–2` (`91.7%` sc) | — | `102–9–9` (`85.0%` WR / `88.8%` sc) |
| `GalconPT-v1 Pure` (0.59M) | `1–11–0` (`8.3%` WR) | — | — | `4–15–1` | `32–25–3` (`53.3%` WR) |

## Quickstart

### 1. Train & Evaluate Locally (Modular Gen 3 Flywheel)
```bash
# Gen 3 Reactive-Search Expert Iteration (optional --lua-opp for external Lua sparring):
python3 -m galcon.train --mode gen2 --stage ei \
  --resume checkpoints/galcon_gen2_champion.pt \
  --search-games 44 --epochs 6 --lr 2.2e-4 \
  --out checkpoints/galcon_gen3_ei_r1.pt

# Gen 3 Paired-Seed Mirror League PPO (compact + classic + large 22-28pl asymmetric maps):
python3 -m galcon.train --mode gen2 --stage mirror_ppo \
  --resume checkpoints/galcon_gen3_ei_r1.pt \
  --gen2-anchor checkpoints/galcon_gen2_champion.pt \
  --ppo-iters 10 --mirror-pairs 8 --lr 4.0e-5 \
  --out checkpoints/galcon_transformer_best.pt

# Optional: pass --lua-opp /path/to/bot_sparky_v1.6.lua (requires lua5.3 on PATH)

# Record showcase replays & Elo viewer:
python3 -m galcon.record_replays --checkpoint checkpoints/galcon_transformer_best.pt --out-html docs/galcon_viewer.html

# Run unit tests:
pytest -v tests/test_galcon.py
```

### 2. Plug Straight Into Phil Hassey's Official `gbots`
```bash
# Run our fast Gen 3 Pure PlanetTimelineTransformer (with native /REDIR) inside gbots:
./gbots pipe -name OurGalconGen3 -key YOUR_KEY \
  -exec "python3 -m galcon.gbots_client --bot transformer --checkpoint checkpoints/galcon_transformer_best.pt"

# Or run our 1831.8 Elo 2-Ply Reactive Fan-Out Search agent (~129 ms/tick):
./gbots pipe -name OurGalconGen3Search -key YOUR_KEY \
  -exec "python3 -m galcon.gbots_client --bot search --checkpoint checkpoints/galcon_transformer_best.pt"

# Zero-weights sparring bots (tactical / classic / rusher / macro):
./gbots pipe -name OurGalconTactical -key YOUR_KEY \
  -exec "python3 -m galcon.gbots_client --bot tactical"
```

Interactive HTML5 Canvas Replay Viewer & Elo Ladder: [`docs/galcon_viewer.html`](../docs/galcon_viewer.html).
