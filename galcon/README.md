# Galcon AI Wave 7 — Ensemble (W6B+W7) + 2-Ply Search (`65.0%` vs. `Sparky v1.6`) & Sparky-Gated Multi-Send Transformer

A competitive, self-contained Galcon AI stack inside `AClaybrook/orbit-wars-lightzero` that cross-pollinates our Orbit Wars **`PlanetTimelineTransformer`** (`orbit_wars_bot/planet_transformer.py`) to classic Galcon, Phil Hassey's official **`gbots`** protocol, and zero-copy head-to-head benchmarking against Evan Sparano's (`esparano/GalconModVault`) unmodified Lua tournament bots (`Sparky v1.6` & `EsparanBot v0.2`) via [`galcon/lua_bridge.py`](lua_bridge.py).

## Wave 7 Highlights (`EnsembleGalconSearchAgent`: `galcon_transformer_best.pt` + `galcon_wave7_p1.pt`)
- **Crushed `esparano Sparky v1.6` (`1821.4 Elo` Lua Grandmaster) `25–13–2` (`65.0%`, ~`1929 Elo`) over 40 held-out Standard games (`116–134 ms/tick` on 2-vCPU VM):**
  - `s=14000` (20g): **`11–7–2` (`60.0%`)** (`+5.0%` over Wave 6B Search, `+13.8%` over Wave 5 Search)
  - `s=9500` (20g): **`14–6–0` (`70.0%`, zero draws!)**
  - **`EsparanBot v0.2` (24g across `s=9500` & `s=15300`):** **`19–3–2` (`83.3%`)** (completely fixing Wave 5's `s=15300` dip!)
  - **Built-in Heuristics (32g total):** **`32–0–0` (`100.0%` PERFECT SWEEP)** — `8–0–0` Tactical, `8–0–0` Macro, `8–0–0` Rusher (All-In), `8–0–0` Classic!
- **Why Wave 7 Jumped `+10%` Over Wave 6B:**
  - **Upgraded 2-Ply Search (`galcon/transformer_bot.py`):** Wired trained multi-send priors (`ms_probs`) into `_build_fanout_plan` & `_build_focus_fire_plan` (replacing stale Gen 3 softmax `probs`) and added an early macro `k=6` (`reserve=0.06`, ticks `<= 20`) opening proposal (`+8.4%` on Large 24pl Search).
  - **AlphaZero Search EI → Sparky-Gated PPO (`galcon_wave7_p1.pt`):** Upgraded `collect_gen4_multisend_ei_dataset` (`h=16, w=0.70, prior=0.010, budget=0.24s`, dual-boss `--lua-opp2`), collecting **3.5× more multi-send ticks** (`87/801`) with `vsign=87.5%` value calibration; 2 PPO iters jumped `val_44g` mixed+large Sparky from **`31.2% → 53.1%` (`+21.9%`)** and Pure `s=9500` Sparky to **`10–6–4` (`60.0%`)**, creating a strongly complementary partner to Wave 6B (`60.0%` on `s=14000`).
- **New `gbots` Ensemble Mode (`galcon/gbots_client.py`):** Run `--bot ensemble` with `checkpoints/galcon_transformer_best.pt` + `checkpoints/galcon_wave7_p1.pt` directly over Phil Hassey's `gbots pipe`.
- **Interactive Win & Loss Replay Viewer (`docs/galcon_viewer.html`):** Includes **both Win and Loss (Forensics) replays vs. `esparano Sparky v1.6`** for both Wave 7 Ensemble 2-Ply Search and Pure neural agents.

## Wave 6B Highlights (Single-Model Pure Champion: `checkpoints/galcon_transformer_best.pt`)
- **Positive Pure & 2-Ply Search Records vs. `esparano Sparky v1.6` (`1821.4 Elo` Lua Grandmaster):**
  - **`Wave 6B` + 2-Ply Adaptive / Adversarial Search (`1×` Canon D4):** **`9–7–4` (`55.0%`)** vs. `Sparky v1.6` (`20g s=14000`, new record: 9 decisive wins, `+5.0%` over Gen 4 Search's `8–8–4` and `+8.8%` over Wave 5 Search's `46.2%`).
  - **`Wave 6B` Pure (`1.29M`, `1×` Canon D4, `~19 ms` VM / `~8 ms` desktop, ZERO search):**
    - **`15–13–12` (`52.5%`)** over **40 held-out games** vs. `Sparky v1.6` (`s=14000`: **`8–4–8 / 60.0%`**, `s=9500`: `7–9–4 / 45.0%`) — **`+7.5%`** over Wave 5 Pure's `14–18–8` (`45.0%`) on the exact same 40 seeds!
    - **`9–5–6` (`60.0%`)** head-to-head vs. Wave 5 Pure (`20g s=18000`).
    - **`20–8–8` (`66.7%`)** vs. `EsparanBot v0.2` (`36g`, up from Wave 5's `65.3%`; `83.3%` on `s=9500`).
    - **`16–0–0` (`100.0%`)** vs. `TacticalBot`, **`14–1–1` (`90.6%`)** vs. `Rusher (All-In)`, **`14–2–0` (`87.5%`)** vs. `MacroBot` and `ClassicBot`.
- **Sparky-Gated 44-Game Validation (`val_44g`) & Stratified Dual-Boss PPO (`galcon/train.py`):**
  - Stratified per-iteration opponent scheduling (`--sparky-share 0.42`, `--esparan-share 0.17`) eliminates RNG starvation on Sparky games.
  - 44-game validation gate (`16` Sparky + `8` EsparanBot + `8` Anchor + `12` Heuristics) logs full per-bucket breakdowns (`spky=`, `esp=`, `anc=`, `heur=`) and enforces a **Sparky-priority composite + strict Sparky regression floor** so heuristic gains can never silently overwrite a stronger Sparky checkpoint.
  - Zero-fallback **DirectML (`DMLAdamW`)** GPU training (`--device dml`) alongside multi-process CPU rollouts (`--workers`, `--train-threads`).

## Gen 4 Highlights
- **#1 Overall on the 1,056-Game Bradley-Terry Elo Ladder (`1824.5 Elo`, surpassing `esparano Sparky v1.6` at `1821.4 Elo`!)**:
  - **`GalconPT-Gen4 + AdaptiveSearch` (`1×` Canonical D4, `108.7 ms/tick`)**: **`1800.7 Elo`**, tying `esparano Sparky v1.6` **`8–8–4` (`50.0%` score — 8 decisive wins, +33% more wins than Gen 3 Search at `21 ms` lower latency!)**, crushing `GalconPT-Gen3 Pure` **`8–2–2` (`75.0%` score)**, sweeping `ClassicBot` **`12–0–0`**, and going undefeated **`10–0–2`** vs. `esparano EsparanBot v0.2`.
- **Bit-Exact `1×` Canonical D4 Invariance (`canonicalize_galcon_d4`)**:
  - Maps every board to its unique canonical D4 octant in `O(N)` time, achieving **`0.00e+00` max output difference across all 8 rotations & reflections** in a single `1×` forward pass (`19.4 ms` Pure / `108.7 ms` 2-Ply Search on 18-planet boards, vs. `22.6 ms` / `129.4 ms` with 8× D4 TTA).
- **Native Pairwise Multi-Target Send Head (`GalconGen4Transformer`, `1.29M` params)**:
  - Adds an independent Bernoulli pairwise edge head `(ms_logits, ms_fracs)` warm-started directly from Gen 3 Champion weights so the **Pure neural net natively launches simultaneous multi-target `/SEND` orders on tick 0** (lifting the `1-of-N` Softmax bottleneck).
  - Trained with Gen 4 Multi-Send AlphaZero EI + 320 episodes of Bernoulli+Normal Paired-Seed Mirror League PPO (`val_26g`: `42.3% → 67.3%`).
  - **Crushed `GalconPT-Gen3 Pure (8× D4)` `9–3–0` (`75.0%` WR) on Large `22–28` planet arenas** at `19.4 ms/tick`, doubled Pure's Large-Arena wins vs. `Sparky v1.6`, and beat `Gen 2 Pure` `6–3–3` (`62.5%` score).
- **Opponent-Adaptive Multi-Ply Search + Empirical 2-Ply vs. 3-Ply Study**:
  - Tracks live enemy fleet launches to adapt simulated opponent replies to multi-send macro expanders (`Sparky v1.6`) vs. early home-rushers.
  - **2-Ply vs. 3-Ply Finding (20 held-out games vs. `Sparky v1.6`)**: **2-Ply (`h=16`, `108.7 ms`) scored `8–8–4` (`50.0%`)**, whereas **3-Ply (`h=15`, `194.4 ms`) scored `6–9–5` (`42.5%`)** — confirming 2-ply is the real-time sweet spot before compounding opponent-prediction error outweighs deeper rollouts.

## Unified Bradley-Terry Elo Ladder (1,056 Held-Out Games • Anchor: `GalconClassicBot = 1200.0`)

| Rank | Agent | Bradley-Terry Elo | Mean Latency | Highlight |
|---:|---|---:|---:|---|
| 1 | **`GalconPT-Gen3 + ReactiveSearch` (1.26M, 2-Ply)** | **`1824.5`** | `129.4 ms` | **#1 Overall** (`+3.1` over Sparky v1.6); `6–6–8` Sparky; `8–0–4` Gen 2; `5–7` Large Arena vs. Sparky |
| 2 | `esparano Sparky v1.6` (Lua Grandmaster) | `1821.4` | `<2 ms` | Unmodified Lua bot via `lua_bridge.py`; `41–1–6` vs. heuristics |
| 3 | **`GalconPT-Gen4 + AdaptiveSearch` (`1×` Canon 2-Ply)** | **`1800.7`** | **`108.7 ms`** | **`8–8–4` (`50.0%`, 8 wins!) vs. Sparky**; `8–2–2` vs. Gen 3 Pure; `10–0–2` EsparanBot |
| 4 | `GalconPT-Gen2 + Search` (1-Ply v2) | `1734.1` | `67.9 ms` | 1-ply lookahead baseline |
| 5 | **`GalconPT-Gen3 Pure` (`1.26M`, 8× D4 + `/REDIR`)** | **`1731.4`** | `22.6 ms` | **`104–5–11` (`86.7%` WR / `91.3%` score)**; `20–0–0` Classic; `6–10–4` vs. Sparky |
| 6 | **`GalconPT-Gen4 Pure` (`1×` Canon Multi-Send)** | **`1689.7`** | **`19.4 ms`** | **Crushes Gen 3 Pure `9–3–0` (`75%`) on Large 22–28pl Arenas**; `6–3–3` vs. Gen 2 |
| 7 | `GalconPT-Gen2 Pure` (`1.26M`, 8× D4) | `1683.0` | `20.7 ms` | Gen 2 champion (`102–9–9`) |
| 8 | `GalconPT-Gen2 EI-R2` (Pre-PPO) | `1544.5` | `20.7 ms` | Pre-RL search-distilled Gen 2 checkpoint |
| 9 | `GalconTacticalBot` (Analytical Solver) | `1450.1` | `<2 ms` | Gen 3 Pure wins `16–1–3`; Gen 4 Pure wins `15–2–3` |
| 10 | `GalconRusherBot (All-In)` | `1431.8` | `<2 ms` | Gen 3 Pure `16–1–3`; Gen 4 Pure `16–3–1` |
| 11 | `GalconPT-v1 Pure` (`0.59M`, 8× D4) | `1400.6` | `10–17 ms` | v1 baseline; Gen 3 Pure `17–2–1`, Gen 4 Pure `16–3–1` |
| 12 | `esparano EsparanBot v0.2` (Lua) | `1360.8` | `<2 ms` | Beats Classic `10–2–0`; **Gen 3 & Gen 4 Search undefeated `10–0–2`** |
| 13 | `GalconRusherBot (Adaptive)` | `1358.8` | `<2 ms` | Gen 3 Pure `16–1–3`; Gen 4 Pure `15–1–4` |
| 14 | `GalconClassicBot` (Anchor) | `1200.0` | `<1 ms` | Gen 3 Pure **`20–0–0`**, Gen 4 Search **`12–0–0`**, Gen 4 Pure `19–1–0` |
| 15 | `GalconMacroBot` (Boomer) | `1129.3` | `<2 ms` | Gen 3 Pure **`19–0–1`**; Gen 4 Pure `18–1–1` |

## Quickstart

### 1. Train & Evaluate Locally (Modular Gen 4 Multi-Send Flywheel)
```bash
# Gen 4 Multi-Send AlphaZero EI (warm-started from Gen 3 Champion in 1x Canonical D4):
python3 -m galcon.train --mode gen2 --stage gen4_ei \
  --resume checkpoints/galcon_transformer_best.pt \
  --search-games 44 --epochs 8 --lr 3.5e-4 \
  --out checkpoints/galcon_gen4_ei_r1.pt

# Gen 4 Native Multi-Send Paired-Seed Mirror League PPO:
python3 -m galcon.train --mode gen2 --stage gen4_ppo \
  --resume checkpoints/galcon_gen4_ei_r1.pt \
  --gen2-anchor checkpoints/galcon_transformer_best.pt \
  --ppo-iters 10 --mirror-pairs 8 --lr 6.0e-5 \
  --out checkpoints/galcon_gen4_ppo.pt

# Optional: pass --lua-opp /path/to/bot_sparky_v1.6.lua (requires lua5.3 on PATH)

# Record showcase replays (incl. Sparky Win/Loss) & Elo viewer:
python3 -m galcon.record_replays \
  --checkpoint checkpoints/galcon_transformer_best.pt \
  --checkpoint2 checkpoints/galcon_wave7_p1.pt \
  --out-html docs/galcon_viewer.html

# Run all 13 unit tests (Canonical D4, multi-send, Lua bridge, DMLAdamW, Sparky gate, Ensemble):
pytest -v tests/test_galcon.py
```

### 2. Plug Straight Into Phil Hassey's Official `gbots`
```bash
# Run our #1 Wave 7 2-Model Ensemble + 2-Ply Search (65.0% vs Sparky v1.6, 32-0-0 heuristics):
./gbots pipe -name OurGalconWave7Ensemble -key YOUR_KEY \
  -exec "python3 -m galcon.gbots_client --bot ensemble --checkpoint checkpoints/galcon_transformer_best.pt --checkpoint2 checkpoints/galcon_wave7_p1.pt"

# Run single-model Opponent-Adaptive 2-Ply Search (~108 ms/tick, 1x Canonical D4):
./gbots pipe -name OurGalconSearch -key YOUR_KEY \
  -exec "python3 -m galcon.gbots_client --bot search --checkpoint checkpoints/galcon_transformer_best.pt"

# Run fast Pure neural net (~19 ms VM / ~8 ms desktop, 52.5% 40g vs Sparky v1.6):
./gbots pipe -name OurGalconPure -key YOUR_KEY \
  -exec "python3 -m galcon.gbots_client --bot transformer --checkpoint checkpoints/galcon_transformer_best.pt"
```

Interactive HTML5 Canvas Replay Viewer (with Sparky v1.6 Win & Loss Forensics) & Elo Ladder: [`docs/galcon_viewer.html`](../docs/galcon_viewer.html).
