# Galcon AI Wave 10 — Waffle3z-GA (`bots_waffle3z_optimized`) & Triple-Boss Champions (`66.3%` Sparky • `68.8%–70.0%` Waffle3z-GA • `66.7%–83.3%` Large)

A competitive, self-contained Galcon AI stack inside `AClaybrook/orbit-wars-lightzero` that cross-pollinates our Orbit Wars **`PlanetTimelineTransformer`** (`orbit_wars_bot/planet_transformer.py`) to classic Galcon, Phil Hassey's official **`gbots`** protocol, and zero-copy head-to-head benchmarking against Evan Sparano's (`esparano/GalconModVault`) unmodified Lua tournament bots (`Sparky v1.6`, `Waffle3z-GA / bots_waffle3z_optimized`, & `EsparanBot v0.2`) via [`galcon/lua_bridge.py`](lua_bridge.py).

## Wave 10 Highlights — Waffle3z-GA Integration, Triple-Boss Distillation & New All-Round Champion
- **Galcon `botwar` Functional API Support in [`galcon/lua_bridge.py`](lua_bridge.py):**
  - Auto-detects `botwar` sandboxed scripts (`_sandbox_init`), adds Lua 5.3 `math.pow`/`math.atan2` shims, tracks a real per-tick `sb_stats()` call+time budget counter (matching Galcon 2's `g2_sandbox(..., 64, 64)` cap), maintains per-player `memory`, and supports cross-platform `path.lua::func_name` selectors so `external/GalconModVault/esparano/bots/bot_hivemind/mod_botwar_testing.lua::bots_waffle3z_optimized` runs **unmodified with zero Lua errors**.
  - **External Lua Boss Hierarchy:** `Waffle3z-GA` crushes `EsparanBot v0.2` **`12–5–3` (`67.5%`)** and pulls **`5–7–0` (`41.7%`)** on Large Paired vs. `Sparky v1.6` (`36.3%` on 40g 18pl Std) — clear **#2 external Lua boss (~`1745 Elo`)**.
- **#1 All-Round Triple-Boss Champion — `Ens(W9S + W6B)` (`galcon_wave9_search.pt` + `galcon_transformer_best.pt`, `43–49 ms/tick`, new `--bot ensemble` default!):**
  - **18pl Std `Sparky v1.6` (40g, 1-tick):** **`24–11–5` (`66.3%`, ~`1945 Elo`)** (`s=14000: 60.0%`, `s=9500: 72.5%`, `49.5 ms/t`)
  - **18pl Std `Waffle3z-GA` (40g, 1-tick):** **`26–11–3` (`68.8%` — `+5.0%` over Wave 9 Ensemble!)** (`s=14000: 65.0%`, `s=9500: 72.5%`, `43.4 ms/t`)
  - **Large 24pl Asymmetric:** **`8–4–0` (`66.7%`)** vs. `Sparky v1.6` | **`9–3–0` (`75.0%`)** vs. `Waffle3z-GA` (`8–4–0` / `66.7%` Large Paired vs. `Waffle3z-GA`)!
- **New Triple-Boss Distilled Checkpoint — [`checkpoints/galcon_wave10_search.pt`](../checkpoints/galcon_wave10_search.pt) (`28–30 ms/tick`):**
  - Distilled 64 games of `Ens(W9S + W6B)` across all three Lua bosses (`23–13–8` / `61.4%` vs. triple bosses, `223` multi-send ticks, `915` redir states) into `galcon_wave8_pure.pt` (`prec=67.6%`, `rec=92.5%`, `aux=96.9%`, `vsign=83.1%–87.5%`):
  - **New Single-Model Records:** **`24–11–5` (`66.3%`, `s=9500: 77.5%`!)** vs. `Waffle3z-GA` (`+3.8%` over `W9S`) and **`9–3–8` (`65.0%`)** vs. `Sparky v1.6` `s=14000`!
  - **Mirror-Image Ensemble `Ens(W9S + W10S)` (`41–51 ms/t`):** **`23–10–7` (`66.3%`, fewest Sparky losses ever, `s=14000: 67.5%` record!)** vs. `Sparky v1.6` and **`24–11–5` (`66.3%`)** vs. `Waffle3z-GA`; 3-model `Ens(W9S + W10S + W6B)` hits **`26–10–4` (`70.0%`!)** vs. `Waffle3z-GA`!
- **Triple-Boss Training & 18pl-Aligned `val_56g` Gate (`--lua-opp3`, `--waffle-share` in [`galcon/train.py`](train.py)):**
  - Restored Wave 8's uniform `0.05` PPO potential shaping (removing Wave 9's `0.085` opening boost that caused greedy early over-expansion), aligned `val_56g` with paired 18pl Standard boards alongside Compact/Mixed & Large, and enforced dual promotion floors on Sparky + `(0.60*spky + 0.40*waff)`.
- **Upgraded Interactive Viewer ([`docs/galcon_viewer.html`](../docs/galcon_viewer.html)):** Now includes **Waffle3z-GA Win/Loss Forensics**, **Wave 10 Ensemble & Single-Model Sparky/Large replays**, an **External Lua Bot War (`Sparky v1.6 vs. Waffle3z-GA`)**, and upgraded **Triple-Boss Win-Rate Comparison Bars & Elo Ladder**.

## Wave 9 Highlights (Map-Adaptive Adversarial/3-Ply + `33 ms` Single-Model Champion `galcon_wave9_search.pt`)
- **#1 Overall on 18pl Standard `Sparky v1.6` — Single-Model `33 ms` Search (`checkpoints/galcon_wave9_search.pt`, ~`1939 Elo`!):**
  - Distilled 60 games of our Wave 9 Map-Adaptive Adversarial+3-Ply Ensemble teacher (`19–13–4` vs. dual bosses, `226` multi-send ticks, `1,155` redir states) into `galcon_wave8_pure.pt` via `--student-init`, achieving all-time record calibration (**`aux=97.0%`**, **`prec=69.9%–71.4%`**, **`vsign=82.5%–85.0%`**).
  - **18pl Standard `Sparky v1.6` (40g, 1-tick real `gbots` cadence):** **`24–11–5` (`66.3%`)** at just **`29.8–37.7 ms/tick`** (**~4× faster** than Wave 7 Ensemble!):
    - `s=14000` (20g): **`9–6–5` (`57.5%`)** (`29.8 ms/tick`)
    - `s=9500` (20g): **`15–5–0` (`75.0%`, zero draws!)** (`37.7 ms/tick`)
  - **Large 24pl Asymmetric `Sparky v1.6` (12g):** **`6–6–0` (`50.0%`)** at `66.2 ms/tick` (tripled Wave 7 Ensemble's `16.7%` with a single model!).
  - **Built-in Heuristics (32g):** **`31–1–0` (`96.9%`)** — `8–0–0` Tactical, `8–0–0` Macro, `8–0–0` Classic, `7–1–0` Rusher!
  - Defaulted `--bot search` in [`galcon/gbots_client.py`](gbots_client.py) to `checkpoints/galcon_wave9_search.pt`.
- **All-Scale 2-Model Map-Adaptive Ensemble (`EnsembleGalconSearchAgent([W6B, W7p1])`, `48–51 ms/tick`):**
  - Made adversarial focus-fire/soft-minimax **map-adaptive** (`use_adv = adversarial_search and n < 22`, enabled by default; `3-ply h=22` on `n >= 22`) + added optional decoupled `value_weights`:
    - **18pl Standard `Sparky v1.6` (40g, 1-tick):** **`25–13–2` (`65.0%`, `+2.5%` over Wave 8)** at **`47.7–50.9 ms/tick`** (`s=14000`: `11–8–1 / 57.5%`, `s=9500`: `14–5–1 / 72.5%`).
    - **Compact / Mixed `8–18pl` `Sparky v1.6` (40g):** **`+10.0%`** (`40.0% → 50.0%`).
    - **Large 24pl Asymmetric (`12g`):** **`8–4–0` (`66.7%`)** | **Large 22–28pl Paired (`12g`):** **`6–6–0` (`50.0%`)**.
- **Training Pipeline (`galcon/train.py`):** Added `--student-init`, `--v-anchor-coef` (zero-overhead EMA value-head anchor tether), `--gen2-anchor2` (dual-anchor pool), and tick `<= 20` opening potential shaping, adopting an **~85% RL / ~15% in-loop `val_44g`/`val_56g`** cadence.

## Wave 8 Highlights (Vectorized Engine + Adaptive Large 3-Ply + `8 ms` Ensemble-Distilled Pure)
- **Stage 8A — `10.8×–19.0×` Faster Feature Extraction (`0.00e+00` Bit-Exact Parity):**
  - Vectorized [`project_all_planet_futures`](sim.py) and [`extract_galcon_features`](transformer_bot.py) in pure NumPy:
    - **18pl Std:** feature extraction dropped `10.29 ms → 0.95 ms` (**`10.8×` faster**); Pure total `17.0 ms → 8.0 ms/tick`; Ensemble Search dropped `134.1 ms → 62.7–78.3 ms/tick` (**`~2×` faster**!).
    - **24pl Large:** feature extraction dropped `19.77 ms → 1.04 ms` (**`19.0×` faster**); Pure total `27.7 ms → 9.65 ms/tick`.
    - Cut Mirror PPO rollout time roughly in half (`~11–15s` per 24 episodes across 2 workers).
- **Stage 8B/D — Map-Adaptive `2-Ply` Std / `3-Ply h=22` Large Search (`EnsembleGalconSearchAgent`):**
  - Keeps exact `2-ply h=16` on `<22pl` maps (`24–14–2` / `62.5%` 1-tick, `25–13–2` / `65.0%` 2-tick vs. `Sparky v1.6` at `63–78 ms/tick`; `19–3–2` / `83.3%` vs. `EsparanBot v0.2`; `32–0–0` heuristics), and switches to `3-ply h=22` (`0.30s` ceiling) on Large `>=22pl` arenas:
    - **Large 24pl Asymmetric `Sparky v1.6` (12g `s=14700`):** jumped **`16.7% → 66.7%`** (**`8–4–0`, `+50.0%`**, `101.4 ms/tick` — quadrupled wins over Wave 7!).
    - **Large 22–28pl Paired-Mirror `Sparky v1.6` (12g `s=14700`):** jumped **`41.7% → 50.0%`** (**`6–6–0`**, avg `91.2` ticks!).
- **Stage 8C — Ensemble-to-Pure Distillation → Tighter-KL PPO (`checkpoints/galcon_wave8_pure.pt`, `8.0 ms/tick` ZERO Search!):**
  - Added `--resume2` (2-model Ensemble Search teacher), `--large-map-prob`, and `--kl-coef` to [`galcon/train.py`](train.py), adopting a **`~82%` RL / `~18%` in-loop `val_44g`** training cadence (`--ppo-iters 9 --eval-every 3 --kl-coef 0.055 --lr 1.6e-5`).
  - **18pl Standard `Sparky v1.6` (40g Pure, ZERO search):** **`18–9–13` (`61.3%`, ~`1901 Elo` at `8.0 ms/tick`!)** — **`+15.0%`** over Wave 6B Pure (`46.3%`) and **`+7.5%`** over Wave 7 `p1` Pure (`53.8%`), while running **2.3× faster** (`s=14000`: `7–4–9 / 57.5%`, `s=9500`: `11–5–4 / 65.0%`).
  - **`val_44g` Sparky:** **`43.8%`** (`+12.6%` over Wave 6B's `31.2%`); **`EsparanBot v0.2` (24g):** **`16–2–6` (`79.2%`, `87.5%` on `s=9500`)**; **`ClassicBot`:** **`16–0–0` (`100%`)**; **H2H vs. Wave 6B Pure:** **`7–6–7` (`52.5%`)**.
- **Interactive Replay Viewer (`docs/galcon_viewer.html`):** Updated with **Wave 8 Adaptive Ensemble (`18pl` Win, `24pl Large Asymmetric` Win, and `18pl` Loss Forensics)** and **Wave 8 `8 ms` Pure (Win & Loss Forensics)** vs. `Sparky v1.6`.

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

# Optional: clone unmodified external Lua tournament bosses (gitignored under external/)
git clone --depth 1 https://github.com/esparano/GalconModVault external/GalconModVault

# Wave 10 Triple-Boss Mirror League PPO (Sparky v1.6 + Waffle3z-GA + EsparanBot v0.2, ~85% RL / ~15% val_56g):
# Use --device cuda (NVIDIA) or --device dml (AMD/Intel Windows) for GPU updates; requires lua5.3/lua on PATH.
python3 -m galcon.train --mode gen2 --stage gen4_ppo \
  --resume checkpoints/galcon_wave8_pure.pt \
  --gen2-anchor checkpoints/galcon_wave8_pure.pt \
  --gen2-anchor2 checkpoints/galcon_wave9_search.pt \
  --lua-opp external/GalconModVault/esparano/bots/sparky/bot_sparky_v1.6.lua \
  --lua-opp2 external/GalconModVault/esparano/bots/bot_esparanbot_v0.2/bot_esparanbot.lua \
  --lua-opp3 external/GalconModVault/esparano/bots/bot_hivemind/mod_botwar_testing.lua::bots_waffle3z_optimized \
  --sparky-share 0.35 --waffle-share 0.20 --esparan-share 0.10 \
  --ppo-iters 10 --mirror-pairs 20 --eval-every 3 \
  --kl-coef 0.055 --v-anchor-coef 0.08 --lr 1.6e-5 \
  --workers 4 --train-threads 4 --device cuda \
  --out checkpoints/galcon_wave10_pure.pt

# Record Wave 10 Triple-Boss showcase replays (Sparky, Waffle3z-GA, Large 24pl, Loss Forensics & External Lua Bot War) & Elo viewer:
python3 -m galcon.record_replays \
  --checkpoint checkpoints/galcon_wave9_search.pt \
  --checkpoint2 checkpoints/galcon_transformer_best.pt \
  --search-checkpoint checkpoints/galcon_wave9_search.pt \
  --w10-checkpoint checkpoints/galcon_wave10_search.pt \
  --pure-checkpoint checkpoints/galcon_wave8_pure.pt \
  --lua-sparky external/GalconModVault/esparano/bots/sparky/bot_sparky_v1.6.lua \
  --lua-waffle external/GalconModVault/esparano/bots/bot_hivemind/mod_botwar_testing.lua::bots_waffle3z_optimized \
  --lua-esparan external/GalconModVault/esparano/bots/bot_esparanbot_v0.2/bot_esparanbot.lua \
  --out-html docs/galcon_viewer.html

# Run all 17 unit tests (Waffle3z botwar Lua bridge, Triple-boss schedule/val gate, Map-adaptive adversarial + decoupled value_weights, Vectorized futures parity, Canonical D4, multi-send, DMLAdamW, Ensemble):
pytest -v tests/test_galcon.py
```

### 2. Plug Straight Into Phil Hassey's Official `gbots`
```bash
# Tier 1 — #1 All-Round Triple-Boss Ensemble (66.3% Sparky / 68.8% Waffle3z-GA / 66.7%–75.0% Large @ 43–49 ms!):
./gbots pipe -name OurGalconWave10Ensemble -key YOUR_KEY \
  -exec "python3 -m galcon.gbots_client --bot ensemble"

# Tier 2 — #1 Fast Single-Model Map-Adaptive Search (66.3% Sparky @ 30–38 ms; or use galcon_wave10_search.pt for 66.3% Waffle3z @ 29 ms!):
./gbots pipe -name OurGalconWave9Search -key YOUR_KEY \
  -exec "python3 -m galcon.gbots_client --bot search --checkpoint checkpoints/galcon_wave9_search.pt"

# Tier 3 — Ultra-Fast Wave 8 Pure neural net (~8.0 ms VM, 61.3% Sparky / 52.5% Waffle3z-GA, ZERO search!):
./gbots pipe -name OurGalconWave8Pure -key YOUR_KEY \
  -exec "python3 -m galcon.gbots_client --bot transformer --checkpoint checkpoints/galcon_wave8_pure.pt"
```

Interactive HTML5 Canvas Replay Viewer (with Sparky v1.6 & Waffle3z-GA Win/Loss Forensics, Large 24pl Wins, External Lua Bot War & Triple-Boss Scoreboard): [`docs/galcon_viewer.html`](../docs/galcon_viewer.html).
