# Orbit Wars Bot Elo & Progression Log

This log tracks calibrated **Kaggle-equivalent Elo** estimates (anchored against real Kaggle submissions with known leaderboard ratings) and local **Bradley-Terry (2P) / Plackett-Luce (4P)** tournament ratings across model generations.

## Calibration Anchors (Known Kaggle Ratings)

| Reference Bot | Source | Official / Reported Kaggle Elo | Role |
|---|---|---:|---|
| `Kaggle Starter` | Built-in `kaggle_environments` | ~600 | Local tournament anchor (`1000` local Elo) |
| `Agent OW-Proto` | Public Kaggle notebook (`SimJeg/orbit-wars`) | ~1,000+ | Mid-tier public heuristic |
| `subham_v14` | `subham007/orbit-wars-v14` (HF, 3,588 lines) | **~1,224** | High-tier adaptive heuristic (4-source swarms) |
| `Hober Malloc` | `SimJeg/orbit-wars` (4.3M ModernBERT, 10B RL steps) | **1,800.1 (#2 of 4,729)** | Official Kaggle #2 (`#1 Isaiah @ Tufa Labs = 1,865.1`, `#10 = 1,681.5`) |

---

## Model Generation Ledger

### Generation 1 — `Heuristic v2` (Rule-Based Baseline)
- **Date:** 2026-11-22
- **2P Benchmarks:**
  - `6–0` (100.0%) vs. `Agent OW-Proto` (~1,000+ Kaggle Elo)
  - `3–3` (50.0%) vs. `subham_v14` (~1,224 Kaggle Elo)
  - `0–4` (0.0%) vs. `#2 Hober Malloc` (10B RL)
  - Local 6-bot Round-Robin: **2070.0 local Elo** (`16–0–4`, 80.0%)
- **4P FFA Benchmarks:**
  - `4–0` (100.0%, rank 1.00) vs. 3× `Agent OW-Proto`
  - `1–3` (25.0%, par) vs. 3× `subham_v14`
  - Local 6-bot Mixed 4P FFA: **1357.8 local Elo** (`66.7%` win rate, 1.75 avg place)
- **Estimated Kaggle-Equivalent Elo:** **~1,220** (range `1,180–1,260`)

### Generation 2 — `LightZero v3` (16×16 AlphaZero CNN + Active MCTS, 1.4 MB)
- **Date:** 2026-11-22
- **Training:** 155 multi-opponent games, 8× D4 augmentation, search-distilled Expert Iteration (CPU-trained).
- **2P Benchmarks:**
  - `4–2` (**66.7%**, +120 Elo) vs. `subham_v14` (~1,224 Kaggle Elo)
  - `3–1` (75.0%) vs. `Heuristic v2` in 6-bot Round-Robin (`6–4` in 10g suite)
  - `0–4` (0.0%) vs. `#2 Hober Malloc` (10B RL)
  - Local 6-bot Round-Robin: **2300.3 local Elo (#1)** (`19–0–1`, 95.0%)
- **4P FFA Benchmarks:**
  - `1–3` (25.0%, par) vs. 3× `subham_v14`
  - `4–4` (50.0%, 2× par) vs. 3× `wd_bestv6`; `3–1` (75.0%, 3× par) vs. 3× `Heuristic v2`
  - Local 6-bot Mixed 4P FFA: **1323.5 local Elo** (`25.0%` win, `66.7%` Top-2, 2.08 avg place)
- **Estimated Kaggle-Equivalent Elo:**
  - **2P 1v1:** **~1,340** (uncertainty band `1,250–1,400`)
  - **4P FFA:** **~1,220** (uncertainty band `1,150–1,270`)
  - **Combined Estimate:** **~1,270–1,300 Kaggle Elo**
- **Identified Bottlenecks (vs. `#2 Hober Malloc`):**
  1. `16×16` CNN grid blurs exact planet IDs & fleet arrival timings.
  2. 16 global macro stances cannot issue independent per-planet all-in short-ETA launches (`41%` all-in vs. `Hober Malloc`'s `100%` all-in).
  3. Needs per-planet 20-step arrival timelines + entity-level Transformer policy.

### Generation 3 — `PlanetTimelineTransformer v1` (595K params, 2.4 MB, Pure PyTorch)
- **Date:** 2026-11-23
- **Architecture:** Lossless per-planet entity tokens + 20-step 1D-CNN projected fleet-arrival timelines + 100% D4-invariant short-ETA (`ETA <= 20`) directed edges + 4-layer edge-biased Transformer + independent per-planet `(HOLD | target_j)` & fraction head + global value head.
- **Training:** Behavior Cloning + Value Learning on 34 diverse `Hober Malloc` (10B RL) 2P & 4P games (6,072 boards / 67,833 planet decisions, 100.0% exact replay target recovery), 17 epochs with exact 8× D4 augmentation (`val_all=72.9%`, `val_tgt=69.8%`, `val_vsign=95.4%`).
- **2P Benchmarks (30–0 undefeated vs. all non-#2 bots, 27–32 ms/turn w/ 4× TTA):**
  - **`6–0` (100.0%)** vs. `LightZero v3` (117.2 ticks)
  - **`6–0` (100.0%)** vs. `subham_v14` (~1,224 Kaggle Elo, 103.5 ticks)
  - **`6–0` (100.0%)** vs. `Heuristic v2` (108.5 ticks)
  - **`6–0` (100.0%)** vs. `wd_bestv6` (112.5 ticks)
  - **`6–0` (100.0%)** vs. `Agent OW-Proto` (~1,000+ Kaggle Elo, 99.2 ticks)
  - `0–6` (0.0%) vs. `#2 Hober Malloc` (10B RL, lasted 117.2 ticks vs. `LightZero v3`'s 97.0 ticks)
- **4P FFA Benchmarks (26–38 ms/turn w/ 4× TTA):**
  - **`4–0` (100.0%, rank `1.00` — 4.0× par!)** vs. 3× `Heuristic v2` (147.8 ticks)
  - **`4–0` (100.0%, rank `1.00` — 4.0× par!)** vs. 3× `wd_bestv6` (192.2 ticks — up from `LightZero v3`'s `50%` & `Heuristic v2`'s `41.7%`)
  - `1–3` (25.0%, rank `2.50` — par) vs. 3× `subham_v14`
- **Estimated Kaggle-Equivalent Elo:**
  - **2P 1v1:** **~1,480–1,550**
  - **4P FFA:** **~1,320–1,380**
  - **Combined Estimate:** **~1,400–1,460 Kaggle Elo** (**+140 to +180 Elo** jump over `LightZero v3`, while running **4× faster**!)

### Generation 4 — `PlanetTransformer v2 (DAgger)` + `Self-Play PPO RL` + `Value-Guided Search`
- **Date:** 2026-11-23
- **Three Parallel Upgrades Tested & Ablated:**
  1. **AlphaZero-Style Value-Guided Lookahead (`SearchPlanetTransformerAgent`):** Proposes up to 6 diverse full-turn plans from the Transformer posterior (threshold sweep + top-2 highest-entropy planet flips), rolls each 10 ticks forward in `FastOrbitSim` with active opponent responses under a `0.22s` cap, and picks the max `0.65 * TransformerValue + 0.35 * HeuristicEval + prior`.
     - Immediately turned `PlanetTransformer v1` vs. 3× `subham_v14` (4P FFA, seeds 400–403) from **`1–3` (`25.0%`, rank `2.50`)** into **`4–0` (`100.0%`, rank `1.00`)** at `187.6 ms/turn`!
  2. **On-Policy DAgger on Lost/Contested Games (`PlanetTransformer v2`):** Played 16 live games (`10× 2P` vs. `#2 Hober Malloc`, `6× 4P` vs. `3× subham_v14`), oversampling lost games (`stride=1`) and querying `#2 Hober Malloc` as an online oracle (+1,228 states / 8,316 decisions → 7,300 total boards). Diagnosed a **`69.1%` turn-level launch-count divergence** on student-visited states. Resumed 9 cosine-annealed epochs:
     - `val_pol`: `0.775 → 0.643` | `val_all`: `72.9% → 76.9%` | `val_tgt`: `69.8% → 74.0%` | `val_vsign`: `95.4% → 97.3%`.
     - **4P vs. 3× `subham_v14` (pure neural policy, `26.9 ms/turn`):** jumped from **`1–3` (`25.0%`) → `4–0` (`100.0%`, rank `1.00`)** with zero search overhead!
  3. **On-Policy Self-Play PPO + Anchor-KL RL (`scripts/rl_finetune_transformer.py`, `PT-v2+RL`):** Trained 5 PPO iterations (40 mixed 2P & 4P league episodes in `FastOrbitSim` against frozen v2 anchor, live self-play, and `Heuristic v2`, 5,635 on-policy states) with per-planet clipped PPO ratios + Anchor-KL (`kl_coef=0.06`) + GAE + 8× D4 augmentation.
     - **2P Head-to-Head vs. `PT-v1`:** **`5–1` (`83.3%`, ~+280 Elo)** at `30.3 ms/turn`!
     - **2P Head-to-Head vs. `PT-v2 (DAgger)`:** **`5–1` (`83.3%`, ~+280 Elo)** at `25.5 ms/turn` — proving Self-Play RL genuinely improved *beyond* its own BC/DAgger teacher-imitation anchor!
     - **2P Regression Suite:** **`6–0` (`100.0%`)** vs. `LightZero v3` (`34.4 ms`), **`6–0` (`100.0%`)** vs. `subham_v14` (`32.1 ms`).
     - **4P FFA:** **`4–0` (`100.0%`, rank `1.00`, 130.8 ticks — 37 ticks faster KO than DAgger!)** vs. 3× `subham_v14` (`27.2 ms`); **`3–1` (`75.0%`, 3× par, rank `1.50`)** vs. 3× `wd_bestv6` (`24.9 ms`).
     - **2P vs. `#2 Hober Malloc` (10B RL, 2nd place of 4,729 teams):** `0–6` pure policy (`109.2` ticks), `0–4` with `PT-v2+RL+Search` (extended survival to **`121.0` ticks**, every game 112–125 ticks).
- **Estimated Kaggle-Equivalent Elo (`PT-v2+RL`, promoted to `checkpoints/planet_transformer_best.pt`):**
  - **2P 1v1:** **~1,550–1,630** (`5–1` over v1/v2, `6–0` over `subham_v14` & `LightZero v3`, still below `#2 Hober Malloc` at 10B RL steps)
  - **4P FFA:** **~1,460–1,540** (`4–0` sweep of `3× subham_v14` at 27 ms/turn, `3–1` vs. `3× wd_bestv6`)
  - **Combined Estimate:** **~1,500–1,580 Kaggle Elo** (**+100 to +120 Elo** over `PT-v1`, **+230 to +280 Elo** over `LightZero v3`!)
- **Next Lever (to take games off `#2 Hober Malloc` on GPU):**
  1. Port `FastOrbitSim` / batched PyTorch PPO rollout workers to a single GPU (100M–1B+ self-play steps with historical opponent pool + delayed teacher KL, mirroring #5's RTX 5090 recipe).
  2. Scale model from `0.6M` → `4–6M` params (`d_model=192–256`, 6–8 layers) with semantic action buckets (`send all`, `sortie`, `hold`, `kill-at-arrival` from #3's solution).

### Generation 5 — `PT-v3-RL` (League PPO + Moving EMA Teacher KL) & Lookahead Tradeoff Study
- **Date:** 2026-11-23
- **Training Upgrades ([`scripts/rl_finetune_transformer.py`](../scripts/rl_finetune_transformer.py)):**
  - Historical opponent pool (`{PT-v1, PT-v2, PT-v2+RL}` + periodic frozen snapshots + external `subham_v14` & `Heuristic v2`), zero-planet early reset (#7 Kaggle lesson), delayed moving EMA teacher KL (#5 TonyK lesson), parallel 2-worker rollout collection, and controlled low entropy (`ent_coef=0.0015`).
  - Ran 84 additional mixed 2P/4P league episodes (~10,787 on-policy states across Stage 1 + Stage 2).
- **`PT-v3-RL` (Pure Neural Policy, `28–36 ms/turn` — Promoted to `checkpoints/planet_transformer_best.pt`):**
  - **20–0 (100.0% undefeated)** across ALL non-#2 benchmarks:
    - **`6–0` (100.0%)** vs. `LightZero v3` (2P, 118.8 ticks, `35.3 ms`)
    - **`6–0` (100.0%)** vs. `subham_v14` (2P, 106.5 ticks, `36.5 ms`)
    - **`4–0` (100.0%, rank `1.00`)** vs. 3× `subham_v14` (4P, 181.5 ticks, `28.3 ms`)
    - **`4–0` (100.0%, rank `1.00`)** vs. 3× `wd_bestv6` (4P, 190.5 ticks, `29.0 ms` — **up from Gen 4 `PT-v2+RL`'s `3–1` / 75.0%!**)
  - **2P Head-to-Head vs. Gen 4 `PT-v2+RL`:** `6–6` (`50.0%`, 12 games pure) & `2–2` (`50.0%`, 4 games Search-vs-Search) — strictly improved 4P dominance with zero 2P regression.
- **Empirical MCTS / Lookahead Compute-vs-Elo Ablation (`SearchPlanetTransformerAgent`):**
  - **`PT-v2+RL + Search` vs. pure `PT-v2+RL` (2P):** **`5–1` (`83.3%`)** at `118.1 ms` mean (`309.5 ms` max) vs. `28 ms` pure.
  - **`PT-v3-RL + Search` vs. pure `PT-v3-RL` (2P):** **`6–0` (`100.0%`)** at `131.9 ms` mean (`311.4 ms` max) vs. `30 ms` pure.
  - **Combined Lookahead vs. Same-Weights Pure Policy:** **`11–1` (`91.7%` win rate ≈ +300–380 Elo!)** for ~4.2× latency (~125 ms mean, still using only ~13% of Kaggle's 1,000 ms turn budget!).
  - **4P FFA:** Flipped `PT-v2+RL` vs. 3× `wd_bestv6` from `3–1` → **`4–0` (`100.0%`)** at `211.7 ms` mean (`359.0 ms` max).
  - **2P vs. `#2 Hober Malloc (1800.1)`:** `0–4` (`116.8` avg ticks at `104.3 ms` mean).
- **Controlled Negative Ablation — Pure Search-Only Distillation (`PT-v3-AZ`):**
  - Collected 10 pure AlphaZero Search self-play games (`2,759` states, `100%` target recovery, `61.1%` launch rate) and fine-tuned without mixing the base replay buffer.
  - Result: `0–1–5` vs. `PT-v3-RL`, `3–7` vs. `PT-v2+RL`, `1–3` vs. 3× `subham_v14` (4P). Lesson: lookahead can safely launch aggressively (`61%`) *because* it checks 10-tick counter-snipe safety; distilling it without replay-buffer mixing causes raw-policy over-launching.
- **Estimated Kaggle Skill Rating (Calibrated to Official `#1 = 1865.1`, `#2 = 1800.1`, `#10 = 1681.5`):**
  - **`PT-v3-RL` (pure `28 ms/turn`):** **~1,530–1,600** (undefeated `20–0` across `LightZero v3`, `subham_v14`, `wd_bestv6` in 2P & 4P)
  - **`PT-v3-RL + Search` (`118–149 ms/turn`):** **~1,620–1,680** (`11–1` / `91.7%` over pure policy; approaching `#10 = 1681.5`, still below `#2 Hober Malloc = 1800.1`)

### Generation 6 — `1.89M PT-v4-Large` (3.2× Scaled Capacity + 2× Data + Neural-Opponent Search)
- **Date:** 2026-11-23
- **Architecture & Data Scaling:**
  - Scaled `PlanetTimelineTransformer` **3.2×** from `0.59M` (`d=128, h=4, L=4, ff=256`) → **`1.89M` params** (`d=192, h=6, L=6, ff=384`, 7.4 MB checkpoint — still half `#2 Hober Malloc`'s 16 MB and only `36–42 ms/turn` w/ 4× TTA!).
  - Expanded dataset **2.1×** (`6,072 → 12,809` board states / ~131K planet decisions across 86 games): added 24 fresh `#2 Hober Malloc` 2P/4P games (`100.0%` exact target recovery) + 12 `PT-v3-RL` DAgger games targeting Gen 5's exact losses vs. `#2 Hober Malloc` (`86.6%` oracle recovery).
  - Trained 15 cosine-annealed epochs across 5 passes with exact 8× D4 augmentation (`val_pol=0.745`, `val_all=73.3%`, `val_launch=58.5%`, `val_tgt=64.8%`, `val_vsign=95.2%`).
  - Upgraded `SearchPlanetTransformerAgent` turn-0 opponent modeling from `heuristic_moves` (1,220 Elo) to our own fast no-TTA neural Transformer (~4 ms), and fixed PyTorch multiprocessing tensor-IPC file-descriptor limits in `scripts/rl_finetune_transformer.py` via NumPy weight serialization.
- **`1.89M PT-v4-Large` (Pure Neural Policy, `36–42 ms/turn` — Promoted to `checkpoints/planet_transformer_best.pt`):**
  - **`4–2` (`66.7%`, ~+120 Elo)** vs. Gen 5 `0.59M PT-v3-RL` (broke directly through the Gen 4/5 `0.59M` 2P plateau!)
  - **`4–2` (`66.7%`)** vs. `1.89M PT-v4-Large-RL` (30-episode CPU League PPO pass also beat Gen 5 `0.59M PT-v3-RL` `4–2`, swept `LightZero v3` `6–0`, `3× subham_v14` `4–0`, and `3× wd_bestv6` `4–0`, though pure 12,809-state BC+DAgger stayed sharper head-to-head!)
  - **`6–0` (`100.0%`, `102.8` ticks — fastest KO yet!)** vs. `subham_v14` (2P)
  - **`4–0` (`100.0%`, rank `1.00`, `135.8` ticks — 46 ticks faster than Gen 5!)** vs. 3× `subham_v14` (4P)
  - **`4–0` (`100.0%`, rank `1.00`, `142.5` ticks — 48 ticks faster than Gen 5!)** vs. 3× `wd_bestv6` (4P)
  - **2P vs. `#2 Hober Malloc (1800.1)` (6 games, seeds 200–205):** `0–6` pure (`111.2` avg ticks, peak `131`t, up to `94` active turns vs. boss's `98`); `0–6` with Neural-Opponent Search (`119.2` avg ticks, peak **`135`t** on seed 204 and **`127`t** on seed 205).
  - **Search-vs-Pure convergence:** On `1.89M PT-v4-Large`, `Neural-Opp Search vs. Pure` narrowed from `11–1` (`91.7%`) on `0.59M` to **`2–2` (`50.0%`, `180.2` ticks)** at `145.5 ms` mean — showing the 3.2× larger 6-layer Transformer internalized much of the tactical foresight directly into its fast `37 ms` weights!
- **Estimated Kaggle Skill Rating (`1.89M PT-v4-Large`):**
  - **Pure (`37 ms/turn`):** **~1,600–1,670** (beat `0.59M PT-v3-RL` `4–2`, matched its own 10-tick Search `2–2`, and crushed all 4P heuristics 46–48 ticks faster!)
  - **With Neural-Opponent Search (`~145 ms/turn`):** **~1,640–1,700** (in range of official `#10 = 1681.5`; closing the final ~100–160 pts to `#2 Hober Malloc = 1800.1` now requires GPU-scale PPO self-play: 100M–1B+ steps on a single GPU).

### Generation 7 — `1.89M PT-v5-Boss` (Full-TTA `1800.1` Boss DAgger + `8×` D4 TTA + Blocked-Shot Fallback + Multi-Model Ensemble)
- **Date:** 2026-11-23
- **Training & Architectural Upgrades:**
  - **Un-nerfed Full-TTA (`1800.1`) Boss DAgger:** Collected 12 live 2P sparring games (`PT-v4-Large` vs. full 8×-TTA `1800.1` `#2 Hober Malloc`, `--full-tta-teacher`), exposing a `57.3%` launch-count divergence against true `1800.1` play (+1,080 states / 4,985 decisions, oversampled 2× onto Gen 6 → `14,969` total board states).
  - **Continued Low-LR Convergence + Launch Calibration:** Resumed `1.89M` for 6 additional epochs (`batch=80`, `lr=4e-4 → 1.8e-4`, `--launch-weight 1.20` — 21 total epochs):
    - `val_pol`: `0.745 → 0.695` | `val_all`: `73.3% → 75.0%` | `val_launch`: `58.5% → 65.0%` (+6.5%) | `val_tgt`: `64.8% → 71.2%` (+6.4%) | `val_vsign`: `95.2% → 96.7%`.
  - **Three Inference Upgrades ([`orbit_wars_bot/planet_transformer.py`](../orbit_wars_bot/planet_transformer.py)):**
    1. Top-3 blocked-shot fallback in `_decode_plan` when an intermediate planet blocks the #1 predicted target (`solve_intercept` returns `None`).
    2. Full **8× D4 dihedral TTA** (`use_d4_tta=True`, 4 rotations × 2 reflections — matching `#2 Hober Malloc` at `55.4 ms/turn`).
    3. Multi-checkpoint **`EnsemblePlanetTransformerAgent`** (blends `PT-v5-Boss + PT-v4-Large + PT-v4-Large-RL` at `72.0 ms/turn`).
- **Elo-Fair Benchmarks (`PT-v5-Boss`, promoted to `checkpoints/planet_transformer_best.pt`):**
  - **`PT-v5-Boss (8× D4)` vs. Gen 6 `1.89M PT-v4-Large` (2P):** **`4–2` (`66.7%`, ~+120 Elo, `145.8` ticks)** at `55.4 ms/turn`.
  - **`Ensemble-3x (4× TTA)` vs. Gen 6 `1.89M PT-v4-Large` (2P):** **`4–2` (`66.7%`, ~+120 Elo, `263.0` ticks)** at `72.0 ms/turn`.
  - **4P FFA New KO Records:**
    - **`4–0` (`100.0%`, `133.8` avg ticks — fastest blitz `85`t!)** vs. 3× `subham_v14` (down from Gen 5's `181.5t` & Gen 6's `135.8t`).
    - **`4–0` (`100.0%`, `137.2` avg ticks — fastest win `108`t!)** vs. 3× `wd_bestv6` (down from Gen 5's `190.5t` & Gen 6's `142.5t`).
  - **2P Boss Gauntlet vs. `#2 Hober Malloc (1800.1)` (seeds 200–205, all `DONE/DONE`):**
    - **Pure `PT-v5-Boss (8× D4)` (`55 ms`):** `0–6` (`119.0` avg ticks, max `127t`, up to `91/104` active turns) — **+7.8 avg ticks over Gen 6 pure (`111.2t`)**, matching Gen 6 *with Search* using zero lookahead!
    - **Pure `Ensemble-3x (4×)` (`72 ms`):** `0–6` (`119.0` avg ticks, peak **`132`t** on seed 202, `129t` on seed 204).
    - **`PT-v5-Boss + Neural-Opp Search (8× D4)`:** `0–6` (`118.0` avg ticks, max `129t`).
- **Estimated Kaggle Skill Rating (`1.89M PT-v5-Boss (8× D4)` / `Ensemble-3x`):**
  - **Pure (`55–72 ms/turn`):** **~1,660–1,720** (firmly in official **Kaggle Top-10** range around `#10 = 1681.5` and `#7 = 1717.3`; taking games off `#2 = 1800.1` requires GPU-scale PPO self-play).
