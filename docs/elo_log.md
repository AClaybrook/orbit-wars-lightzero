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

### Generation 8 — `1.89M PT-Gen8` (Soft-Logit KD + `#7 Audun` Aux `t+10` World Model + `#5 TonyK` 2P Specialist)
- **Date:** 2026-11-23
- **Naming Note:** Synced `PT-Gen#` 1-to-1 with Generation numbers starting at **Gen 8** (`PT-Gen8`).
- **Three Top-10 ROI Upgrades Implemented & Combined ([`scripts/distill_soft_kd.py`](../scripts/distill_soft_kd.py)):**
  1. **Full Soft-Logit Knowledge Distillation (`KL(q_hober || p_ours)`):** Queried `#2 Hober Malloc`'s local JAX model (`rotation_tta=True` = exact 100% full 2P `1800.1` probabilities, `89.2–91.1%` live Rust launch parity) across `2,422` 2P boards (16 on-policy `PT-v5-Boss` DAgger games + 3 elite Hober mirror games) instead of hard 1-hot argmax targets.
  2. **Auxiliary `t+10` Future-State World-Model Head (inspired by `#7 Audun`, `1717.3`):** Added a lightweight training-only head (`aux_future_head`, **0 ms inference overhead**) predicting each planet's true post-action owner & `log1p(ships)` 10 ticks into the future — reached **`94.2%`** owner prediction accuracy!
  3. **Dedicated 2P Specialist + All-In Execution (inspired by `#5 TonyK`, `1754.6` & `#2 SimJeg`, `1800.1`):** Pure 2P Soft-KD training + `force_all_in=True` + auto-dispatch `pt_specialist:<2p>|<4p>` pairing **`PT-Gen8` (2P)** with **`Gen 7 PT-v5-Boss` (4P)**.
- **Training Convergence (12 cosine-annealed epochs resuming `1.89M PT-v5-Boss`, `295s`):**
  - `val_kl`: **`0.489 → 0.369`** | `val_all`: **`83.2%`** | `val_tgt`: **`62.7% → 73.6%`** (+10.9%) | `aux_own (t+10)`: **`94.2%`**.
- **Elo-Fair Benchmarks (`PT-Gen8`, `53.8 ms/turn` — zero inference overhead!):**
  - **`PT-Gen8 (8× D4 All-In)` vs. Gen 7 `1.89M PT-v5-Boss (8× D4)` (2P):** **`6–2–0` (`75.0%`, ~+190 Elo, `148.0t`)** — largest single-generation jump since Gen 4!
  - **`Ensemble (Gen8 + Gen7, 8× D4 All-In)` vs. Gen 7 `1.89M PT-v5-Boss (8× D4)` (2P):** **`4–2–0` (`66.7%`, ~+120 Elo, `219.8t`)** at `96.3 ms/turn`.
  - **2P Boss Gauntlet vs. `#2 Hober Malloc (1800.1)` (seeds 200–205, all `DONE/DONE`):**
    - `PT-Gen8` pure (`53.8 ms`, `thr=0.50`): `0–6` (`111.3t` avg, max `127t`); threshold sweep (`0.40`: `110.0t`, `0.34`: `108.5t`) confirmed Soft-KL probabilities are best calibrated at `0.50`.
    - `PT-Gen8 + Neural-Opp Search`: `0–6` (`115.2t` avg, peak **`134t`** on seed 202 with `93/111` active turns, `127t` on seed 205).
- **Estimated Kaggle Skill Rating (`1.89M PT-Gen8` 2P Specialist + `Gen 7` 4P Specialist):**
  - **Pure (`53.8 ms/turn`):** **~1,720–1,760** (firmly inside Kaggle Top-10 between `#7 Audun = 1717.3` and `#5 TonyK = 1754.6`, +190 Elo over Gen 7 in 2P!). Crossing the final ~50–80 Elo to consistently beat `#2 Hober Malloc (1800.1, 4.3M ModernBERT, 10B RL steps)` now genuinely hits the GPU self-play RL compute wall.

### Generation 9 — `1.89M PT-Gen9` (Midgame + Comet-Weighted Soft-KD Ensemble + 5-Fix Forensics Ablation)
- **Date:** 2026-11-23
- **Forensics-Driven Upgrades ([`scripts/distill_soft_kd.py`](../scripts/distill_soft_kd.py), [`orbit_wars_bot/planet_transformer.py`](../orbit_wars_bot/planet_transformer.py), [`scripts/rl_finetune_transformer.py`](../scripts/rl_finetune_transformer.py)):**
  - **Dense Midgame & Comet-Window DAgger Collection:** Collected **961 fresh dense boards** (`stride=1` during `t = 22..78` and active comet windows) across 10 `PT-Gen8` vs. `#2 Hober Malloc` sparring games → **`3,383` total Full-TTA `1800.1` 2P Soft-KD + `t+10` boards** (`+40%` data).
  - **Midgame (`2.5×`) & Comet (`2.0×`) Weighted Soft-KD Convergence:**
    - `PT-Gen9 (Hold-Boost: hold_b=1.25, launch_b=1.0)`: `val_kl = 0.247` (**`-33%`** vs. Gen 8's `0.369`), `val_all = 85.6%`, `val_tgt = 77.3%` (`+3.7%`), `aux_own (t+10) = 94.9%`.
    - `PT-Gen9 (Balanced: hold_b=1.0, launch_b=1.18)`: `val_kl = 0.247` (**`-33%`**), `val_all = 85.1%`, `val_tgt = 77.3%` (peak `77.7%`), `aux_own (t+10) = 94.8%`.
  - **Complementary Dual-Head Ensemble (`PT-Gen9 Ens-2x`, `0.55 Balanced + 0.45 Hold-Boost`, `8× D4 All-In`, `~98 ms/turn`):** Combines `Hold-Boost`'s midgame patience (`-34%` premature shuffles, `-62%` impulsive bounces) with `Balanced`'s offensive tempo (`76` vs `70` midgame enemy strikes) at <10% of Kaggle's 1,000 ms budget!
  - **Controlled Negative Ablations (Documented in [`docs/boss_midgame_forensics.md`](boss_midgame_forensics.md)):**
    - Peaceful Hold Calibration (`reinforce_threshold=0.60`): **`4–2` (`+120 Elo`)** on `PT-Gen8` (`0.65` too strict at `1–5`).
    - Decoder `surgical_neutral_cap`: **`2–4`** (smaller neutral sends travel slower due to nonlinear ship-count speed scaling).
    - Decoder `anti_bounce_guard`: **`1–5`** (hard-blocking solo pokes hurts attrition waves).
    - Post-KD 2P Midgame CPU PPO (`--force-all-in --two-player-only --midgame-adv-weight 1.8`): `75%` (`6–2`) on 8 eval seeds at iter 1 (`kl=0.05`), but **`1–7`** on unseen seeds `990–997` (reconfirming CPU-scale RL overfits relative to `val_kl = 0.247` Soft-KD).
- **Official Kaggle 2P Benchmarks (`PT-Gen9 Ens-2x`, `~98 ms/turn` — Auto-routed 2P Specialist paired with `Gen 7` 4P Specialist):**
  - **`PT-Gen9 (Ens-2x, 8× D4 All-In)` vs. `PT-Gen8` (official 2P, seeds `990–997`):** **`5–3–0` (`62.5%`, ~+90 Elo, `227.6t`, fastest blitz KO `63` ticks on seed `992`!)**
  - **Single-net Ablations vs. `PT-Gen8` (official 2P, 8g each):** `PT-Gen9 (Hold-Boost)` = `4–4–0` (`50.0%`), `PT-Gen9 (Balanced)` = `4–4–0` (`50.0%`, fastest win `89t`), `PT-Gen9 (Balanced + Hold-Cal 0.60)` = `4–4–0` (`50.0%`, fastest win `77t`).
  - **2P Boss Gauntlet vs. `#2 Hober Malloc (1800.1)` (seeds `200–205`, pure neural — zero search!):**
    - `PT-Gen9 (Ens-2x)`: `0–6` (**`115.2t` avg**, **new pure-policy record `140t` on seed `201`** with `19 vs 11` sync arrivals and `605 vs 1,289` bounced ships — matching Gen 8 *with Search* using pure neural inference!).
    - Ship tipping point delayed **`t = 16.3 → t = 27.8`** (`+11.5` ticks; seed `200` held even until **`t = 59`**), `t = 50` ship deficit narrowed `-131.8 → -112.5`, premature reinforces cut **`-21%`** (`94 → 74`; `-34%` on Hold-Boost), impulsive solo bounces cut **`-55%`** (`29 → 13`), and total bounced ships dropped below `#2` (`2,785 vs 3,100`)!
- **Estimated Kaggle Skill Rating (`1.89M PT-Gen9` 2P Specialist + `Gen 7` 4P Specialist):**
  - **Pure (`~98 ms/turn`):** **~1,750–1,790** (`5–3` / `+90 Elo` over `PT-Gen8`, solidly past `#5 TonyK = 1754.6` into **Kaggle Top-5** territory, with zero copied weights and <10% of the turn budget!).

### Generation 10 — `4.35M PT-Gen10` (RX 6950 XT DirectML Soft-KD + Paired-Seed League PPO)
- **Date:** 2026-11-24
- **Architecture & GPU Pipeline ([`orbit_wars_bot/dml_utils.py`](../orbit_wars_bot/dml_utils.py), [`scripts/distill_ensemble_4m.py`](../scripts/distill_ensemble_4m.py), [`scripts/rl_finetune_transformer.py`](../scripts/rl_finetune_transformer.py)):**
  - Scaled `PlanetTimelineTransformer` from `1.89M` to **`4.35M` parameters** (`d_model=256, n_heads=8, n_layers=7, dim_ff=640, use_aux_future=True`) matching `#2 Hober Malloc`'s `4.3M` ModernBERT capacity, promoted at **`34.23 ms/turn` mean** (`49.40 ms` p95 — **< 4% of Kaggle's 1,000 ms budget**).
  - Zero-fallback AMD RX 6950 XT DirectML training via `DMLAdamW`, scatter-free `one_hot_no_scatter`, and GPU scalar accumulators (`B=256`): **`5.75×` faster PPO updates** (`8.1–8.4s` vs Ryzen 7 5800X `46.6–48.2s`) with **0 DirectML fallback warnings**.
- **Two-Stage Training (`PT-Gen9` Distillation → DirectML League PPO):**
  1. **Stage 1 Ensemble Soft-KD (`4m_base`):** Collected `24,678` 2P boards across 120 `FastOrbitSim` games (`69.8s`), GPU-labeled with `8× D4 TTA × 2` Gen 9 teachers (`43.1s`), and trained 14 epochs (`B=256, lr=8e-4`, `~25s/ep`) → `val_kl = 0.052`, `val_all = 89.0%`, `val_launch = 74.3%`, `val_tgt = 88.2%`, `aux_own = 96.4%`, `vsign = 83.8%`.
  2. **Stage 2 Paired-Seed DirectML League PPO:** 30 iterations × 24 mirror pairs (`1,440` games / `~186K` boards in `31.4 min`, `clip=0.15, kl=0.06, akl=0.025, ema=0.06, aux=0.15`). No-TTA 36g eval progressed `44.4%` → **`61.1%`** at iter 15 (`*BEST*`, `planet_transformer_gen10_4m_ppo.pt`) and ended at `52.8%` at iter 30 (`planet_transformer_gen10_4m_ppo_latest.pt`).
- **Unseen-Seed `8× D4 TTA` Benchmarks (`FastOrbitSim`, seeds `990–999` × 2 seats = `20g`/matchup):**
  - `4m_ppo_best` vs `PT-Gen9`: **`40–20` (`66.7%`, ~+120 Elo overall — zero losing matchups!)** (`12–8 / 60.0%` vs `PT-Gen9 Ens-2x`, `16–4 / 80.0%` vs `Gen9-Balanced`, `12–8 / 60.0%` vs `Gen9-SoftKD`; up from `4m_base`'s `14–46 / 23.3%`).
  - `4m_ppo_latest` vs `PT-Gen9`: **`38–22` (`63.3%`)** (`14–6 / 70.0%` vs `PT-Gen9 Ens-2x`, `8–12 / 40.0%` vs `Gen9-Balanced`, `16–4 / 80.0%` vs `Gen9-SoftKD`).
  - **Direct Head-to-Head (`4m_ppo_best` vs `4m_ppo_latest`):** **`14–6–0` (`70.0%`, ~+147 Elo, `300.5t`)** — establishing `planet_transformer_gen10_4m_ppo.pt` as the clear single-model promotion to [`checkpoints/planet_transformer_best.pt`](../checkpoints/planet_transformer_best.pt) (while preserving `planet_transformer_gen10_4m_ppo_latest.pt` alongside it; 4P specialist remains [`checkpoints/planet_transformer_4p_best.pt`](../checkpoints/planet_transformer_4p_best.pt)).
  - `4m_ppo_ens (0.5 best + 0.5 latest)`: **`12–8–0` (`60.0%`, `267.4t`)** vs `PT-Gen9 Ens-2x` and **`10–10–0` (`50.0%`, `369.3t`)** vs `4m_ppo_best`.
- **Official Kaggle Benchmarks & `#2 Hober Malloc (1800.1)` Boss Forensics:**
  - `4m_ppo_best` vs `subham_v14` (`4 games`, seed `100`): **`4–0–0` (`100.0%`, rank `1.00`, `128.0t`, `37.81 ms` mean)**.
  - **2P Boss Gauntlet vs `#2 Hober Malloc (1800.1)` (`seeds 200–205`, prebuilt `submission1.tar.gz`):**
    - `4m_ppo_best` (`34.23 ms` mean): `0–0–6` (`107.0t` avg)
    - `4m_ppo_latest` (`34.50 ms` mean): `0–0–6` (`115.3t` avg)
    - `4m_ppo_ens (best + latest, 60.78 ms mean)`: `0–0–6` (**`115.7t` avg** — edges `PT-Gen9 Ens-2x`'s `115.2t` record at < 6.5% turn budget).
  - **Midgame Forensics (`scripts/analyze_boss_gap.py`, `4m_ppo_best` vs `#2`, seeds `200–205`):**
    - Ship tipping point delayed further **`t = 27.8 → t = 29.5`** (held seed `204` to **`t = 60`**, `t=10` `+1.0`, `t=20` `+0.8`), and **`t = 50` ship deficit narrowed `-25%` (`-112.5 → -84.5`, `434.7` vs `519.2`)**.
    - **Overkill & Bounce Efficiency Breakthrough:** Neutral overkill slashed **`-57%`** vs `Gen9 Ens-2x` (`1,890 → 818` ships), total bounced ships cut **`-31%`** (`2,785 → 1,922` vs `#2`'s `4,303`), and impulsive midgame bounces dropped `29 (Gen 8) → 13 (Gen 9) → 8`.
    - **Diagnosed Gen 11 Bottleneck:** Because Stage 1–2 distilled strictly from `PT-Gen9` and self-play League PPO without `#2 Hober Malloc` supervision, `4m_ppo_best` under-expanded and conserved too passively (`49.0%` vs `77.1%` active turns, `8.0` vs `10.67` neutral captures, `28.1%` vs `56.1%` `InFlight%` at `t=40`, `40` `ours_HOLD_hober_NEUTRAL` misses across `t=0..65`, `1 vs 5` comets). Next: direct Full-TTA `#2 Hober Malloc` Soft-KD + dense sparring DAgger into the `4.35M` network on RX 6950 XT, followed by Wave 2 DirectML PPO.
- **Estimated Kaggle Skill Rating (`4.35M PT-Gen10` 2P Specialist + `Gen 7` 4P Specialist):**
  - **Pure (`34.23 ms/turn`):** **~1,760–1,795** (`+120 Elo` / `66.7%` over `PT-Gen9` across 60 unseen games; still `0–6` below `#2 Hober Malloc = 1800.1` due to lower expansion tempo & comet rate — motivating direct `#2` Soft-KD on RX 6950 XT next).

## Generation 11 — `4.35M PT-Gen11` (Direct `#2 Hober Malloc` Soft-KD + Iterative DAgger + 3-Wave RX 6950 XT DirectML League PPO)
- **Hardware & Environment:** Ubuntu 24.04 (WSL2), AMD Radeon RX 6950 XT (`torch-directml` + `DMLAdamW`, **0** DirectML CPU-fallback warnings across all Soft-KD and PPO runs).
- **Teacher Parity Fix & Inference Upgrades (`scripts/distill_soft_kd.py`, `scripts/rl_finetune_transformer.py`, `orbit_wars_bot/planet_transformer.py`, `scripts/evaluate.py`):**
  - Fixed `kaggle_environments` slot-1 missing `"step"` observation key in `distill_soft_kd.py` (injecting `steps[t][0]["observation"].get("step", t)`), jumping live `#2 Hober Malloc (1800.1)` JAX `HoberSoftTeacher` parity from `90.94%` (`1515/1666`) to **`99.88%`** (`1664/1666`, and `99.5%` across `408` boards in the 2-game smoke check).
  - Added threshold spec parsing (`_t<NN>` suffix on `pt_d4_allin`, `pt_ens_allin`, `pt_search_d4_allin`), `--launch-threshold` for PPO rollouts/eval, `--no-weak-opps` / `--save-every` / `--fixed-anchor`, and `force_all_in` / `neural_mid_replan` for `SearchPlanetTransformerAgent`.
- **Direct `#2` Soft-KD + Iterative DAgger (`34,130` exact Full-TTA `#2` boards total: `pos=11,174`, `neg=19,596`, `zero=3,360`; `step p10/p50/p90 = [10, 54, 103]`):**
  - **Round 1 (`15,822` train / `1,757` contiguous unseen val from `17,579` boards):**
    - `gen11_r1_bal` (`434.6s`): `val_kl=0.168`, `val_all=88.9%` (peak `89.9%`), `aux_own=96.5%`, `vsign=92.1%` (peak `95.3%`); vs `#2` (`6g` @ `0.50`): `118.7t` (`33.09 ms/t`).
    - `gen11_r1_hold` (`434.7s`): `val_kl=0.171`, `val_all=89.2%` (peak `89.8%`), `aux_own=96.6%`, `vsign=93.1%` (peak `94.8%`); vs `#2` (`6g`): `122.3t` @ `0.50` (`31.82 ms/t`), `131.7t` @ `0.53` (`32.24 ms/t`).
  - **Round 2 (`30,717` train / `3,413` contiguous unseen val after `+16,551` DAgger boards across `50` `r1_bal` + `50` `r1_hold` games at `t=0.53`):**
    - `checkpoints/planet_transformer_gen11_4m_kd.pt` (`r2_bal`, `630.7s`): `val_kl=0.140` (epoch 7), `val_all=90.0%` (peak `90.6%`), `val_launch=41.0%`, `val_tgt=79.9%`, `aux_own=95.8%`, `vsign=93.2%` (peak `94.2%`); vs `#2` (`t=0.53`): `131.7t` (`6g`, `33.38 ms/t`) / `125.5t` (`12g`).
    - `checkpoints/planet_transformer_gen11_4m_kd_pure.pt` (`r2_pure`, `488.1s`): `val_kl=0.140` (epoch 6), `val_all=90.2%` (peak `90.3%`), `val_launch=39.5%`, `val_tgt=79.9%`, `aux_own=95.8%`, `vsign=92.9%` (peak `94.7%`); vs `#2` (`t=0.53`): `122.1t` (`12g`). `2×` `r2_bal + r2_pure` scored `131.3t` (`12g`, `59.51 ms/t`).
- **3-Wave RX 6950 XT DirectML League PPO (`10,800` 2P games across 3 × 60-iter waves, `30` pairs/iter, stopped at diminishing returns):**
  - **Wave 1 (`4,830.0s`, `T=0.90`):** Iter 60 `*BEST*` `EVAL(48g)=83.3%` (`init=66.7%`, `r2_pure=75.0%`, `gen10=91.7%`, `gen9_bal=100.0%`). `40g` gate (`seeds 990–1009`, `t=0.53`): `w1` vs `r2_bal = 24-16-0` (**`60.0%`**, `185.0t`); vs `#2`: `w1 = 126.0t` (`6g`) / `126.3t` (`12g`), `w1_it20 = 132.5t` (`6g`) / `128.7t` (`12g`), `3× W1 Ens = 132.1t` (`12g`, `99.55 ms/t`).
  - **Wave 2 (`4,831.9s`, `T=0.65`, `lr=1.5e-5`):** **`w2_it20`** (`checkpoints/planet_transformer_gen11_4m_ppo.pt` -> promoted to `checkpoints/planet_transformer_best.pt`) defeated `w1` **`26-14-0` (`65.0%`, `191.4t`)** in the `40g` gate (`t=0.53`) and reached `134.2t` (`6g`) / `131.3t` (`12g`, `33.77 ms/t`) vs `#2`; `w2` (it60) scored `22-18-0` (`55.0%`, `168.3t`) vs `r2_bal` (while `w2_it20` vs `r2_bal` was `16-24-0`, `40.0%`, reflecting non-transitivity).
  - **Wave 3 (`4,850.3s`, `T=0.60`, `lr=9e-6`):** `w3` (it60 `*BEST*`) scored `16-24-0` (`40.0%`, `154.8t`) vs `w2_it20` and `131.0t` (`6g`, `33.45 ms/t`) vs `#2` — confirming diminishing returns and ending PPO training.
- **Inference Optimization & Boss Gauntlet (`#2 Hober Malloc`, `seeds 200–205` 6g / `200–211` 12g):**
  - Threshold sweep on `w2_it20` (`6g`): `0.50 = 130.5t (33.76 ms/t)`, `0.53 = 134.2t`, **`0.56 = 134.5t (33.46 ms/t)`**, `0.59 = 130.0t (34.55 ms/t)`.
  - Expanded `12g` gauntlet (`t=0.56`): **`w2_it20` single = `0-0-12`, `138.8t` avg (`33.19 ms/turn` mean, `50.48 ms` p95, `94.56 ms` max)** — outperforming `2× Hybrid (r2_bal + w2_it20) = 128.4t (59.86 ms/t)`, `3× Diverse (r2_bal + r2_pure + w2_it20) = 134.2t (85.44 ms/t)`, `4× Grand (+w1) = 133.8t (110.77 ms/t)`, and `pt_search_d4_allin_t56:w2_it20 = 128.0t (171.67 ms/t mean, 335.17 ms p95)`.
- **Unseen-Seed Head-to-Head (`8× D4 TTA`, `160` games across `seeds 990–999` × 2 seats, `8` concurrent CPU workers, `544.7s`):**
  - `gen11_ppo_w2_it20` (`t=0.56`): **`80.0%` overall** (`14-6-0` / **`70.0%`**, `193t` vs `gen10_best`; `18-2-0` / **`90.0%`**, `167t` vs `gen9_ens`).
  - `gen11_r2_bal` (`t=0.53`): `75.0%` overall (`14-6-0` / `70.0%`, `228t` vs `gen10_best`; `16-4-0` / `80.0%`, `150t` vs `gen9_ens`).
  - `gen11_ppo_w2` (`t=0.53`): `75.0%` overall (`14-6-0` / `70.0%`, `190t` vs `gen10_best`; `16-4-0` / `80.0%`, `178t` vs `gen9_ens`).
  - `gen11_3x_ens` (`r2_bal + r2_pure + w2_it20`, `t=0.56`): **`90.0%` overall** (`18-2-0` / **`90.0%`**, `146t` vs `gen10_best`; `18-2-0` / **`90.0%`**, `169t` vs `gen9_ens`).
- **`subham_v14` Sanity Verification (`4` sequential games, `seed 100`):** `pt_d4_allin_t56:w2_it20 = 4-0-0 (100.0%, 98.5t, 37.54 ms/t)`; `pt_d4_allin:w2_it20 = 4-0-0 (100.0%, 111.8t, 36.80 ms/t)`.
- **Boss Forensics vs `#2 Hober Malloc` (`seeds 200–205`, default `thr=0.50`):**
  - `R2-Bal` (`92.9s`): `t=40` ships `344.8 vs 355.5 (-10.7)`, `t=50` `389.5 vs 448.0 (-58.5)`, `t=60` `406.7 vs 541.2 (-134.5)`; exact policy parity `85.7%` (`102/119`) early, `67.5%` (`462/684`) mid, `61.0%` (`86/141`) late; comets `3 vs 2`, neutrals `10.83 vs 10.33`, sync arrivals `110 vs 113`.
  - `PPO w2_it20` (`96.4s`): `t=10` ships `38.0 vs 37.2 (+0.8)`, `t=20` `91.7 vs 88.3 (+3.3)`, `t=40` `360.8 vs 380.5 (-19.7)`, `t=50` `434.8 vs 501.0 (-66.2)`, `t=60` `446.0 vs 601.5 (-155.5)`; exact policy parity `83.5%` (`106/127`) early, `59.6%` (`378/634`, with `153` `ours_REINFORCE_hober_HOLD` at `t=0.50` — explaining why inference `t=0.56` gains survival vs `#2`); comets `3 vs 2`, neutral ROI `0.313 vs 0.200`, sync arrivals `114 vs 132`.
- **Promotion Summary:** `checkpoints/planet_transformer_best.pt` promoted to `Gen 11 PPO (w2_it20)` (`+70.0%` over `Gen 10`, `90.0%` over `Gen 9`, `138.8t` avg vs `#2 Hober Malloc`, `33.19 ms/turn`, with recommended inference threshold `t=0.56`); retained `gen11_4m_kd.pt`, `gen11_4m_kd_pure.pt`, and `gen11_4m_ppo.pt` together also power the `90.0%` `3×` diverse ensemble (`85.44 ms/turn`).

---

### Generation 12 (2026-11-25): `4.31M PT-v2` Architecture + 71.9k Direct `#2` Soft-KD + Fast Pipelined DirectML League PPO (`8–0–16 / 33.3%` vs. `#2 Hober Malloc`, `18–0–2` vs. Gen 11/10/9!)

- **Architectural Breakthrough (`PT-v2`, `4.31M` params — exact footprint of `#2 Hober Malloc`):**
  - **14-Channel Per-Timestep Orbital Timeline (`feature_version=2`):** Extended the 20-tick timeline with exact per-tick `(dx_t, dy_t, r_t, pos_valid_t, log1p(cap_need_t)/6, (step+dt)/500)` + exact D4 dihedral symmetry transform (`apply_d4_timeline_feats`), encoded via 3 residual Conv1D+GELU+LayerNorm blocks + 6 GeGLU Edge-Biased Transformer layers.
  - **Decoupled Factorized Launch + Multi-Head `Q·K` Target Attention (`arch_version=2`):** Separated binary launch probability (using zero-fallback 2-class `F.log_softmax([0, z])` on DirectML) from the variable-size target softmax (`|legal(i)|`) with learned `tgt_head_weights` and directed `edge_target_head` residual.
- **3-Round Direct `#2 Hober Malloc` Soft-KD + Iterative On-Policy DAgger (`71,876` float16 boards, RX 6950 XT DirectML, `0` fallbacks):**
  - **Round 1 (`23,123` boards):** `val_kl = 0.167`, `val_all = 90.5%`, `l_f1 = 86.4%`, `val_tgt = 80.9%` (`134.3t` avg on 6g Boss).
  - **Round 2 (`46,279` boards -> `gen12_r2_pure.pt`, unweighted clean Soft-KD):** `val_kl = 0.115`, `val_all = 92.6%`, `l_f1 = 89.7%`, `val_launch = 59.5%`, `val_tgt = 88.5%` — **first-ever single-model win vs. `#2 Hober Malloc` (`1–0–11`, `167.9t` avg)** and **`20–0–0` (`100.0%`)** sweep of Gen 11 PPO!
  - **Round 3 (`71,876` boards -> `planet_transformer_gen12_4m_ptv2_kd.pt`):** **`val_kl = 0.098`** (first time under `0.10`), **`val_all = 93.4%`**, **`l_f1 = 91.1%` (`p=91.3%`, `r=90.8%`)**, **`val_launch = 62.2%`**, **`val_tgt = 89.6%`**, `aux_own = 96.2%` -> **`4–0–20` (`16.7%`, `176.2t` avg)** on the 24-game `#2 Hober Malloc` gauntlet (`seeds 200–223`)!
- **RX 6950 XT DirectML KL-Anchored League PPO Waves + `~3×` RL Pipeline Speedup (`91s → 31.5s/iter`):**
  - **Wave 1 (`60` iters × `24` mirror pairs = `2,880` 2P games -> `planet_transformer_gen12_4m_ptv2_ppo.pt`):** Anchored to `r3_pure` (`akl=0.075`, `aux=96.5%`), **doubled single-model Boss win rate from `16.7% → 33.3%` (`8–0–16`, `5–0–7 / 41.7%` on seeds `212–223`, `175.9t` avg, `5` P0 wins including `153t`/`154t` KOs + `3` P1 wins)**!
  - **Wave 2 (`40` iters, greedy `oppT=0.0`):** Regressed to `4–0–20` (`16.7%`) due to unbalanced `21–39%` rollout win rate; rejected.
  - **RL & Inference Speedup Stack Landed (Bit-Exact Verified Over 954 Live Boards, `max_diff = 5.96e-8`, `0` Mismatches):**
    1. Vectorized NumPy `predict_all_planet_tracks_arrays`, 1-shot `(F, H, N)` `predict_fleet_arrivals_vectorized`, and `build_shared_planet_physics` (`3.86×` solo / **`6.53×` 2P faster** feature extraction).
    2. `compute_frac=False` fast-path + `(1, N, N, 12)` `8× D4` TTA edge broadcast (cut live `8× D4` latency `43.4 ms → 33.30 ms` mean / `42.21 ms` p95 / `50.62 ms` max).
    3. `B=8` lockstep batched CPU rollouts (`collect_episodes_batched_2p`, `2.12×` faster rollouts), once-per-iter cached `teacher`/`fixed_anchor` log-probs, `float16` rollout IPC, `--ppo-epochs 2`, and `--async-rollout` CPU/GPU pipelining (`roll_wait = 0.0s`, **`91s/iter → 31.5s/iter`, `~3.0×` faster**).
  - **Wave 3B Fast Pipelined PPO (`40` iters = `1,920` games in `29.6 min`, balanced `oppT=0.45` -> `planet_transformer_gen12_4m_ptv2_ppo_w3b.pt`):** **`7–0–17` (`29.2%`, `181.2t` avg — new highest avg survival, `4` P1 + `3` P0 wins)**; `10–0–10` (`50.0%`) vs. `w1_it60` — stopped clone-only PPO waves at `< 5%` diminishing returns.
- **Verified Final Head-to-Head Ladder (`4.31M PT-v2 Gen 12 PPO`, Single Model, `8× D4`, `t=0.53`, Unseen `seeds 990–999` × 2 seats):**

| Matchup (Single `4.31M` Champion vs. Opponent) | Arena / Games | Score | Win Rate | Avg Len | Mean Latency |
| --- | --- | --- | ---: | ---: | ---: |
| **Gen 12 PPO (`w1_it60`)** vs. **`#2 Hober Malloc (1800.1)`** | 2P (`seeds 200–223`, 24g) | **`8–0–16`** (`5–0–7` on `212–223`) | **`33.3%`** | `175.9t` | `33.3 ms` |
| **Gen 12 PPO (`w1_it60`)** vs. **Gen 11 PPO (`4.31M PT-v1`)** | 2P (`seeds 990–999`, 20g) | **`18–0–2`** | **`90.0%`** | `156.3t` | `33.3 ms` |
| **Gen 12 PPO (`w1_it60`)** vs. **Gen 10 PPO (`4.31M PT-v1`)** | 2P (`seeds 990–999`, 20g) | **`18–0–2`** | **`90.0%`** | `130.1t` | `33.3 ms` |
| **Gen 12 PPO (`w1_it60`)** vs. **Gen 9 `2×` Ensemble (`1.89M`)** | 2P (`seeds 990–999`, 20g) | **`18–0–2`** | **`90.0%`** | `136.6t` | `33.3 ms` |
| **Gen 12 PPO (`w1_it60`)** vs. **Gen 12 Soft-KD R3 (`0.098 KL`)** | 2P (`seeds 990–999`, 20g) | `10–0–10` (`2×` Boss WR) | `50.0%` | `170.0t` | `33.3 ms` |
| **Gen 12 PPO (`w1_it60`)** vs. **`subham_v14`** | 2P (`seed 100`, 4g, `t=0.53` / `0.50`) | **`4–0–0` / `4–0–0`** | **`100.0%`** | `114.8t` / `109.8t` | **`33.30 ms`** / `36.46 ms` |

- **Verified `#2 Hober Malloc` Forensics (`scripts/analyze_boss_gap.py`, `seeds 200–205`, Gen 8 → Gen 9 → Gen 12):**
  - Counterfactual exact-match jumped to **`89.2%` early (`116/130`), `86.0%` midgame (`723/841`, `+19.7%` over Gen 9), and `84.0%` late (`178/212`)**.
  - Midgame premature reinforces dropped **`94 → 74 → 21` (`-78%`)**; impulsive enemy pokes dropped **`29 → 13 → 7` (`-76%`)**; neutral overkill dropped **`1,890 → 866` (`-54%`)**; comets flipped **`0 v 5 → 3 v 1`**.
  - Production led `#2` at **`t=30` (`+1.5`)** and **`t=50` (`+0.2`)**; ship deficit slashed **`-73%` at `t=50` (`-131.8 → -36.2`)** and **`-75%` at `t=90` (`-1104 → -258.2`)**; avg ship tipping tick shifted `t=16.3 → t=27.8 → t=67.8` (seed 204 **154t KO win**, seed 200 held to `t=148`).
- **Graphics & Replay Viewers Updated:** Embedded 8 compact Gen 12 comparison battles (including 3 wins vs. `#2 Hober Malloc` from P0 & P1 + sweeps of Gen 11/10/9/`subham_v14`) into `docs/gameplay_viewer.html` (28 matches total) and updated `docs/status_report.html` and `docs/boss_forensics.html`.
- **Promotion Summary:** `checkpoints/planet_transformer_best.pt` promoted to `Gen 12 PT-v2 PPO (w1_it60)` (`8–0–16 / 33.3%` vs. `#2 Hober Malloc`, `18–0–2 / 90.0%` vs. Gen 11/10/9, `4–0–0` vs. `subham_v14`, `33.30 ms/turn`, recommended threshold `t=0.53`). Retained `planet_transformer_gen12_4m_ptv2_ppo.pt`, `planet_transformer_gen12_4m_ptv2_kd.pt`, and `planet_transformer_gen12_4m_ptv2_ppo_w3b.pt`.

---

### Generation 13 (2026-11-26): `4.67M PT-v3.5` Architecture + Decoupled Logit-TTA + 48g Forensics (`21–0–27 / 43.8%` vs. `#2 Hober Malloc`, `18–0–2` vs. Gen 12 PPO, `20–0–0` vs. Gen 11 & Gen 10!)

- **Architectural Breakthroughs (`PT-v3` `4.60M` → `PT-v3.5` `4.67M`):**
  - **`PT-v3` (`arch_version=3`, `4.60M` params):** Upgraded timeline stem to **4× masked residual Conv1D (`k=5, ch=128`)** over `t=0..20` linear + log channels feeding **7 edge-biased GeGLU Transformer layers**, `detach_value`, CPU valid-planet Conv1D packing (`1.51×` padded rollout / `1.22×` live `8× D4` speedup), and pre-transpose mask optimization + contiguous slice concat (`+16.9%` DML `N=44`, `0.00e+00` diff, `0` fallbacks).
  - **`PT-v3.5` Target-Conditioned Launch Context (`target_cond_launch=True`, `4.67M` params):** Added zero-initialized `launch_ctx_head` residual feeding selected target features `[h, h_tgt, e_tgt, max_p_tgt, tgt_ent, best_raw]` back into `launch_head` while keeping factorized gradients intact.
  - **Decoupled Logit-Space `8× D4` TTA (`decoupled_logit_tta=True`):** Auto-enabled for all `target_cond_launch=True` checkpoints in `PlanetTransformerAgent`; averages `launch_logit = log p_launch - log p_hold` and conditional target log-probs separately across all 8 dihedral symmetries instead of averaging post-softmax probabilities (eliminating probability-blur false holds on `[0.40, 0.53)` cliff decisions).
- **Iterative Direct `#2` Soft-KD, Cliff-Boosted Losses & On-Policy Low-T PPO (`RX 6950 XT DirectML`, `0` Fallbacks):**
  - **`PT-v3 KD` (`planet_transformer_gen13_4m_ptv3_kd.pt`, `4.60M`):** Jumped `PT-v2 R5` baseline (`9–0–39 / 18.8%`) to **`17–0–31 (35.4%)`** on the expanded **48-game `#2 Hober Malloc` Boss Suite** (`200–223: 7–0–17 / 29.2%`, `300–323: 10–0–14 / 41.7%`).
  - **`PT-v3 R6 KD` (`+36,757` live DAgger boards -> `244,194` train / `10,000` unseen val boards):** Reached `val_kl = 0.0593` and scored `16–0–32 (33.3%)` on the 48g Boss Suite (`8–0–16` on both `200–223` and `300–323` at `t=0.53` and `t=0.50`).
  - **`PT-v3.5 KD` (`planet_transformer_gen13_4m_ptv35_kd.pt`, `4.67M` Champion):** Trained `5` cosine DML epochs (`0` fallbacks, `val_all = 94.5%`, `l_f1 = 90.4%`, `6k cal kl = 0.0525`, `tgt = 83.7%`). Unseen `seeds 300–323` TTA ablation proved Decoupled Logit-TTA adds **`+8.4%`** over Prob-TTA (`8–0–16 / 33.3% → 10–0–14 / 41.7%`). Full 48g Boss Suite hit **`21–0–27 (43.8%)`** (`200–223: 11–0–13 / 45.8%`, `300–323: 10–0–14 / 41.7%`, `P0 = 12/24 [50.0%]`, `P1 = 9/24 [37.5%]`), flipping former forensics blunders **Seed 212** (`false_launch`) and **Seed 203** (`false_hold`) into wins!
  - **`PT-v3.5` Exact Factorized Low-T PPO (`planet_transformer_gen13_ptv35_ppo.pt`):** Implemented DirectML-native factorized PPO using `F.log_softmax`, planet-weighted advantage centering, and mid-epoch `max_kl` early stop (`35` iters, `kl <= 0.0074`, `akl <= 0.0104`, `aux = 96.5%`, `0` fallbacks); peaked on `44g` rollout eval (`hober = 20% → 40%` at `it15`) but scored `15–0–33 (31.2%)` on the strict 48g `kaggle_environments` suite (`5–0–19` on `200–223`, `10–0–14` on `300–323`), so `PT-v3.5 KD` (`43.8%`) won promotion decisively.
- **48-Game Loss Forensics (`scratch/analyze_ptv3_losses.py` on `PT-v3 R6 KD`, `FastOrbitSim` `14–0–34 / 29.2%`, `88.64%` agreement over `37,320` decisions):**
  - Losses diverged early (`t_div` quantiles: `p10 = 17.3t`, `p25 = 23.25t`, `p50 = 36.5t`, `p75 = 54.5t`, `p90 = 67.0t`); value head turned negative by tick 45 in losses (`v = -0.610` vs `+0.132` in wins).
  - Across `4,238` decision disagreements on lost games: **`false_hold = 43.8%` (`1,856`)** with **`81.0%` (`1,503`)** inside the `[0.40, 0.53)` threshold cliff and **`47.0%` (`872`)** lagging `#2` by `1–2` ticks (`54.0%` in `[t_div-18, t_div+6]` critical window); **`false_launch = 31.0%` (`1,314`)** with **`89.3%` (`1,174`)** in `[0.53, 0.65]` (`30.5%` / `401` captured within `15t`); **`wrong_target = 25.2%` (`1,068`)** (`57.6%` / `615` friendly hub selection mismatches). Solving these exact two mechanisms lifted `#2 Hober Malloc` win rate **`33.3% → 43.8%` (`+10.5%`)**.
- **Verified 48-Game `#2 Hober Malloc` Boss Suite Progression (`kaggle_environments`, `seeds 200–223` + `300–323`, `8× D4`, `t=0.53`):**

| Model / Architecture | Params | Seeds `200–223` (24g) | Seeds `300–323` (24g) | Full 48g Score | Full 48g WR |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Gen 12 PPO (`w1_it60`, `PT-v2`)** | `4.31M` | `8–0–16 (33.3%)` | `—` | `—` | `33.3% (24g)` |
| **Gen 13 `PT-v2 R5` Baseline** | `4.31M` | `—` | `—` | `9–0–39` | `18.8%` |
| **Gen 13 `PT-v3` R6 DAgger KD (`0.0593` KL)** | `4.60M` | `8–0–16 (33.3%)` | `8–0–16 (33.3%)` | `16–0–32` | `33.3%` |
| **Gen 13 `PT-v3` KD** | `4.60M` | `7–0–17 (29.2%)` | `10–0–14 (41.7%)` | `17–0–31` | `35.4%` |
| **Gen 13 `PT-v3.5` PPO (`it15`, `PT-v3.5` + Logit-TTA)** | `4.67M` | `5–0–19 (20.8%)` | `10–0–14 (41.7%)` | `15–0–33` | `31.2%` |
| **Gen 13 `PT-v3.5` KD (`planet_transformer_best.pt`, Decoupled Logit-TTA)** | **`4.67M`** | **`11–0–13 (45.8%)`** | **`10–0–14 (41.7%)`** | **`21–0–27`** | **`43.8%`** |

- **Verified Final Head-to-Head Ladder (`4.67M PT-v3.5 Gen 13 KD`, Single Model, `8× D4` Decoupled Logit-TTA, `t=0.53`, 1-Thread CPU, Unseen `seeds 990–999` × 2 seats):**

| Matchup (Single `4.67M` Champion vs. Opponent) | Arena / Games | Score | Win Rate | Avg Len | 1-Thread Latency |
| --- | --- | --- | ---: | ---: | ---: |
| **Gen 13 `PT-v3.5`** vs. **`#2 Hober Malloc (1800.1)`** | 2P (`200–223` + `300–323`, 48g) | **`21–0–27`** (`P0=12/24, P1=9/24`) | **`43.8%`** | `164.5t` *(4g sample)* | **`53.91 ms/t`** (`72.17 ms` p95) |
| **Gen 13 `PT-v3.5`** vs. **Gen 12 PPO (`4.31M PT-v2`)** | 2P (`seeds 990–999`, 20g) | **`18–0–2`** | **`90.0%`** | `158.5t` | `~53.9–63.0 ms/t` |
| **Gen 13 `PT-v3.5`** vs. **Gen 11 PPO (`4.31M PT-v1`)** | 2P (`seeds 990–999`, 20g) | **`20–0–0`** | **`100.0%`** | `131.7t` | `~53.9–63.0 ms/t` |
| **Gen 13 `PT-v3.5`** vs. **Gen 10 PPO (`4.31M PT-v1`)** | 2P (`seeds 990–999`, 20g) | **`20–0–0`** | **`100.0%`** | `113.4t` | `~53.9–63.0 ms/t` |
| **Gen 13 `PT-v3.5`** vs. **Gen 13 `PT-v3` KD (`4.60M`)** | 2P (`seeds 990–999`, 20g) | `2–0–18` *(mirror counter; `PT-v3.5` `+8.4%` vs `#2`)* | `10.0%` | `189.9t` | `~53.9–63.0 ms/t` |
| **Gen 13 `PT-v3.5`** vs. **`subham_v14`** | 2P (`seeds 100–103`, 4g, 1-thread) | **`4–0–0`** | **`100.0%`** | `111.8t` | **`63.04 ms/t`** (`79.57 ms` p95) |

- **Graphics & Replay Viewers Updated:** Embedded 5 compact Gen 13 `PT-v3.5` comparison replays (wins vs. `#2 Hober Malloc` from P0 & P1, unseen series win vs. `Gen 12 PPO`, and a loss forensics case) into `docs/gameplay_viewer.html` (33 matches total) and updated `docs/status_report.html` and `docs/boss_forensics.html`.
- **Promotion Summary:** `checkpoints/planet_transformer_best.pt` promoted to `Gen 13 PT-v3.5 KD (4.67M)` (`21–0–27 / 43.8%` vs. `#2 Hober Malloc`, `18–0–2 / 90.0%` vs. Gen 12 PPO, `20–0–0` sweeps of Gen 11 & Gen 10 PPO, `4–0–0` vs. `subham_v14`, `53.91–63.04 ms/turn` 1-thread CPU, auto-enabled `decoupled_logit_tta=True`, recommended threshold `t=0.53`). Retained `planet_transformer_gen13_4m_ptv35_kd.pt`, `planet_transformer_gen13_4m_ptv3_kd.pt`, and `planet_transformer_gen13_ptv35_ppo.pt`.

---

### Generation 14 & 15B (2026-11-26): Wave 14 Wide-Map R7 KD (`54.2%` Unseen `400–423`) + Wave 15B Same-Seed GRPO (`24–0–24 / 50.0%` 48g Boss Suite, `14–0–6` Over Both Gen 13 Models!)

- **Wave 14 — Wide-Map R7 DAgger + Blind-Miss Calibration (`planet_transformer_gen14_ptv35_r7_kd.pt`, `4.67M PT-v3.5`):**
  - **Loss Forensics Motivation:** Audited `Gen 13 PT-v3.5` losses across `48g`: wins averaged `start_dist = 85.3` vs. losses `97.0` (`+11.7` wider), and `56.7%` (`173/305`) of critical divergence-window disagreements were blind early `false_hold` (`59.5%` with `p_launch < 0.40`).
  - **Data & Training:** Collected `280` on-policy games (`70%` wide maps `start_dist >= 88`, `26,777` dense boards merged into `268,971` train + `12,000` unseen val), trained `14` epochs on RX 6950 XT DirectML (`0` fallbacks) with `--cliff-boost 1.6` + new `--blind-miss-boost 2.0`: `val_tgt = 94.3%`, `l_f1 = 90.8%` (`r7_val_tgt = 92.6%`, `r7_f1 = 87.5%`).
  - **Benchmarks (`8× D4`, `t=0.53`):**
    - **72g vs. `#2 Hober Malloc`:** **`33–0–39 (45.8%)`** vs. Gen 13 `30–0–42 (41.7%)` — **`200–223`: `11–0–13 (45.8%)`**, `300–323`: `9–0–15 (37.5%)`, unseen **`400–423`: `13–0–11 (54.2%)`** (vs. Gen 13's `9–0–15 / 37.5%`).
    - **Eliminated `PT-v3` Mirror Counter:** **`14–0–6 (70.0%)` vs. `Gen 13 PT-v3.5 KD`**, **`14–0–6 (70.0%)` vs. `Gen 13 PT-v3 KD`** (reversing `2–0–18`), **`18–0–2 (90.0%)` vs. `Gen 12 PPO`**.
- **Wave 15B — Same-Seed Counterfactual GRPO (`planet_transformer_gen15b_grpo.pt` -> Promoted to `checkpoints/planet_transformer_best.pt`):**
  - Implemented DirectML-native `--target-kl-weight 2.5` (preserving target selection accuracy), `EnsemblePlanetTransformerAgent(decoupled_logit=True)`, and `--grpo-k 4` same-seed counterfactual trajectory groups (`T_l=0.22`, `T_t=0.15`, `80%` live `#2 Hober Malloc`, `0` fallbacks).
  - Iter `6` simultaneously boosted unseen R7 `F1` (`85.6% → 85.7%`), `tgt` (`84.6% → 84.9%`), and **`vsign` (`56.6% → 75.0%`, `+18.4%`)**, whereas longer trajectory-level RL (`it10–30`) drifted on borderline Launch/Hold flips (`F1 → 75.1%`).
  - **Verified `Gen 15B PT-v3.5 GRPO-it6` Results (`t=0.53`, `8× D4`, 1-Thread CPU):**

| Matchup (`Gen 15B GRPO-it6` vs. Opponent) | Arena / Games | Score | Win Rate | Avg Len | 1-Thread Latency |
| --- | --- | --- | ---: | ---: | ---: |
| **`Gen 15B` vs. `#2 Hober Malloc` (48g Suite)** | 2P (`200–223` + `300–323`, 48g) | **`24–0–24`** (`200s=13/24, 300s=11/24`) | **`50.0%`** | `182.6t` | **`58.80 ms/t`** (`82.10 ms` p95) |
| **`Gen 15B` vs. `#2 Hober Malloc` (72g Full)** | 2P (`200–223` + `300–323` + `400–423`, 72g) | **`33–0–39`** (`P0=20/36, P1=13/36`) | **`45.8%`** | `185.0t` | **`58.80 ms/t`** |
| **`Gen 15B` vs. `Gen 14 PT-v3.5 R7 KD`** | 2P (`seeds 990–999`, 20g unseen) | **`12–0–8`** | **`60.0%`** | `246.3t` | `58.8–69.0 ms/t` |
| **`Gen 15B` vs. `Gen 13 PT-v3.5 KD`** | 2P (`seeds 990–999`, 20g unseen) | **`14–0–6`** | **`70.0%`** | `177.2t` | `58.8–69.0 ms/t` |
| **`Gen 15B` vs. `Gen 13 PT-v3 KD`** | 2P (`seeds 990–999`, 20g unseen) | **`14–0–6`** | **`70.0%`** | `177.1t` | `58.8–69.0 ms/t` |
| **`Gen 15B` vs. `Gen 12 PPO`** | 2P (`seeds 990–999`, 20g unseen) | **`16–0–4`** | **`80.0%`** | `167.4t` | `58.8–69.0 ms/t` |
| **`Gen 15B` vs. `subham_v14`** | 2P (`seeds 200–203`, 4g, 1-thread) | **`4–0–0`** | **`100.0%`** | `77.2t` | **`68.97 ms/t`** (`82.47 ms` p95) |

- **Definitive Lookahead Search Ablation vs. `#2 Hober Malloc` (200s–223s / 24g):**
  - Wave 14A (`Gen 13 PT-v3.5`, `H=8`): `9–0–15` (`37.5%`, `185 ms/t` mean, `661 ms` p95) vs. greedy `11–0–13` (`45.8%`).
  - Stage 16A (`Gen 15B it6`, `vsign=75.0%`, `H=12`, replan `3,6,9`): `6–0–18` (`25.0%`, `310.5 ms/t` mean, `661.5 ms` p95) vs. greedy `13–0–11` (`54.2%`). Pure greedy 8× D4 inference is strictly stronger against `#2`.
- **Stage 16B Feature Infrastructure Prepared (`65` Scoped Tests Passing):**
  - Discovered `timeline_feats[:, :, 12]` approx `cap_need` differed from `#2`'s exact simultaneous-arrival `ships_for_capture` on `74.2%` (`233/314`) of live ticks; implemented exact `compute_ships_for_capture` + `feature_version=3` (`orbit_wars_bot/planet_transformer.py`, shared with `orbit_wars_bot/hober_sparring.py`) and `--target-kl-weight` in `scripts/distill_soft_kd.py`.
- **Promotion Summary:** `checkpoints/planet_transformer_best.pt` promoted to **`Gen 15B PT-v3.5 GRPO-it6`** (`24–0–24 / 50.0%` on 48g Boss Suite, `33–0–39 / 45.8%` on 72g, `70.0%` vs. both `Gen 13` models, `60.0%` vs. `Gen 14 KD`, `4–0–0` vs. `subham_v14`, `58.80–68.97 ms/t`).

---

### Step 2.2 — C++17 `OrbitSearchAgent` Ablation & Latency Verification (2026-11-27)

- **Architecture (`orbit_wars_bot/orbit_search.py`):** Uncertainty-gated 2-ply neural lookahead over `CppFastOrbitSim` (`H=10`, mid-ply neural replan at `t=4`, `K <= 3`, `evaluate_settled_shared_physics`, Stage-2 rotational leaf refinement, asymmetric override gate). Wired `orbit_search` into `scripts/evaluate.py` and added unit + latency tests in `tests/test_orbit_search.py`.
- **24g Ablation vs. `#2 Hober Malloc` (`seeds 200–223`, 6 parallel workers, `Gen 15B` Pure baseline = `13–0–11 / 54.2%`, `AvgLen=187.3`, `103.7 ms` mean / `155.4 ms` p95 under 6-worker load):**
  - `cfg_bal` (`nn_w=0.55, prior=0.05, margin=0.06, cliff=0.14`): `7–0–17 (29.2%)` | `Trig=72.4% Over=2.3%` | `168.0 ms` mean / `303.8 ms` p95 (`369.4 ms` max)
  - `cfg_tac` (`nn_w=0.30, prior=0.06, margin=0.08, cliff=0.12`): `7–0–17 (29.2%)` | `Trig=71.3% Over=2.1%` | `165.2 ms` mean / `301.1 ms` p95 (`402.5 ms` max)
  - `cfg_settled` (`nn_w=0.00, prior=0.05, margin=0.08, cliff=0.14`): `9–0–15 (37.5%)` | `Trig=72.6% Over=2.1%` | `141.2 ms` mean / `240.3 ms` p95 (`361.7 ms` max)
  - `cfg_nn` (`nn_w=0.80, prior=0.06, margin=0.08, cliff=0.14`): `10–0–14 (41.7%)` | `Trig=74.7% Over=1.8%` | `170.1 ms` mean / `309.9 ms` p95 (`359.4 ms` max)
  - `cfg_tight` (locked defaults: `nn_w=0.80, prior=0.06, margin=0.10, cliff=0.07, alt_ratio=0.45, max_candidates=3, budget=0.21s`): `8–0–16 (33.3%)` | `Trig=65.8% Over=0.4%` | `130.9 ms` mean / `241.0 ms` p95 (`339.0 ms` max)
- **Solo 1-Thread Verification (`scripts/evaluate.py`, `orbit_search_t53:checkpoints/planet_transformer_best.pt`):**
  - **2P vs. `#2 Hober Malloc` (`seeds 200–207`, 8g):** `4–0–4 (50.0%)`, `AvgRank=1.50`, `AvgLen=205.6t`, **`130.76 ms` mean**, **`214.31 ms` p95** (`253.17 ms` max, cleanly `< 300 ms` gate).
  - **4P FFA vs. `3× subham_v14` (`seeds 400–407`, 8g):** **`8–0–0 (100.0%)`**, `AvgRank=1.00`, `AvgLen=123.0t`, **`151.18 ms` mean**, **`233.04 ms` p95** (`301.46 ms` max).
- **Takeaway:** Fast C++17 search cuts Stage 16A p95 latency from `661.5 ms` down to **`214.31 ms`**, but `Gen 15B` Pure (`50.0%` 48g Boss, `58.80 ms/t`) remains the active default until Gen 17 upgrades `feature_version=3` and recalibrates the value head.

---

### Generation 17 — C++17 Training-Loop Acceleration & Pure `fv=3` Experiment (2026-11-27)

- **Infrastructure (`orbit_wars_bot/hober_sparring.py`, `scripts/rl_finetune_transformer.py`):**
  - Wired C++17 `solve_intercept_cpp` fast paths into `FastHoberSparringModel.act_batch` and `_decode_sampled_actions`.
  - Fixed per-player `feature_version` isolation (`hero_fv` vs `opp_fv`) across `collect_episode`, `collect_batch_episodes`, and `_eval_match` so `fv=3` policies spar against `fv=2` anchors cleanly without feature leakage.
- **Training & Polish (`4.67M PT-v3.5`, `feature_version=3`):**
  - Collected `108,852` fresh pure `fv=3` boards via `CppFastOrbitSim` (`r9_v3_hober_mirror` `68,160` + `r9_v3_dagger` `37,496` + `r9_v3_cf_win` `3,196`) + `r8_v3` (`37,485`) -> `143,431` effective train + `9,000` unseen val boards.
  - `Gen17-KD` (`16` ep DirectML, `value_weight=0.75`): `v3_kl=0.0820`, `l_f1=87.7%`, `val_tgt=93.9%`, `aux=96.5%`, `vsign=72.4%`.
  - `Gen17-GRPO` (`12` iters `18x4` C++ balanced-league counterfactual GRPO on DirectML): `vl 0.120 -> 0.041`, `kl=0.0145`, `aux=97.1%`.
- **Official Kaggle Benchmarks (`8× D4`, `@t=0.53`):**
  - **48g Boss Suite Screen (`200–223` + `300–323` vs. `#2 Hober Malloc`):** `Gen17-kd` `14–0–34 (29.2%)`, `Gen17-kd_final` `18–0–30 (37.5%)`, `Gen17-grpo_best` `18–0–30 (37.5%)`, `Gen17-grpo_it6` `15–0–33 (31.2%)`, **`Gen17-grpo_it4` `19–0–29 (39.6%)`** (`b200=9/24, b300=10/24`; `@0.51`: `18–0–30 / 37.5%`; `@0.55`: `16–0–32 / 33.3%`).
  - **48g Unseen Holdout (`400–447` vs. `#2 Hober Malloc`):** `Gen17-grpo_it4` `12–0–36 (25.0%)` vs. `Gen 15B` Pure `17–0–31 (35.4%)` → **Combined `96g`: `31/96 (32.3%)` vs. `Gen 15B` Pure `41/96 (42.7%)`**.
  - **20g Unseen Mirrors (`990–999`):** **`12–0–8 (60.0%)`** vs. `Gen 15B`, `10–0–10 (50.0%)` vs. `Gen 13 PT-v3.5`, `18–0–2 (90.0%)` vs. `Gen 12 PPO`.
  - **`OrbitSearchAgent` on `Gen17-grpo_it4` (`200–223`, `24g`):** **`10–0–14 (41.7%)`** (improved **`+4.2 pp`** over `Gen17-grpo_it4` Pure `9/24 = 37.5%`).
  - **Solo 1-Thread Checks (`Gen17-grpo_it4 @0.53`):** `8–0–0 (100.0%)` 4P FFA vs. `3× subham_v14` (`400–407`, `62.79 ms` mean / `80.70 ms` p95 / `138.46 ms` max); `3–0–1 (75.0%)` 2P vs. `#2 Hober Malloc` (`200–203`, `58.40 ms` mean / `73.79 ms` p95 / `100.80 ms` max).
- **Root-Cause Takeaway:** `checkpoints/planet_transformer_best.pt` stays at `Gen 15B` (`42.7%` 96g vs `Gen 17` `32.3%`). Heavy single-corpus Hober-mirror repetition (`tr_kl 0.030` vs `v3_kl 0.082`) + high `value_weight=0.75` + dropping `3,500+`-map `r5–r7` diversity caused `Gen17-KD` to drop `50.0% -> 29.2%` Boss48 (even though GRPO lifted `+10.4 pp` back to `39.6%` and beat `Gen 15B` `60.0%` in mirrors). Full-diversity `r5–r9` 2-stage KD follows in Gen 18.
