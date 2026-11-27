# Galcon AI Wave 24 — 3-Size Guard75 League EI + On-Policy `--ppo-temp` Breakthrough (`78.33%` 3-Size Search • `93.8%` Compact • `1946.6 Elo` Zero-Search Pure!)

A competitive, self-contained Galcon AI stack inside `AClaybrook/orbit-wars-lightzero` featuring our **Gen 5 Edge-Gated Residual-Edge WDL Transformer** ([`galcon/gen5_model.py`](gen5_model.py), `2.93M` params + fused TorchScript [`FastGen5Infer`](gen5_model.py)), **`esparano`'s real Galcon 2 C++ multi-ship physics simulator** ([`galcon/sim_core.cpp`](sim_core.cpp)), Phil Hassey's official **`gbots`** protocol, and zero-copy head-to-head benchmarking against Evan Sparano's (`esparano/GalconModVault`) unmodified Lua tournament bots (`Sparky v1.6`, `Waffle3z-GA / bots_waffle3z_optimized`, & `EsparanBot v0.2`) via [`galcon/lua_bridge.py`](lua_bridge.py).

## Wave 24 Highlights — Unbiased `--pos-weight 1.0`, Exact On-Policy `--ppo-temp`, 3-Size Guard75 League EI (`78.33%` 3-Size Search & `1946.6 Elo` Zero-Search Pure!)
- **Training Pipeline Upgrades ([`galcon/train.py`](train.py)):**
  - **Unbiased Multi-Send Supervision (`--pos-weight`, `--opening-pol-weight`):** Eliminated `pos_weight=2.2` logit inflation that caused early tick `0–10` over-firing (`prec 48.7% → 70.1%`, cutting Large-map early-rush losses in half) and up-weighted critical opening ticks `0–12` (`sim.tick <= 12` 100% 1-tick EI coverage).
  - **Exact On-Policy Bernoulli Temperature (`--ppo-temp`, e.g. `0.55`):** Scales multi-edge logits `ms_l / T` identically in both PPO rollouts and PPO log-prob updates while keeping unscaled EMA KL — suppressing ~270-edge tail sampling noise `30×` and turning 1-tick Multi-Size League PPO from regression into monotonic `+8.6%` / `+96.8 Elo` gains.
- **New 3-Size Search Champion ([`checkpoints/galcon_gen5_w24.pt`](../checkpoints/galcon_gen5_w24.pt), default `--bot search`, `45.6 ms/tick`):**
  - Collected `15,935` fresh 1-tick balanced 3-size Guard75 Neural-League EI states (`Compact 1,943 | Std 6,126 | Large 7,866`, bringing our 100% real-C++ corpus to **`66,193` states**), trained `W24_Unified_e2` on `44,077` 1-tick states (`vloss=0.365, vsign=87.7%, prec=63.5%`), recalibrated `W19`'s WDL value head across all `66,193` states (`val_ce: 0.5531 → 0.5248, vsign: 88.7% → 91.3%`), and formed a zero-latency 3-way Model Soup (`0.40 W21 + 0.30 W19_VCal66k + 0.30 W24_Unified_e2`).
  - **Jumped `+11.81%` in 3-Size Boss Avg (`66.52% → 78.33%`) over `W21_Search`:**
    - **Compact 6pl (`32g 1t`):** **`93.8%`** (`Sparky 96.9% [15–0–1]`, `Waffle3z 90.6% [14–1–1]` — BOTH >90%, `+6.3%` vs W21!)
    - **Standard 18pl (`104g`):** **`78.75%`** Dual-Boss (`Sparky 80.0% [s14k 85.0%]`, `Waffle3z 77.5% [s9.5k 80.0%]`, `EsparanBot 58.3%`, `1918.5 Elo`)
    - **Large 24pl (`24g / 48g 2t`):** **`62.5%`** canonical 24g (`Sparky 50.0% [6–6]`, `Waffle3z 75.0% [9–3]`, **`+29.2%`** vs W21's `33.3%`!) • **`47.9%`** on full 48g (`Sparky 37.5%`, `Waffle3z 58.3%`)
    - **Heuristics (`32g 2t`):** **`96.9%` (`31–1–0`)**
- **New Zero-Search Pure Champion ([`checkpoints/galcon_gen5_w24_pure.pt`](../checkpoints/galcon_gen5_w24_pure.pt), default `--bot transformer`, `7.5 ms/tick`, ZERO Search!):**
  - Combined unbiased `--pos-weight 1.0 --opening-pol-weight 2.5` supervision, exact logit bias calibration, and `--ppo-temp 0.55` Multi-Size 1-tick League PPO (`--boss-draw-penalty -0.45`), beating `W21_Pure` on **EVERY map size with ZERO regression** and matching `W21_Search`'s Standard 18pl Elo at **7.5 ms/tick**:
    - **Standard 18pl (`104g`, ZERO SEARCH!):** **`1946.6` True MLE Elo (`+103.2 Elo` vs W21_Pure!)** — **`82.5%` Waffle3z (`33–7–0`, `s14k 90.0%`!)**, **`79.2%` EsparanBot (`19–5–0`, `+33.4%`!)**, **`70.0%` Sparky (`24–8–8`)**, `76.25%` Dual-Boss
    - **Compact 6pl (`32g 1t`):** **`56.2%`** (`+7.8%` vs W21_Pure, `68.8%` Sparky) • **Large 24pl (`24g 2t`):** **`33.3%`** (`+8.3%` vs W21_Pure, `41.7%` Waffle3z) • **Heuristics:** **`90.6%`**
    - **3-Size Boss Avg:** **`55.28%` (`+7.05%` vs W21_Pure's `48.23%`)**

## Wave 22 Highlights — Guard75 Search Upgrade (`1961.6` MLE Elo), DirectML Optimizer Fix & EI Teacher Parity
- **Search Engine Upgrades ([`galcon/transformer_bot.py`](transformer_bot.py), pushed `W21` to `80.00%` Elo-Weighted / `1961.6` True MLE Elo on our 160-game real-C++ suite!):**
  - **Enabled Focus-Fire & Tunnel Candidates on Large `>=22pl` Maps + `0.55s` Budget:** Removing the `n < 22` gate on `_build_focus_fire_plan` / `_build_tunnel_feed_plan` jumped `W21` on `Large 24pl Sparky` from **`25.0% → 41.7%`** and `EsparanBot` from **`66.7% → 70.8%`** (`1951.0` MLE, `90.6%` Heuristics).
  - **Opening Defensive Soft-Minimax `Guard75`:** Evaluating opening strikes (`sim.tick <= 10`) against BOTH a tight-defending opponent (`reserve_ratio=0.75, max_targets=1`) AND pure-hold `[]` lifted `galcon_gen5_w21.pt` cleanly to **`80.00%` Elo-Weighted / `1961.6` True MLE Elo** (**`85.0%` Sparky `34–6–0`** all-time record, `75.0%` Waffle3z-GA `30–10–0`, **`75.0%` EsparanBot `18–6–0`**, **`50.0% / 58.3%` Large 24pl**, and **`90.6%` Heuristics `29–3–0`**).
- **Training & DirectML Fixes ([`galcon/train.py`](train.py), [`tests/test_galcon.py`](tests/test_galcon.py)):**
  - **Fixed Silent DirectML Zero-Update Bug (`train_gen4_multisend_supervised`):** Moved `param_groups` construction AFTER `model.to(train_dev)` (on `torch-directml`, `.to(dml)` creates new `Parameter` handles, leaving earlier optimizer groups stepping over dead CPU tensors) + added `--freeze-trunk` for fast `~60s` GPU WDL recalibration (`val_bce=0.028 → 0.021`, `W22_S1 vsign=96.7%`) and subprocess-isolated DirectML regression test (`40/40` tests passing).
  - **Fixed `-58.3 Elo` Gen 5 EI Teacher Mismatch (`collect_gen4_multisend_ei_dataset`):** Ablation proved `neural_value_weight=0.75` + `enable_rescue_snipe=True` + `enable_opp_direct_rush=True` dropped `W21` from `1951.0 → 1892.7` MLE (`-58.3` Elo); removed those overrides so Gen 5 EI now collects from the clean `1961+` Elo Search config.
- **Wave 22 Checkpoint Ablation Takeaways (Keeping `galcon_gen5_w21.pt` + `galcon_gen5_w21_pure.pt` Promoted):**
  - Training/recalibrating exclusively on `19,016` Large-skewed (`53.5%` `>=22pl`) EI states boosted targeted buckets (`W22_S3`: `85.0%` Sparky & `41.7% / 66.7%` Lg24; `W22_S2`: `75.0%` EsparanBot) while softening `18pl` `Waffle3z` (`80.0% → 67.5%`) and Pure precision (`74.3% → 59.8%`), so Wave 22 promotes verified engine/training fixes while keeping `galcon_gen5_w21.pt` + `galcon_gen5_w21_pure.pt` until balanced Wave 23 collection.

## Wave 21 Highlights — Frozen-Trunk WDL Recalibration (`1946.6 Elo` Search!) & Clean Guard-v2 Pure Distillation (`71.25%` Dual-Boss Pure!)
- **Engine & Efficiency Upgrades ([`galcon/train.py`](train.py), [`galcon/transformer_bot.py`](transformer_bot.py)):**
  - **2× Faster PPO Rollouts & `val_56g`:** Wired cached fused TorchScript [`FastGen5Infer`](gen5_model.py) + exact-`N` planet-count micro-batching into `_run_gen4_ppo_episodes_batched`, `_run_gen4_val_wave_batched`, and `_fast_gen4_neural_act` (bit-exact `max_diff <= 2.86e-6`, cutting rollout time to `8–15s/iter`!).
  - **Zero-Drift Fixed-Anchor Value Regularization & Asymmetric Large Maps:** Anchored `--freeze-value-head` PPO scalar value MSE to an immutable starting anchor (`fixed_val_anchor`), doubled asymmetric Large `22–28pl` training arenas (`15% → 30%`), always recorded critical opening ticks `0–6` in 1-tick EI, and extended Large-map (`>=22pl`) search horizon `22 → 28` & budget `0.30s → 0.45s`.
- **New Real-C++ Search Champion ([`checkpoints/galcon_gen5_w21.pt`](../checkpoints/galcon_gen5_w21.pt), default `--bot search`):**
  - Collected `9,699` fresh `1-tick` real-C++ Guard-v2 Search states (`160` games across `W20` + Large-heavy `W19`, bringing our 100% real-C++ pool to **`31,815` states**) and recalibrated ONLY the 4-query Attention-Pooling WDL head (`val_*`) on `Gen5_W20`'s frozen trunk (`val_ce: 0.5438 → 0.5238`, `vsign: 89.5% → 92.5%`, zero policy risk!).
  - Vaulted **`+54.5 Elo` to `1946.6` True MLE Elo (`+188.5 Elo` since C++ reset!)**:
    - **`Sparky v1.6` 18pl (40g, 1t):** **`32–8–0` (`80.0%`, `s9.5k: 17–3 / 85.0%`!)**
    - **`Waffle3z-GA` 18pl (40g, 1t):** **`31–9–0` (`77.5%` — ALL-TIME OVERALL RECORD, `s14k: 17–3 / 85.0%`!)**
    - **`EsparanBot v0.2` (24g, 2t):** **`17–7–0` (`70.8%`, `+16.6%` vs W20, `s15.3k: 10–2 / 83.3%`!)**
    - **Elo-Weighted (50/40/10):** **`78.08%`** (`+3.66%` vs W20, **`+24.16%` vs W16!**) • **Dual-Boss Avg:** **`78.75%` (`63–17`)** • **Heuristics:** `84.4%`.
- **New Zero-Search Pure Champion ([`checkpoints/galcon_gen5_w21_pure.pt`](../checkpoints/galcon_gen5_w21_pure.pt), default `--bot transformer`, `7.5 ms/tick`, ZERO Search!):**
  - Caught & eliminated `--kd-alpha` pre-search prior contamination (`--kd-alpha 0.0` on our `9,699` Guard-v2 Search states → `val_bce=0.020`, `vloss=0.337`, `1851.3 Elo` Pure, `67.5%` Waffle3z / `95%` on `s14k`!), then ran 5 clean Fixed-Anchor Frozen-V PPO iterations:
    - **`Sparky v1.6` 18pl (40g, 1t, ZERO SEARCH!):** **`30–10–0` (`75.0%` PURE, balanced `75%/75%`!)**
    - **`Waffle3z-GA` 18pl (40g, 1t, ZERO SEARCH!):** **`27–13–0` (`67.5%` PURE, `+27.5%` vs W20_Pure, balanced `70%/65%`!)**
    - **Dual-Boss Avg (80g, ZERO SEARCH!):** **`71.25%` (`57–23` — BEATS `Gen5_W19_Search` with ZERO search at `7.5 ms/tick`!)** • **Elo-Weighted:** **`69.08%` (`+8.50%` vs W20_Pure!)** • **Heuristics:** **`90.6%` (`29–3`)**.

## Wave 20 Highlights — 3-Stage Frozen-Value-Head `1-Tick` C++ PPO (`1892.1 Elo` Search & `1805.5 Elo` Pure!)
- **Heavy Multi-Stage Frozen-Value-Head `1-Tick` C++ Mirror League PPO (21 Iters / 336 Full Paired-Seed Episodes):**
  - Ran 3 annealed stages (`20A: 7 iters lr=2.2e-5 kl=0.022`, `20B: 7 iters lr=1.5e-5 kl=0.025`, `20C: 7 iters lr=9.0e-6 kl=0.030`) directly from `Gen5_W19` with `--freeze-value-head --v-anchor-coef 0.50` on real Galcon 2 C++ physics + `3,495` 100% real-C++ redirect replay states.
  - **Stage 20B (`checkpoints/galcon_gen5_w20.pt`):** **`74.42%` Elo-Weighted / `1892.1` True MLE Elo Search** (`80.0%` Sparky, `72.5%` Waffle3z, `84.4%` Heuristics).
  - **Stage 20C (`checkpoints/galcon_gen5_w20_pure.pt`):** **`60.58%` Elo-Weighted / `1805.5` True MLE Elo Pure** (`75.0%` Pure Sparky, `70.8%` EsparanBot).

## Wave 19 Highlights — Unified Single-Model Champion (`galcon_gen5_w19.pt`: `1851.3 Elo` Search & `1765.3 Elo` Pure!)
- **Loss-Autopsy Search Upgrades ([`galcon/transformer_bot.py`](transformer_bot.py)):**
  - Autopsy of Wave 18's remaining losses revealed `12/12` (`100%`) involved early tick `0–2` home rushes where `opening_def_guard` (`eta >= 14`, `def_horizon = eta + 2`, passive `opp=[]`) missed `eta=6..13` rushes and truncated C++ gradual flock landings.
  - Upgraded **Opening Defensive Soft-Minimax Guard v2** (all early strikes `eta >= 4, pct >= 35, tick <= 10`, gradual-arrival horizon `min(36, eta + 10)`, calm defending+expanding `opp_def_orders`, clean non-rush neutral expansion candidates), raised default `rollout_horizon = 20`, and tuned `prior_bonus = 0.040` to properly trust our `63.8%` Pure Sparky prior (`s14k` Dual-Boss A/B jumped `62.5% → 70.0%` at `40–42 ms/t`).
- **Continued Frozen-Value-Head `1-Tick` C++ PPO ([`checkpoints/galcon_gen5_w19.pt`](../checkpoints/galcon_gen5_w19.pt)):**
  - Continued 7 gentle iterations (`lr=2.2e-5`, `--freeze-value-head --v-anchor-coef 0.50`, `42%` Sparky / `38%` Waffle3z) directly on top of our sharpest `vloss=0.370` C++ WDL value head + `3,495` 100% real C++ redirs, pushing `val_56g` to `67.9%` (`comp = 63.1%`, `87.5%` / `7–1` vs `Gen5_W18`) → **`67.46%` Elo-Weighted / `1851.3` True MLE Elo Search** (`71.2%` Waffle3z, `66.2%` Sparky, `50.0% / 50.0%` Large 24pl) & **`55.71%` / `1765.3 Elo` Pure**.

## Unified Real-C++ Cross-Bot Bradley-Terry Elo Leaderboards (2,860 Games · Overall + Per-Map-Size)

To put **all** model generations (`Gen 3` → `Wave 22`, Pure & Search), built-in heuristics (`Random`, `Classic`, `Macro`, `Rusher`, `Tactical`), unmodified Lua tournament bosses (`Sparky v1.6`, `Waffle3z-GA`, `EsparanBot v0.2`), and future external bots (`esparano/LightZero` checkpoints via `py:mod:factory`) on **one shared leaderboard system**, [`galcon/elo_rating.py`](elo_rating.py) + [`galcon/tournament.py`](tournament.py) use **Evan Sparano's exact Bradley-Terry MLE solver** (`esparano/LightZero` `lzero/worker/elo_rating.py` — vectorized damped Newton-Raphson, played-edge prior `prior_draws=0.1`, `l2_reg=1e-4`, and analytical Laplace `±Std`) on **Evan's real Galcon 2 C++ simulator** ([`galcon/sim_core.cpp`](sim_core.cpp)).

Each map size has its own dedicated JSON ladder (`Compact 6pl`, `Standard 18pl`, `Large 24pl`), aggregated equal-weight into an **Overall Cross-Map Leaderboard**:

### Overall Cross-Map Leaderboard (`Compact 6pl` + `Standard 18pl` + `Large 24pl` · 2,860 Total Games)

| Rank | Bot | Overall Elo (`bot:classic=1200`) | ±Std | Compact 6pl | Standard 18pl | Large 24pl | Overall vs `bot:random` (0) | Sizes | Total Games | Score% |
|---:|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|
| 1 | `w22_search` | **1889.1** | ±46.6 | `1965.6` | `1925.2` | `1776.5` | `1297.9` | 3/3 | 296 | 63.3% |
| 2 | `w20_search` | **1882.8** | ±45.5 | `1919.9` | `1927.6` | `1801.0` | `1291.6` | 3/3 | 340 | 62.9% |
| 3 | `w19_search` | **1882.3** | ±45.8 | `1870.8` | `1909.7` | `1866.5` | `1291.1` | 3/3 | 340 | 60.6% |
| 4 | `w18_search` | **1855.2** | ±45.3 | `1925.2` | `1852.7` | `1787.8` | `1264.0` | 3/3 | 340 | 57.8% |
| 5 | `w14_search` | **1786.7** | ±44.9 | `1737.9` | `1830.3` | `1792.0` | `1195.6` | 3/3 | 340 | 45.9% |
| 6 | `w21_pure` | **1782.4** | ±45.0 | `1910.7` | `1808.3` | `1628.1` | `1191.2` | 3/3 | 340 | 53.2% |
| 7 | `w20_pure` | **1765.9** | ±44.7 | `1845.4` | `1804.3` | `1648.1` | `1174.8` | 3/3 | 340 | 48.4% |
| 8 | `bot:sparky` | **1760.1** | ±43.1 | `1687.0` | `1768.1` | `1825.1` | `1168.9` | 3/3 | 340 | 55.9% |
| 9 | `w18_pure` | **1718.0** | ±43.8 | `1727.2` | `1763.6` | `1663.1` | `1126.8` | 3/3 | 340 | 43.8% |
| 10 | `bot:waffle` | **1701.2** | ±41.7 | `1636.9` | `1783.3` | `1683.5` | `1110.0` | 3/3 | 340 | 56.8% |
| 11 | `gen3_pure` | **1514.1** | ±36.3 | `1445.6` | `1541.5` | `1555.1` | `922.9` | 3/3 | 340 | 56.5% |
| 12 | `gen4_pure` | **1492.5** | ±36.7 | `1425.4` | `1531.4` | `1520.7` | `901.3` | 3/3 | 340 | 48.1% |
| 13 | `bot:esparan` | **1396.7** | ±33.9 | `1189.7` | `1446.4` | `1553.9` | `805.5` | 3/3 | 324 | 57.4% |
| 14 | `bot:classic` | **1200.0** | ±0.0 | `1200.0` | `1200.0` | `1200.0` | `608.9` | 3/3 | 328 | 38.0% |
| 15 | `bot:macro` | **1192.7** | ±34.2 | `1007.8` | `1211.2` | `1358.9` | `601.6` | 3/3 | 260 | 46.9% |
| 16 | `bot:rusher` | **1181.3** | ±33.3 | `1104.5` | `1147.2` | `1292.1` | `590.2` | 3/3 | 292 | 37.0% |
| 17 | `bot:tactical` | **1168.3** | ±32.9 | `1036.7` | `1203.8` | `1264.3` | `577.2` | 3/3 | 308 | 39.4% |
| 18 | `bot:random` | **590.0** | ±84.9 | `583.2` | `553.0` | `633.8` | `0.0` | 3/3 | 172 | 2.9% |

### Compact 6pl (200×200 · Evan LightZero Geometry · 67 Pairs / 1,072 Games · [`galcon_ladder_compact6.json`](../docs/galcon_ladder_compact6.json))

| Rank | Bot | Elo (`bot:classic=1200`) | ±Std | vs `bot:random` (0) | Games | W-D-L | Score% | Mean ms |
|---:|---|---:|---:|---:|---:|:---:|---:|---:|
| 1 | `w22_search` | **1965.6** | ±77.2 | `1381.8` | 112 | 65-17-30 | 65.6% | `16.47 ms` |
| 2 | `w18_search` | **1925.2** | ±74.6 | `1341.4` | 128 | 76-7-45 | 62.1% | `15.68 ms` |
| 3 | `w20_search` | **1919.9** | ±74.6 | `1336.0` | 128 | 74-9-45 | 61.3% | `15.49 ms` |
| 4 | `w21_pure` | **1910.7** | ±74.4 | `1326.9` | 128 | 65-25-38 | 60.5% | `7.39 ms` |
| 5 | `w19_search` | **1870.8** | ±74.8 | `1287.0` | 128 | 63-18-47 | 56.2% | `16.24 ms` |
| 6 | `w20_pure` | **1845.4** | ±74.8 | `1261.5` | 128 | 62-11-55 | 52.7% | `7.13 ms` |
| 7 | `w14_search` | **1737.9** | ±73.7 | `1154.0` | 128 | 44-2-82 | 35.2% | `10.26 ms` |
| 8 | `w18_pure` | **1727.2** | ±74.1 | `1143.4` | 128 | 41-2-85 | 32.8% | `5.77 ms` |
| 9 | `bot:sparky` | **1687.0** | ±64.3 | `1103.2` | 128 | 77-4-47 | 61.7% | `0.33 ms` |
| 10 | `bot:waffle` | **1636.9** | ±62.0 | `1053.1` | 128 | 81-8-39 | 66.4% | `0.41 ms` |
| 11 | `gen3_pure` | **1445.6** | ±54.3 | `861.8` | 128 | 69-5-54 | 55.9% | `4.13 ms` |
| 12 | `gen4_pure` | **1425.4** | ±53.7 | `841.6` | 128 | 68-2-58 | 53.9% | `3.87 ms` |
| 13 | `bot:classic` | **1200.0** | ±0.0 | `616.2` | 128 | 50-1-77 | 39.5% | `0.01 ms` |
| 14 | `bot:esparan` | **1189.7** | ±48.9 | `605.9` | 128 | 48-2-78 | 38.3% | `0.55 ms` |
| 15 | `bot:rusher` | **1104.5** | ±50.6 | `520.7` | 112 | 50-2-60 | 45.5% | `0.23 ms` |
| 16 | `bot:tactical` | **1036.7** | ±51.9 | `453.0` | 128 | 41-3-84 | 33.2% | `0.49 ms` |
| 17 | `bot:macro` | **1007.8** | ±56.1 | `424.1` | 80 | 36-0-44 | 45.0% | `0.30 ms` |
| 18 | `bot:random` | **583.2** | ±110.3 | `0.0` | 48 | 3-0-45 | 6.2% | `0.01 ms` |

### Standard 18pl (400×240 Symmetric · Classic 1v1 · 66 Pairs / 1,056 Games · [`galcon_unified_ladder.json`](../docs/galcon_unified_ladder.json))

| Rank | Bot | Elo (`bot:classic=1200`) | ±Std | vs `bot:random` (0) | Games | W-D-L | Score% | Mean ms |
|---:|---|---:|---:|---:|---:|:---:|---:|---:|
| 1 | `w20_search` | **1927.6** | ±75.4 | `1373.0` | 128 | 78-11-39 | 65.2% | `46.17 ms` |
| 2 | `w22_search` | **1925.2** | ±78.0 | `1370.6` | 112 | 67-14-31 | 66.1% | `44.12 ms` |
| 3 | `w19_search` | **1909.7** | ±75.3 | `1355.1` | 128 | 73-8-47 | 60.2% | `46.91 ms` |
| 4 | `w18_search` | **1852.7** | ±74.8 | `1298.1` | 128 | 64-6-58 | 52.3% | `41.84 ms` |
| 5 | `w14_search` | **1830.3** | ±74.8 | `1275.7` | 128 | 60-6-62 | 49.2% | `20.73 ms` |
| 6 | `w21_pure` | **1808.3** | ±75.7 | `1253.6` | 128 | 56-15-57 | 49.6% | `9.98 ms` |
| 7 | `w20_pure` | **1804.3** | ±73.6 | `1249.7` | 128 | 61-6-61 | 50.0% | `10.04 ms` |
| 8 | `bot:waffle` | **1783.3** | ±71.9 | `1228.7` | 128 | 64-4-60 | 51.6% | `0.62 ms` |
| 9 | `bot:sparky` | **1768.1** | ±73.3 | `1213.5` | 128 | 55-10-63 | 46.9% | `0.61 ms` |
| 10 | `w18_pure` | **1763.6** | ±71.5 | `1209.0` | 128 | 64-2-62 | 50.8% | `8.84 ms` |
| 11 | `gen3_pure` | **1541.5** | ±59.2 | `986.9` | 128 | 70-1-57 | 55.1% | `5.45 ms` |
| 12 | `gen4_pure` | **1531.4** | ±60.5 | `976.8` | 128 | 56-1-71 | 44.1% | `5.25 ms` |
| 13 | `bot:esparan` | **1446.4** | ±55.0 | `891.9` | 112 | 80-0-32 | 71.4% | `0.92 ms` |
| 14 | `bot:macro` | **1211.2** | ±52.8 | `656.7` | 96 | 47-0-49 | 49.0% | `0.45 ms` |
| 15 | `bot:tactical` | **1203.8** | ±52.9 | `649.3` | 96 | 46-0-50 | 47.9% | `0.82 ms` |
| 16 | `bot:classic` | **1200.0** | ±0.0 | `645.5` | 128 | 48-0-80 | 37.5% | `0.01 ms` |
| 17 | `bot:rusher` | **1147.2** | ±54.1 | `592.7` | 96 | 24-0-72 | 25.0% | `0.35 ms` |
| 18 | `bot:random` | **553.0** | ±163.1 | `0.0` | 64 | 1-0-63 | 1.6% | `0.01 ms` |

### Large 24pl (560×340 Asymmetric Macro · 61 Pairs / 732 Games · [`galcon_ladder_large24.json`](../docs/galcon_ladder_large24.json))

| Rank | Bot | Elo (`bot:classic=1200`) | ±Std | vs `bot:random` (0) | Games | W-D-L | Score% | Mean ms |
|---:|---|---:|---:|---:|---:|:---:|---:|---:|
| 1 | `w19_search` | **1866.5** | ±87.0 | `1231.4` | 84 | 57-0-27 | 67.9% | `110.44 ms` |
| 2 | `bot:sparky` | **1825.1** | ±85.1 | `1190.0` | 84 | 51-0-33 | 60.7% | `0.85 ms` |
| 3 | `w20_search` | **1801.0** | ±85.9 | `1165.8` | 84 | 52-0-32 | 61.9% | `114.06 ms` |
| 4 | `w14_search` | **1792.0** | ±84.3 | `1156.9` | 84 | 48-0-36 | 57.1% | `43.10 ms` |
| 5 | `w18_search` | **1787.8** | ±85.6 | `1152.6` | 84 | 50-0-34 | 59.5% | `113.48 ms` |
| 6 | `w22_search` | **1776.5** | ±86.7 | `1141.3` | 72 | 40-0-32 | 55.6% | `107.06 ms` |
| 7 | `bot:waffle` | **1683.5** | ±81.4 | `1048.4` | 84 | 42-0-42 | 50.0% | `0.70 ms` |
| 8 | `w18_pure` | **1663.1** | ±81.7 | `1028.0` | 84 | 42-0-42 | 50.0% | `9.80 ms` |
| 9 | `w20_pure` | **1648.1** | ±83.3 | `1013.0` | 84 | 33-0-51 | 39.3% | `10.26 ms` |
| 10 | `w21_pure` | **1628.1** | ±83.4 | `993.0` | 84 | 40-0-44 | 47.6% | `9.75 ms` |
| 11 | `gen3_pure` | **1555.1** | ±73.4 | `920.0` | 84 | 50-0-34 | 59.5% | `5.71 ms` |
| 12 | `bot:esparan` | **1553.9** | ±70.3 | `918.8` | 84 | 57-0-27 | 67.9% | `1.33 ms` |
| 13 | `gen4_pure` | **1520.7** | ±74.6 | `885.6` | 84 | 38-0-46 | 45.2% | `5.66 ms` |
| 14 | `bot:macro` | **1358.9** | ±67.7 | `723.8` | 84 | 39-0-45 | 46.4% | `0.60 ms` |
| 15 | `bot:rusher` | **1292.1** | ±67.1 | `657.1` | 84 | 33-0-51 | 39.3% | `0.52 ms` |
| 16 | `bot:tactical` | **1264.3** | ±65.3 | `629.2` | 84 | 33-0-51 | 39.3% | `1.06 ms` |
| 17 | `bot:classic` | **1200.0** | ±0.0 | `564.9` | 72 | 26-0-46 | 36.1% | `0.01 ms` |
| 18 | `bot:random` | **633.8** | ±161.5 | `0.0` | 60 | 1-0-59 | 1.7% | `0.01 ms` |

```bash
# Re-anchor & print any per-size leaderboard without replaying games:
python3 -m galcon.tournament --report-only --state docs/galcon_ladder_compact6.json \
  --anchor bot:classic --anchor-elo 1200 --secondary-anchor bot:random --secondary-anchor-elo 0

# Regenerate the Overall Cross-Map table across all 3 map sizes:
python3 -m galcon.tournament --combine \
  "Compact 6pl=docs/galcon_ladder_compact6.json" \
  "Standard 18pl=docs/galcon_unified_ladder.json" \
  "Large 24pl=docs/galcon_ladder_large24.json" \
  --anchor bot:classic --anchor-elo 1200 --secondary-anchor bot:random --secondary-anchor-elo 0

# Add a new bot/checkpoint cleanly via catch-up Swiss on any map size (--arena compact6|standard18|large24):
python3 -m galcon.tournament --resume --arena compact6 --bots evan_sez=py:my_adapter:make_bot \
  --state docs/galcon_ladder_compact6.json --games-per-pair 16 --rounds 7 --workers 14
```

## Real Galcon 2 C++ Physics Scoreboard (`50% Sparky v1.6 / 40% Waffle3z-GA / 10% EsparanBot v0.2` + 104g True MLE Elo)

| Model (Real C++ Galcon 2 Physics) | 18pl Sparky (40g 1t) | 18pl Waffle3z (40g 1t) | 18pl Esparan (24g 2t) | **Elo-Wtd (50/40/10)** | **104g MLE Elo** | Large 24pl (Sp/Wf) | Heur (32g) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Gen5_W16_Pure` (old physics, `5ms`) | 50.0% | 20.0% | **75.0%** | 40.50% | 1682.4 | — | — |
| `Gen5_W17_Pure` (old physics, `6ms`) | 47.5% | 30.0% | 58.3% | 41.58% | 1675.1 | — | — |
| `Gen5_W18_Pure` (`galcon_gen5_w18_pure.pt`, `6ms`) | 58.8% | 40.0% | 62.5% | 51.62% | 1743.7 | — | — |
| `Gen5_W19_Pure` (`galcon_gen5_w19.pt`, `7ms`) | 63.8% | 45.0% | 58.3% | 55.71% | 1765.3 | — | — |
| `Gen5_W20_Pure` (`galcon_gen5_w20_pure.pt`, `7.5ms`) | **75.0%** (`30–10`) | 40.0% | 70.8% | 60.58% | 1805.5 | 16.7% / — | — |
| `Gen5_W21_PureDistill` (zero-search, `7.5ms`) | 67.5% (`27–13`) | **67.5%** (`s14k: 95%`!) | 62.5% | 67.00% | **1851.3** | 25.0% / 33.3% | 84.4% |
| **`Gen5_W21_Pure` (`galcon_gen5_w21_pure.pt`, `7.5ms`)** | **75.0%** (`30–10`, `75/75`) | **67.5%** (`27–13`, `70/65`) | 45.8% | **69.08%** (`+28.6%`!) | 1843.4 (`71.25%` Dual!) | 16.7% / 33.3% | **90.6%** |
| `Gen5_W16_Search` (old physics, `25–33ms`) | 67.5% (`25–11–4`) | 33.8% (`11–24–5`) | 66.7% | 53.92% | 1758.1 | 16.7% / 33.3% | 87.5% |
| `Gen5_W17_Search` (old physics, `27–42ms`) | 50.0% (`17–17–6`) | 41.2% (`14–21–5`) | 75.0% | 49.00% | 1743.7 | 41.7% / 58.3% | 84.4% |
| `Gen5_W18_R1_Search` (`7.7k` C++ EI, `31–53ms`) | 60.0% | 46.2% | 62.5% | 54.75% | 1765.3 | 33.3% / 50.0% | 81.2% |
| `Gen5_W18_R2_Search` (`13.4k` C++ EI, `30–31ms`) | 53.8% | 53.8% | **83.3%** | 56.71% | 1805.5 | **50.0%** / 58.3% | 87.5% |
| `Gen5_W18_Search` (`galcon_gen5_w18.pt`, `29–31ms`) | 66.2% (`24–11–5`) | 52.5% | 70.8% | 61.21% | 1816.7 | 41.7% / **66.7%** | 84.4% |
| `Gen5_W19_Search` (`galcon_gen5_w19.pt`, `42ms`) | 66.2% (`26–13–1`) | 71.2% (`28–11–1`) | 58.3% | 67.46% | 1851.3 | **50.0%** / 50.0% | 84.4% |
| `Gen5_W20_Search` (`galcon_gen5_w20.pt`, `49ms`) | **80.0%** (`32–8–0`) | 72.5% (`29–11–0`) | 54.2% | 74.42% | 1892.1 | 33.3% / 50.0% (`h=28`) | 84.4% |
| `Gen5_W21_PureDistill_Search` (`47ms`) | 72.5% (`29–11–0`) | 65.0% (`26–14–0`) | 50.0% | 67.25% | 1835.7 | **50.0%** / **66.7%** | **90.6%** |
| `Gen5_W21_Search` (`galcon_gen5_w21.pt`, `49ms`) | 80.0% (`32–8`, `s9.5k: 85%`) | **77.5%** (`31–9`, `s14k: 85%`) | 70.8% | 78.08% (`+24.2%`!) | 1946.6 (`+188.5`!) | 25.0% / 41.7% | 84.4% |
| **`Gen5_W22_Search` (`galcon_gen5_w21.pt` + Guard75 Engine)** | **85.0%** (`34–6–0`) | 75.0% (`30–10–0`) | 75.0% (`18–6–0`) | **80.00%** (`+1.9%`!) | **1961.6** (`+15.0`!) | **50.0%** / 58.3% | **90.6%** |

## Wave 18 Highlights — Real Galcon 2 C++ Physics Transition, `FastGen5Infer` Speedup & Frozen-Value-Head `1-Tick` PPO
- **Real Galcon 2 C++ Physics Shift Autopsy:** Switching from simplified point-mass Python physics to real multi-ship Galcon 2 C++ physics (flock separation, planet-edge steering, gradual multi-tick arrivals, fractional ship packing) exposed a massive physics-shift gap against `Waffle3z-GA` (genetically evolved on real Galcon 2 physics): old-physics `Gen5_W16_Search` held `67.5%` vs `Sparky v1.6` but dropped `72.5% → 33.8%` vs `Waffle3z-GA` (`Gen5_W16_Pure` dropped to `20.0%`), repeatedly launching giant tick-0 home rushes that spread out over 6–10 ticks of gradual landing and lost `-68` to `-166` production by tick 20.
- **Engine Breakthroughs:**
  - **C++ + Bridge + `FastGen5Infer` TorchScript Acceleration:** Fleet-slice AABB broadphase + zero-copy C++ features + dynamic-shape TorchScript [`FastGen5Infer`](gen5_model.py) with 6 fused linear projection pairs & `B=2` fused root batching cut live `Gen5` Search `32.5 → 22.5 ms/t` (`1.44×`) and end-to-end `1-tick` C++ EI `65.1s → 47.9s` (`1.36×`), enabling **240 full `1-tick` real C++ EI games (`13,419` states)** in ~13 minutes.
  - **Frozen-Value-Head `1-Tick` C++ PPO (`--freeze-value-head` in [`galcon/train.py`](train.py)):** Directly solved Wave 17's PPO-for-Search regression by freezing Gen 5's C++-calibrated WDL value head (`vloss=0.370`, `vsign=87.5%`) and anchoring trunk value outputs (`--v-anchor-coef 0.35–0.50`) while sharpening the multi-send policy via GRPO paired-seed RL directly against Elo-weighted real C++ Lua bosses.
- **Promoted Wave 20 Defaults ([`galcon/gbots_client.py`](gbots_client.py)):**
  - **`--bot search` → [`checkpoints/galcon_gen5_w20.pt`](../checkpoints/galcon_gen5_w20.pt):** **`1892.1` True MLE Elo** (`74.42%` Elo-Weighted, `80.0%` Sparky v1.6, `72.5%` Waffle3z-GA) on real Galcon 2 C++ physics (`--bot ensemble` pairs `galcon_gen5_w20.pt` + `galcon_gen5_w19.pt`).
  - **`--bot transformer` (Pure) → [`checkpoints/galcon_gen5_w20_pure.pt`](../checkpoints/galcon_gen5_w20_pure.pt):** First Pure neural net over **`1800+` Elo (`1805.5`)** on real C++ physics (`75.0%` Pure Sparky v1.6, `70.8%` EsparanBot) at `7.5 ms/tick` with zero search.

## Wave 17 Highlights — `50/50` Balanced `19,586`-State Gen 5 Self-Play, Native `1-Tick` WDL PPO & Supervised-EI Plateau Analysis
- **Key Empirical Finding (`100% 1t` vs. `50/50` Balanced Pooling):**
  - Training Gen 5 exclusively on sparse `1-tick` trajectories (`9,832` states) sharpened value accuracy (`vsign=87.0%`) but starved policy recall (`rec: 82.2% → 67.1%`, Pure Sparky `60.0% → 50.0%`).
  - Pooling **50% dense high-recall policy anchors (`9,754` states)** + **50% full-speed `1-tick` Gen 5 self-play (`9,832` states across 3 rounds / `156` games)** restored full policy precision (`rec=78.8%`, `prec=67.5%`, `aux=98.0%`) while smashing our all-time value calibration record (**`vsign = 93.6%`**, **`vloss = 0.466`**, **`val_bce = 0.037`**)!
- **New All-Round Gen 5 Triple-Boss Champion — [`checkpoints/galcon_gen5_w17.pt`](../checkpoints/galcon_gen5_w17.pt) (`74.6%` Triple-Boss Avg — `+6.7%` over `Gen5_W16`, `+2.5%` over `Gen5_W15`!):**
  - **`Sparky v1.6` 18pl (40g, 1t):** **`20–7–13` (`66.2%`)** — **`s=14000: 10–0–10` (`75.0%` UNDEFEATED across 20 games, ZERO losses!)**
  - **`Waffle3z-GA` 18pl (40g, 1t):** **`25–9–6` (`70.0%`, perfectly balanced `70.0% / 70.0%`)**
  - **`EsparanBot v0.2` (24g, 2t):** **`20–2–2` (`87.5%` — `+25.0%` over `Gen5_W16`, `s=9500: 12–0–0` PERFECT SWEEP!)**
  - **Built-in Heuristics (32g, 2t):** **`30–2–0` (`93.8%`, `+9.4%` over `Gen5_W16`)** | **Large 24pl Asym `Waffle3z-GA`:** **`7–5–0` (`58.3%`, `+16.6%`)**
  - **Pure Neural Net (`9.6 ms/t`, ZERO Search!):** **`22–12–6` (`62.5%` — NEW ALL-TIME PURE WAFFLE3Z RECORD, `s=14000: 70.0%`!)** + **`70.0%` (`10–2–8`) Pure vs. `Sparky s=14000`** (`52.5%` overall).
  - *(Note: `checkpoints/galcon_gen5_w16.pt` remains default `--bot search` / `--bot transformer` for peak 18pl Dual-Boss Avg `70.6%` & `60.0%` Pure Sparky; `checkpoints/galcon_gen5_w15.pt` remains `#1` Sparky `73.8% / 3L`.)*
- **Engine Upgrades ([`galcon/train.py`](train.py), [`galcon/transformer_bot.py`](transformer_bot.py)):**
  - Added **native `GalconGen5Transformer` Quadratic Soft-WDL PPO value loss** + **full-speed `--act-every 1` cadence** across PPO rollouts, parallel workers, and batched `val_56g`.
  - Fixed Gen 5 `contempt` propagation to secondary members of [`EnsembleGalconSearchAgent`](transformer_bot.py).
  - **Diagnostic Finding for Wave 18 (Breaking the `~71%–75%` Supervised-EI Plateau):** Our first `1-tick` Gen 5 PPO run boosted Pure `val_56g` (`74.1%`, iter 6 `68.8%` Sparky / `79.2%` Waffle3z), but updating the WDL value head on discounted potential-shaped GAE returns miscalibrated Search (`48.8%`). Wave 18 will target **Policy-Only / Frozen-WDL-Critic `1-tick` PPO** (preserving `Gen5_W17`'s `93.6%` WDL value head!) and/or **Gen 6 Architecture + PUCT Tree Search** upgrades to break past `80%–90%`.

## Wave 16 Highlights — Deep Loss Autopsy → Root-Cause Search Fixes & Full-Speed `1-Tick` Gen 5 Self-Play
- **Quantitative 80-Game Loss & Draw Autopsy (Why We Were Losing):**
  1. **Midgame Passive Bleeding vs. `Sparky v1.6` (`77.8%` of losses in `t=16–60`):** In losses vs. wins, Sparky fired `5.6×` more `/REDIR`s (`56.8` vs. `10.2`) and caused **`1.44` undefended planet falls/loss vs. `0.29` in wins (`5.0×`)** while our bot sat idle (`[]`) for 10+ ticks even when ahead on ships/production.
  2. **Blind Long-ETA Opening Collapses vs. `Waffle3z-GA` (`50%` of losses in `t<=15`):** Sending `73%–89%` opening rushes across `eta > rollout_horizon(16)` assumed the opponent also emptied its home planet (`opp_pred` / `opp_adv_fan`), letting a defending `Waffle3z` effortlessly crush the rush.
  3. **Training-vs.-Tournament Cadence Bug:** `44.6%–48.4%` of tournament orders occur on odd ticks, yet **100% of Waves 1–15 EI/PPO training ran at `act_every=2` (`0.5×` APM bosses)**!
- **Root-Cause Fixes Shipped & Defaulted in Wave 16:**
  - **Default `DefStream+Guard` in [`GalconSearchTransformerAgent`](transformer_bot.py) (`enable_def_stream=True, enable_opening_def_guard=True`, zero latency overhead):**
    - [`_build_defense_and_pressure_plan`](transformer_bot.py): Multi-helper exact defense for every threatened owned planet (`po[t] <= 0`) + rear-to-front relay streaming (`tunnel_const=1.22`) + safe frontline enemy pressure (`sim.tick > 10`) with `smart_redirs`.
    - **Opening Deep-Strike Soft-Minimax Guard:** For `sim.tick <= 8` when any candidate launches `>=45%` at an enemy planet with `eta >= 14`, evaluates top-6 candidates against a holding opponent (`opp = []`) out to `min(28, max_eta + 2)` ticks (`0.55 * scores + 0.45 * min(scores, s_def)`).
  - **Full-Speed `1-Tick` EI Cadence (`--act-every 1` in [`galcon/train.py`](train.py)):** Plays both teacher and Lua bosses at true `1.0×` tournament speed (recording alternating even/odd ticks for a 50/50 split while staying lean) + **3× peak RAM reduction** (zero-copy `TensorDataset` + `Subset`, immediate list release, 4k redir cap: `~7.5 GiB → ~4.2 GiB`).
- **4 NEW ALL-TIME SINGLE-MODEL RECORDS (Shattering Every Previous Single Model AND 2-Model Ensemble!):**
  - **`checkpoints/galcon_gen5_w16.pt` (Default `--bot search` & `--bot transformer`, 1t-Pooled `13,381` states, `aux=98.0%` all-time record, `val_bce=0.038`, `vsign=85.8%`):**
    - **`Waffle3z-GA` 18pl (40g, 1t):** **`26–8–6` (`72.5%` — NEW ALL-TIME OVERALL RECORD, beats every single model AND 2-model ensemble; `s=14000: 70.0%`, `s=9500: 75.0%`!)**
    - **`Sparky v1.6` 18pl (40g, 1t):** **`20–5–15` (`68.8%` — only 5 losses / `87.5%` non-loss, `s=9500: 14–3–3 / 77.5%`!)**
    - **NEW ALL-TIME OVERALL DUAL-BOSS AVG: `70.6%` (`46–13–21`, FIRST MODEL IN PROJECT HISTORY >70%!!)** | **Heuristics:** `84.4%` (`26–4–2`) | **Large 24pl Asym:** `41.7%` Spky / `41.7%` Waff | **`EsparanBot`:** `62.5%`.
    - **Pure Neural Net (`7.5–7.8 ms/t`, ZERO Search!):** **`18–10–12` (`60.0%` Sparky — NEW ALL-TIME 1-TICK PURE RECORD, `60.0% / 60.0%` balanced!)** & **`20–14–6` (`57.5%` Waffle3z)** → **`58.8%` NEW ALL-TIME PURE DUAL-BOSS AVG!**
  - **`checkpoints/galcon_gen5_w15.pt` + Default Wave 16 `DefStream+Guard` (`29.9 ms/t`, ~`2001 Elo`):**
    - **`Sparky v1.6` 18pl (40g, 1t):** **`22–3–15` (`73.8%` — NEW ALL-TIME OVERALL SPARKY RECORD, `+8.8%` jump! ONLY 3 LOSSES IN 40 GAMES = `92.5%` NON-LOSS RATE! `s=14000: 9–0–11` UNDEFEATED! `s=9500: 13–3–4 / 75.0%`!)**
    - **`EsparanBot v0.2` (24g):** **`18–3–3` (`81.2%`, `+6.2%`, `s=15300: 9–0–3 / 87.5%` undefeated!)** | **`Waffle3z-GA`:** `61.3%` (`20–11–9`) | **Heuristics:** `79.7%`.
  - **`checkpoints/galcon_wave14_kd.pt` + Default Wave 16 `DefStream+Guard` (`14.9–19.6 ms/t`, ~`1963 Elo`):**
    - **`Sparky v1.6`:** **`21–8–11` (`66.2%`, `+1.2%`)** | **`Waffle3z-GA`:** **`24–8–8` (`70.0%`, `s=9500: 82.5%`)** → **`68.1%` 20ms Dual-Boss Avg** + **`89.6%` `#1` EsparanBot** + **`50% / 75%` Large 24pl**!

## Wave 15 Highlights — Gen 5 Architecture (`2.93M` Edge-Gated Residual-Edge SwiGLU + Quadratic Soft-WDL)
- **Why We Upgraded Beyond Gen 4 (`1.29M`):** Controlled A/B ablations on `W14_KD` confirmed Gen 4's static-edge value ceiling (`3-ply h=24` dropped to `42.5%` Sparky; raising `neural_value_weight` `0.70 → 0.85` dropped `65.0% → 55.0%`).
- **New Self-Contained Gen 5 Architecture ([`galcon/gen5_model.py`](gen5_model.py), `2.93M` params, `11.7 MB` FP32, `0` edits to `orbit_wars_bot/`):**
  - **Edge-Conditioned Value Message Passing (`torch.bmm` AVX2) + 6-Layer Residual Pairwise Edge Track (`e_ij` refined every layer)** for 2-hop / pincer / rescue-chain coordination.
  - **Pre-Norm SwiGLU FFN (`d_model=192, n_heads=6, n_layers=6, dim_ff=384, edge_hidden=24`)** with state-dict-compatible residual `dropout=0.08` (zero inference cost at `.eval()`).
  - **Multi-Query 4-Token Attention Pooling + Stockfish/Lc0-Style 3-Class WDL Value Head `(P_loss, P_draw, P_win)`** with configurable **`contempt`** (`v = P_win - P_loss - c * P_draw`, defaulting to `0.30` in [`GalconSearchTransformerAgent`](transformer_bot.py)).
  - **Calibrated Training Engine Upgrades ([`galcon/train.py`](train.py), `--arch gen5`):** Exact-mean-preserving **Quadratic Soft-WDL targets** (`t_draw = 0.65 * (1 - |v|)^2`, guaranteeing `t_win - t_loss == v` without bogus draw mass on wins), **Game-Block Held-Out Validation** (`64`-state blocks, zero tick-level leakage), calibrated `pos_weight=2.2`, `weight_decay=2.5e-3`, and composite early stopping (`val_bce + 0.15 * val_v_loss`).
- **Gen 5 Foundation Checkpoint — [`checkpoints/galcon_gen5_w15.pt`](../checkpoints/galcon_gen5_w15.pt) (ONLY 9 Scratch Epochs on `15,017` Gen 4 Replay States — ZERO Gen 5 Self-Play Yet!):**
  - **Pure (`10.0 ms/t`, ZERO Search!):**
    - **`Waffle3z-GA` (40g):** **`20–12–8` (`60.0%` — NEW ALL-TIME PURE RECORD, `+7.5%` over Wave 8 Pure!)**
    - **`Sparky v1.6` (40g):** **`13–12–15` (`51.2%` Pure — beats unmodified Sparky >50% with zero search; `s=9500: 10–5–5 / 62.5%`!)**
  - **Single-Model Search (`c=0.30, nv=0.75`, Surgical, `33.0 ms/t`):**
    - **`Sparky v1.6` 18pl (40g, 1t):** **`19–7–14` (`65.0%` — ALL-TIME FEWEST SPARKY LOSSES AT 7 / `82.5%` NON-LOSS RATE; `s=14000: 8–2–10`, `s=9500: 11–5–4`, balanced `65.0% / 65.0%`!)**
    - **`Waffle3z-GA` 18pl (40g, 1t):** **`21–10–9` (`63.7%`, `s=9500: 67.5%`)** | **Large 24pl Asym:** `66.7%` Waffle3z / `41.7%` Sparky | **`EsparanBot v0.2`:** `75.0%` (`s=15300: 83.3%`) | **Heuristics:** `85.9%`.
  - *(Note: `checkpoints/galcon_wave14_kd.pt` remains the default `--bot search` in `galcon/gbots_client.py` at `20 ms/t` while Wave 16 launches Gen 5's own self-bootstrapping AlphaZero Search-EI flywheel!)*

## Wave 14 Highlights — Replay-Pooled AlphaZero Search-EI (`10,297` States → `#1` Single-Model Champion at `20 ms/t`)
- **#1 Single-Model All-Rounder — [`checkpoints/galcon_wave14_kd.pt`](../checkpoints/galcon_wave14_kd.pt) (default `--bot search`, `19.9–20.0 ms/tick`, ~`1955 Elo`!):**
  - First multi-round **Replay-Pooled AlphaZero Search-EI** wave using Wave 13's C99 fast engine + `--save-ei-npz` / `--load-ei-npz`:
    - **Round 14A:** `84` Large-heavy (`--large-map-prob 0.42`) Triple-Boss games (`5,231` states, `boss=31–16–9 / 63.4%`, `317.3s`).
    - **Round 14B:** `76` pure single-model (`21 ms`) `W13_KD` Search games (`5,066` states, `boss=29–20–3 / 58.7%`, `164.9s`).
    - **Pooled Distillation (`10,297` states across `160` games — `2.3×` larger than any previous wave!):** Achieved our lowest validation BCE ever (**`val_bce=0.027`**, `rec=95.3%`, `prec=72.3%`, `aux=97.0%`, `vsign=83.8%`).
  - **18pl Std `Waffle3z-GA` (40g, 1t):** **`24–8–8` (`70.0%` — NEW ALL-TIME SINGLE-MODEL RECORD, only 8 losses! `s=9500: 77.5%` Std / `82.5%` Surgical!)** @ **`19.9 ms/t`**.
  - **18pl Std `Sparky v1.6` (40g, 1t):** **`21–9–10` (`65.0%` — single-digit losses, perfectly balanced `s=14000: 65.0%` & `s=9500: 65.0%`)** @ **`20.0 ms/t`**.
  - **New All-Time Single-Model Dual-Boss Average:** **`67.5%`** (**`+3.1%`** over `W9S` & `W13_KD`, matching our 2-model `Ens(W9S+W6B)` at **`2×` lower latency!**).
  - **`EsparanBot v0.2` (24g):** **`20–1–3` (`89.6%` — NEW ALL-TIME RECORD, only 1 loss! `s=9500: 91.7%`, `s=15300: 9–0–3 / 87.5%` undefeated!)** — right at our **90%** boss mastery target!
  - **Large 24pl Asymmetric:** **`6–6–0` (`50.0%` — doubled `W13_KD`'s `25.0%`!)** vs. `Sparky v1.6` | **`9–3–0` (`75.0%`)** vs. `Waffle3z-GA`.

## Wave 13 Highlights — Profiling-Driven C99 Engine (`5.3ms` Pure / `21ms` Search) & `W13_KD` Single-Model Records
- **Profiling-Driven Codebase Acceleration (`0.00e+00` / `<2.4e-7` Bit-Exact Parity, `21/21` Unit Tests):**
  - **Fused C99 Feature + Timeline Kernel ([`galcon/fast_c.py`](fast_c.py)):** Auto-compiles a self-contained C99 shared library (`gcc -O3 -march=native -shared -fPIC`, hash-cached via atomic rename, zero-wrapper `c_void_p` pointers, graceful NumPy fallback on Windows/no-gcc) computing both `project_all_planet_futures` and all planet/timeline/edge features in one C call — cutting 18pl `extract_galcon_features` **`1.690 ms → 0.080–0.101 ms` (`16.7×` faster!)**.
  - **Lazy 2D-Centroid [`canonicalize_galcon_d4`](transformer_bot.py):** Evaluates all 8 D4 transforms on the 2D centroid vector in O(1), computes the sorted-coordinate tie-break lazily only on exact ties (<0.1% of states), and calls `apply_galcon_d4` once — cutting `1.743s` in Search to **`<0.02s` (`~80×` faster!)**.
  - **Vectorized [`extract_galcon_redir_features`](transformer_bot.py) (`3.8×` faster)** & vectorized `project_all_planet_futures` across [`galcon/heuristic_bot.py`](heuristic_bot.py) (`GalconTacticalBot`, `GalconRusherBot`, `GalconMacroBot`).
  - **`ms_only=True` Fast Paths (`_predict_all`, `_batch_forward_policy_and_val`):** Skips the unused Gen 2/3 legacy pairwise head (`decode_from_hidden`) and unnecessary value heads during Pure inference (`act` / `act_pure`) and Search mid-ply rollouts.
  - **Training Engine Upgrades ([`galcon/train.py`](train.py)):** AlphaZero EI Replay Buffer caching (`--save-ei-npz` / `--load-ei-npz` supporting comma-separated multi-wave pooling and `--search-games 0` instant retraining) + `B=10` PPO/validation lockstep waves with 3 cached Lua bridges per boss.
  - **End-to-End Measured Speedups on 2-vCPU Xeon VM:**
    - **Pure Neural Net (`18pl`):** **`5.3–5.7 ms/tick`** (`7.4–7.9 ms/t` Large 24pl — **`~1.7×` faster** than Wave 12!).
    - **Single-Model Search (`18pl`):** **`19.2–24.2 ms/tick`** (down from `32–36.6 ms/t` — **`~1.5×` faster**!).
    - **Surgical Ensemble Search (`cProfile`):** **`13.07s → 9.37s` (`1.40×` faster**, now `70.2%` pure PyTorch BLAS).
    - **PPO Rollouts (`B=6` `cProfile`):** **`3.49s → 2.36s` (`1.48×` faster)** — **512 episodes (16 iters + 5× `val_56g` at 35% Large maps)** complete in **`9.1 min`** (`~2.7×` faster than Wave 10!).
- **New Single-Model Soft-KD Champion — [`checkpoints/galcon_wave13_kd.pt`](../checkpoints/galcon_wave13_kd.pt) (`21–24 ms/tick`, ~`1941 Elo`!):**
  - Distilled 72 Triple-Boss games of our Wave 12 Surgical Ensemble (`boss=28–10–10 / 68.8%`) with `--kd-alpha 0.35` (`rec=95.4%`, `prec=76.5%–77.5%`, `aux=96.2%`, `vsign=85.8%`):
  - **18pl Std `Sparky v1.6` (40g, 1t):** **`22–12–6` (`62.5%`)** — **`s=14000: 12–4–4 (70.0%!!)` — NEW ALL-TIME `s=14000` RECORD across every single model AND 2-model ensemble in history!**
  - **18pl Std `Waffle3z-GA` (40g, 1t):** **`25–12–3` (`66.2%`)** (`67.5%` / `25–11–4` in Surgical single-model mode @ `21.0 ms/t`!) — **ties `W9S` for #1 Single-Model Dual-Boss Average (`64.4%`) at `21–24 ms/t`!**
  - **Large 24pl Asym `Waffle3z-GA` (12g):** **`10–2–0` (`83.3%` — ALL-TIME RECORD!)** | **`EsparanBot v0.2` (24g):** **`19–2–3` (`85.4%`, `s=9500: 91.7%`)** | **Heuristics (32g):** **`29–3–0` (`90.6%`)**.

## Wave 12 Highlights — Shattering 70% vs. `Sparky v1.6` & `Waffle3z-GA` + `2.3×` Faster VM Training
- **#1 Overall Champion — Wave 12 Surgical Ensemble (`EnsembleGalconSearchAgent([W9S, W6B])`, default `--bot ensemble`, `35–42 ms/tick` W13 / `46–52 ms/tick` W12, ~`1979 Elo`!):**
  - Driven directly by loss forensics against `esparano Sparky v1.6` (which exposed passive symmetric stalemate draws and empty opening soft-minimax replies), added two surgical search helpers in [`galcon/transformer_bot.py`](transformer_bot.py) enabled by default on `EnsembleGalconSearchAgent`:
    1. **`_build_rescue_and_snipe_plan`:** Exact-need defensive rescue for any friendly planet projected to fall + single-target high-ROI snipe/stalemate-breaker whenever conservative multi-target fan-out returns empty.
    2. **`_build_opp_direct_rush_plan`:** Soft-minimax fallback stress-testing an 85% strike from the opponent's strongest planet at our closest/weakest planet when opponent fan-out/focus-fire is empty (e.g. `t=0` openings).
  - **18pl Std `Sparky v1.6` (40g, 1-tick):** **`26–9–5` (`71.2%` — NEW ALL-TIME RECORD, `+5.0%` jump, single-digit losses!)** (`s=14000: 11–5–4 / 65.0%`, `s=9500: 15–4–1 / 77.5%`, `52.4 ms/t`).
  - **Built-in Heuristics (32g):** **`29–2–1` (`92.2%`, `+3.1%`)** | **Large 24pl Asym:** **`8–4–0` (`66.7%`)** Sparky & **`9–3–0` (`75.0%`)** Waffle3z-GA (zero Large regression) | **18pl `Waffle3z-GA`:** `24–13–3` (`63.7%`, `46.3 ms/t`).
  - **Direct-Rush Mode (`enable_rescue_snipe=False`):** Sets a **new all-time `Waffle3z-GA` record** at **`27–10–3` (`71.2%`**, `s=14000: 70.0%`, `s=9500: 72.5%` @ `47.3 ms/t`) while holding `66.2%` vs. `Sparky v1.6` (`68.8%` dual-boss average).
  - **Bit-Exact `need_value=False` Skip & Leaf Value Path:** Decoupled value-critic ensembles (`Ens(W9S+W6B+W11_KD)` → `67.5%` Sparky, `83.3%` Large Asym Waffle3z) skip value-only members on `opp_enc` & mid-ply rollouts (~30% faster), and `_batch_evaluate_leaves` calls `encode_planets + decode_value_from_hidden` directly.
- **Wave 12 VM Training Engine Speedups (~`1.55×` faster over Wave 11 / `~2.1×–2.3×` over Wave 10 on 2-vCPU VM!):**
  - **`forward_multi_send` ([`GalconPlanetTransformer`](transformer_bot.py)):** Skips the unused Gen 2/3 $O(B \cdot N^2 \cdot D)$ `decode_from_hidden` pairwise head across Gen 4 EI training, PPO rollouts, `ema_teacher` precompute, PPO updates, and snapshot actors (100% bit-exact).
  - **Length-Bucketed Minibatching (`_LengthBucketBatchSampler` in [`galcon/train.py`](train.py)):** Groups `<=14pl`, `15–18pl`, and `>=19pl` states into tight minibatches so `_trim_batch_n` avoids $O(N^2)$ padding waste, cutting PPO `upd` from **`25–29s → 14.5–18.3s`** (`~1.55×` faster than Wave 11, `~2.1×` faster than Wave 10!).
  - **Persistent Worker Lua/Snapshot Caches + `B=6` Lockstep Batched `val_56g`:** Eliminates per-iteration `lua5.3` respawns & snapshot reloads inside `ProcessPoolExecutor` workers and batches validation games in lockstep (`val_56g`: `~19s → ~12s`).

## Wave 11 Highlights — `1.8×–2.0×` Faster Training Engine & Hybrid Soft-KD Records
- **`1.8×–2.0×` Faster Triple-Boss Training Engine in [`galcon/train.py`](train.py):**
  - **Batched Lockstep Multi-Game PPO Rollouts (`_run_gen4_ppo_episodes_batched`, `B=6`):** Batches 6 simultaneous games inside each rollout worker (with dual Lua bridges per boss so both P1 & P2 run lockstep in one forward pass), cutting rollout time from `17.5–20.4s → 8.9–11.8s` (**`~1.6×` faster**).
  - **Precomputed Frozen `ema_teacher` + `batch_size=96` Updates:** Caches KL teacher probabilities & value targets once per PPO iteration instead of re-forwarding on every minibatch (`30–35s → 14.9–21.0s`, **`~1.5×` faster**).
  - **Parallel Worker-Sharded `val_56g` (`_gen4_val_worker`):** Shards the 56-game validation gate across `pool_exec` workers (`~35–40s → ~18–20s`, **`~2×` faster** on 2-vCPU VM and **`4×–8×` faster** on multi-core desktops).
  - **End-to-End Measured Speedup:** 10 full Triple-Boss PPO iterations (`320` episodes + `3× val_56g`) now finish in **`6.0 minutes`** (`28–32 s/iter`, down from `50–55 s/iter`) on a 2-vCPU VM!
- **Hybrid Search-EI + Ensemble Soft-KD (`--kd-alpha 0.35`, [`checkpoints/galcon_wave11_kd.pt`](../checkpoints/galcon_wave11_kd.pt)):**
  - Caches root `ms_probs` & `val` inside [`GalconSearchTransformerAgent.act_with_search_info`](transformer_bot.py) (`0.0 ms` overhead) and distills `Ens(W9S + W6B)` across 64 Triple-Boss games into `galcon_wave8_pure.pt` with Soft-KD BCE + blended value targets (`0.55*z + 0.25*best_sc + 0.20*t_val`):
  - **All-Time Record Calibration:** **`aux = 97.4%`**, **`vsign = 88.2%`** (`+3.2%` over Wave 9, `+5.1%` over Wave 10 hard-only EI!), `rec = 96.2%`, `prec = 70.3%`, jumping `val_56g` Pure `+7.2%` (`59.8% → 67.0%`) and raw EI Pure Heuristics **`76.6% → 92.2%`** (`+15.6%`) / Pure Sparky **`38.8% → 47.5%`** (`+8.7%`)!
  - **3 New Held-Out Benchmark Records (`W11_KD_Search`, `28–33 ms/t`, ~`1925 Elo`):**
    - **18pl Std `Waffle3z-GA` (40g, 1t):** **`26–11–3` (`68.8%` @ `27.9 ms/t`)** (`s=14000: 13–5–2 / 70.0%`!, `s=9500: 67.5%`) — **new Single-Model Record**, tying 2-model `#1` Ensemble `Ens(W9S+W6B)` at `~1.6×` lower latency!
    - **Large `22–28pl` Paired `Sparky v1.6` (12g):** **`7–5–0` (`58.3%`)** — **new All-Time Record** across all single models and 2-model 3-ply ensembles (`Ens(W6B+W7p1)` was `50.0%`)!
    - **`EsparanBot v0.2` (24g):** **`19–2–3` (`85.4%`)** (`s=9500: 11–1–0 / 91.7%`!, `s=15300: 79.2%`) — **new All-Time Record**!
    - **18pl Std `Sparky v1.6` (40g, 1t):** `21–15–4` (`57.5%`, balanced `57.5%/57.5%`, `32.9 ms/t`); **Large `24pl` Asym:** `41.7%` Sparky / `66.7%` Waffle3z-GA. Cross-lineage `Ens(W11_KD + W7p1)` scores `60.0%` Sparky & `66.2%` Waffle3z-GA (`42–48 ms/t`).
- **Fast 10-Iter Triple-Boss PPO ([`checkpoints/galcon_wave11_pure.pt`](../checkpoints/galcon_wave11_pure.pt), `6.0 min`):**
  - **New `10 ms` Pure (ZERO Search!) Large Paired `Sparky v1.6` Record:** **`6–6–0` (`50.0%`)** — **doubling `galcon_wave8_pure.pt`'s `25.0%`** and matching our 3-ply Ensemble with zero search!
  - Under Single-Model Search (`25–32 ms/t`): **`21–12–7` (`61.3%`, `s=9500: 70.0%`)** vs. `Sparky v1.6`, **`50.0%`** Large Asym Sparky, and **`83.3%`** Large Asym Waffle3z-GA.

## Wave 10 Highlights — Waffle3z-GA Integration, Triple-Boss Distillation & New All-Round Champion
- **Galcon `botwar` Functional API Support in [`galcon/lua_bridge.py`](lua_bridge.py):**
  - Auto-detects `botwar` sandboxed scripts (`_sandbox_init`), adds Lua 5.3 `math.pow`/`math.atan2` shims, tracks a real per-tick `sb_stats()` call+time budget counter (matching Galcon 2's `g2_sandbox(..., 64, 64)` cap), maintains per-player `memory`, and supports cross-platform `path.lua::func_name` selectors so `external/GalconModVault/esparano/bots/bot_hivemind/mod_botwar_testing.lua::bots_waffle3z_optimized` runs **unmodified with zero Lua errors**.
  - **External Lua Boss Hierarchy:** `Waffle3z-GA` crushes `EsparanBot v0.2` **`12–5–3` (`67.5%`)** and pulls **`5–7–0` (`41.7%`)** on Large Paired vs. `Sparky v1.6` (`36.3%` on 40g 18pl Std) — clear **#2 external Lua boss (~`1745 Elo`)**.
- **#1 All-Round Triple-Boss Champion — `Ens(W9S + W6B)` (`galcon_wave9_search.pt` + `galcon_transformer_best.pt`, `43–49 ms/tick`, `--bot ensemble` default):**
  - **18pl Std `Sparky v1.6` (40g, 1-tick):** **`24–11–5` (`66.3%`, ~`1945 Elo`)** (`s=14000: 60.0%`, `s=9500: 72.5%`, `49.5 ms/t`)
  - **18pl Std `Waffle3z-GA` (40g, 1-tick):** **`26–11–3` (`68.8%` — `+5.0%` over Wave 9 Ensemble!)** (`s=14000: 65.0%`, `s=9500: 72.5%`, `43.4 ms/t`)
  - **Large 24pl Asymmetric:** **`8–4–0` (`66.7%`)** vs. `Sparky v1.6` | **`9–3–0` (`75.0%`)** vs. `Waffle3z-GA` (`8–4–0` / `66.7%` Large Paired vs. `Waffle3z-GA`)!
- **Triple-Boss Distilled Checkpoint — [`checkpoints/galcon_wave10_search.pt`](../checkpoints/galcon_wave10_search.pt) (`28–30 ms/tick`):**
  - Distilled 64 games of `Ens(W9S + W6B)` across all three Lua bosses (`23–13–8` / `61.4%` vs. triple bosses, `223` multi-send ticks, `915` redir states) into `galcon_wave8_pure.pt` (`prec=67.6%`, `rec=92.5%`, `aux=96.9%`, `vsign=83.1%–87.5%`):
  - **Single-Model Results:** **`24–11–5` (`66.3%`, `s=9500: 77.5%`!)** vs. `Waffle3z-GA` and **`9–3–8` (`65.0%`)** vs. `Sparky v1.6` `s=14000`!
  - **Mirror-Image Ensemble `Ens(W9S + W10S)` (`41–51 ms/t`):** **`23–10–7` (`66.3%`, fewest Sparky losses ever, `s=14000: 67.5%` record!)** vs. `Sparky v1.6` and **`24–11–5` (`66.3%`)** vs. `Waffle3z-GA`; 3-model `Ens(W9S + W10S + W6B)` hits **`26–10–4` (`70.0%`!)** vs. `Waffle3z-GA`!
- **Triple-Boss Training & 18pl-Aligned `val_56g` Gate (`--lua-opp3`, `--waffle-share` in [`galcon/train.py`](train.py)):**
  - Restored Wave 8's uniform `0.05` PPO potential shaping (removing Wave 9's `0.085` opening boost that caused greedy early over-expansion), aligned `val_56g` with paired 18pl Standard boards alongside Compact/Mixed & Large, and enforced dual promotion floors on Sparky + `(0.60*spky + 0.40*waff)`.
- **Upgraded Interactive Viewer ([`docs/galcon_viewer.html`](../docs/galcon_viewer.html)):** Includes **Wave 11 Soft-KD & Pure Large showcase wins**, **Waffle3z-GA & Sparky Win/Loss Forensics**, an **External Lua Bot War (`Sparky v1.6 vs. Waffle3z-GA`)**, and upgraded **Triple-Boss Win-Rate Comparison Bars & Elo Ladder**.

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

# Wave 13 Fast Replay-Pooled Search-EI + Soft-KD (--save-ei-npz / --load-ei-npz, C99 16.7x feature engine!):
python3 -m galcon.train --mode gen2 --stage gen4_ei \
  --resume checkpoints/galcon_wave9_search.pt \
  --resume2 checkpoints/galcon_transformer_best.pt \
  --student-init checkpoints/galcon_wave8_pure.pt \
  --lua-opp external/GalconModVault/esparano/bots/sparky/bot_sparky_v1.6.lua \
  --lua-opp2 external/GalconModVault/esparano/bots/bot_esparanbot_v0.2/bot_esparanbot.lua \
  --lua-opp3 external/GalconModVault/esparano/bots/bot_hivemind/mod_botwar_testing.lua::bots_waffle3z_optimized \
  --search-games 72 --epochs 8 --lr 1.8e-4 --kd-alpha 0.35 \
  --save-ei-npz checkpoints/ei_wave13.npz \
  --workers 2 --train-threads 2 --device cpu \
  --out checkpoints/galcon_wave13_kd.pt

# Wave 13 Fast Batched Triple-Boss PPO (C99 fast_c + lazy D4 + B=10 lockstep waves + precomputed EMA teacher):
python3 -m galcon.train --mode gen2 --stage gen4_ppo \
  --resume checkpoints/galcon_wave13_kd.pt \
  --gen2-anchor checkpoints/galcon_wave8_pure.pt \
  --gen2-anchor2 checkpoints/galcon_wave13_kd.pt \
  --lua-opp external/GalconModVault/esparano/bots/sparky/bot_sparky_v1.6.lua \
  --lua-opp2 external/GalconModVault/esparano/bots/bot_esparanbot_v0.2/bot_esparanbot.lua \
  --lua-opp3 external/GalconModVault/esparano/bots/bot_hivemind/mod_botwar_testing.lua::bots_waffle3z_optimized \
  --sparky-share 0.45 --waffle-share 0.22 --esparan-share 0.13 --large-map-prob 0.35 \
  --ppo-iters 16 --mirror-pairs 16 --eval-every 4 \
  --kl-coef 0.035 --v-anchor-coef 0.15 --lr 3.5e-5 \
  --workers 2 --train-threads 2 --device cpu \
  --out checkpoints/galcon_wave13_pure.pt

# Record Wave 13 Triple-Boss showcase replays & Elo viewer:
python3 -m galcon.record_replays \
  --checkpoint checkpoints/galcon_wave9_search.pt \
  --checkpoint2 checkpoints/galcon_transformer_best.pt \
  --search-checkpoint checkpoints/galcon_wave9_search.pt \
  --w10-checkpoint checkpoints/galcon_wave10_search.pt \
  --w11-kd-checkpoint checkpoints/galcon_wave11_kd.pt \
  --w11-pure-checkpoint checkpoints/galcon_wave11_pure.pt \
  --w13-kd-checkpoint checkpoints/galcon_wave13_kd.pt \
  --pure-checkpoint checkpoints/galcon_wave8_pure.pt \
  --lua-sparky external/GalconModVault/esparano/bots/sparky/bot_sparky_v1.6.lua \
  --lua-waffle external/GalconModVault/esparano/bots/bot_hivemind/mod_botwar_testing.lua::bots_waffle3z_optimized \
  --lua-esparan external/GalconModVault/esparano/bots/bot_esparanbot_v0.2/bot_esparanbot.lua \
  --out-html docs/galcon_viewer.html

# Run all 21 unit tests (Wave 13 C99 fast_c parity + lazy D4 + vectorized redir + ms_only + EI cache, Wave 12 Surgical Search + VM speedups, Wave 11 Soft-KD + Batched PPO, Waffle3z Lua bridge, Triple-boss val gate, Map-adaptive adversarial, Canonical D4, DMLAdamW, Ensemble):
pytest -v tests/test_galcon.py
```

### 2. Plug Straight Into Phil Hassey's Official `gbots`
```bash
# Tier 1 — #1 Overall Triple-Boss Ensemble (71.2% Sparky / 66.7%–75.0% Large @ 35–42 ms W13!):
./gbots pipe -name OurGalconWave12Ensemble -key YOUR_KEY \
  -exec "python3 -m galcon.gbots_client --bot ensemble"

# Tier 2 — #1 Fast Single-Model Search (21–24 ms W13! Use galcon_wave13_kd.pt for 70.0% s14k Sparky & 83.3% Large Waffle3z, or galcon_wave9_search.pt for 66.3% 40g Sparky):
./gbots pipe -name OurGalconWave13Search -key YOUR_KEY \
  -exec "python3 -m galcon.gbots_client --bot search --checkpoint checkpoints/galcon_wave13_kd.pt"

# Tier 3 — Ultra-Fast Wave 8 Pure neural net (~5.5 ms W13 VM, 61.3% Sparky / 52.5% Waffle3z-GA, ZERO search!):
./gbots pipe -name OurGalconWave8Pure -key YOUR_KEY \
  -exec "python3 -m galcon.gbots_client --bot transformer --checkpoint checkpoints/galcon_wave8_pure.pt"
```

Interactive HTML5 Canvas Replay Viewer (with Wave 13 `21 ms` Single-Model & Wave 12 Surgical Showcase Wins, Sparky v1.6 & Waffle3z-GA Forensics, External Lua Bot War & Triple-Boss Scoreboard): [`docs/galcon_viewer.html`](../docs/galcon_viewer.html).
