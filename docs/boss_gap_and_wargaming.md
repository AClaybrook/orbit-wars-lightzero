# Orbit Wars: Midgame Boss Forensics (`PT-Gen8` vs. `#2 Hober Malloc`) & 3v1 Coalition Wargaming Study

> **Scope:** Deep empirical diagnosis of why `1.89M PT-Gen8` (~1,740 Elo) diverges from `#2 Hober Malloc` (`1800.1`) in the midgame, paired with a multi-dimensional 3v1 wargaming study quantifying tradeoffs between **Intelligence Gap ($\Delta\text{Elo}$)**, **Coordination Fidelity (Truce, Latency, Compliance, Info Noise)**, and **Starting Resource Asymmetry**.

---

## Part 1 — Midgame Forensics: Why `PT-Gen8` Loses to `#2 Hober Malloc` Around `t = 40–60`

Measured across all 6 official 2P Boss Gauntlet seeds (`200–205`, via [`scripts/analyze_boss_gap.py`](../scripts/analyze_boss_gap.py)), paired with **860 exact counterfactual policy queries** against `#2 Hober Malloc`'s full-TTA JAX policy on `PT-Gen8`'s live board states.

### 1.1 Economy & Fleet Trajectory (Mean Across Seeds `200–205`)

| Tick | Planets (`PT-Gen8` vs `#2`) | Prod/tick (`PT-Gen8` vs `#2`) | Total Ships (`PT-Gen8` vs `#2`) | In-Flight % (`PT-Gen8` vs `#2`) | Phase |
|---:|---:|---:|---:|---:|---|
| `t=0` | `1.0` vs `1.0` | `3.0` vs `3.0` (`+0.0`) | `10.0` vs `10.0` (`+0.0`) | `0.0%` vs `0.0%` | Opening |
| `t=10` | `1.5` vs `1.7` | `4.3` vs `5.2` (`-0.8`) | `37.7` vs `37.2` (**`+0.5`**) | `56.9%` vs `57.4%` | Opening (Dead Even) |
| `t=20` | `3.0` vs `3.5` | `10.5` vs `12.7` (`-2.2`) | `87.0` vs `88.3` (`-1.3`) | `58.4%` vs `50.4%` | Opening (`79.7%` exact policy match) |
| `t=30` | `4.8` vs `6.3` | `17.3` vs `21.3` (`-4.0`) | `190.5` vs `202.2` (`-11.7`) | `44.4%` vs `42.6%` | Early Midgame Drift |
| `t=40` | `7.0` vs `8.7` | `24.5` vs `29.0` (`-4.5`) | `313.8` vs `367.2` (`-53.3`) | `53.4%` vs `59.0%` | Ship Bleed Accelerates |
| `t=50` | `8.7` vs `9.7` | `29.7` vs `32.5` (`-2.8`) | `414.0` vs `545.8` (**`-131.8`**) | `59.3%` vs `62.6%` | **Tipping Point** (Prod nearly equal, `-132` ships down!) |
| `t=60` | `8.2` vs `12.3` | `27.2` vs `39.7` (`-12.5`) | `373.7` vs `582.0` (`-208.3`) | `68.2%` vs `68.6%` | Frontline Collapse |
| `t=75` | `5.7` vs `15.2` | `18.2` vs `49.3` (`-31.2`) | `389.7` vs `865.7` (`-476.0`) | `59.8%` vs `78.5%` | Snowball Blowout |
| `t=90` | `2.7` vs `17.5` | `6.8` vs `60.0` (`-53.2`) | `294.8` vs `1398.8` (`-1104.0`) | `43.8%` vs `80.6%` | Endgame Cleanup |

### 1.2 The 5 Concrete Root Causes (and What Was Ruled Out)

- **Ruled Out — Small-Planet Expansion Bias:** `PT-Gen8` captures virtually the same number of neutrals (`9.67` vs. `10.00`/game) with slightly *higher* average production (`3.28` vs. `3.10`) and an almost identical production tier split (`prod <= 2 / 3 / >= 4`: `18 / 11 / 29` vs. `21 / 11 / 28`). Colonization selection is **not** the bottleneck.

1. **Impulsive Midgame All-In Shuffling (`58.3%` of All Midgame Policy Errors):**
   - Exact counterfactual match with `#2 Hober Malloc` drops from **`79.7%`** (`98/123` decisions in `t = 0..25`) down to **`66.3%`** (`416/627` decisions in `t = 26..65`).
   - Breakdown of all `211` midgame disagreements:
     - **`94` (`44.5%`):** `PT-Gen8 REINFORCE` when Hober would **`HOLD`** — premature all-in garrison shuffles that leave frontline planets empty and put ships out of action in transit.
     - **`41` (`19.4%`):** `PT-Gen8 HOLD` when Hober would **`REINFORCE`** — mistiming the actual defensive/staging transfer.
     - **`29` (`13.7%`):** `PT-Gen8 ENEMY` attack when Hober would **`HOLD`** — impulsive solo attacks that bounce instead of accumulating for a decisive push.
     - **`18` (`8.5%`):** `PT-Gen8 HOLD` when Hober would expand to a **`NEUTRAL`**.
     - **`10` (`4.7%`):** `PT-Gen8 HOLD` when Hober would launch an **`ENEMY`** strike.
   - Combined, **`58.3%`** (`123/211`) of midgame mistakes are **launching all-in when `#2 Hober` would HOLD & accumulate**.
2. **`3.14×` More Neutral Overkill (`1,521` vs. `485` Excess Ships):**
   - Because `PT-Gen8` uses `force_all_in=True` (`1.0` send fraction) without surgical timing/partial-send discipline, it dumps **`3.14×`** more excess ships into neutral captures (`1,521` vs. `485` surplus ships across 6 games). That idle capital directly explains the **`-131.8` ship deficit at `t=50`** even when production is almost tied (`29.7` vs. `32.5`).
3. **`+55%` More Bounced / Failed Attacks (`235` vs. `152` Bounces; `3,074` vs. `2,573` Ships Lost):**
   - `PT-Gen8` bounces `235` fleets (`3,074` ships) off enemy/neutral targets without flipping them, whereas Hober bounces only `152` fleets while capturing **`2.03×` more enemy planets (`124` vs. `61`)**.
4. **Half the Midgame Offensive Pressure (`70` vs. `135` Enemy Attacks in `t = 26..65`):**
   - In `t = 26..65`, neutral launches (`63` vs. `64`) and reinforcements (`344` vs. `337`) are nearly identical, **but `#2 Hober` launches `1.93×` more direct offensive strikes at our planets (`135` vs. `70`)**, constantly forcing us onto the back foot.
5. **Comet Blindness (`0` vs. `5`) & Late-Game Tempo (`3.12` vs. `9.46` Fleets/Active Turn):**
   - `#2 Hober` captured **`5` comets vs. `PT-Gen8`'s `0`** across the 6 games — comets spawn right at `t = 50` (the exact tipping point) and act as free mobile production/ship boosts.
   - Once ahead (`t = 60+`), Hober sustains `78.5%–82.5%` airborne fleet utilization (`9.46` fleets/active turn vs. our `3.12`), choking every planet before counterplay can build.

### 1.3 Concrete Roadmap for Gen 9 (`PT-Gen9`) — Direct Fixes for the 5 Bottlenecks

> *Queued for the next phase immediately after finalizing the Wargaming study PR:*

1. **Midgame-Weighted Soft-KD + Hold-Calibration (`t = 25..75` upweight `2.5×`):** Upweight the exact midgame tipping window (`t = 25..75`) and `HOLD`-vs-`REINFORCE` boundary states in `distill_soft_kd.py`, plus test per-action-type launch thresholds (higher bar for `REINFORCE` shuffling, e.g., `thr_reinforce = 0.60`, while keeping `thr_enemy = 0.45`).
2. **Surgical Send-Fraction / Anti-Overkill Decoder:** Instead of blanket `force_all_in=True` on neutrals, cap neutral launches at `needed_ships + safety_margin` (using `ships_needed_to_capture` from `physics.py`) while keeping full All-In for contested enemy frontline assaults — immediately reclaiming ~150+ wasted ships per game by `t = 50`.
3. **Anti-Bounce Attack Filter / Multi-Wave Accumulation:** Prevent solo attacks guaranteed to bounce (`send_ships <= target_projected_defense`) unless another friendly fleet is already inbound to that target within `±3` ticks — directly attacking the `235` bounced fleets & `29` impulsive `ours_ENEMY_hober_HOLD` errors.
4. **Comet-Aware Training Data & Feature Emphasis:** Sample more `t >= 48` comet-spawn states in Soft-KD DAgger rollouts so `PT-Gen9` actively snipes `t = 50` comets (`0 → 3+` per 6 games).
5. **Post-KD On-Policy Midgame Self-Play RL:** Fine-tune `PT-Gen9` with PPO anchored to Soft-KD probabilities (`KL` anchor) specifically against `PT-Gen8` + `#2 Hober` + `Gen 7` to sharpen midgame offensive pressure (`70 → 120+` strikes) without reverting to passive shuffling.

---

## Part 2 — 4P Gen 8 vs. Gen 7 & 3v1 Coalition Wargaming Results

### 2.1 4-Player FFA: `PT-Gen8` (2P Specialist) vs. `Gen 7 PT-v5-Boss` (4P Specialist)

| Matchup (Kaggle Environments 4P) | Result | Top-2 % | Key Insight |
|---|---|---:|---|
| **`2× Gen 7` vs. `2× PT-Gen8`** (6 balanced seats, seeds `600–605`) | **Gen 7 wins `4–2`** | `58.3%` vs. `41.7%` | Validates our `#5 TonyK` 2P/4P Specialist router (`pt_specialist`): 2P All-In commits leave flanks exposed in 4P neural tables! |
| **`1× PT-Gen8` vs. `3× subham_v14`** (seeds `400–403`) | **`PT-Gen8` `4–0` (`114.8t` avg)** | `100.0%` | Beats Gen 7's `133.8t` record by 19 ticks (fastest wipe: `85t` on seed 401). |

### 2.2 Official Kaggle 3v1 Coalition Gauntlet (`3× subham_v14`, `~1,224` Elo, Seeds `500–503`)

| Boss Agent | Mode | Coalition Takedown Rate | Boss Elim Tick(s) | Avg Sync Volleys |
|---|---|---:|---:|---:|
| **`1.89M PT-Gen8` (~1,740)** | FFA (`none`) / Truce (`lat=0`) | `0.0%` (`0/4`) / `0.0%` (`0/4`) | — | `0` |
| **`1.89M PT-Gen8` (~1,740)** | **Focus-Fire (`lat=0`)** | **`50.0%` (`2/4`)** | `t=240` (#3), `t=207` (#4) | `113.3` |
| **`1.89M PT-Gen8` (~1,740)** | Focus-Fire (`lat=5`) | `25.0%` (`1/4`) | `t=124` (#4) | `77.0` |
| **`Gen 7 PT-v5-Boss` (~1,700, 4P)** | Truce (`lat=0`) | `0.0%` (`0/4`) | — | `0` |
| **`Gen 7 PT-v5-Boss` (~1,700, 4P)** | Focus-Fire (`lat=0` / `lat=5`) | `25.0%` (`1/4`) / `25.0%` (`1/4`) | `t=100` / `t=115` (#4) | `52.8` / `46.3` |
| **`#2 Hober Malloc` (`1800.1`, seeds `500–502`)** | **Focus-Fire (`lat=0`)** | **`33.3%` (`1/3`)** | **`t=482` (#4, 504 volleys!)** | `182.7` |

### 2.3 Multi-Dimensional Wargaming Phase-Transition Sweep (`200` `FastOrbitSim` 4P Matches, Seeds `700–703`)

#### Grid A — Intelligence Gap ($\Delta\text{Elo}$) × Coordination Fidelity (`3:1` Standard Resources vs. `PT-Gen8`)

| Ally Tier ($\Delta\text{Elo}$) | `C0` FFA | `C1` Truce | `C2` Imperfect (`Lat=10, 50% Comp, 30% Noise`) | `C3` Delayed (`Lat=5`) | `C4` Zero-Lat (`Lat=0`) |
|---|---:|---:|---:|---:|---:|
| **`Gen 3 PT-v1`** (`+310`) | `25%` (`1/4`) | `25%` (`1/4`) | **`100%` (`4/4`, `129.8t`)** | **`100%` (`4/4`, `115.2t`)** | **`75%` (`3/4`, `114.7t`)** |
| **`subham_v14`** (`+516`) | `0%` (`0/4`) | `25%` (`1/4`) | `25%` (`1/4`) | **`50%` (`2/4`)** | **`50%` (`2/4`, `156.5t`)** |
| **`wd_heuristic`** (`+635`) | `0%` (`0/4`) | `0%` (`0/4`) | `0%` (`0/4`) | `0%` (`0/4`) | `0%` (`0/4`) |
| **`starter`** (`+740`) | `0%` (`0/4`) | `0%` (`0/4`) | `0%` (`0/4`) | `0%` (`0/4`) | `0%` (`0/4`) |

#### Grid B — Resource Ratio (`Coalition / Boss Starting Ships`) × Coordination (`subham_v14`, $\Delta\text{Elo} = +516$)

| Mode | `6.0×` (`Boss 0.5×`) | `3.0×` (`Std 4P`) | `1.5×` (`Boss 2×`) | `1.0×` (`Equal Ships`) | `0.6×` (`Boss 5×`) |
|---|---:|---:|---:|---:|---:|
| **Uncoordinated FFA (`C0`)** | `0%` (`0/4`) | `0%` (`0/4`) | `0%` (`0/4`) | `0%` (`0/4`) | `0%` (`0/4`) |
| **Zero-Lat Focus-Fire (`C4`)** | **`75%` (`3/4`)** | **`50%` (`2/4`)** | **`50%` (`2/4`)** | **`25%` (`1/4`)** | `0%` (`0/4`) |

*(Note: `wd_heuristic (+635)` and `starter (+740)` scored **`0/80` (`0%`)** across ALL 5 resource ratios under both FFA and Focus-Fire.)*

### 2.4 Fitted Logistic Power-Law Tradeoff Equation (`N = 200`, Weighted MAE = `6.2%`)

$$\text{logit}\big(P_{\text{coalition}}\big) = 3.523 - 1.653 \cdot \frac{\Delta\text{Elo}}{100} + 3.529 \cdot C + 0.723 \cdot \log_2(R)$$

- **Elo Value of Perfect Coordination (`C0 → C4`):** **`+213.5 Elo`** (lifts the 50% takedown boundary from **`+282 Elo`** in FFA to **`+496 Elo`** under Zero-Latency Focus-Fire!).
- **Elo Value of Doubling Starting Resources (`2× R`):** **`+43.7 Elo`** — meaning perfect coordination is worth roughly a **`~29×`** starting resource boost compared to an uncoordinated FFA!
- **Intelligence Wall:** Beyond **`ΔElo ≥ +600 Elo`**, neither `6×` starting resources nor zero-latency coordination can overcome fundamental tactical incompetence (`0/80` observed wins).

Interactive 2D gradient heatmaps, rotatable 3D surfaces, and live tradeoff sliders are available in [`docs/wargame_report.html`](wargame_report.html), and the 3v1 takedown replays (including `3× subham_v14` eliminating `#2 Hober Malloc`) are playable in [`docs/gameplay_viewer.html`](gameplay_viewer.html).

