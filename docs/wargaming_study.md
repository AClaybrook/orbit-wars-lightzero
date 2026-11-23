# Orbit Wars: 3v1 Coalition Wargaming & Power-Dynamics Study

> **Research Track:** Quantifying asymmetric multi-agent power dynamics — how **Intelligence Gap ($\Delta\text{Elo}$)**, **Coordination Fidelity (Truce, Comm Latency, Partial Compliance, Intel Noise)**, and **Starting Resource Asymmetry** trade off when three independent weaker actors team up against a single stronger Boss (without merging resources).
>
> *Interactive 2D gradient heatmaps, rotatable 3D surfaces, and live tradeoff calculator:* [`docs/wargame_report.html`](wargame_report.html)  
> *Playable 3v1 takedown replays:* [`docs/gameplay_viewer.html`](gameplay_viewer.html)

---

## 1. Framework (`orbit_wars_bot/coalition.py`)

Each coalition ally remains a **strictly independent Orbit Wars player** (separate planets, separate fleets, zero resource sharing):
- **`C0` (`none`):** Uncoordinated 4P FFA baseline.
- **`C1` (`truce`):** Non-aggression pact (blocks friendly fire between allies) + neutral-capture claim deconfliction.
- **`C2` (Imperfect Focus-Fire):** Truce + synchronized boss targeting with `comm_latency = 10` ticks, `compliance = 0.50`, and `info_noise = 0.30`.
- **`C3` (Delayed Focus-Fire):** Truce + synchronized boss targeting with `comm_latency = 5` ticks, `compliance = 1.0`.
- **`C4` (Zero-Latency Focus-Fire):** Truce + exact zero-latency synchronized multi-angle volleys (`comm_latency = 0`, `compliance = 1.0`).
- Coalition automatically dissolves once the Boss reaches 0 planets and 0 fleets.

---

## 2. Multi-Dimensional Phase-Transition Sweep (`200` `FastOrbitSim` Matches, Seeds `700–703`)

### 2.1 Grid A — Intelligence Gap ($\Delta\text{Elo}$) × Coordination Fidelity (`3:1` Standard Resources vs. `PT-Gen8`)

| Ally Tier ($\Delta\text{Elo}$) | `C0` FFA | `C1` Truce | `C2` Imperfect (`Lat=10, 50% Comp, 30% Noise`) | `C3` Delayed (`Lat=5`) | `C4` Zero-Lat (`Lat=0`) |
|---|---:|---:|---:|---:|---:|
| **`Gen 3 PT-v1`** (`+310`) | `25%` (`1/4`) | `25%` (`1/4`) | **`100%` (`4/4`, `129.8t`)** | **`100%` (`4/4`, `115.2t`)** | **`75%` (`3/4`, `114.7t`)** |
| **`subham_v14`** (`+516`) | `0%` (`0/4`) | `25%` (`1/4`) | `25%` (`1/4`) | **`50%` (`2/4`)** | **`50%` (`2/4`, `156.5t`)** |
| **`wd_heuristic`** (`+635`) | `0%` (`0/4`) | `0%` (`0/4`) | `0%` (`0/4`) | `0%` (`0/4`) | `0%` (`0/4`) |
| **`starter`** (`+740`) | `0%` (`0/4`) | `0%` (`0/4`) | `0%` (`0/4`) | `0%` (`0/4`) | `0%` (`0/4`) |

### 2.2 Grid B — Resource Ratio (`Coalition / Boss Starting Ships`) × Coordination (`subham_v14`, $\Delta\text{Elo} = +516$)

| Mode | `6.0×` (`Boss 0.5×`) | `3.0×` (`Std 4P`) | `1.5×` (`Boss 2×`) | `1.0×` (`Equal Ships`) | `0.6×` (`Boss 5×`) |
|---|---:|---:|---:|---:|---:|
| **Uncoordinated FFA (`C0`)** | `0%` (`0/4`) | `0%` (`0/4`) | `0%` (`0/4`) | `0%` (`0/4`) | `0%` (`0/4`) |
| **Zero-Lat Focus-Fire (`C4`)** | **`75%` (`3/4`)** | **`50%` (`2/4`)** | **`50%` (`2/4`)** | **`25%` (`1/4`)** | `0%` (`0/4`) |

*(Both `wd_heuristic (+635)` and `starter (+740)` scored **`0/80` (`0%`)** across all 5 resource ratios under both FFA and Zero-Latency Focus-Fire.)*

---

## 3. Fitted Logistic Power-Law Tradeoff Model (`N = 200`, Weighted MAE = `6.2%`)

$$\text{logit}\big(P_{\text{coalition}}\big) = 3.523 - 1.653 \cdot \frac{\Delta\text{Elo}}{100} + 3.529 \cdot C + 0.723 \cdot \log_2(R)$$

- **Elo Value of Perfect Coordination (`C0 → C4`):** **`+213.5 Elo`** — lifts the 50% takedown boundary at standard `3:1` resources from **`+282 Elo` (FFA)** → **`+336` (Truce)** → **`+389` (Imperfect)** → **`+443` (`Lat=5`)** → **`+496 Elo` (Zero-Lat Focus-Fire)**.
- **Elo Value of Doubling Starting Resources (`2× R`):** **`+43.7 Elo`** — making perfect coordination equivalent to roughly a **`~29×`** starting resource multiplier over an uncoordinated FFA.
- **Unbreakable Intelligence Wall:** Beyond **$\Delta\text{Elo} \ge +600$**, neither `6×` starting resources nor zero-latency coordination can overcome fundamental tactical incompetence (`0/80` empirical wins).

---

## 4. Official Kaggle 3v1 Gauntlets & 4P Multi-Front Benchmarks

| Matchup (Kaggle Environments 4P) | Mode | Coalition / Winner Result | Boss Elim Tick(s) | Avg Volleys |
|---|---|---:|---:|---:|
| `3× subham_v14` vs. **`PT-Gen8` (~1,740)** (seeds `500–503`) | FFA / Truce (`lat=0`) | `0%` (`0/4`) / `0%` (`0/4`) | — | `0` |
| `3× subham_v14` vs. **`PT-Gen8` (~1,740)** (seeds `500–503`) | **Focus-Fire (`lat=0`)** | **`50%` (`2/4`)** | `t=240` (#3), `t=207` (#4) | `113.3` |
| `3× subham_v14` vs. **`PT-Gen8` (~1,740)** (seeds `500–503`) | Focus-Fire (`lat=5`) | `25%` (`1/4`) | `t=124` (#4) | `77.0` |
| `3× subham_v14` vs. **`Gen 7 PT-v5-Boss` (~1,700, 4P)** (seeds `500–503`) | Truce / Focus (`lat=0` / `5`) | `0%` / `25%` / `25%` | `t=100` / `t=115` (#4) | `52.8` / `46.3` |
| `3× subham_v14` vs. **`#2 Hober Malloc` (`1800.1`)** (seeds `500–502`) | **Focus-Fire (`lat=0`)** | **`33%` (`1/3`)** | **`t=482` (#4, 504 volleys!)** | `182.7` |
| **`2× Gen 7` vs. `2× PT-Gen8`** (6 balanced 4P FFA tables, seeds `600–605`) | 4P FFA | **Gen 7 wins `4–2`** (`58.3%` vs `41.7%` Top-2) | — | — |
