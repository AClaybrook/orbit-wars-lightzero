"""Fit a logistic power-dynamics tradeoff model across all 3v1 wargaming sweeps.

Models:
  logit(P(Coalition Takedown)) = b0 + b_elo * (DeltaElo / 100) + b_coord * C + b_res * log2(R)

Outputs exact exchange rates:
  - Elo-equivalent value of Perfect Coordination (C0 -> C4)
  - Elo-equivalent value of Doubling Starting Resources (2x R)
  - Maximum overcomeable Intelligence Gap (DeltaElo at 50% phase boundary) per coordination tier
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def fit_logistic(X: np.ndarray, y: np.ndarray, weights: np.ndarray, l2: float = 0.05) -> np.ndarray:
    """IRLS / Newton-Raphson solver for weighted L2-regularized logistic regression."""
    n, d = X.shape
    w = np.zeros(d, dtype=np.float64)
    reg = np.eye(d, dtype=np.float64) * l2
    reg[0, 0] = 0.0  # don't regularize intercept

    for _ in range(40):
        z = np.clip(X @ w, -20.0, 20.0)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = X.T @ (weights * (p - y)) + reg @ w
        S = weights * np.maximum(p * (1.0 - p), 1e-6)
        H = (X.T * S) @ X + reg
        step = np.linalg.solve(H, grad)
        w -= step
        if float(np.linalg.norm(step)) < 1e-7:
            break
    return w


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=str, nargs="+", required=True)
    parser.add_argument("--out-json", type=str, required=True)
    args = parser.parse_args()

    coord_cells: List[Dict[str, Any]] = []
    res_cells: List[Dict[str, Any]] = []
    rows_X = []
    rows_y = []
    rows_n = []

    for path_str in args.inputs:
        p = Path(path_str)
        if not p.is_file():
            continue
        data = json.loads(p.read_text())
        for c in data.get("cells", []):
            if "coord_score" in c:
                coord_cells.append(c)
                c_score = float(c["coord_score"])
                r_ratio = 3.0
            else:
                res_cells.append(c)
                c_score = 1.0 if c["mode"] == "focus_fire" else 0.0
                r_ratio = float(c["resource_ratio"])

            d_elo = float(c["delta_elo"]) / 100.0
            n = float(c["games"])
            k = float(c["coalition_wins"])
            rows_X.append([1.0, d_elo, c_score, math.log2(max(0.1, r_ratio))])
            rows_y.append(k / n)
            rows_n.append(n)

    X = np.array(rows_X, dtype=np.float64)
    y = np.array(rows_y, dtype=np.float64)
    n_w = np.array(rows_n, dtype=np.float64)
    w = fit_logistic(X, y, n_w, l2=0.15)
    b0, b_elo, b_coord, b_res = [float(v) for v in w]

    # Compute exchange rates (note b_elo < 0 because larger Boss Elo advantage lowers Coalition win prob)
    elo_per_full_coord = round(-100.0 * b_coord / b_elo, 1) if abs(b_elo) > 1e-6 else 0.0
    elo_per_2x_res = round(-100.0 * b_res / b_elo, 1) if abs(b_elo) > 1e-6 else 0.0

    # Compute 50% phase-transition DeltaElo boundary at standard 3:1 resources (log2(3) = 1.585)
    # 0 = b0 + b_elo*(dElo/100) + b_coord*C + b_res*log2(3)
    boundary_by_coord = {}
    for label, c_val in (
        ("C0_ffa", 0.0),
        ("C1_truce", 0.25),
        ("C2_imperfect", 0.50),
        ("C3_lat5", 0.75),
        ("C4_perfect", 1.0),
    ):
        d_elo_50 = -100.0 * (b0 + b_coord * c_val + b_res * math.log2(3.0)) / b_elo
        boundary_by_coord[label] = round(float(d_elo_50), 1)

    # Model goodness-of-fit (weighted MAE on win probability)
    preds = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -20.0, 20.0)))
    mae_pct = round(100.0 * float(np.average(np.abs(preds - y), weights=n_w)), 1)

    summary = {
        "total_cells": len(rows_X),
        "total_matches": int(np.sum(n_w)),
        "coefficients": {
            "intercept": round(b0, 4),
            "b_elo_per_100": round(b_elo, 4),
            "b_coord_0_to_1": round(b_coord, 4),
            "b_log2_res_ratio": round(b_res, 4),
        },
        "exchange_rates": {
            "perfect_coord_elo_equiv": elo_per_full_coord,
            "doubling_resources_elo_equiv": elo_per_2x_res,
            "perfect_coord_resource_mult_equiv": round(2.0 ** (b_coord / max(1e-6, b_res)), 2) if b_res > 0 else None,
        },
        "phase_boundary_50pct_delta_elo_at_3to1_res": boundary_by_coord,
        "weighted_mae_pct": mae_pct,
        "coord_cells": coord_cells,
        "resource_cells": res_cells,
    }
    Path(args.out_json).write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if not k.endswith("_cells")}, indent=2))


if __name__ == "__main__":
    main()
