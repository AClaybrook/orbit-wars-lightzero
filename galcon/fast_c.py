"""Optional auto-compiled C99 accelerator for Galcon feature extraction & timeline projection.

Compiles a tiny self-contained C99 shared library via ``gcc -O3 -shared -fPIC`` into a hash-keyed
cache in the OS temp directory (shared across worker processes via atomic rename). If ``gcc`` is
unavailable on the host platform, gracefully sets ``HAS_FAST_C = False`` so callers fall back
transparently to the pure-NumPy implementation.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

_C_SOURCE = r"""
#include <math.h>
#include <stdint.h>

static inline double dmax(double a, double b) { return a > b ? a : b; }
static inline double dmin(double a, double b) { return a < b ? a : b; }

void extract_galcon_features_c(
    int n,
    int H,
    int feature_version,
    int player,
    double tick_dt,
    double step_dist,
    double prod_div,
    double step_prog,
    const double* xs,
    const double* ys,
    const double* rs,
    const double* ships,
    const double* prods,
    const int32_t* owners,
    int n_fleets,
    const int32_t* fl_tgt_idx,
    const int32_t* fl_owner,
    const double* fl_ships,
    const double* fl_x,
    const double* fl_y,
    float* my_arrs,
    float* en_arrs,
    float* po_all,
    float* ps_all,
    float* pf,
    float* tf,
    float* ef,
    uint8_t* legal_edges,
    uint8_t* owned_mask,
    float* safe_surplus
) {
    if (n <= 0) return;
    int p_dim = (feature_version >= 2) ? 22 : 18;
    int e_dim = (feature_version >= 2) ? 12 : 10;
    int opp = (player == 1) ? 2 : 1;

    // 0. Compute ship_ratio & prod_ratio inside C
    double my_tot_ships = 0.0, en_tot_ships = 0.0;
    double my_tot_prod = 0.0, en_tot_prod = 0.0;
    for (int i = 0; i < n; ++i) {
        if (owners[i] == player) {
            my_tot_ships += ships[i];
            my_tot_prod += prods[i];
        } else if (owners[i] == opp) {
            en_tot_ships += ships[i];
            en_tot_prod += prods[i];
        }
    }

    // 1. Zero arrival arrays and accumulate incoming fleets
    for (int i = 0; i < n * H; ++i) {
        my_arrs[i] = 0.0f;
        en_arrs[i] = 0.0f;
    }
    uint8_t has_inc[64];
    for (int i = 0; i < n; ++i) has_inc[i] = 0;

    for (int f = 0; f < n_fleets; ++f) {
        if (fl_owner[f] == player) my_tot_ships += fl_ships[f];
        else if (fl_owner[f] == opp) en_tot_ships += fl_ships[f];
        int idx_p = fl_tgt_idx[f];
        if (idx_p < 0 || idx_p >= n) continue;
        double dx = xs[idx_p] - fl_x[f];
        double dy = ys[idx_p] - fl_y[f];
        double dist = dmax(0.0, hypot(dx, dy) - rs[idx_p]);
        int eta = (int)ceil(dist / step_dist);
        if (eta < 1) eta = 1;
        int t_i = eta - 1;
        if (t_i >= 0 && t_i < H) {
            has_inc[idx_p] = 1;
            if (fl_owner[f] == player) {
                my_arrs[idx_p * H + t_i] += (float)fl_ships[f];
            } else {
                en_arrs[idx_p * H + t_i] += (float)fl_ships[f];
            }
        }
    }
    double ship_ratio = (my_tot_ships - en_tot_ships) / dmax(30.0, my_tot_ships + en_tot_ships);
    double prod_ratio = (my_tot_prod - en_tot_prod) / dmax(40.0, my_tot_prod + en_tot_prod);

    // 2. Project exact planet futures
    for (int i = 0; i < n; ++i) {
        int cur_owner = (owners[i] == player) ? 1 : ((owners[i] == 0) ? 0 : -1);
        double cur_ships = ships[i];
        double g_i = (prods[i] / prod_div) * tick_dt;
        int base = i * H;
        if (!has_inc[i]) {
            float fo = (float)cur_owner;
            for (int t = 0; t < H; ++t) {
                po_all[base + t] = fo;
                ps_all[base + t] = (float)(cur_ships + ((cur_owner != 0) ? g_i * (double)(t + 1) : 0.0));
            }
        } else {
            for (int t = 0; t < H; ++t) {
                if (cur_owner != 0) cur_ships += g_i;
                double ma = (double)my_arrs[base + t];
                double ea = (double)en_arrs[base + t];
                if (ma > 0.0 || ea > 0.0) {
                    double f_my = ma + ((cur_owner == 1) ? cur_ships : 0.0);
                    double f_en = ea + ((cur_owner == -1) ? cur_ships : 0.0);
                    double f_nu = (cur_owner == 0) ? cur_ships : 0.0;
                    double top_v, sec_v;
                    int top_o;
                    if (f_my >= f_en && f_my >= f_nu) {
                        top_v = f_my; top_o = 1;
                        sec_v = (f_en >= f_nu) ? f_en : f_nu;
                    } else if (f_en >= f_my && f_en >= f_nu) {
                        top_v = f_en; top_o = -1;
                        sec_v = (f_my >= f_nu) ? f_my : f_nu;
                    } else {
                        top_v = f_nu; top_o = 0;
                        sec_v = (f_my >= f_en) ? f_my : f_en;
                    }
                    if (top_v == sec_v) {
                        cur_ships = 0.0;
                    } else {
                        cur_owner = top_o;
                        cur_ships = top_v - sec_v;
                    }
                }
                po_all[base + t] = (float)cur_owner;
                ps_all[base + t] = (float)cur_ships;
            }
        }
    }

    // 3. Centering and pairwise ETA matrix
    double min_x = xs[0], max_x = xs[0], min_y = ys[0], max_y = ys[0];
    for (int i = 1; i < n; ++i) {
        if (xs[i] < min_x) min_x = xs[i];
        if (xs[i] > max_x) max_x = xs[i];
        if (ys[i] < min_y) min_y = ys[i];
        if (ys[i] > max_y) max_y = ys[i];
    }
    double cx = 0.5 * (min_x + max_x);
    double cy = 0.5 * (min_y + max_y);
    double half_span = dmax(50.0, 0.5 * dmax(max_x - min_x, max_y - min_y));

    float etas[64 * 64];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                etas[i * n + j] = 0.0f;
            } else {
                double d = dmax(0.0, hypot(xs[j] - xs[i], ys[j] - ys[i]) - rs[i] - rs[j]);
                double e = ceil(d / step_dist);
                if (e < 1.0) e = 1.0;
                etas[i * n + j] = (float)e;
            }
        }
    }

    // 4. Per-planet ownership, safe_surplus, threats, nearest ETAs
    int my_idx[64], en_idx[64];
    int n_my = 0, n_en = 0;
    for (int i = 0; i < n; ++i) {
        if (owners[i] == player) {
            my_idx[n_my++] = i;
            owned_mask[i] = (ships[i] >= 2.0) ? 1 : 0;
        } else {
            owned_mask[i] = 0;
            if (owners[i] != 0) en_idx[n_en++] = i;
        }
    }

    float min_en_eta[64], min_my_eta[64];
    for (int i = 0; i < n; ++i) {
        if (n_en > 0) {
            float m = etas[i * n + en_idx[0]];
            for (int k = 1; k < n_en; ++k) {
                float v = etas[i * n + en_idx[k]];
                if (v < m) m = v;
            }
            min_en_eta[i] = m;
        } else {
            min_en_eta[i] = 40.0f;
        }
        if (n_my == 0) {
            min_my_eta[i] = 40.0f;
        } else if (n_my == 1) {
            min_my_eta[i] = (i == my_idx[0]) ? 40.0f : etas[i * n + my_idx[0]];
        } else {
            float m = 1e9f;
            for (int k = 0; k < n_my; ++k) {
                int j = my_idx[k];
                if (j == i) continue;
                float v = etas[i * n + j];
                if (v < m) m = v;
            }
            min_my_eta[i] = m;
        }
    }

    for (int i = 0; i < n; ++i) {
        int is_mine = (owners[i] == player) ? 1 : 0;
        int is_en = (owners[i] != 0 && owners[i] != player) ? 1 : 0;
        int is_neu = (owners[i] == 0) ? 1 : 0;
        int base = i * H;

        int first_lost = -1;
        float min_hold = ps_all[base];
        float sum_my = 0.0f, sum_en = 0.0f;
        for (int t = 0; t < H; ++t) {
            if (first_lost < 0 && po_all[base + t] <= 0.0f) first_lost = t;
            if (ps_all[base + t] < min_hold) min_hold = ps_all[base + t];
            sum_my += my_arrs[base + t];
            sum_en += en_arrs[base + t];
        }
        int any_lost = (first_lost >= 0) ? 1 : 0;
        float ss = 0.0f;
        if (is_mine && !any_lost) {
            float sh_f = (float)ships[i];
            float mh = (sh_f < min_hold) ? sh_f : min_hold;
            ss = (mh > 0.0f) ? mh : 0.0f;
        }
        safe_surplus[i] = ss;

        float fl_val = (min_my_eta[i] - min_en_eta[i]) / (float)dmax(10.0, (double)(min_my_eta[i] + min_en_eta[i]));
        float* row = pf + i * p_dim;
        row[0] = (float)((xs[i] - cx) / half_span);
        row[1] = (float)((ys[i] - cy) / half_span);
        row[2] = (float)is_mine;
        row[3] = (float)is_en;
        row[4] = (float)is_neu;
        row[5] = (float)(log1p(dmax(0.0, ships[i])) / 5.5);
        row[6] = (float)(prods[i] / 100.0);
        row[7] = (float)(rs[i] / 35.0);
        row[8] = (float)(log1p((double)ss) / 5.5);
        row[9] = (float)(log1p((double)sum_my) / 5.5);
        row[10] = (float)(log1p((double)sum_en) / 5.5);
        row[11] = po_all[base + H - 1] * (float)(log1p(dmax(0.0, (double)ps_all[base + H - 1])) / 5.5);
        row[12] = (float)dmin(1.0, (double)min_en_eta[i] / 35.0);
        row[13] = (float)dmin(1.0, (double)min_my_eta[i] / 35.0);
        row[14] = fl_val;
        row[15] = (float)ship_ratio;
        row[16] = (float)prod_ratio;
        row[17] = (float)step_prog;

        if (feature_version >= 2) {
            row[18] = (is_mine && any_lost) ? 1.0f : 0.0f;
            row[19] = (is_mine && any_lost) ? (float)(1.0 / (1.0 + 0.25 * (double)(first_lost + 1))) : 0.0f;
            row[20] = (is_mine && any_lost) ? (float)(log1p(dmax(0.0, (double)ps_all[base + first_lost] + 2.0)) / 5.5) : 0.0f;
            float rt_out = 0.0f;
            if (is_mine && n_en > 0) {
                int best_en = en_idx[0];
                float best_eta = etas[best_en * n + i];
                for (int k = 1; k < n_en; ++k) {
                    int ej = en_idx[k];
                    float e_val = etas[ej * n + i];
                    if (e_val < best_eta) {
                        best_eta = e_val;
                        best_en = ej;
                    }
                }
                double growth_dur = (prods[i] / prod_div) * tick_dt * (double)best_eta;
                double rt = (ships[best_en] * 0.85 - ships[i] - growth_dur) / 60.0;
                if (rt < 0.0) rt = 0.0;
                if (rt > 2.0) rt = 2.0;
                rt_out = (float)rt;
            }
            row[21] = rt_out;
        }

        // 5. Timeline features (N, H, 8)
        for (int t = 0; t < H; ++t) {
            float* trow = tf + (base + t) * 8;
            float po_t = po_all[base + t];
            float ps_t = ps_all[base + t];
            trow[0] = (float)(log1p((double)my_arrs[base + t]) / 5.0);
            trow[1] = (float)(log1p((double)en_arrs[base + t]) / 5.0);
            trow[2] = (po_t > 0.0f) ? 1.0f : 0.0f;
            trow[3] = (po_t < 0.0f) ? 1.0f : 0.0f;
            trow[4] = (po_t == 0.0f) ? 1.0f : 0.0f;
            trow[5] = (float)(log1p(dmax(0.0, (double)ps_t)) / 5.5);
            trow[6] = (t > 0 && po_t != po_all[base + t - 1]) ? 1.0f : 0.0f;
            trow[7] = (float)t / (float)H;
        }
    }

    // 6. Edge features (N, N, e_dim) and legal_edges
    for (int i = 0; i < n; ++i) {
        float src_s = (float)ships[i];
        float src_ss = safe_surplus[i];
        float pf_i12 = pf[i * p_dim + 12];
        for (int j = 0; j < n; ++j) {
            int ij = i * n + j;
            float* erow = ef + ij * e_dim;
            if (i == j) {
                legal_edges[ij] = 0;
                for (int c = 0; c < e_dim; ++c) erow[c] = 0.0f;
                continue;
            }
            legal_edges[ij] = 1;
            float eta = etas[ij];
            int t_idx = (int)eta - 1;
            if (t_idx < 0) t_idx = 0;
            if (t_idx >= H) t_idx = H - 1;
            float owner_eta = po_all[j * H + t_idx];
            float ships_eta = ps_all[j * H + t_idx];
            float prod_rate_j = (float)(prods[j] / prod_div);
            float extra = 0.0f;
            if (owner_eta < 0.0f && eta > (float)H) {
                extra = prod_rate_j * (float)tick_dt * (eta - (float)H);
            }
            float need = ships_eta + extra + 1.5f;
            float denom_need = (need > 1.0f) ? need : 1.0f;
            float denom_src = (src_s > 1.0f) ? src_s : 1.0f;
            float pr_clamp = (prod_rate_j > 0.3f) ? prod_rate_j : 0.3f;
            float roi_den = need + 0.45f * eta * pr_clamp;
            if (roi_den < 3.0f) roi_den = 3.0f;
            float roi = (prod_rate_j * 20.0f) / roi_den;

            int j_is_neu = (owners[j] == 0) ? 1 : 0;
            int j_is_mine = (owners[j] == player) ? 1 : 0;
            int j_is_en = (owners[j] != 0 && owners[j] != player) ? 1 : 0;

            float ratio_s = src_s / denom_need;
            erow[0] = (eta / 35.0f < 1.0f) ? (eta / 35.0f) : 1.0f;
            erow[1] = 1.0f / (1.0f + 0.12f * eta);
            erow[2] = (ratio_s < 2.0f) ? ratio_s : 2.0f;
            erow[3] = (owner_eta <= 0.0f && src_s > need) ? 1.0f : 0.0f;
            erow[4] = (owner_eta <= 0.0f && src_ss > need) ? 1.0f : 0.0f;
            erow[5] = (j_is_neu && owner_eta < 0.0f) ? 1.0f : 0.0f;
            erow[6] = (j_is_mine && owner_eta <= 0.0f) ? 1.0f : 0.0f;
            erow[7] = (roi < 2.0f) ? roi : 2.0f;
            erow[8] = owner_eta;
            erow[9] = pf_i12 - pf[j * p_dim + 12];
            if (feature_version >= 2) {
                float r2 = need / denom_src;
                erow[10] = (r2 < 1.0f) ? r2 : 1.0f;
                erow[11] = (j_is_en && prods[j] >= 70.0 && src_s > need) ? 1.0f : 0.0f;
            }
        }
    }
}
"""

_C_VP = ctypes.c_void_p
_MAX_FL = 512
_SCRATCH_XS = (ctypes.c_double * 64)()
_SCRATCH_YS = (ctypes.c_double * 64)()
_SCRATCH_RS = (ctypes.c_double * 64)()
_SCRATCH_SHIPS = (ctypes.c_double * 64)()
_SCRATCH_PRODS = (ctypes.c_double * 64)()
_SCRATCH_OWNERS = (ctypes.c_int32 * 64)()
_SCRATCH_FL_TGT = (ctypes.c_int32 * _MAX_FL)()
_SCRATCH_FL_OWN = (ctypes.c_int32 * _MAX_FL)()
_SCRATCH_FL_SH = (ctypes.c_double * _MAX_FL)()
_SCRATCH_FL_X = (ctypes.c_double * _MAX_FL)()
_SCRATCH_FL_Y = (ctypes.c_double * _MAX_FL)()
_PTR_XS = ctypes.addressof(_SCRATCH_XS)
_PTR_YS = ctypes.addressof(_SCRATCH_YS)
_PTR_RS = ctypes.addressof(_SCRATCH_RS)
_PTR_SHIPS = ctypes.addressof(_SCRATCH_SHIPS)
_PTR_PRODS = ctypes.addressof(_SCRATCH_PRODS)
_PTR_OWNERS = ctypes.addressof(_SCRATCH_OWNERS)
_PTR_FL_TGT = ctypes.addressof(_SCRATCH_FL_TGT)
_PTR_FL_OWN = ctypes.addressof(_SCRATCH_FL_OWN)
_PTR_FL_SH = ctypes.addressof(_SCRATCH_FL_SH)
_PTR_FL_X = ctypes.addressof(_SCRATCH_FL_X)
_PTR_FL_Y = ctypes.addressof(_SCRATCH_FL_Y)

_LIB: Optional[ctypes.CDLL] = None
_EXTRACT_FN = None
HAS_FAST_C: bool = False


def _load_or_build_lib() -> Optional[ctypes.CDLL]:
    if sys.platform == "win32":
        return None
    gcc = shutil.which("gcc")
    if not gcc:
        return None
    h = hashlib.sha256(_C_SOURCE.encode("utf-8")).hexdigest()[:16]
    cache_dir = Path(tempfile.gettempdir()) / "galcon_fast_c_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    so_path = cache_dir / f"galcon_fast_{h}.so"
    if not so_path.is_file():
        pid = os.getpid()
        c_tmp = cache_dir / f"galcon_fast_{h}_{pid}.c"
        so_tmp = cache_dir / f"galcon_fast_{h}_{pid}.so"
        try:
            c_tmp.write_text(_C_SOURCE, encoding="utf-8")
            subprocess.run(
                [gcc, "-O3", "-march=native", "-shared", "-fPIC", str(c_tmp), "-lm", "-o", str(so_tmp)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            os.replace(so_tmp, so_path)
        except Exception:
            try:
                so_tmp.unlink(missing_ok=True)
            except Exception:
                pass
            return None
        finally:
            try:
                c_tmp.unlink(missing_ok=True)
            except Exception:
                pass
    try:
        lib = ctypes.CDLL(str(so_path))
        lib.extract_galcon_features_c.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
            _C_VP, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP,
            ctypes.c_int, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP,
            _C_VP, _C_VP, _C_VP, _C_VP,
            _C_VP, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP,
        ]
        lib.extract_galcon_features_c.restype = None
        return lib
    except Exception:
        return None


_LIB = _load_or_build_lib()
_EXTRACT_FN = _LIB.extract_galcon_features_c if _LIB is not None else None
HAS_FAST_C = _EXTRACT_FN is not None


def extract_features_fast_c(
    sim,
    planets,
    player: int,
    horizon: int,
    feature_version: int,
    prod_div: float,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Run the fused C99 feature + future-timeline kernel. Returns ``None`` if C lib is unavailable or N > 64."""
    if _EXTRACT_FN is None:
        return None
    n = len(planets)
    if n == 0 or n > 64:
        return None

    xs = _SCRATCH_XS
    ys = _SCRATCH_YS
    rs = _SCRATCH_RS
    ships = _SCRATCH_SHIPS
    prods = _SCRATCH_PRODS
    owners = _SCRATCH_OWNERS
    for i, p in enumerate(planets):
        xs[i] = p.x
        ys[i] = p.y
        rs[i] = p.radius
        ships[i] = p.ships
        prods[i] = p.production
        owners[i] = p.owner

    fleets_dict = sim.fleets
    nf = len(fleets_dict)
    if nf > 0:
        id_to_idx = {p.id: i for i, p in enumerate(planets)}
        if nf <= _MAX_FL:
            fl_tgt = _SCRATCH_FL_TGT
            fl_own = _SCRATCH_FL_OWN
            fl_sh = _SCRATCH_FL_SH
            fl_x = _SCRATCH_FL_X
            fl_y = _SCRATCH_FL_Y
            for fi, fl in enumerate(fleets_dict.values()):
                fl_tgt[fi] = id_to_idx.get(fl.target, -1)
                fl_own[fi] = fl.owner
                fl_sh[fi] = fl.ships
                fl_x[fi] = fl.x
                fl_y[fi] = fl.y
            p_ft, p_fo, p_fs, p_fx, p_fy = _PTR_FL_TGT, _PTR_FL_OWN, _PTR_FL_SH, _PTR_FL_X, _PTR_FL_Y
        else:
            fl_tgt_a = np.empty((nf,), dtype=np.int32)
            fl_own_a = np.empty((nf,), dtype=np.int32)
            fl_sh_a = np.empty((nf,), dtype=np.float64)
            fl_x_a = np.empty((nf,), dtype=np.float64)
            fl_y_a = np.empty((nf,), dtype=np.float64)
            for fi, fl in enumerate(fleets_dict.values()):
                fl_tgt_a[fi] = id_to_idx.get(fl.target, -1)
                fl_own_a[fi] = fl.owner
                fl_sh_a[fi] = fl.ships
                fl_x_a[fi] = fl.x
                fl_y_a[fi] = fl.y
            p_ft, p_fo = fl_tgt_a.ctypes.data, fl_own_a.ctypes.data
            p_fs, p_fx, p_fy = fl_sh_a.ctypes.data, fl_x_a.ctypes.data, fl_y_a.ctypes.data
    else:
        p_ft, p_fo, p_fs, p_fx, p_fy = _PTR_FL_TGT, _PTR_FL_OWN, _PTR_FL_SH, _PTR_FL_X, _PTR_FL_Y

    step_prog = min(1.0, float(sim.tick) / max(1.0, float(sim.max_ticks)))
    step_dist = max(1e-6, sim.fleet_speed * sim.tick_dt)

    p_dim = 22 if feature_version >= 2 else 18
    e_dim = 12 if feature_version >= 2 else 10
    my_arrs = np.empty((n, horizon), dtype=np.float32)
    en_arrs = np.empty((n, horizon), dtype=np.float32)
    po_all = np.empty((n, horizon), dtype=np.float32)
    ps_all = np.empty((n, horizon), dtype=np.float32)
    pf = np.empty((n, p_dim), dtype=np.float32)
    tf = np.empty((n, horizon, 8), dtype=np.float32)
    ef = np.empty((n, n, e_dim), dtype=np.float32)
    legal_u8 = np.empty((n, n), dtype=np.uint8)
    owned_u8 = np.empty((n,), dtype=np.uint8)
    safe_surplus = np.empty((n,), dtype=np.float32)

    _EXTRACT_FN(
        n, horizon, feature_version, player,
        float(sim.tick_dt), float(step_dist), float(prod_div), float(step_prog),
        _PTR_XS, _PTR_YS, _PTR_RS, _PTR_SHIPS, _PTR_PRODS, _PTR_OWNERS,
        nf, p_ft, p_fo, p_fs, p_fx, p_fy,
        my_arrs.ctypes.data,
        en_arrs.ctypes.data,
        po_all.ctypes.data,
        ps_all.ctypes.data,
        pf.ctypes.data,
        tf.ctypes.data,
        ef.ctypes.data,
        legal_u8.ctypes.data,
        owned_u8.ctypes.data,
        safe_surplus.ctypes.data,
    )
    return (
        my_arrs,
        en_arrs,
        po_all,
        ps_all,
        pf,
        tf,
        ef,
        legal_u8.view(np.bool_),
        owned_u8.view(np.bool_),
        safe_surplus,
    )
