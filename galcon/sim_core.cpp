// Auto-compiled C++17 Galcon physical-ship 20 Hz physics engine & standalone POD SimState.
// Compile:
//   g++ -O3 -march=native -ffp-contract=off -shared -fPIC -std=c++17 sim_core.cpp -lm -o sim_core.so
//
// Important:
//   - Compiled with -ffp-contract=off (no FMA contraction) and uses std::sqrt(dx*dx + dy*dy)
//     to guarantee bit-exact IEEE-754 float64 parity with GalconSim._step_physics_python().
//   - Uses static thread_local scratch buffers for <=64 planets / <=512 fleets / <=4096 units
//     and automatically falls back to std::vector scratch buffers for arbitrarily large worlds.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

constexpr double PRODUCTION_DIVISOR = 50.0;
constexpr double DEFAULT_SHIP_RADIUS = 6.0;
constexpr double TIE_EPS = 1e-9;
constexpr double ARRIVAL_EPS = 1e-9;
constexpr double GOLDEN_ANGLE = 2.399963229728653; // pi * (3 - sqrt(5))
constexpr double SPAWN_SPIRAL_STEP = 0.05;
constexpr double EPS_DIST2 = 1e-12;
constexpr double ANTIPODE_DOT_THRESHOLD = -0.995;

constexpr int32_t ITEMS_MAX = 4096;
constexpr int32_t ITEMS_RESERVED = 16;
constexpr int32_t BASE_ITEM_USERS = 3;

// Max distinct owners fighting on a single planet in one substep.
constexpr int32_t MAX_ARR_OWNERS = 16;

// Limits for fast POD SimState & thread-local scratch (~101 KB < 128 KB glibc mmap threshold).
constexpr int32_t SIM_MAX_PLANETS = 64;
constexpr int32_t SIM_MAX_FLEETS = 512;
constexpr int32_t SIM_MAX_UNITS = 4096;

struct ArrivalBucket {
    int32_t owners[MAX_ARR_OWNERS];
    double ships[MAX_ARR_OWNERS];
    int32_t count;
};

static inline double dist2d(double dx, double dy) {
    return std::sqrt(dx * dx + dy * dy);
}

// Exact Galcon 2 / Galcon 3 physical_ship_count formula matching galcon/sim.py.
static inline int32_t physical_ship_count_cpp(
    double ships,
    double total_planet_ships,
    int32_t item_count,
    int32_t items_max = ITEMS_MAX,
    int32_t reserved = ITEMS_RESERVED
) {
    if (ships < 1.0 - TIE_EPS) {
        return 0;
    }
    int32_t avail = items_max - item_count - reserved;
    if (avail <= 0) {
        avail = 0;
    }
    avail /= 2;

    double parts_d = std::floor(ships);
    const double sqrt_cap = std::sqrt(17.0 * ships);
    if (sqrt_cap < parts_d) {
        parts_d = sqrt_cap;
    }
    parts_d = std::floor(parts_d);

    if (total_planet_ships > 0.0) {
        const double avail_cap = ((double)avail * ships) / total_planet_ships;
        if (avail_cap < parts_d) {
            parts_d = avail_cap;
        }
        parts_d = std::floor(parts_d);
    }

    if (parts_d > 2000000000.0) {
        parts_d = 2000000000.0;
    }
    const int32_t parts = (int32_t)parts_d;
    return parts < 1 ? 1 : parts;
}

template <bool CheckAlive, bool PairCap>
static inline void separate_fleet_gs(
    int32_t start,
    int32_t end,
    double* __restrict__ ux,
    double* __restrict__ uy,
    const uint8_t* __restrict__ u_alive,
    double min_sep,
    double min_sep2,
    double half_k,
    double max_sep_push
) {
    for (int32_t a = start; a < end; ++a) {
        if constexpr (CheckAlive) {
            if (!u_alive[a]) continue;
        }
        double ax = ux[a];
        double ay = uy[a];
        for (int32_t b = a + 1; b < end; ++b) {
            if constexpr (CheckAlive) {
                if (!u_alive[b]) continue;
            }
            const double dx = ux[b] - ax;
            const double dy = uy[b] - ay;
            const double d2 = dx * dx + dy * dy;
            if (d2 < min_sep2) {
                double nx, ny, dist;
                if (d2 > EPS_DIST2) {
                    dist = std::sqrt(d2);
                    nx = dx / dist;
                    ny = dy / dist;
                } else {
                    dist = 0.0;
                    const double ang = GOLDEN_ANGLE * (double)((a - start) * 31 + (b - start) + 1);
                    nx = std::cos(ang);
                    ny = std::sin(ang);
                }
                double push = half_k * (min_sep - dist);
                if constexpr (PairCap) {
                    if (push > max_sep_push) push = max_sep_push;
                }
                const double px_d = nx * push;
                const double py_d = ny * push;
                ax -= px_d;
                ay -= py_d;
                ux[b] += px_d;
                uy[b] += py_d;
            }
        }
        ux[a] = ax;
        uy[a] = ay;
    }
}

// Core physics step operating on flat SoA arrays.
// Shared identically by both step_physics_cpp() and SimState::step_one_tick().
//
// sep_mode:
//   0 = Gauss-Seidel pairwise pushout (optional per-pair cap if max_sep_push > 0)
//   1 = Gauss-Seidel pairwise pushout + per-ship substep total displacement cap (default)
//   2 = Jacobi pairwise pushout (optional per-ship cap)
//
// obstacle_escape_eps:
//   0.0 = pure radial pushout
//   >0  = tiny deterministic tangential nudge only when a unit is near-antipodal
//         to target on an obstacle planet (dot(radial, to_target) < -0.995)
int32_t run_physics_substeps(
    int32_t n_planets,
    int32_t* __restrict__ p_owner,
    double* __restrict__ p_ships,
    const double* __restrict__ p_x,
    const double* __restrict__ p_y,
    const double* __restrict__ p_prod,
    const double* __restrict__ p_radius,
    int32_t n_fleets,
    const int32_t* __restrict__ fl_owner,
    const int32_t* __restrict__ fl_target_idx,
    const double* __restrict__ fl_spu,
    int32_t* __restrict__ fl_unit_start,
    int32_t* __restrict__ fl_unit_count,
    double* __restrict__ fl_out_ships,
    double* __restrict__ fl_out_cx,
    double* __restrict__ fl_out_cy,
    double* __restrict__ fl_out_radius,
    int32_t total_units,
    double* __restrict__ ux,
    double* __restrict__ uy,
    double tick_dt,
    double fleet_speed,
    int32_t substeps,
    double ship_radius,
    double sep_stiffness,
    double max_sep_push,
    int32_t sep_mode,
    double prod_divisor,
    double obstacle_escape_eps,
    // Scratch buffers
    uint8_t* __restrict__ u_alive,
    int32_t* __restrict__ fl_alive_cnt,
    double* __restrict__ tmp_x,
    double* __restrict__ tmp_y,
    ArrivalBucket* __restrict__ arr_buckets,
    int32_t* __restrict__ dirty_planets,
    int32_t* __restrict__ near_planets,
    double* __restrict__ p_min_r,
    double* __restrict__ p_min_r2,
    double* __restrict__ p_land_r,
    double* __restrict__ p_land_r2_guard,
    double* __restrict__ p_prod_sub
) {
    if (n_planets <= 0) return 0;
    if (substeps < 1) substeps = 1;

    const double sub_dt = tick_dt / (double)substeps;

    // Fast path when no fleets exist: only planet production runs across substeps.
    if (n_fleets <= 0) {
        for (int32_t p = 0; p < n_planets; ++p) {
            p_prod_sub[p] = (p_prod[p] / prod_divisor) * sub_dt;
        }
        for (int32_t s = 0; s < substeps; ++s) {
            for (int32_t p = 0; p < n_planets; ++p) {
                if (p_owner[p] > 0) {
                    p_ships[p] += p_prod_sub[p];
                }
            }
        }
        return 0;
    }

    const double sub_step_dist = fleet_speed * sub_dt;
    const double min_sep = 2.0 * ship_radius;
    const double min_sep2 = min_sep * min_sep;
    const double half_k = 0.5 * sep_stiffness;

    double max_mr = 0.0;
    for (int32_t p = 0; p < n_planets; ++p) {
        const double mr = p_radius[p] + ship_radius;
        if (mr > max_mr) max_mr = mr;
        p_min_r[p] = mr;
        p_min_r2[p] = mr * mr;
        const double lr = mr + ARRIVAL_EPS;
        p_land_r[p] = lr;
        const double lrg = lr + 1e-6;
        p_land_r2_guard[p] = lrg * lrg;
        p_prod_sub[p] = (p_prod[p] / prod_divisor) * sub_dt;
        arr_buckets[p].count = 0;
    }

    if (total_units > 0) {
        std::memset(u_alive, 1, (size_t)total_units);
    }
    for (int32_t f = 0; f < n_fleets; ++f) {
        fl_alive_cnt[f] = fl_unit_count[f];
    }

    for (int32_t s = 0; s < substeps; ++s) {
        // 1. Production on owned planets
        for (int32_t p = 0; p < n_planets; ++p) {
            if (p_owner[p] > 0) {
                p_ships[p] += p_prod_sub[p];
            }
        }

        // 2. Same-fleet ship-ship circle separation
        if (sep_stiffness > 0.0) {
            for (int32_t f = 0; f < n_fleets; ++f) {
                const int32_t alive_c = fl_alive_cnt[f];
                if (alive_c <= 1) continue;
                const int32_t start = fl_unit_start[f];
                const int32_t cnt = fl_unit_count[f];
                const int32_t end = start + cnt;
                const bool all_alive = (alive_c == cnt);

                if (sep_mode == 2) {
                    for (int32_t u = start; u < end; ++u) {
                        tmp_x[u] = 0.0;
                        tmp_y[u] = 0.0;
                    }
                    for (int32_t a = start; a < end; ++a) {
                        if (!all_alive && !u_alive[a]) continue;
                        const double ax = ux[a];
                        const double ay = uy[a];
                        for (int32_t b = a + 1; b < end; ++b) {
                            if (!all_alive && !u_alive[b]) continue;
                            const double dx = ux[b] - ax;
                            const double dy = uy[b] - ay;
                            const double d2 = dx * dx + dy * dy;
                            if (d2 < min_sep2) {
                                double nx, ny, dist;
                                if (d2 > EPS_DIST2) {
                                    dist = std::sqrt(d2);
                                    nx = dx / dist;
                                    ny = dy / dist;
                                } else {
                                    dist = 0.0;
                                    const double ang = GOLDEN_ANGLE * (double)((a - start) * 31 + (b - start) + 1);
                                    nx = std::cos(ang);
                                    ny = std::sin(ang);
                                }
                                const double push = half_k * (min_sep - dist);
                                tmp_x[a] -= nx * push;
                                tmp_y[a] -= ny * push;
                                tmp_x[b] += nx * push;
                                tmp_y[b] += ny * push;
                            }
                        }
                    }
                    const double cap2 = (max_sep_push > 0.0) ? (max_sep_push * max_sep_push) : 0.0;
                    for (int32_t u = start; u < end; ++u) {
                        if (!all_alive && !u_alive[u]) continue;
                        double dx = tmp_x[u];
                        double dy = tmp_y[u];
                        if (cap2 > 0.0) {
                            const double d2 = dx * dx + dy * dy;
                            if (d2 > cap2) {
                                const double sc = max_sep_push / std::sqrt(d2);
                                dx *= sc;
                                dy *= sc;
                            }
                        }
                        ux[u] += dx;
                        uy[u] += dy;
                    }
                } else {
                    const bool ship_cap = (sep_mode == 1 && max_sep_push > 0.0);
                    const bool pair_cap = (sep_mode == 0 && max_sep_push > 0.0);
                    if (ship_cap) {
                        std::memcpy(tmp_x + start, ux + start, (size_t)cnt * sizeof(double));
                        std::memcpy(tmp_y + start, uy + start, (size_t)cnt * sizeof(double));
                    }

                    if (all_alive) {
                        if (pair_cap) {
                            separate_fleet_gs<false, true>(start, end, ux, uy, u_alive, min_sep, min_sep2, half_k, max_sep_push);
                        } else {
                            separate_fleet_gs<false, false>(start, end, ux, uy, u_alive, min_sep, min_sep2, half_k, max_sep_push);
                        }
                    } else {
                        if (pair_cap) {
                            separate_fleet_gs<true, true>(start, end, ux, uy, u_alive, min_sep, min_sep2, half_k, max_sep_push);
                        } else {
                            separate_fleet_gs<true, false>(start, end, ux, uy, u_alive, min_sep, min_sep2, half_k, max_sep_push);
                        }
                    }

                    if (ship_cap) {
                        const double cap2 = max_sep_push * max_sep_push;
                        for (int32_t u = start; u < end; ++u) {
                            if (!all_alive && !u_alive[u]) continue;
                            const double dx = ux[u] - tmp_x[u];
                            const double dy = uy[u] - tmp_y[u];
                            const double d2 = dx * dx + dy * dy;
                            if (d2 > cap2) {
                                const double sc = max_sep_push / std::sqrt(d2);
                                ux[u] = tmp_x[u] + dx * sc;
                                uy[u] = tmp_y[u] + dy * sc;
                            }
                        }
                    }
                }
            }
        }

        // 3. Move toward target + 4. Planet collider pushout + 5a. Touch-to-land
        int32_t n_dirty = 0;

        for (int32_t f = 0; f < n_fleets; ++f) {
            const int32_t alive_c = fl_alive_cnt[f];
            if (alive_c <= 0) continue;
            const int32_t tgt = fl_target_idx[f];
            if (tgt < 0 || tgt >= n_planets) {
                fl_alive_cnt[f] = 0;
                continue;
            }
            const double tx = p_x[tgt];
            const double ty = p_y[tgt];
            const int32_t start = fl_unit_start[f];
            const int32_t cnt = fl_unit_count[f];
            const int32_t end = start + cnt;
            const bool all_alive = (alive_c == cnt);

            // Step 3: Move toward target + track exact fleet AABB
            double min_x = 1e300, max_x = -1e300, min_y = 1e300, max_y = -1e300;
            for (int32_t u = start; u < end; ++u) {
                if (!all_alive && !u_alive[u]) continue;
                double x = ux[u];
                double y = uy[u];
                const double dx = tx - x;
                const double dy = ty - y;
                const double dist = dist2d(dx, dy);
                if (dist <= sub_step_dist) {
                    x = tx;
                    y = ty;
                } else if (dist > 0.0) {
                    x += (dx / dist) * sub_step_dist;
                    y += (dy / dist) * sub_step_dist;
                }
                ux[u] = x;
                uy[u] = y;
                if (x < min_x) min_x = x;
                if (x > max_x) max_x = x;
                if (y < min_y) min_y = y;
                if (y > max_y) max_y = y;
            }

            // Exact zero-margin broadphase: check if any planet p != tgt overlaps the fleet AABB
            bool any_exact_hit = false;
            for (int32_t p = 0; p < n_planets; ++p) {
                if (p == tgt) continue;
                const double mr = p_min_r[p];
                if (p_x[p] + mr >= min_x && p_x[p] - mr <= max_x &&
                    p_y[p] + mr >= min_y && p_y[p] - mr <= max_y) {
                    any_exact_hit = true;
                    break;
                }
            }

            // Step 4: Planet collider radial pushout (padded candidate list preserves exactness)
            if (any_exact_hit) {
                int32_t n_near = 0;
                const double pad_min_x = min_x - max_mr;
                const double pad_max_x = max_x + max_mr;
                const double pad_min_y = min_y - max_mr;
                const double pad_max_y = max_y + max_mr;
                for (int32_t p = 0; p < n_planets; ++p) {
                    if (p == tgt) continue;
                    const double mr = p_min_r[p];
                    if (p_x[p] + mr >= pad_min_x && p_x[p] - mr <= pad_max_x &&
                        p_y[p] + mr >= pad_min_y && p_y[p] - mr <= pad_max_y) {
                        near_planets[n_near++] = p;
                    }
                }
                for (int32_t u = start; u < end; ++u) {
                    if (!all_alive && !u_alive[u]) continue;
                    double x = ux[u];
                    double y = uy[u];
                    for (int32_t ni = 0; ni < n_near; ++ni) {
                        const int32_t p = near_planets[ni];
                        const double dx = x - p_x[p];
                        const double dy = y - p_y[p];
                        const double d2 = dx * dx + dy * dy;
                        if (d2 < p_min_r2[p]) {
                            const double mr = p_min_r[p];
                            if (d2 > EPS_DIST2) {
                                const double dist = std::sqrt(d2);
                                double nx = dx / dist;
                                double ny = dy / dist;
                                if (obstacle_escape_eps > 0.0) {
                                    const double tdx = tx - p_x[p];
                                    const double tdy = ty - p_y[p];
                                    const double td = std::sqrt(tdx * tdx + tdy * tdy);
                                    if (td > 1e-6) {
                                        const double tnx = tdx / td;
                                        const double tny = tdy / td;
                                        const double dot = nx * tnx + ny * tny;
                                        if (dot < ANTIPODE_DOT_THRESHOLD) {
                                            const double cross = tnx * ny - tny * nx;
                                            const double sign = (std::fabs(cross) > 1e-9)
                                                ? (cross > 0.0 ? 1.0 : -1.0)
                                                : ((((u - start) & 1) == 0) ? 1.0 : -1.0);
                                            nx += (-tny) * sign * obstacle_escape_eps;
                                            ny += (tnx) * sign * obstacle_escape_eps;
                                            const double nn = std::sqrt(nx * nx + ny * ny);
                                            nx /= nn;
                                            ny /= nn;
                                        }
                                    }
                                }
                                x = p_x[p] + nx * mr;
                                y = p_y[p] + ny * mr;
                            } else {
                                const double tdx = tx - p_x[p];
                                const double tdy = ty - p_y[p];
                                const double td2 = tdx * tdx + tdy * tdy;
                                if (td2 > EPS_DIST2) {
                                    const double td = std::sqrt(td2);
                                    x = p_x[p] + (tdx / td) * mr;
                                    y = p_y[p] + (tdy / td) * mr;
                                } else {
                                    x = p_x[p] + mr;
                                    y = p_y[p];
                                }
                            }
                        }
                    }
                    ux[u] = x;
                    uy[u] = y;
                }
            }

            // Step 5a: Touch-to-land check
            const double land_r = p_land_r[tgt];
            const double land_r2_g = p_land_r2_guard[tgt];
            const int32_t owner = fl_owner[f];
            const double spu = fl_spu[f];
            int32_t alive_left = alive_c;

            for (int32_t u = start; u < end; ++u) {
                if (!all_alive && !u_alive[u]) continue;
                const double dx = tx - ux[u];
                const double dy = ty - uy[u];
                const double d2 = dx * dx + dy * dy;
                if (d2 <= land_r2_g && std::sqrt(d2) <= land_r) {
                    u_alive[u] = 0;
                    --alive_left;

                    ArrivalBucket& b = arr_buckets[tgt];
                    if (b.count == 0) {
                        dirty_planets[n_dirty++] = tgt;
                        if (p_ships[tgt] > 0.0) {
                            b.owners[0] = p_owner[tgt];
                            b.ships[0] = p_ships[tgt];
                            b.count = 1;
                        }
                    }
                    bool found = false;
                    for (int32_t k = 0; k < b.count; ++k) {
                        if (b.owners[k] == owner) {
                            b.ships[k] += spu;
                            found = true;
                            break;
                        }
                    }
                    if (!found && b.count < MAX_ARR_OWNERS) {
                        b.owners[b.count] = owner;
                        b.ships[b.count] = spu;
                        b.count++;
                    }
                }
            }
            fl_alive_cnt[f] = alive_left;
        }

        // Step 5b: Resolve gradual combat on planets with arrivals
        for (int32_t di = 0; di < n_dirty; ++di) {
            const int32_t p = dirty_planets[di];
            ArrivalBucket& b = arr_buckets[p];
            const int32_t cnt = b.count;
            if (cnt == 1) {
                p_owner[p] = b.owners[0];
                p_ships[p] = b.ships[0];
            } else if (cnt >= 2) {
                double top_v = -1.0;
                double sec_v = -1.0;
                int32_t top_o = p_owner[p];
                for (int32_t k = 0; k < cnt; ++k) {
                    const double v = b.ships[k];
                    const int32_t o = b.owners[k];
                    if (v > top_v) {
                        sec_v = top_v;
                        top_v = v;
                        top_o = o;
                    } else if (v > sec_v) {
                        sec_v = v;
                    }
                }
                if (std::fabs(top_v - sec_v) <= TIE_EPS) {
                    p_ships[p] = 0.0;
                } else {
                    p_owner[p] = top_o;
                    p_ships[p] = top_v - sec_v;
                }
            }
            b.count = 0;
        }
    }

    // 6. Compact alive units in-place & compute fleet summaries
    int32_t write_pos = 0;
    for (int32_t f = 0; f < n_fleets; ++f) {
        const int32_t old_start = fl_unit_start[f];
        const int32_t old_count = fl_unit_count[f];
        const int32_t new_start = write_pos;
        double sum_x = 0.0;
        double sum_y = 0.0;
        int32_t kept = 0;

        if (fl_alive_cnt[f] == old_count && old_count > 0) {
            if (write_pos != old_start) {
                std::memmove(ux + write_pos, ux + old_start, (size_t)old_count * sizeof(double));
                std::memmove(uy + write_pos, uy + old_start, (size_t)old_count * sizeof(double));
            }
            for (int32_t u = new_start; u < new_start + old_count; ++u) {
                sum_x += ux[u];
                sum_y += uy[u];
            }
            kept = old_count;
            write_pos += old_count;
        } else if (fl_alive_cnt[f] > 0) {
            for (int32_t u = old_start; u < old_start + old_count; ++u) {
                if (u_alive[u]) {
                    const double x = ux[u];
                    const double y = uy[u];
                    ux[write_pos] = x;
                    uy[write_pos] = y;
                    sum_x += x;
                    sum_y += y;
                    ++write_pos;
                    ++kept;
                }
            }
        }

        fl_unit_start[f] = new_start;
        fl_unit_count[f] = kept;
        if (kept > 0) {
            const double cx = sum_x / (double)kept;
            const double cy = sum_y / (double)kept;
            double max_d2 = 0.0;
            for (int32_t u = new_start; u < new_start + kept; ++u) {
                const double dx = ux[u] - cx;
                const double dy = uy[u] - cy;
                const double d2 = dx * dx + dy * dy;
                if (d2 > max_d2) max_d2 = d2;
            }
            fl_out_ships[f] = (double)kept * fl_spu[f];
            fl_out_cx[f] = cx;
            fl_out_cy[f] = cy;
            fl_out_radius[f] = std::sqrt(max_d2) + ship_radius;
        } else {
            fl_out_ships[f] = 0.0;
            fl_out_cx[f] = 0.0;
            fl_out_cy[f] = 0.0;
            fl_out_radius[f] = ship_radius;
        }
    }

    return write_pos;
}

} // namespace

// ============================================================================
// Standalone POD SimState (inline SoA buffers for ultra-fast clone & rollouts)
// ============================================================================

struct COrder {
    int32_t kind;    // 0 = SendOrder, 1 = RedirOrder
    int32_t player;  // player ID
    int32_t a;       // Send: src_id, Redir: fleet_id
    int32_t b;       // Send: dst_id, Redir: dst_id
    int32_t percent; // Send: 1..100
};

struct SimState {
    // Config
    double width;
    double height;
    double fleet_speed;
    double tick_dt;
    double ship_radius;
    double sep_stiffness;
    double max_sep_push;
    double prod_divisor;
    double obstacle_escape_eps;
    int32_t max_ticks;
    int32_t substeps;
    int32_t sep_mode;
    int32_t allow_redirect;
    int32_t items_max;
    int32_t items_reserved;
    int32_t base_item_users;

    // Dynamic scalar state
    int32_t tick;
    int32_t next_fleet_id;
    int32_t done;
    int32_t winner;

    // Planets SoA (sorted by planet id)
    int32_t n_planets;
    int32_t p_id[SIM_MAX_PLANETS];
    int32_t p_owner[SIM_MAX_PLANETS];
    double p_ships[SIM_MAX_PLANETS];
    double p_x[SIM_MAX_PLANETS];
    double p_y[SIM_MAX_PLANETS];
    double p_prod[SIM_MAX_PLANETS];
    double p_radius[SIM_MAX_PLANETS];

    // Fleets SoA (insertion order, matching Python dict iteration order)
    int32_t n_fleets;
    int32_t fl_id[SIM_MAX_FLEETS];
    int32_t fl_owner[SIM_MAX_FLEETS];
    int32_t fl_source_id[SIM_MAX_FLEETS];
    int32_t fl_target_idx[SIM_MAX_FLEETS];
    double fl_spu[SIM_MAX_FLEETS];
    int32_t fl_unit_start[SIM_MAX_FLEETS];
    int32_t fl_unit_count[SIM_MAX_FLEETS];
    double fl_ships[SIM_MAX_FLEETS];
    double fl_cx[SIM_MAX_FLEETS];
    double fl_cy[SIM_MAX_FLEETS];
    double fl_radius[SIM_MAX_FLEETS];

    // Physical ship units SoA
    int32_t n_units;
    double ux[SIM_MAX_UNITS];
    double uy[SIM_MAX_UNITS];

    int32_t find_planet_idx(int32_t planet_id) const {
        for (int32_t i = 0; i < n_planets; ++i) {
            if (p_id[i] == planet_id) return i;
        }
        return -1;
    }

    int32_t item_count() const {
        return base_item_users + n_planets + n_fleets + n_units;
    }

    double total_ships(int32_t player) const {
        double s = 0.0;
        for (int32_t i = 0; i < n_planets; ++i) {
            if (p_owner[i] == player) s += p_ships[i];
        }
        for (int32_t f = 0; f < n_fleets; ++f) {
            if (fl_owner[f] == player) s += fl_ships[f];
        }
        return s;
    }

    double total_production(int32_t player) const {
        double s = 0.0;
        for (int32_t i = 0; i < n_planets; ++i) {
            if (p_owner[i] == player) s += p_prod[i];
        }
        return s;
    }

    void check_terminal() {
        bool alive1 = false, alive2 = false;
        for (int32_t i = 0; i < n_planets; ++i) {
            if (p_owner[i] == 1) alive1 = true;
            else if (p_owner[i] == 2) alive2 = true;
        }
        for (int32_t f = 0; f < n_fleets; ++f) {
            if (fl_ships[f] > 0.0) {
                if (fl_owner[f] == 1) alive1 = true;
                else if (fl_owner[f] == 2) alive2 = true;
            }
        }
        if (alive1 && !alive2) {
            done = 1;
            winner = 1;
            return;
        }
        if (alive2 && !alive1) {
            done = 1;
            winner = 2;
            return;
        }
        if (!alive1 && !alive2) {
            done = 1;
            winner = 0;
            return;
        }
        if (tick >= max_ticks) {
            done = 1;
            const double s1 = total_ships(1);
            const double s2 = total_ships(2);
            const double pr1 = total_production(1);
            const double pr2 = total_production(2);
            const double sc1 = s1 + 12.0 * (pr1 / prod_divisor);
            const double sc2 = s2 + 12.0 * (pr2 / prod_divisor);
            if (std::fabs(sc1 - sc2) < 1e-3) {
                winner = 0;
            } else {
                winner = (sc1 > sc2) ? 1 : 2;
            }
        }
    }

    int32_t apply_send(int32_t pid, int32_t src_id, int32_t dst_id, int32_t percent) {
        if (src_id == dst_id) return -1;
        const int32_t si = find_planet_idx(src_id);
        const int32_t di = find_planet_idx(dst_id);
        if (si < 0 || di < 0 || p_owner[si] != pid) return -1;
        if (p_ships[si] < 1.0 - TIE_EPS) return -1;

        const int32_t pct = percent < 1 ? 1 : (percent > 100 ? 100 : percent);
        const double frac_ships = p_ships[si] * ((double)pct / 100.0);
        const double send_ships = (frac_ships < p_ships[si]) ? frac_ships : p_ships[si];
        if (send_ships < 1.0 - TIE_EPS) return -1;

        double total_planet_ships = 0.0;
        for (int32_t i = 0; i < n_planets; ++i) {
            if (p_ships[i] > 0.0) {
                total_planet_ships += p_ships[i];
            }
        }
        const int32_t parts = physical_ship_count_cpp(
            send_ships, total_planet_ships, item_count(), items_max, items_reserved
        );
        if (parts <= 0) return -1;
        if (n_fleets >= SIM_MAX_FLEETS || n_units + parts > SIM_MAX_UNITS) return -1;

        const double rem = p_ships[si] - send_ships;
        p_ships[si] = rem > 0.0 ? rem : 0.0;

        const double dx = p_x[di] - p_x[si];
        const double dy = p_y[di] - p_y[si];
        const double raw_d = dist2d(dx, dy);
        const double d = (raw_d > 1e-6) ? raw_d : 1e-6;
        const double spawn_r = p_radius[si] + ship_radius;
        const double sx = p_x[si] + (dx / d) * spawn_r;
        const double sy = p_y[si] + (dy / d) * spawn_r;

        const int32_t f = n_fleets++;
        const int32_t fid = next_fleet_id++;
        const int32_t u_start = n_units;
        fl_id[f] = fid;
        fl_owner[f] = pid;
        fl_source_id[f] = src_id;
        fl_target_idx[f] = di;
        fl_spu[f] = send_ships / (double)parts;
        fl_unit_start[f] = u_start;
        fl_unit_count[f] = parts;
        fl_ships[f] = send_ships;

        if (parts == 1) {
            ux[u_start] = sx;
            uy[u_start] = sy;
            fl_cx[f] = sx;
            fl_cy[f] = sy;
            fl_radius[f] = ship_radius;
        } else {
            double sum_x = 0.0;
            double sum_y = 0.0;
            for (int32_t j = 0; j < parts; ++j) {
                const double rj = SPAWN_SPIRAL_STEP * std::sqrt((double)j);
                const double ang = GOLDEN_ANGLE * (double)j;
                const double x = sx + rj * std::cos(ang);
                const double y = sy + rj * std::sin(ang);
                ux[u_start + j] = x;
                uy[u_start + j] = y;
                sum_x += x;
                sum_y += y;
            }
            const double cx = sum_x / (double)parts;
            const double cy = sum_y / (double)parts;
            double max_d2 = 0.0;
            for (int32_t j = 0; j < parts; ++j) {
                const double ddx = ux[u_start + j] - cx;
                const double ddy = uy[u_start + j] - cy;
                const double dd2 = ddx * ddx + ddy * ddy;
                if (dd2 > max_d2) max_d2 = dd2;
            }
            fl_cx[f] = cx;
            fl_cy[f] = cy;
            fl_radius[f] = std::sqrt(max_d2) + ship_radius;
        }
        n_units += parts;
        return fid;
    }

    void apply_redir(int32_t pid, int32_t fleet_id, int32_t dst_id) {
        if (!allow_redirect) return;
        const int32_t di = find_planet_idx(dst_id);
        if (di < 0) return;
        for (int32_t f = 0; f < n_fleets; ++f) {
            if (fl_id[f] == fleet_id && fl_owner[f] == pid && fl_target_idx[f] != di) {
                fl_target_idx[f] = di;
                return;
            }
        }
    }

    void step_one_tick() {
        if (done) return;

        // Recompute ships_per_unit identically to Python's Fleet.ships_per_unit
        // (fl.ships / max(1, fl.num_units)) for bit-exact parity across partial landings.
        for (int32_t f = 0; f < n_fleets; ++f) {
            const int32_t c = fl_unit_count[f];
            fl_spu[f] = fl_ships[f] / (double)(c > 1 ? c : 1);
        }

        static thread_local uint8_t s_u_alive[SIM_MAX_UNITS];
        static thread_local int32_t s_fl_alive_cnt[SIM_MAX_FLEETS];
        static thread_local double s_tmp_x[SIM_MAX_UNITS];
        static thread_local double s_tmp_y[SIM_MAX_UNITS];
        static thread_local ArrivalBucket s_arr_buckets[SIM_MAX_PLANETS];
        static thread_local int32_t s_dirty_planets[SIM_MAX_PLANETS];
        static thread_local int32_t s_near_planets[SIM_MAX_PLANETS];
        static thread_local double s_p_min_r[SIM_MAX_PLANETS];
        static thread_local double s_p_min_r2[SIM_MAX_PLANETS];
        static thread_local double s_p_land_r[SIM_MAX_PLANETS];
        static thread_local double s_p_land_r2_guard[SIM_MAX_PLANETS];
        static thread_local double s_p_prod_sub[SIM_MAX_PLANETS];

        n_units = run_physics_substeps(
            n_planets,
            p_owner, p_ships, p_x, p_y, p_prod, p_radius,
            n_fleets,
            fl_owner, fl_target_idx, fl_spu, fl_unit_start, fl_unit_count,
            fl_ships, fl_cx, fl_cy, fl_radius,
            n_units, ux, uy,
            tick_dt, fleet_speed, substeps, ship_radius,
            sep_stiffness, max_sep_push, sep_mode, prod_divisor, obstacle_escape_eps,
            s_u_alive, s_fl_alive_cnt, s_tmp_x, s_tmp_y,
            s_arr_buckets, s_dirty_planets, s_near_planets,
            s_p_min_r, s_p_min_r2, s_p_land_r, s_p_land_r2_guard, s_p_prod_sub
        );

        int32_t wf = 0;
        for (int32_t f = 0; f < n_fleets; ++f) {
            if (fl_unit_count[f] > 0) {
                if (wf != f) {
                    fl_id[wf] = fl_id[f];
                    fl_owner[wf] = fl_owner[f];
                    fl_source_id[wf] = fl_source_id[f];
                    fl_target_idx[wf] = fl_target_idx[f];
                    fl_spu[wf] = fl_spu[f];
                    fl_unit_start[wf] = fl_unit_start[f];
                    fl_unit_count[wf] = fl_unit_count[f];
                    fl_ships[wf] = fl_ships[f];
                    fl_cx[wf] = fl_cx[f];
                    fl_cy[wf] = fl_cy[f];
                    fl_radius[wf] = fl_radius[f];
                }
                ++wf;
            }
        }
        n_fleets = wf;

        tick += 1;
        check_terminal();
    }
};

static inline void copy_sim_state_fast(const SimState* src, SimState* dst) {
    dst->width = src->width;
    dst->height = src->height;
    dst->fleet_speed = src->fleet_speed;
    dst->tick_dt = src->tick_dt;
    dst->ship_radius = src->ship_radius;
    dst->sep_stiffness = src->sep_stiffness;
    dst->max_sep_push = src->max_sep_push;
    dst->prod_divisor = src->prod_divisor;
    dst->obstacle_escape_eps = src->obstacle_escape_eps;
    dst->max_ticks = src->max_ticks;
    dst->substeps = src->substeps;
    dst->sep_mode = src->sep_mode;
    dst->allow_redirect = src->allow_redirect;
    dst->items_max = src->items_max;
    dst->items_reserved = src->items_reserved;
    dst->base_item_users = src->base_item_users;
    dst->tick = src->tick;
    dst->next_fleet_id = src->next_fleet_id;
    dst->done = src->done;
    dst->winner = src->winner;

    const int32_t np = src->n_planets;
    dst->n_planets = np;
    if (np > 0) {
        std::memcpy(dst->p_id, src->p_id, (size_t)np * sizeof(int32_t));
        std::memcpy(dst->p_owner, src->p_owner, (size_t)np * sizeof(int32_t));
        std::memcpy(dst->p_ships, src->p_ships, (size_t)np * sizeof(double));
        std::memcpy(dst->p_x, src->p_x, (size_t)np * sizeof(double));
        std::memcpy(dst->p_y, src->p_y, (size_t)np * sizeof(double));
        std::memcpy(dst->p_prod, src->p_prod, (size_t)np * sizeof(double));
        std::memcpy(dst->p_radius, src->p_radius, (size_t)np * sizeof(double));
    }

    const int32_t nf = src->n_fleets;
    dst->n_fleets = nf;
    if (nf > 0) {
        std::memcpy(dst->fl_id, src->fl_id, (size_t)nf * sizeof(int32_t));
        std::memcpy(dst->fl_owner, src->fl_owner, (size_t)nf * sizeof(int32_t));
        std::memcpy(dst->fl_source_id, src->fl_source_id, (size_t)nf * sizeof(int32_t));
        std::memcpy(dst->fl_target_idx, src->fl_target_idx, (size_t)nf * sizeof(int32_t));
        std::memcpy(dst->fl_spu, src->fl_spu, (size_t)nf * sizeof(double));
        std::memcpy(dst->fl_unit_start, src->fl_unit_start, (size_t)nf * sizeof(int32_t));
        std::memcpy(dst->fl_unit_count, src->fl_unit_count, (size_t)nf * sizeof(int32_t));
        std::memcpy(dst->fl_ships, src->fl_ships, (size_t)nf * sizeof(double));
        std::memcpy(dst->fl_cx, src->fl_cx, (size_t)nf * sizeof(double));
        std::memcpy(dst->fl_cy, src->fl_cy, (size_t)nf * sizeof(double));
        std::memcpy(dst->fl_radius, src->fl_radius, (size_t)nf * sizeof(double));
    }

    const int32_t nu = src->n_units;
    dst->n_units = nu;
    if (nu > 0) {
        std::memcpy(dst->ux, src->ux, (size_t)nu * sizeof(double));
        std::memcpy(dst->uy, src->uy, (size_t)nu * sizeof(double));
    }
}

// ============================================================================
// C ABI Exports
// ============================================================================

extern "C" {

int32_t step_physics_cpp(
    int32_t n_planets,
    int32_t* p_owner,
    double* p_ships,
    const double* p_x,
    const double* p_y,
    const double* p_prod,
    const double* p_radius,
    int32_t n_fleets,
    const int32_t* fl_owner,
    const int32_t* fl_target_idx,
    const double* fl_spu,
    int32_t* fl_unit_start,
    int32_t* fl_unit_count,
    double* fl_out_ships,
    double* fl_out_cx,
    double* fl_out_cy,
    double* fl_out_radius,
    int32_t total_units,
    double* ux,
    double* uy,
    double tick_dt,
    double fleet_speed,
    int32_t substeps,
    double ship_radius,
    double sep_stiffness,
    double max_sep_push,
    int32_t sep_mode,
    double prod_divisor,
    double obstacle_escape_eps
) {
    if (n_planets <= 0) return 0;

    if (n_planets <= SIM_MAX_PLANETS && n_fleets <= SIM_MAX_FLEETS && total_units <= SIM_MAX_UNITS) {
        static thread_local uint8_t s_u_alive[SIM_MAX_UNITS];
        static thread_local int32_t s_fl_alive_cnt[SIM_MAX_FLEETS];
        static thread_local double s_tmp_x[SIM_MAX_UNITS];
        static thread_local double s_tmp_y[SIM_MAX_UNITS];
        static thread_local ArrivalBucket s_arr_buckets[SIM_MAX_PLANETS];
        static thread_local int32_t s_dirty_planets[SIM_MAX_PLANETS];
        static thread_local int32_t s_near_planets[SIM_MAX_PLANETS];
        static thread_local double s_p_min_r[SIM_MAX_PLANETS];
        static thread_local double s_p_min_r2[SIM_MAX_PLANETS];
        static thread_local double s_p_land_r[SIM_MAX_PLANETS];
        static thread_local double s_p_land_r2_guard[SIM_MAX_PLANETS];
        static thread_local double s_p_prod_sub[SIM_MAX_PLANETS];

        return run_physics_substeps(
            n_planets, p_owner, p_ships, p_x, p_y, p_prod, p_radius,
            n_fleets, fl_owner, fl_target_idx, fl_spu, fl_unit_start, fl_unit_count,
            fl_out_ships, fl_out_cx, fl_out_cy, fl_out_radius,
            total_units, ux, uy,
            tick_dt, fleet_speed, substeps, ship_radius,
            sep_stiffness, max_sep_push, sep_mode, prod_divisor, obstacle_escape_eps,
            s_u_alive, s_fl_alive_cnt, s_tmp_x, s_tmp_y,
            s_arr_buckets, s_dirty_planets, s_near_planets,
            s_p_min_r, s_p_min_r2, s_p_land_r, s_p_land_r2_guard, s_p_prod_sub
        );
    }

    const size_t sz_p = (size_t)n_planets;
    const size_t sz_f = (size_t)(n_fleets > 0 ? n_fleets : 0);
    const size_t sz_u = (size_t)(total_units > 0 ? total_units : 0);

    std::vector<uint8_t> u_alive(sz_u);
    std::vector<int32_t> fl_alive_cnt(sz_f);
    std::vector<double> tmp_x(sz_u);
    std::vector<double> tmp_y(sz_u);
    std::vector<ArrivalBucket> arr_buckets(sz_p);
    std::vector<int32_t> dirty_planets(sz_p);
    std::vector<int32_t> near_planets(sz_p);
    std::vector<double> p_min_r(sz_p);
    std::vector<double> p_min_r2(sz_p);
    std::vector<double> p_land_r(sz_p);
    std::vector<double> p_land_r2_guard(sz_p);
    std::vector<double> p_prod_sub(sz_p);

    return run_physics_substeps(
        n_planets, p_owner, p_ships, p_x, p_y, p_prod, p_radius,
        n_fleets, fl_owner, fl_target_idx, fl_spu, fl_unit_start, fl_unit_count,
        fl_out_ships, fl_out_cx, fl_out_cy, fl_out_radius,
        total_units, ux, uy,
        tick_dt, fleet_speed, substeps, ship_radius,
        sep_stiffness, max_sep_push, sep_mode, prod_divisor, obstacle_escape_eps,
        u_alive.data(), fl_alive_cnt.data(), tmp_x.data(), tmp_y.data(),
        arr_buckets.data(), dirty_planets.data(), near_planets.data(),
        p_min_r.data(), p_min_r2.data(), p_land_r.data(), p_land_r2_guard.data(), p_prod_sub.data()
    );
}

SimState* sim_create(
    int32_t n_planets,
    const int32_t* p_id,
    const int32_t* p_owner,
    const double* p_ships,
    const double* p_x,
    const double* p_y,
    const double* p_prod,
    const double* p_radius,
    double width,
    double height,
    double fleet_speed,
    double tick_dt,
    int32_t max_ticks,
    int32_t substeps,
    double ship_radius,
    double sep_stiffness,
    double max_sep_push,
    int32_t sep_mode,
    int32_t allow_redirect,
    double obstacle_escape_eps
) {
    if (n_planets < 0 || n_planets > SIM_MAX_PLANETS) return nullptr;
    SimState* st = new SimState();
    st->width = width;
    st->height = height;
    st->fleet_speed = fleet_speed;
    st->tick_dt = tick_dt;
    st->ship_radius = ship_radius;
    st->sep_stiffness = sep_stiffness;
    st->max_sep_push = max_sep_push;
    st->prod_divisor = PRODUCTION_DIVISOR;
    st->obstacle_escape_eps = obstacle_escape_eps;
    st->max_ticks = max_ticks;
    st->substeps = substeps < 1 ? 1 : substeps;
    st->sep_mode = sep_mode;
    st->allow_redirect = allow_redirect;
    st->items_max = ITEMS_MAX;
    st->items_reserved = ITEMS_RESERVED;
    st->base_item_users = BASE_ITEM_USERS;
    st->tick = 0;
    st->next_fleet_id = 1000;
    st->done = 0;
    st->winner = 0;

    st->n_planets = n_planets;
    for (int32_t i = 0; i < n_planets; ++i) {
        st->p_id[i] = p_id[i];
        st->p_owner[i] = p_owner[i];
        st->p_ships[i] = p_ships[i];
        st->p_x[i] = p_x[i];
        st->p_y[i] = p_y[i];
        st->p_prod[i] = p_prod[i];
        st->p_radius[i] = p_radius[i];
    }
    st->n_fleets = 0;
    st->n_units = 0;
    return st;
}

void sim_set_meta(SimState* st, int32_t tick, int32_t next_fleet_id, int32_t done, int32_t winner) {
    if (!st) return;
    st->tick = tick;
    st->next_fleet_id = next_fleet_id;
    st->done = done;
    st->winner = winner;
}

int32_t sim_load_fleets(
    SimState* st,
    int32_t n_fleets,
    const int32_t* fl_id,
    const int32_t* fl_owner,
    const int32_t* fl_source_id,
    const int32_t* fl_target_idx,
    const double* fl_ships,
    const double* fl_cx,
    const double* fl_cy,
    const double* fl_radius,
    const int32_t* fl_unit_start,
    const int32_t* fl_unit_count,
    int32_t n_units,
    const double* ux,
    const double* uy
) {
    if (!st) return -1;
    if (n_fleets < 0 || n_fleets > SIM_MAX_FLEETS) return -1;
    if (n_units < 0 || n_units > SIM_MAX_UNITS) return -1;
    st->n_fleets = n_fleets;
    st->n_units = n_units;
    for (int32_t f = 0; f < n_fleets; ++f) {
        st->fl_id[f] = fl_id[f];
        st->fl_owner[f] = fl_owner[f];
        st->fl_source_id[f] = fl_source_id[f];
        st->fl_target_idx[f] = fl_target_idx[f];
        st->fl_ships[f] = fl_ships[f];
        st->fl_cx[f] = fl_cx[f];
        st->fl_cy[f] = fl_cy[f];
        st->fl_radius[f] = fl_radius[f];
        st->fl_unit_start[f] = fl_unit_start[f];
        const int32_t cnt = fl_unit_count[f];
        st->fl_unit_count[f] = cnt;
        st->fl_spu[f] = fl_ships[f] / (double)(cnt > 1 ? cnt : 1);
    }
    if (n_units > 0) {
        std::memcpy(st->ux, ux, (size_t)n_units * sizeof(double));
        std::memcpy(st->uy, uy, (size_t)n_units * sizeof(double));
    }
    return 0;
}

void sim_destroy(SimState* st) {
    delete st;
}

SimState* sim_clone(const SimState* src) {
    if (!src) return nullptr;
    SimState* dst = new SimState();
    copy_sim_state_fast(src, dst);
    return dst;
}

void sim_copy_into(const SimState* src, SimState* dst) {
    if (!src || !dst) return;
    copy_sim_state_fast(src, dst);
}

int32_t sim_send_order(SimState* st, int32_t player, int32_t src_id, int32_t dst_id, int32_t percent) {
    if (!st || st->done) return -1;
    return st->apply_send(player, src_id, dst_id, percent);
}

void sim_redir_order(SimState* st, int32_t player, int32_t fleet_id, int32_t dst_id) {
    if (!st || st->done) return;
    st->apply_redir(player, fleet_id, dst_id);
}

void sim_step_with_orders(SimState* st, int32_t n_orders, const COrder* orders) {
    if (!st || st->done) return;
    for (int32_t i = 0; i < n_orders; ++i) {
        const COrder& o = orders[i];
        if (o.kind == 0) {
            st->apply_send(o.player, o.a, o.b, o.percent);
        } else if (o.kind == 1) {
            st->apply_redir(o.player, o.a, o.b);
        }
    }
    st->step_one_tick();
}

int32_t sim_step_ticks(SimState* st, int32_t n_ticks) {
    if (!st) return 0;
    int32_t stepped = 0;
    for (int32_t t = 0; t < n_ticks && !st->done; ++t) {
        st->step_one_tick();
        ++stepped;
    }
    return stepped;
}

void sim_get_meta(const SimState* st, int32_t* out_meta) {
    if (!st || !out_meta) return;
    out_meta[0] = st->tick;
    out_meta[1] = st->done;
    out_meta[2] = st->winner;
    out_meta[3] = st->n_planets;
    out_meta[4] = st->n_fleets;
    out_meta[5] = st->n_units;
    out_meta[6] = st->next_fleet_id;
}

void sim_get_config(const SimState* st, double* out_f64, int32_t* out_i32) {
    if (!st) return;
    if (out_f64) {
        out_f64[0] = st->width;
        out_f64[1] = st->height;
        out_f64[2] = st->fleet_speed;
        out_f64[3] = st->tick_dt;
        out_f64[4] = st->ship_radius;
        out_f64[5] = st->sep_stiffness;
        out_f64[6] = st->max_sep_push;
        out_f64[7] = st->obstacle_escape_eps;
    }
    if (out_i32) {
        out_i32[0] = st->max_ticks;
        out_i32[1] = st->substeps;
        out_i32[2] = st->sep_mode;
        out_i32[3] = st->allow_redirect;
    }
}

void sim_read_planets(const SimState* st, int32_t* out_owner, double* out_ships) {
    if (!st || st->n_planets <= 0) return;
    const int32_t np = st->n_planets;
    std::memcpy(out_owner, st->p_owner, (size_t)np * sizeof(int32_t));
    std::memcpy(out_ships, st->p_ships, (size_t)np * sizeof(double));
}

void sim_read_planets_full(
    const SimState* st,
    int32_t* out_id,
    int32_t* out_owner,
    double* out_ships,
    double* out_x,
    double* out_y,
    double* out_prod,
    double* out_radius
) {
    if (!st || st->n_planets <= 0) return;
    const int32_t np = st->n_planets;
    std::memcpy(out_id, st->p_id, (size_t)np * sizeof(int32_t));
    std::memcpy(out_owner, st->p_owner, (size_t)np * sizeof(int32_t));
    std::memcpy(out_ships, st->p_ships, (size_t)np * sizeof(double));
    std::memcpy(out_x, st->p_x, (size_t)np * sizeof(double));
    std::memcpy(out_y, st->p_y, (size_t)np * sizeof(double));
    std::memcpy(out_prod, st->p_prod, (size_t)np * sizeof(double));
    std::memcpy(out_radius, st->p_radius, (size_t)np * sizeof(double));
}

void sim_read_fleets(
    const SimState* st,
    int32_t* out_id,
    int32_t* out_owner,
    int32_t* out_source_id,
    int32_t* out_target_idx,
    int32_t* out_unit_start,
    int32_t* out_unit_count,
    double* out_ships,
    double* out_cx,
    double* out_cy,
    double* out_radius
) {
    if (!st || st->n_fleets <= 0) return;
    const int32_t nf = st->n_fleets;
    std::memcpy(out_id, st->fl_id, (size_t)nf * sizeof(int32_t));
    std::memcpy(out_owner, st->fl_owner, (size_t)nf * sizeof(int32_t));
    if (out_source_id) {
        std::memcpy(out_source_id, st->fl_source_id, (size_t)nf * sizeof(int32_t));
    }
    std::memcpy(out_target_idx, st->fl_target_idx, (size_t)nf * sizeof(int32_t));
    std::memcpy(out_unit_start, st->fl_unit_start, (size_t)nf * sizeof(int32_t));
    std::memcpy(out_unit_count, st->fl_unit_count, (size_t)nf * sizeof(int32_t));
    std::memcpy(out_ships, st->fl_ships, (size_t)nf * sizeof(double));
    std::memcpy(out_cx, st->fl_cx, (size_t)nf * sizeof(double));
    std::memcpy(out_cy, st->fl_cy, (size_t)nf * sizeof(double));
    std::memcpy(out_radius, st->fl_radius, (size_t)nf * sizeof(double));
}

void sim_read_units(const SimState* st, double* out_ux, double* out_uy) {
    if (!st || st->n_units <= 0) return;
    const int32_t nu = st->n_units;
    std::memcpy(out_ux, st->ux, (size_t)nu * sizeof(double));
    std::memcpy(out_uy, st->uy, (size_t)nu * sizeof(double));
}

} // extern "C"
