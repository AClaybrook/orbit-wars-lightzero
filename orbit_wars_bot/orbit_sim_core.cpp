#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

constexpr double OW_BOARD_SIZE = 100.0;
constexpr double OW_CENTER = 50.0;
constexpr double OW_SUN_RADIUS = 10.0;
constexpr int32_t OW_MAX_AGENTS = 8;

inline double ow_ln_1000() {
    static const double val = std::log(1000.0);
    return val;
}

inline double ow_fleet_speed(int32_t ships, double max_speed) {
    if (ships <= 1) {
        return 1.0;
    }
    const double ratio = std::log(static_cast<double>(ships)) / ow_ln_1000();
    const double spd = 1.0 + (max_speed - 1.0) * std::pow(ratio, 1.5);
    return (spd < max_speed) ? spd : max_speed;
}

inline double ow_point_to_segment_distance(
    double px, double py,
    double vx, double vy,
    double wx, double wy
) {
    const double dx = wx - vx;
    const double dy = wy - vy;
    const double l2 = dx * dx + dy * dy;
    if (l2 == 0.0) {
        return std::hypot(px - vx, py - vy);
    }
    double t = ((px - vx) * dx + (py - vy) * dy) / l2;
    if (t < 0.0) {
        t = 0.0;
    } else if (t > 1.0) {
        t = 1.0;
    }
    const double proj_x = vx + t * dx;
    const double proj_y = vy + t * dy;
    return std::hypot(px - proj_x, py - proj_y);
}

inline bool ow_swept_pair_hit(
    double ax, double ay,
    double bx, double by,
    double p0x, double p0y,
    double p1x, double p1y,
    double r
) {
    const double d0x = ax - p0x;
    const double d0y = ay - p0y;
    const double dvx = (bx - ax) - (p1x - p0x);
    const double dvy = (by - ay) - (p1y - p0y);
    const double qa = dvx * dvx + dvy * dvy;
    const double qb = 2.0 * (d0x * dvx + d0y * dvy);
    const double qc = d0x * d0x + d0y * d0y - r * r;
    if (qa < 1e-12) {
        return qc <= 0.0;
    }
    const double disc = qb * qb - 4.0 * qa * qc;
    if (disc < 0.0) {
        return false;
    }
    const double sq = std::sqrt(disc);
    const double t1 = (-qb - sq) / (2.0 * qa);
    const double t2 = (-qb + sq) / (2.0 * qa);
    return (t2 >= 0.0) && (t1 <= 1.0);
}

struct CPlanet {
    int32_t id;
    int32_t owner;
    double x;
    double y;
    double radius;
    int32_t ships;
    int32_t prod;
    double orb_r;
    double init_angle;
    int32_t is_orbiting;
    int32_t is_comet;
};

struct CFleet {
    int32_t id;
    int32_t owner;
    double x;
    double y;
    double angle;
    int32_t from_id;
    int32_t ships;
};

struct CCometGroup {
    int32_t path_index;
    std::vector<int32_t> planet_ids;
    std::vector<std::vector<double>> paths_x;
    std::vector<std::vector<double>> paths_y;
};

struct CMove {
    int32_t player_id;
    int32_t from_id;
    double angle;
    int32_t ships;
};

struct OrbitSimState {
    std::vector<CPlanet> planets;
    std::vector<CFleet> fleets;
    std::vector<CCometGroup> comets;
    double angular_velocity = 0.0;
    double ship_speed = 6.0;
    double comet_speed = 4.0;
    int32_t next_fleet_id = 0;
    int32_t step_count = 0;
    int32_t num_agents = 2;
    int32_t max_steps = 500;
    int32_t done = 0;
    int32_t rewards[OW_MAX_AGENTS]{};

    // Scratch buffers reused per tick to avoid heap churn during MCTS rollouts
    std::vector<double> tmp_old_x;
    std::vector<double> tmp_old_y;
    std::vector<double> tmp_new_x;
    std::vector<double> tmp_new_y;
    std::vector<uint8_t> tmp_collidable;
    std::vector<uint8_t> tmp_expired;
    std::vector<uint8_t> tmp_has_combat;
    std::vector<int32_t> tmp_combat_ships;
    std::vector<CFleet> tmp_surviving_fleets;
};

inline int32_t find_planet_index(const std::vector<CPlanet>& planets, int32_t pid) {
    const int32_t n = static_cast<int32_t>(planets.size());
    for (int32_t i = 0; i < n; ++i) {
        if (planets[i].id == pid) {
            return i;
        }
    }
    return -1;
}

inline void purge_comets_by_expired_ids(OrbitSimState* s, const std::vector<int32_t>& expired_pids) {
    if (expired_pids.empty()) {
        return;
    }
    auto is_exp = [&expired_pids](int32_t pid) -> bool {
        for (int32_t e : expired_pids) {
            if (e == pid) {
                return true;
            }
        }
        return false;
    };

    std::vector<CPlanet> kept_planets;
    kept_planets.reserve(s->planets.size());
    for (const auto& p : s->planets) {
        if (!is_exp(p.id)) {
            kept_planets.push_back(p);
        }
    }
    s->planets.swap(kept_planets);

    std::vector<CCometGroup> kept_groups;
    kept_groups.reserve(s->comets.size());
    for (auto& g : s->comets) {
        CCometGroup ng;
        ng.path_index = g.path_index;
        const size_t m = g.planet_ids.size();
        for (size_t i = 0; i < m; ++i) {
            if (!is_exp(g.planet_ids[i])) {
                ng.planet_ids.push_back(g.planet_ids[i]);
                ng.paths_x.push_back(std::move(g.paths_x[i]));
                ng.paths_y.push_back(std::move(g.paths_y[i]));
            }
        }
        if (!ng.planet_ids.empty()) {
            kept_groups.push_back(std::move(ng));
        }
    }
    s->comets.swap(kept_groups);
}

inline void resolve_combat_at_planet(CPlanet& pl, const int32_t* counts) {
    int32_t top_player = -1;
    int32_t top_ships = -1;
    int32_t second_ships = -1;
    for (int32_t p = 0; p < OW_MAX_AGENTS; ++p) {
        const int32_t c = counts[p];
        if (c <= 0) {
            continue;
        }
        if (c > top_ships) {
            second_ships = top_ships;
            top_ships = c;
            top_player = p;
        } else if (c > second_ships) {
            second_ships = c;
        }
    }
    if (top_ships <= 0 || top_player < 0) {
        return;
    }
    int32_t survivor_ships = (second_ships > 0) ? (top_ships - second_ships) : top_ships;
    if (survivor_ships <= 0) {
        return;
    }
    if (pl.owner == top_player) {
        pl.ships += survivor_ships;
    } else {
        pl.ships -= survivor_ships;
        if (pl.ships < 0) {
            pl.owner = top_player;
            pl.ships = -pl.ships;
        }
    }
}

int32_t step_internal(OrbitSimState* s, const CMove* moves, int32_t n_moves) {
    if (s->done) {
        return 1;
    }

    // Pre-launch expired comet cleanup (exact FastOrbitSim parity)
    if (!s->comets.empty()) {
        std::vector<int32_t> pre_exp;
        for (const auto& g : s->comets) {
            const int32_t idx = g.path_index;
            if (idx < 0) {
                continue;
            }
            for (size_t i = 0; i < g.planet_ids.size(); ++i) {
                if (idx >= static_cast<int32_t>(g.paths_x[i].size())) {
                    pre_exp.push_back(g.planet_ids[i]);
                }
            }
        }
        if (!pre_exp.empty()) {
            purge_comets_by_expired_ids(s, pre_exp);
        }
    }

    const int32_t step = s->step_count;

    // 0. Fleet launch (player order 0..num_agents-1)
    if (moves != nullptr && n_moves > 0) {
        const int32_t na = std::min(s->num_agents, OW_MAX_AGENTS);
        bool in_player_order = true;
        int32_t prev_p = 0;
        for (int32_t i = 0; i < n_moves; ++i) {
            if (moves[i].player_id < prev_p) {
                in_player_order = false;
                break;
            }
            prev_p = moves[i].player_id;
        }

        auto apply_move = [s](const CMove& mv) {
            if (mv.ships <= 0) {
                return;
            }
            const int32_t idx = find_planet_index(s->planets, mv.from_id);
            if (idx < 0) {
                return;
            }
            CPlanet& src = s->planets[idx];
            if (src.owner == mv.player_id && src.ships >= mv.ships) {
                src.ships -= mv.ships;
                const double start_x = src.x + std::cos(mv.angle) * (src.radius + 0.1);
                const double start_y = src.y + std::sin(mv.angle) * (src.radius + 0.1);
                s->fleets.push_back(CFleet{
                    s->next_fleet_id++,
                    mv.player_id,
                    start_x,
                    start_y,
                    mv.angle,
                    mv.from_id,
                    mv.ships,
                });
            }
        };

        if (in_player_order) {
            for (int32_t i = 0; i < n_moves; ++i) {
                if (moves[i].player_id >= 0 && moves[i].player_id < na) {
                    apply_move(moves[i]);
                }
            }
        } else {
            for (int32_t pid = 0; pid < na; ++pid) {
                for (int32_t i = 0; i < n_moves; ++i) {
                    if (moves[i].player_id == pid) {
                        apply_move(moves[i]);
                    }
                }
            }
        }
    }

    const int32_t np = static_cast<int32_t>(s->planets.size());

    // 1. Production
    for (int32_t i = 0; i < np; ++i) {
        if (s->planets[i].owner != -1) {
            s->planets[i].ships += s->planets[i].prod;
        }
    }

    // 2. Compute planet end-of-tick positions
    s->tmp_old_x.resize(np);
    s->tmp_old_y.resize(np);
    s->tmp_new_x.resize(np);
    s->tmp_new_y.resize(np);
    s->tmp_collidable.resize(np);
    s->tmp_expired.assign(np, 0);

    const double ang_vel = s->angular_velocity;
    for (int32_t i = 0; i < np; ++i) {
        const CPlanet& p = s->planets[i];
        s->tmp_old_x[i] = p.x;
        s->tmp_old_y[i] = p.y;
        if (p.is_comet) {
            continue;
        }
        if (p.is_orbiting) {
            const double cur_ang = p.init_angle + ang_vel * static_cast<double>(step);
            s->tmp_new_x[i] = OW_CENTER + p.orb_r * std::cos(cur_ang);
            s->tmp_new_y[i] = OW_CENTER + p.orb_r * std::sin(cur_ang);
        } else {
            s->tmp_new_x[i] = p.x;
            s->tmp_new_y[i] = p.y;
        }
        s->tmp_collidable[i] = 1;
    }

    std::vector<int32_t> expired_pids;
    if (!s->comets.empty()) {
        for (auto& g : s->comets) {
            g.path_index += 1;
            const int32_t c_idx = g.path_index;
            const size_t m = g.planet_ids.size();
            for (size_t k = 0; k < m; ++k) {
                const int32_t pid = g.planet_ids[k];
                const int32_t pi = find_planet_index(s->planets, pid);
                if (pi < 0) {
                    continue;
                }
                const int32_t plen = static_cast<int32_t>(g.paths_x[k].size());
                if (c_idx >= plen) {
                    s->tmp_expired[pi] = 1;
                    s->tmp_new_x[pi] = s->tmp_old_x[pi];
                    s->tmp_new_y[pi] = s->tmp_old_y[pi];
                    s->tmp_collidable[pi] = 1;
                    expired_pids.push_back(pid);
                } else {
                    s->tmp_new_x[pi] = g.paths_x[k][c_idx];
                    s->tmp_new_y[pi] = g.paths_y[k][c_idx];
                    s->tmp_collidable[pi] = (s->tmp_old_x[pi] >= 0.0) ? 1 : 0;
                }
            }
        }
    }

    // 3. Fleet movement & swept-pair collision detection
    s->tmp_has_combat.assign(np, 0);
    s->tmp_combat_ships.assign(static_cast<size_t>(np) * OW_MAX_AGENTS, 0);
    s->tmp_surviving_fleets.clear();
    s->tmp_surviving_fleets.reserve(s->fleets.size());

    const double max_spd = s->ship_speed;
    const int32_t nf = static_cast<int32_t>(s->fleets.size());
    for (int32_t fi = 0; fi < nf; ++fi) {
        CFleet fl = s->fleets[fi];
        const double spd = ow_fleet_speed(fl.ships, max_spd);
        const double ox = fl.x;
        const double oy = fl.y;
        const double nx = ox + std::cos(fl.angle) * spd;
        const double ny = oy + std::sin(fl.angle) * spd;
        fl.x = nx;
        fl.y = ny;

        bool hit_planet = false;
        for (int32_t j = 0; j < np; ++j) {
            if (!s->tmp_collidable[j]) {
                continue;
            }
            if (ow_swept_pair_hit(
                    ox, oy, nx, ny,
                    s->tmp_old_x[j], s->tmp_old_y[j],
                    s->tmp_new_x[j], s->tmp_new_y[j],
                    s->planets[j].radius)) {
                if (fl.owner >= 0 && fl.owner < OW_MAX_AGENTS) {
                    s->tmp_combat_ships[static_cast<size_t>(j) * OW_MAX_AGENTS + fl.owner] += fl.ships;
                    s->tmp_has_combat[j] = 1;
                }
                hit_planet = true;
                break;
            }
        }
        if (hit_planet) {
            continue;
        }
        if (nx < 0.0 || nx > OW_BOARD_SIZE || ny < 0.0 || ny > OW_BOARD_SIZE) {
            continue;
        }
        if (ow_point_to_segment_distance(OW_CENTER, OW_CENTER, ox, oy, nx, ny) < OW_SUN_RADIUS) {
            continue;
        }
        s->tmp_surviving_fleets.push_back(fl);
    }
    s->fleets.swap(s->tmp_surviving_fleets);

    // 4. Apply planet movement
    for (int32_t i = 0; i < np; ++i) {
        s->planets[i].x = s->tmp_new_x[i];
        s->planets[i].y = s->tmp_new_y[i];
    }

    // 5. Combat resolution on surviving planets (skip expired comets matching pre-combat purge)
    for (int32_t i = 0; i < np; ++i) {
        if (!s->tmp_expired[i] && s->tmp_has_combat[i]) {
            resolve_combat_at_planet(
                s->planets[i],
                &s->tmp_combat_ships[static_cast<size_t>(i) * OW_MAX_AGENTS]
            );
        }
    }

    if (!expired_pids.empty()) {
        purge_comets_by_expired_ids(s, expired_pids);
    }

    // 6. Termination check
    bool terminated = (step >= s->max_steps - 2);
    if (!terminated) {
        uint32_t alive_mask = 0;
        for (const auto& p : s->planets) {
            if (p.owner >= 0 && p.owner < 32) {
                alive_mask |= (1u << p.owner);
            }
        }
        for (const auto& f : s->fleets) {
            if (f.owner >= 0 && f.owner < 32) {
                alive_mask |= (1u << f.owner);
            }
        }
        // Check <= 1 set bit
        if ((alive_mask & (alive_mask - 1u)) == 0u) {
            terminated = true;
        }
    }

    s->step_count = step + 1;
    if (terminated) {
        s->done = 1;
        int32_t sc[OW_MAX_AGENTS]{};
        const int32_t na = std::min(s->num_agents, OW_MAX_AGENTS);
        for (const auto& p : s->planets) {
            if (p.owner >= 0 && p.owner < na) {
                sc[p.owner] += p.ships;
            }
        }
        for (const auto& f : s->fleets) {
            if (f.owner >= 0 && f.owner < na) {
                sc[f.owner] += f.ships;
            }
        }
        int32_t max_score = 0;
        for (int32_t i = 0; i < na; ++i) {
            if (sc[i] > max_score) {
                max_score = sc[i];
            }
        }
        for (int32_t i = 0; i < OW_MAX_AGENTS; ++i) {
            if (i < na) {
                s->rewards[i] = (sc[i] == max_score && max_score > 0) ? 1 : -1;
            } else {
                s->rewards[i] = 0;
            }
        }
    }
    return s->done;
}

inline void build_comet_track_lookup(
    const OrbitSimState* s,
    int32_t pid,
    const CCometGroup*& out_group,
    int32_t& out_member_idx
) {
    out_group = nullptr;
    out_member_idx = -1;
    for (const auto& g : s->comets) {
        const int32_t m = static_cast<int32_t>(g.planet_ids.size());
        for (int32_t k = 0; k < m; ++k) {
            if (g.planet_ids[k] == pid) {
                out_group = &g;
                out_member_idx = k;
                return;
            }
        }
    }
}

inline int32_t compute_shared_physics_core(
    int32_t n_planets,
    const int32_t* p_id,
    const int32_t* p_owner,
    const double* p_x,
    const double* p_y,
    const double* p_r,
    const int32_t* p_ships,
    const int32_t* p_prod,
    const int32_t* p_is_orbiting,
    const double* p_orb_r,
    const double* p_init_angle,
    const int32_t* p_is_comet,
    int32_t n_fleets,
    const int32_t* f_id,
    const int32_t* f_owner,
    const double* f_x,
    const double* f_y,
    const double* f_angle,
    const int32_t* f_ships,
    int32_t current_step,
    double ang_vel,
    double max_speed,
    int32_t max_horizon,
    int32_t max_route_eta,
    const double* comet_tx,    // [n_planets, max_horizon + 1] or nullptr if using tr_x prefilled
    const double* comet_ty,    // [n_planets, max_horizon + 1] or nullptr
    const uint8_t* comet_tval, // [n_planets, max_horizon + 1] (0 = none/nan, 1 = valid, 2 = hidden -99)
    double* out_tr_x,          // [n_planets, max_horizon + 1]
    double* out_tr_y,          // [n_planets, max_horizon + 1]
    uint8_t* out_tr_valid,     // [n_planets, max_horizon + 1]
    int32_t* out_fut_owner,    // [n_planets, max_horizon + 1]
    int32_t* out_fut_ships,    // [n_planets, max_horizon + 1]
    uint8_t* out_valid_route,  // [n_planets, n_planets]
    int32_t* out_best_eta,     // [n_planets, n_planets]
    double* out_best_dist,     // [n_planets, n_planets]
    int32_t max_arrivals,
    int32_t* out_arr_count,
    int32_t* out_arr_fid,
    int32_t* out_arr_owner,
    int32_t* out_arr_ships,
    int32_t* out_arr_pidx,
    int32_t* out_arr_eta
) {
    const int32_t h1 = max_horizon + 1;
    const double nan_val = std::nan("");
    const double sun_r2 = OW_SUN_RADIUS * OW_SUN_RADIUS;

    if (out_arr_count != nullptr) {
        *out_arr_count = 0;
    }
    if (n_planets <= 0) {
        return 0;
    }

    // 1. Planet tracks [n_planets, max_horizon + 1]
    std::vector<uint8_t> is_none_track(static_cast<size_t>(n_planets) * h1, 0);
    for (int32_t i = 0; i < n_planets; ++i) {
        const size_t base = static_cast<size_t>(i) * h1;
        out_tr_x[base] = p_x[i];
        out_tr_y[base] = p_y[i];
        out_tr_valid[base] = (p_x[i] >= 0.0 && !std::isnan(p_x[i])) ? 1 : 0;

        if (p_is_comet[i] && comet_tval != nullptr) {
            for (int32_t dt = 1; dt <= max_horizon; ++dt) {
                const uint8_t st = comet_tval[base + dt];
                if (st == 1) {
                    out_tr_x[base + dt] = comet_tx[base + dt];
                    out_tr_y[base + dt] = comet_ty[base + dt];
                    out_tr_valid[base + dt] = (comet_tx[base + dt] >= 0.0) ? 1 : 0;
                } else if (st == 2) {
                    out_tr_x[base + dt] = -99.0;
                    out_tr_y[base + dt] = -99.0;
                    out_tr_valid[base + dt] = 0;
                } else {
                    out_tr_x[base + dt] = nan_val;
                    out_tr_y[base + dt] = nan_val;
                    out_tr_valid[base + dt] = 0;
                    is_none_track[base + dt] = 1;
                }
            }
        } else if (p_is_orbiting[i]) {
            const double orb_r = p_orb_r[i];
            const double init_ang = p_init_angle[i];
            for (int32_t dt = 1; dt <= max_horizon; ++dt) {
                const double ang = init_ang + ang_vel * static_cast<double>(current_step + dt - 1);
                const double ox = OW_CENTER + orb_r * std::cos(ang);
                const double oy = OW_CENTER + orb_r * std::sin(ang);
                out_tr_x[base + dt] = ox;
                out_tr_y[base + dt] = oy;
                out_tr_valid[base + dt] = (ox >= 0.0) ? 1 : 0;
            }
        } else {
            const uint8_t ok0 = out_tr_valid[base];
            for (int32_t dt = 1; dt <= max_horizon; ++dt) {
                out_tr_x[base + dt] = p_x[i];
                out_tr_y[base + dt] = p_y[i];
                out_tr_valid[base + dt] = ok0;
            }
        }
    }

    // 2. Fleet arrivals across horizon
    std::vector<int32_t> combat_by_pt(static_cast<size_t>(n_planets) * (max_horizon + 1) * OW_MAX_AGENTS, 0);
    std::vector<uint8_t> has_combat_by_pt(static_cast<size_t>(n_planets) * (max_horizon + 1), 0);

    int32_t arr_cnt = 0;
    for (int32_t fi = 0; fi < n_fleets; ++fi) {
        const int32_t fl_ships = f_ships[fi];
        const int32_t fl_owner = f_owner[fi];
        const double fspd = ow_fleet_speed(fl_ships, max_speed);
        const double vx = std::cos(f_angle[fi]) * fspd;
        const double vy = std::sin(f_angle[fi]) * fspd;
        const double fx0 = f_x[fi];
        const double fy0 = f_y[fi];
        const double l2 = vx * vx + vy * vy;
        const double base_cull = fspd + 4.5;

        for (int32_t t = 0; t < max_horizon; ++t) {
            const double ox = fx0 + vx * static_cast<double>(t);
            const double oy = fy0 + vy * static_cast<double>(t);
            const double nx = ox + vx;
            const double ny = oy + vy;

            bool hit = false;
            int32_t hit_p = -1;
            for (int32_t pj = 0; pj < n_planets; ++pj) {
                const size_t t_idx = static_cast<size_t>(pj) * h1 + t;
                if (!out_tr_valid[t_idx]) {
                    continue;
                }
                const double p_old_x = out_tr_x[t_idx];
                const double p_old_y = out_tr_y[t_idx];
                const double cull = base_cull + p_r[pj];
                if (std::abs(ox - p_old_x) > cull || std::abs(oy - p_old_y) > cull) {
                    continue;
                }
                const bool p_new_none = (is_none_track[t_idx + 1] != 0);
                const double p_new_x = p_new_none ? p_old_x : out_tr_x[t_idx + 1];
                const double p_new_y = p_new_none ? p_old_y : out_tr_y[t_idx + 1];
                if (ow_swept_pair_hit(ox, oy, nx, ny, p_old_x, p_old_y, p_new_x, p_new_y, p_r[pj])) {
                    hit = true;
                    hit_p = pj;
                    break;
                }
            }

            if (hit) {
                const size_t t_idx = static_cast<size_t>(hit_p) * h1 + t + 1;
                if (!is_none_track[t_idx]) {
                    const int32_t eta = t + 1;
                    if (fl_owner >= 0 && fl_owner < OW_MAX_AGENTS) {
                        const size_t cb_off = (static_cast<size_t>(hit_p) * h1 + eta) * OW_MAX_AGENTS + fl_owner;
                        combat_by_pt[cb_off] += fl_ships;
                        has_combat_by_pt[static_cast<size_t>(hit_p) * h1 + eta] = 1;
                    }
                    if (arr_cnt < max_arrivals && out_arr_fid != nullptr) {
                        out_arr_fid[arr_cnt] = f_id[fi];
                        out_arr_owner[arr_cnt] = fl_owner;
                        out_arr_ships[arr_cnt] = fl_ships;
                        out_arr_pidx[arr_cnt] = hit_p;
                        out_arr_eta[arr_cnt] = eta;
                    }
                    arr_cnt++;
                }
                break;
            }

            if (nx < 0.0 || nx > OW_BOARD_SIZE || ny < 0.0 || ny > OW_BOARD_SIZE) {
                break;
            }
            double tt = 0.0;
            if (l2 > 0.0) {
                tt = ((OW_CENTER - ox) * vx + (OW_CENTER - oy) * vy) / l2;
                if (tt < 0.0) {
                    tt = 0.0;
                } else if (tt > 1.0) {
                    tt = 1.0;
                }
            }
            const double dx_s = OW_CENTER - (ox + tt * vx);
            const double dy_s = OW_CENTER - (oy + tt * vy);
            if (dx_s * dx_s + dy_s * dy_s < sun_r2) {
                break;
            }
        }
    }
    if (out_arr_count != nullptr) {
        *out_arr_count = arr_cnt;
    }

    // 3. Per-planet future simulation [n_planets, max_horizon + 1]
    for (int32_t i = 0; i < n_planets; ++i) {
        int32_t owner = p_owner[i];
        int32_t ships = p_ships[i];
        const int32_t prod = p_prod[i];
        const size_t base = static_cast<size_t>(i) * h1;
        out_fut_owner[base] = owner;
        out_fut_ships[base] = ships;

        for (int32_t dt = 1; dt <= max_horizon; ++dt) {
            if (owner != -1) {
                ships += prod;
            }
            if (has_combat_by_pt[base + dt]) {
                const int32_t* counts = &combat_by_pt[(base + dt) * OW_MAX_AGENTS];
                int32_t top_player = -1;
                int32_t top_ships = -1;
                int32_t second_ships = -1;
                for (int32_t op = 0; op < OW_MAX_AGENTS; ++op) {
                    const int32_t c = counts[op];
                    if (c <= 0) {
                        continue;
                    }
                    if (c > top_ships) {
                        second_ships = top_ships;
                        top_ships = c;
                        top_player = op;
                    } else if (c > second_ships) {
                        second_ships = c;
                    }
                }
                if (top_ships > 0 && top_player >= 0) {
                    const int32_t surv = (second_ships > 0) ? (top_ships - second_ships) : top_ships;
                    if (surv > 0) {
                        if (owner == top_player) {
                            ships += surv;
                        } else {
                            ships -= surv;
                            if (ships < 0) {
                                owner = top_player;
                                ships = -ships;
                            }
                        }
                    }
                }
            }
            out_fut_owner[base + dt] = owner;
            out_fut_ships[base + dt] = ships;
        }
    }

    // 4. Directed route matrix [n_planets, n_planets]
    const int32_t r_eta = std::min(max_route_eta, max_horizon);
    std::vector<uint8_t> tval_prefix(static_cast<size_t>(n_planets) * std::max(1, r_eta), 0);
    for (int32_t j = 0; j < n_planets; ++j) {
        uint8_t ok = 1;
        const size_t base_tr = static_cast<size_t>(j) * h1;
        const size_t base_pv = static_cast<size_t>(j) * std::max(1, r_eta);
        for (int32_t k = 0; k < r_eta; ++k) {
            if (!out_tr_valid[base_tr + 1 + k]) {
                ok = 0;
            }
            tval_prefix[base_pv + k] = ok;
        }
    }

    for (int32_t i = 0; i < n_planets; ++i) {
        const int32_t s_clamp = (p_ships[i] > 1) ? p_ships[i] : 1;
        const double spd_i = ow_fleet_speed(s_clamp, max_speed);
        const double pxi = p_x[i];
        const double pyi = p_y[i];
        const double pri = p_r[i];

        for (int32_t j = 0; j < n_planets; ++j) {
            const size_t eidx = static_cast<size_t>(i) * n_planets + j;
            if (i == j) {
                out_valid_route[eidx] = 0;
                out_best_eta[eidx] = 0;
                out_best_dist[eidx] = 0.0;
                continue;
            }

            const double pad_r = pri + 0.1 + p_r[j];
            const size_t base_tr_j = static_cast<size_t>(j) * h1;
            const size_t base_pv_j = static_cast<size_t>(j) * std::max(1, r_eta);
            int32_t first_k = -1;
            for (int32_t k = 0; k < r_eta; ++k) {
                if (!tval_prefix[base_pv_j + k]) {
                    break;
                }
                const double dx = out_tr_x[base_tr_j + 1 + k] - pxi;
                const double dy = out_tr_y[base_tr_j + 1 + k] - pyi;
                const double reach = spd_i * static_cast<double>(k + 1) + pad_r;
                if (dx * dx + dy * dy <= reach * reach) {
                    first_k = k;
                    break;
                }
            }

            if (first_k < 0) {
                out_valid_route[eidx] = 0;
                out_best_eta[eidx] = 0;
                out_best_dist[eidx] = 0.0;
                continue;
            }

            const double imp_x = out_tr_x[base_tr_j + 1 + first_k];
            const double imp_y = out_tr_y[base_tr_j + 1 + first_k];
            const double sdx = imp_x - pxi;
            const double sdy = imp_y - pyi;
            const double sl2 = sdx * sdx + sdy * sdy;
            const double dist = std::sqrt(sl2);
            double st = 0.0;
            if (sl2 > 0.0) {
                st = ((OW_CENTER - pxi) * sdx + (OW_CENTER - pyi) * sdy) / sl2;
                if (st < 0.0) {
                    st = 0.0;
                } else if (st > 1.0) {
                    st = 1.0;
                }
            }
            const double spx = pxi + st * sdx;
            const double spy = pyi + st * sdy;
            const bool clear_sun = std::hypot(OW_CENTER - spx, OW_CENTER - spy) >= (OW_SUN_RADIUS + 0.25);

            out_best_dist[eidx] = dist;
            if (clear_sun) {
                out_valid_route[eidx] = 1;
                out_best_eta[eidx] = first_k + 1;
            } else {
                out_valid_route[eidx] = 0;
                out_best_eta[eidx] = 0;
            }
        }
    }

    (void)p_id;
    return 0;
}

inline int32_t verify_launch_trajectory_core(
    double fx0, double fy0, double fr,
    int32_t target_idx,
    double angle,
    int32_t ships,
    int32_t n_planets,
    int32_t track_stride,
    const double* tr_x,
    const double* tr_y,
    const uint8_t* tr_none,
    const double* p_r,
    int32_t max_dt,
    double max_speed,
    int32_t* out_eta,
    double* out_ix,
    double* out_iy
) {
    const double spd = ow_fleet_speed(ships, max_speed);
    const double cos_a = std::cos(angle);
    const double sin_a = std::sin(angle);
    double fx = fx0 + cos_a * (fr + 0.1);
    double fy = fy0 + sin_a * (fr + 0.1);
    const double vx = cos_a * spd;
    const double vy = sin_a * spd;
    const double base_cull = spd + 4.5;

    for (int32_t dt = 1; dt <= max_dt; ++dt) {
        const double ox = fx;
        const double oy = fy;
        fx += vx;
        fy += vy;

        for (int32_t j = 0; j < n_planets; ++j) {
            const size_t idx_prev = static_cast<size_t>(j) * track_stride + (dt - 1);
            if (tr_none[idx_prev]) {
                continue;
            }
            const double p_old_x = tr_x[idx_prev];
            if (p_old_x < 0.0 || std::isnan(p_old_x)) {
                continue;
            }
            const double p_old_y = tr_y[idx_prev];
            const double pr = p_r[j];
            const double cull = base_cull + pr;
            const double dx = ox - p_old_x;
            if (dx > cull || dx < -cull) {
                continue;
            }
            const double dy = oy - p_old_y;
            if (dy > cull || dy < -cull) {
                continue;
            }
            const size_t idx_cur = idx_prev + 1;
            if (tr_none[idx_cur] || std::isnan(tr_x[idx_cur])) {
                if (ow_swept_pair_hit(ox, oy, fx, fy, p_old_x, p_old_y, p_old_x, p_old_y, pr)) {
                    return 0;
                }
                continue;
            }
            const double p_new_x = tr_x[idx_cur];
            const double p_new_y = tr_y[idx_cur];
            if (ow_swept_pair_hit(ox, oy, fx, fy, p_old_x, p_old_y, p_new_x, p_new_y, pr)) {
                if (j == target_idx) {
                    *out_eta = dt;
                    *out_ix = p_new_x;
                    *out_iy = p_new_y;
                    return 1;
                }
                return 0;
            }
        }

        if (fx < 0.0 || fx > OW_BOARD_SIZE || fy < 0.0 || fy > OW_BOARD_SIZE) {
            return 0;
        }
        if (ow_point_to_segment_distance(OW_CENTER, OW_CENTER, ox, oy, fx, fy) < OW_SUN_RADIUS) {
            return 0;
        }
    }
    return 0;
}

}  // namespace

extern "C" {

void* orbit_sim_create(
    int32_t n_planets,
    const int32_t* p_id,
    const int32_t* p_owner,
    const double* p_x,
    const double* p_y,
    const double* p_r,
    const int32_t* p_ships,
    const int32_t* p_prod,
    const int32_t* p_is_orbiting,
    const double* p_orb_r,
    const double* p_init_angle,
    const int32_t* p_is_comet,
    int32_t n_fleets,
    const int32_t* f_id,
    const int32_t* f_owner,
    const double* f_x,
    const double* f_y,
    const double* f_angle,
    const int32_t* f_from_id,
    const int32_t* f_ships,
    double angular_velocity,
    int32_t next_fleet_id,
    int32_t step_count,
    int32_t num_agents,
    int32_t max_steps,
    double ship_speed,
    double comet_speed,
    int32_t done,
    const int32_t* rewards
) {
    auto* s = new OrbitSimState();
    s->planets.resize(std::max(0, n_planets));
    for (int32_t i = 0; i < n_planets; ++i) {
        s->planets[i] = CPlanet{
            p_id[i],
            p_owner[i],
            p_x[i],
            p_y[i],
            p_r[i],
            p_ships[i],
            p_prod[i],
            p_orb_r[i],
            p_init_angle[i],
            p_is_orbiting[i],
            p_is_comet[i],
        };
    }
    s->fleets.resize(std::max(0, n_fleets));
    for (int32_t i = 0; i < n_fleets; ++i) {
        s->fleets[i] = CFleet{
            f_id[i],
            f_owner[i],
            f_x[i],
            f_y[i],
            f_angle[i],
            f_from_id[i],
            f_ships[i],
        };
    }
    s->angular_velocity = angular_velocity;
    s->next_fleet_id = next_fleet_id;
    s->step_count = step_count;
    s->num_agents = std::max(1, std::min(OW_MAX_AGENTS, num_agents));
    s->max_steps = max_steps;
    s->ship_speed = ship_speed;
    s->comet_speed = comet_speed;
    s->done = done ? 1 : 0;
    if (rewards != nullptr) {
        for (int32_t i = 0; i < s->num_agents; ++i) {
            s->rewards[i] = rewards[i];
        }
    }
    return static_cast<void*>(s);
}

void orbit_sim_destroy(void* ptr) {
    delete static_cast<OrbitSimState*>(ptr);
}

void* orbit_sim_clone(const void* src_ptr) {
    const auto* src = static_cast<const OrbitSimState*>(src_ptr);
    if (src == nullptr) {
        return nullptr;
    }
    auto* dst = new OrbitSimState();
    dst->planets = src->planets;
    dst->fleets = src->fleets;
    dst->comets = src->comets;
    dst->angular_velocity = src->angular_velocity;
    dst->ship_speed = src->ship_speed;
    dst->comet_speed = src->comet_speed;
    dst->next_fleet_id = src->next_fleet_id;
    dst->step_count = src->step_count;
    dst->num_agents = src->num_agents;
    dst->max_steps = src->max_steps;
    dst->done = src->done;
    std::memcpy(dst->rewards, src->rewards, sizeof(dst->rewards));
    return static_cast<void*>(dst);
}

void orbit_sim_copy_into(const void* src_ptr, void* dst_ptr) {
    const auto* src = static_cast<const OrbitSimState*>(src_ptr);
    auto* dst = static_cast<OrbitSimState*>(dst_ptr);
    if (src == nullptr || dst == nullptr) {
        return;
    }
    dst->planets = src->planets;
    dst->fleets = src->fleets;
    dst->comets = src->comets;
    dst->angular_velocity = src->angular_velocity;
    dst->ship_speed = src->ship_speed;
    dst->comet_speed = src->comet_speed;
    dst->next_fleet_id = src->next_fleet_id;
    dst->step_count = src->step_count;
    dst->num_agents = src->num_agents;
    dst->max_steps = src->max_steps;
    dst->done = src->done;
    std::memcpy(dst->rewards, src->rewards, sizeof(dst->rewards));
}

void orbit_sim_add_comet_group(
    void* ptr,
    int32_t path_index,
    int32_t n_members,
    int32_t path_len,
    const int32_t* member_pids,
    const double* paths_x,
    const double* paths_y,
    int32_t add_planets,
    int32_t comet_ships,
    double comet_radius,
    int32_t comet_prod
) {
    auto* s = static_cast<OrbitSimState*>(ptr);
    if (s == nullptr || n_members <= 0 || path_len <= 0) {
        return;
    }
    CCometGroup g;
    g.path_index = path_index;
    g.planet_ids.reserve(n_members);
    g.paths_x.resize(n_members);
    g.paths_y.resize(n_members);
    for (int32_t k = 0; k < n_members; ++k) {
        const int32_t pid = member_pids[k];
        g.planet_ids.push_back(pid);
        const double* rx = paths_x + static_cast<size_t>(k) * path_len;
        const double* ry = paths_y + static_cast<size_t>(k) * path_len;
        g.paths_x[k].assign(rx, rx + path_len);
        g.paths_y[k].assign(ry, ry + path_len);
        if (add_planets) {
            s->planets.push_back(CPlanet{
                pid,
                -1,
                -99.0,
                -99.0,
                comet_radius,
                comet_ships,
                comet_prod,
                0.0,
                0.0,
                0,
                1,
            });
        }
    }
    s->comets.push_back(std::move(g));
}

int32_t orbit_sim_step(void* ptr, const CMove* moves, int32_t n_moves) {
    auto* s = static_cast<OrbitSimState*>(ptr);
    if (s == nullptr) {
        return 1;
    }
    return step_internal(s, moves, n_moves);
}

int32_t orbit_sim_step_n(void* ptr, int32_t n_ticks) {
    auto* s = static_cast<OrbitSimState*>(ptr);
    if (s == nullptr) {
        return 1;
    }
    for (int32_t t = 0; t < n_ticks; ++t) {
        if (s->done) {
            break;
        }
        step_internal(s, nullptr, 0);
    }
    return s->done;
}

void orbit_sim_step_batch(
    void** sims,
    int32_t n_sims,
    const CMove* all_moves,
    const int32_t* move_offsets,
    int32_t n_empty_after
) {
    for (int32_t i = 0; i < n_sims; ++i) {
        auto* s = static_cast<OrbitSimState*>(sims[i]);
        if (s == nullptr || s->done) {
            continue;
        }
        if (all_moves != nullptr && move_offsets != nullptr) {
            const int32_t st = move_offsets[i];
            const int32_t ed = move_offsets[i + 1];
            step_internal(s, all_moves + st, ed - st);
        } else {
            step_internal(s, nullptr, 0);
        }
        for (int32_t k = 0; k < n_empty_after; ++k) {
            if (s->done) {
                break;
            }
            step_internal(s, nullptr, 0);
        }
    }
}

void orbit_sim_get_meta(const void* ptr, int32_t* out_meta) {
    const auto* s = static_cast<const OrbitSimState*>(ptr);
    if (s == nullptr || out_meta == nullptr) {
        return;
    }
    out_meta[0] = s->step_count;
    out_meta[1] = s->next_fleet_id;
    out_meta[2] = s->num_agents;
    out_meta[3] = s->max_steps;
    out_meta[4] = s->done;
    out_meta[5] = static_cast<int32_t>(s->planets.size());
    out_meta[6] = static_cast<int32_t>(s->fleets.size());
    out_meta[7] = static_cast<int32_t>(s->comets.size());
}

void orbit_sim_get_scores(const void* ptr, int32_t* out_scores, int32_t* out_rewards) {
    const auto* s = static_cast<const OrbitSimState*>(ptr);
    if (s == nullptr) {
        return;
    }
    const int32_t na = std::min(s->num_agents, OW_MAX_AGENTS);
    if (out_scores != nullptr) {
        for (int32_t i = 0; i < na; ++i) {
            out_scores[i] = 0;
        }
        for (const auto& p : s->planets) {
            if (p.owner >= 0 && p.owner < na) {
                out_scores[p.owner] += p.ships;
            }
        }
        for (const auto& f : s->fleets) {
            if (f.owner >= 0 && f.owner < na) {
                out_scores[f.owner] += f.ships;
            }
        }
    }
    if (out_rewards != nullptr) {
        for (int32_t i = 0; i < na; ++i) {
            out_rewards[i] = s->rewards[i];
        }
    }
}

int32_t orbit_sim_winner(const void* ptr) {
    const auto* s = static_cast<const OrbitSimState*>(ptr);
    if (s == nullptr || !s->done) {
        return -1;
    }
    const int32_t na = std::min(s->num_agents, OW_MAX_AGENTS);
    int32_t sc[OW_MAX_AGENTS]{};
    for (const auto& p : s->planets) {
        if (p.owner >= 0 && p.owner < na) {
            sc[p.owner] += p.ships;
        }
    }
    for (const auto& f : s->fleets) {
        if (f.owner >= 0 && f.owner < na) {
            sc[f.owner] += f.ships;
        }
    }
    int32_t max_s = sc[0];
    int32_t winner = 0;
    int32_t ties = 1;
    for (int32_t i = 1; i < na; ++i) {
        if (sc[i] > max_s) {
            max_s = sc[i];
            winner = i;
            ties = 1;
        } else if (sc[i] == max_s) {
            ties++;
        }
    }
    return (ties == 1) ? winner : -1;
}

void orbit_sim_export_planets(
    const void* ptr,
    int32_t* out_id,
    int32_t* out_owner,
    double* out_x,
    double* out_y,
    double* out_r,
    int32_t* out_ships,
    int32_t* out_prod,
    int32_t* out_is_comet
) {
    const auto* s = static_cast<const OrbitSimState*>(ptr);
    if (s == nullptr) {
        return;
    }
    const int32_t np = static_cast<int32_t>(s->planets.size());
    for (int32_t i = 0; i < np; ++i) {
        const auto& p = s->planets[i];
        out_id[i] = p.id;
        out_owner[i] = p.owner;
        out_x[i] = p.x;
        out_y[i] = p.y;
        out_r[i] = p.radius;
        out_ships[i] = p.ships;
        out_prod[i] = p.prod;
        if (out_is_comet != nullptr) {
            out_is_comet[i] = p.is_comet;
        }
    }
}

void orbit_sim_export_fleets(
    const void* ptr,
    int32_t* out_id,
    int32_t* out_owner,
    double* out_x,
    double* out_y,
    double* out_angle,
    int32_t* out_from_id,
    int32_t* out_ships
) {
    const auto* s = static_cast<const OrbitSimState*>(ptr);
    if (s == nullptr) {
        return;
    }
    const int32_t nf = static_cast<int32_t>(s->fleets.size());
    for (int32_t i = 0; i < nf; ++i) {
        const auto& f = s->fleets[i];
        out_id[i] = f.id;
        out_owner[i] = f.owner;
        out_x[i] = f.x;
        out_y[i] = f.y;
        out_angle[i] = f.angle;
        out_from_id[i] = f.from_id;
        out_ships[i] = f.ships;
    }
}

void orbit_sim_export_comet_indices(const void* ptr, int32_t* out_path_indices) {
    const auto* s = static_cast<const OrbitSimState*>(ptr);
    if (s == nullptr || out_path_indices == nullptr) {
        return;
    }
    for (size_t i = 0; i < s->comets.size(); ++i) {
        out_path_indices[i] = s->comets[i].path_index;
    }
}

int32_t orbit_sim_build_shared_physics(
    const void* ptr,
    int32_t max_planets,
    int32_t max_horizon,
    int32_t max_route_eta,
    double* out_tr_x,
    double* out_tr_y,
    uint8_t* out_tr_valid,
    int32_t* out_fut_owner,
    int32_t* out_fut_ships,
    uint8_t* out_valid_route,
    int32_t* out_best_eta,
    double* out_best_dist,
    int32_t max_arrivals,
    int32_t* out_arr_count,
    int32_t* out_arr_fid,
    int32_t* out_arr_owner,
    int32_t* out_arr_ships,
    int32_t* out_arr_pidx,
    int32_t* out_arr_eta
) {
    const auto* s = static_cast<const OrbitSimState*>(ptr);
    if (s == nullptr) {
        return -1;
    }
    const int32_t np = std::min(static_cast<int32_t>(s->planets.size()), max_planets);
    const int32_t h1 = max_horizon + 1;

    std::vector<int32_t> pid(np), pown(np), psh(np), pprd(np), porb(np), pic(np);
    std::vector<double> px(np), py(np), pr(np), por(np), pang(np);
    std::vector<double> comet_tx(static_cast<size_t>(np) * h1, 0.0);
    std::vector<double> comet_ty(static_cast<size_t>(np) * h1, 0.0);
    std::vector<uint8_t> comet_tval(static_cast<size_t>(np) * h1, 0);

    for (int32_t i = 0; i < np; ++i) {
        const auto& p = s->planets[i];
        pid[i] = p.id;
        pown[i] = p.owner;
        px[i] = p.x;
        py[i] = p.y;
        pr[i] = p.radius;
        psh[i] = p.ships;
        pprd[i] = p.prod;
        porb[i] = p.is_orbiting;
        por[i] = p.orb_r;
        pang[i] = p.init_angle;
        const CCometGroup* cg = nullptr;
        int32_t cm = -1;
        if (!s->comets.empty()) {
            build_comet_track_lookup(s, p.id, cg, cm);
        }
        pic[i] = (p.is_comet || cg != nullptr) ? 1 : 0;
        if (cg != nullptr && cm >= 0) {
            const int32_t cur_idx = cg->path_index;
            const int32_t plen = static_cast<int32_t>(cg->paths_x[cm].size());
            const size_t base = static_cast<size_t>(i) * h1;
            for (int32_t dt = 1; dt <= max_horizon; ++dt) {
                const int32_t fidx = cur_idx + dt;
                if (fidx < 0) {
                    comet_tval[base + dt] = 2;
                } else if (fidx >= plen) {
                    comet_tval[base + dt] = 0;
                } else {
                    comet_tx[base + dt] = cg->paths_x[cm][fidx];
                    comet_ty[base + dt] = cg->paths_y[cm][fidx];
                    comet_tval[base + dt] = 1;
                }
            }
        }
    }

    const int32_t nf = static_cast<int32_t>(s->fleets.size());
    std::vector<int32_t> fid(nf), fown(nf), fsh(nf);
    std::vector<double> fx(nf), fy(nf), fang(nf);
    for (int32_t i = 0; i < nf; ++i) {
        const auto& f = s->fleets[i];
        fid[i] = f.id;
        fown[i] = f.owner;
        fx[i] = f.x;
        fy[i] = f.y;
        fang[i] = f.angle;
        fsh[i] = f.ships;
    }

    return compute_shared_physics_core(
        np, pid.data(), pown.data(), px.data(), py.data(), pr.data(), psh.data(), pprd.data(),
        porb.data(), por.data(), pang.data(), pic.data(),
        nf, fid.data(), fown.data(), fx.data(), fy.data(), fang.data(), fsh.data(),
        s->step_count, s->angular_velocity, s->ship_speed, max_horizon, max_route_eta,
        comet_tx.data(), comet_ty.data(), comet_tval.data(),
        out_tr_x, out_tr_y, out_tr_valid,
        out_fut_owner, out_fut_ships,
        out_valid_route, out_best_eta, out_best_dist,
        max_arrivals, out_arr_count,
        out_arr_fid, out_arr_owner, out_arr_ships, out_arr_pidx, out_arr_eta
    );
}

int32_t orbit_compute_shared_physics_arrays(
    int32_t n_planets,
    const int32_t* p_id,
    const int32_t* p_owner,
    const double* p_x,
    const double* p_y,
    const double* p_r,
    const int32_t* p_ships,
    const int32_t* p_prod,
    const int32_t* p_is_orbiting,
    const double* p_orb_r,
    const double* p_init_angle,
    const int32_t* p_is_comet,
    const double* comet_tx,
    const double* comet_ty,
    const uint8_t* comet_tval,
    int32_t n_fleets,
    const int32_t* f_id,
    const int32_t* f_owner,
    const double* f_x,
    const double* f_y,
    const double* f_angle,
    const int32_t* f_ships,
    int32_t current_step,
    double angular_velocity,
    double max_speed,
    int32_t max_horizon,
    int32_t max_route_eta,
    double* out_tr_x,
    double* out_tr_y,
    uint8_t* out_tr_valid,
    int32_t* out_fut_owner,
    int32_t* out_fut_ships,
    uint8_t* out_valid_route,
    int32_t* out_best_eta,
    double* out_best_dist,
    int32_t max_arrivals,
    int32_t* out_arr_count,
    int32_t* out_arr_fid,
    int32_t* out_arr_owner,
    int32_t* out_arr_ships,
    int32_t* out_arr_pidx,
    int32_t* out_arr_eta
) {
    return compute_shared_physics_core(
        n_planets, p_id, p_owner, p_x, p_y, p_r, p_ships, p_prod,
        p_is_orbiting, p_orb_r, p_init_angle, p_is_comet,
        n_fleets, f_id, f_owner, f_x, f_y, f_angle, f_ships,
        current_step, angular_velocity, max_speed, max_horizon, max_route_eta,
        comet_tx, comet_ty, comet_tval,
        out_tr_x, out_tr_y, out_tr_valid,
        out_fut_owner, out_fut_ships,
        out_valid_route, out_best_eta, out_best_dist,
        max_arrivals, out_arr_count,
        out_arr_fid, out_arr_owner, out_arr_ships, out_arr_pidx, out_arr_eta
    );
}

int32_t orbit_solve_intercept(
    int32_t from_idx,
    int32_t target_idx,
    int32_t ships,
    int32_t n_planets,
    int32_t track_stride,
    const double* tr_x,
    const double* tr_y,
    const double* p_r,
    int32_t max_eta,
    double max_speed,
    double* out_angle,
    int32_t* out_eta,
    double* out_ix,
    double* out_iy,
    double* out_speed
) {
    if (ships <= 0 || from_idx < 0 || from_idx >= n_planets || target_idx < 0 || target_idx >= n_planets || from_idx == target_idx) {
        return 0;
    }
    const int32_t eff_max_eta = std::min(max_eta, track_stride - 1);
    if (eff_max_eta < 1) {
        return 0;
    }

    const double spd = ow_fleet_speed(ships, max_speed);
    const double fx0 = tr_x[static_cast<size_t>(from_idx) * track_stride];
    const double fy0 = tr_y[static_cast<size_t>(from_idx) * track_stride];
    const double fr = p_r[from_idx];
    const double tr_r = p_r[target_idx];

    std::vector<uint8_t> tr_none(static_cast<size_t>(n_planets) * track_stride, 0);
    for (int32_t j = 0; j < n_planets; ++j) {
        const size_t base = static_cast<size_t>(j) * track_stride;
        for (int32_t dt = 0; dt < track_stride; ++dt) {
            if (std::isnan(tr_x[base + dt])) {
                tr_none[base + dt] = 1;
            }
        }
    }

    struct Cand {
        double err;
        int32_t dt;
        double ang;
    };
    std::vector<Cand> cands;
    cands.reserve(eff_max_eta);

    const size_t t_base = static_cast<size_t>(target_idx) * track_stride;
    for (int32_t dt = 1; dt <= eff_max_eta; ++dt) {
        if (tr_none[t_base + dt]) {
            break;
        }
        if (tr_none[t_base + dt - 1]) {
            continue;
        }
        const double p_old_x = tr_x[t_base + dt - 1];
        if (p_old_x < 0.0) {
            continue;
        }
        const double p_new_x = tr_x[t_base + dt];
        const double p_new_y = tr_y[t_base + dt];
        const double dist_end = std::hypot(p_new_x - fx0, p_new_y - fy0);
        const double reach_lo = (fr + 0.1) + spd * static_cast<double>(dt - 1) - tr_r - 1.5;
        const double reach_hi = (fr + 0.1) + spd * static_cast<double>(dt) + tr_r + 1.5;
        if (reach_lo <= dist_end && dist_end <= reach_hi) {
            if (ow_point_to_segment_distance(OW_CENTER, OW_CENTER, fx0, fy0, p_new_x, p_new_y) < OW_SUN_RADIUS - 0.5) {
                continue;
            }
            const double expected_reach = (fr + 0.1) + spd * (static_cast<double>(dt) - 0.5);
            const double err = std::abs(dist_end - expected_reach);
            const double ang_end = std::atan2(p_new_y - fy0, p_new_x - fx0);
            cands.push_back(Cand{err, dt, ang_end});
        }
    }

    if (cands.empty()) {
        if (!tr_none[t_base + 1]) {
            const double nx = tr_x[t_base + 1];
            const double ny = tr_y[t_base + 1];
            if (ow_point_to_segment_distance(OW_CENTER, OW_CENTER, fx0, fy0, nx, ny) >= OW_SUN_RADIUS - 0.5) {
                cands.push_back(Cand{0.0, 1, std::atan2(ny - fy0, nx - fx0)});
            }
        }
    }

    std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) {
        if (a.dt != b.dt) {
            return a.dt < b.dt;
        }
        return a.err < b.err;
    });

    const int32_t top_k = std::min<int32_t>(4, static_cast<int32_t>(cands.size()));
    for (int32_t k = 0; k < top_k; ++k) {
        int32_t res_eta = 0;
        double ix = 0.0, iy = 0.0;
        const int32_t horizon_chk = std::min(eff_max_eta, cands[k].dt + 3);
        if (verify_launch_trajectory_core(
                fx0, fy0, fr, target_idx, cands[k].ang, ships, n_planets, track_stride,
                tr_x, tr_y, tr_none.data(), p_r, horizon_chk, max_speed,
                &res_eta, &ix, &iy)) {
            *out_angle = cands[k].ang;
            *out_eta = res_eta;
            *out_ix = ix;
            *out_iy = iy;
            *out_speed = spd;
            return 1;
        }
    }

    const int32_t nudge_k = std::min<int32_t>(2, top_k);
    const double deltas[2] = {0.04, -0.04};
    for (int32_t k = 0; k < nudge_k; ++k) {
        const int32_t horizon_chk = std::min(eff_max_eta, cands[k].dt + 3);
        for (int32_t di = 0; di < 2; ++di) {
            const double ang = cands[k].ang + deltas[di];
            int32_t res_eta = 0;
            double ix = 0.0, iy = 0.0;
            if (verify_launch_trajectory_core(
                    fx0, fy0, fr, target_idx, ang, ships, n_planets, track_stride,
                    tr_x, tr_y, tr_none.data(), p_r, horizon_chk, max_speed,
                    &res_eta, &ix, &iy)) {
                *out_angle = ang;
                *out_eta = res_eta;
                *out_ix = ix;
                *out_iy = iy;
                *out_speed = spd;
                return 1;
            }
        }
    }
    return 0;
}

}  // extern "C"
