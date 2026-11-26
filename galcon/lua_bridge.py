"""Galcon 2 ``g2`` Lua bridge for benchmarking external Galcon 2 Lua bots inside ``GalconSim``.

Runs any standard Galcon 2 Lua bot (e.g. ``esparano/GalconModVault``'s ``Sparky v1.6``)
unmodified in a persistent ``lua5.3`` subprocess over stdin/stdout without copying third-party
bot source code into this repository.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from galcon.sim import GalconSim, Order, RedirOrder, SendOrder

_LUA_HARNESS = r"""
-- Minimal Galcon 2 `g2` API shim for running standard Galcon 2 Lua bots against GalconSim
local bot_file = arg[1]
local loop_fn_name = arg[2] or "bot_sparky"
local reset_fn_name = arg[3] or "reset_sparky"

-- Stub Galcon 2 global() helper, require("foo.lua") compatibility & UI no-ops
function global(name) end
package.path = "./?.lua;../?.lua;../../utils/elo/?.lua;../../utils/genetics/?.lua;" .. package.path
local _orig_require = require
function require(mod)
    local clean = string.gsub(mod, "%.lua$", "")
    local ok, res = pcall(_orig_require, clean)
    if ok then return res end
    return {}
end

local _items = {}
local _users = {}
local _planets = {}
local _fleets = {}
local _orders = {}
local _pending_fleet_id = -1000

local ItemMT = {}
ItemMT.__index = ItemMT
ItemMT.__tostring = function(self) return tostring(self.n) end
ItemMT.__concat = function(a, b) return tostring(a) .. tostring(b) end

local function make_item(tbl)
    return setmetatable(tbl, ItemMT)
end

function ItemMT:distance(other)
    local dx = (self.position_x or 0) - (other.position_x or 0)
    local dy = (self.position_y or 0) - (other.position_y or 0)
    return math.sqrt(dx * dx + dy * dy)
end

function ItemMT:owner()
    return _items[self.owner_n]
end

function ItemMT:fleet_send(perc, target)
    if not self.has_planet or not target or not target.has_planet then return end
    if self.n == target.n then return end
    local p = math.floor(tonumber(perc) + 0.5)
    if p < 1 then return end
    if p > 100 then p = 100 end
    local send_s = math.min(self.ships_value, self.ships_value * (p / 100.0))
    if send_s < 1.0 - 1e-9 then return end
    self.ships_value = math.max(0.0, self.ships_value - send_s)
    local ord_idx = #_orders + 1
    _orders[ord_idx] = string.format("S %d %d %d", self.n, target.n, p)

    -- Expose newly launched fleet immediately to same-tick g2.search("fleet") calls
    local dx = (target.position_x or 0) - (self.position_x or 0)
    local dy = (target.position_y or 0) - (self.position_y or 0)
    local dist = math.sqrt(dx * dx + dy * dy)
    local spawn_r = (self.planet_r or 12.0) + 6.0
    local sx = (self.position_x or 0) + ((dist > 1e-6) and (dx / dist) * spawn_r or spawn_r)
    local sy = (self.position_y or 0) + ((dist > 1e-6) and (dy / dist) * spawn_r or 0.0)
    _pending_fleet_id = _pending_fleet_id - 1
    local fid = _pending_fleet_id
    local fl = make_item({
        n = fid,
        has_fleet = true,
        owner_n = self.owner_n,
        fleet_ships = send_s,
        position_x = sx,
        position_y = sy,
        fleet_target = target.n,
        fleet_r = 6.0,
        _order_idx = ord_idx,
        _src_n = self.n,
        _pct = p,
    })
    _items[fid] = fl
    _fleets[#_fleets + 1] = fl
    return fl
end

function ItemMT:fleet_redirect(target)
    if not self.has_fleet or not target or not target.has_planet then return end
    if self.fleet_target ~= target.n then
        self.fleet_target = target.n
        if self._order_idx ~= nil then
            if self._src_n == target.n then
                _orders[self._order_idx] = ""
            else
                _orders[self._order_idx] = string.format("S %d %d %d", self._src_n, target.n, self._pct)
            end
        else
            _orders[#_orders + 1] = string.format("R %d %d", self.n, target.n)
        end
    end
end

g2 = {}
g2.state = "play"
g2.status = ""
g2.headless = true
g2.uid = 1
g2.data = ""
g2.html = ""
g2.form = {}
function g2.item(n) return _items[tonumber(n)] end
function g2.distance(a, b)
    local dx = (a.position_x or 0) - (b.position_x or 0)
    local dy = (a.position_y or 0) - (b.position_y or 0)
    return math.sqrt(dx * dx + dy * dy)
end
function g2.new_label(...) return make_item({n = -1, label_text = ""}) end
function g2.game_reset() end
function g2.planets_settle() end
function g2.new_user() return make_item({n = 0}) end
function g2.new_planet() return make_item({n = 0}) end
function g2.net_send() end

-- Botwar registration stubs (for mods like mod_botwar_testing.lua)
function bots_register(...) end

function g2.search(query)
    local tokens = {}
    for tok in string.gmatch(query, "%S+") do
        tokens[#tokens + 1] = tok
    end
    local kind = tokens[1]
    local filters = {}
    for i = 2, #tokens do
        local k, v = string.match(tokens[i], "^([^:]+):(.+)$")
        if k and v then
            filters[k] = v
        else
            filters[tokens[i]] = true
        end
    end

    local pool
    if kind == "planet" then
        pool = _planets
    elseif kind == "fleet" then
        pool = _fleets
    elseif kind == "user" then
        pool = _users
    else
        pool = {}
    end

    local out = {}
    for i = 1, #pool do
        local obj = pool[i]
        local ok = true
        for k, v in pairs(filters) do
            if k == "owner" then
                if obj.owner_n ~= tonumber(v) then ok = false break end
            elseif k == "target" then
                if obj.fleet_target ~= tonumber(v) then ok = false break end
            elseif k == "neutral" then
                if not obj.user_neutral then ok = false break end
            elseif k == "-neutral" then
                if obj.user_neutral then ok = false break end
            elseif k == "team" then
                if obj.user_team_n ~= tonumber(v) then ok = false break end
            end
        end
        if ok then
            out[#out + 1] = obj
        end
    end
    return out
end

-- Load external bot script
dofile(bot_file)

-- Support both standard Galcon 2 `g2` bots and `botwar` functional-API bots (e.g. bots_waffle3z_optimized)
local _botwar_mode = false
local _botwar_env = nil
local _botwar_mem = {}
local _sb_calls = 0
local _sb_t0 = 0.0

local function _make_botwar_sandbox()
    local _pow = math.pow or function(a, b) return a ^ b end
    local _atan2 = math.atan2 or math.atan
    return {
        type = type, pairs = pairs, ipairs = ipairs, next = next,
        tostring = tostring, tonumber = tonumber,
        print = function() end,
        strsub = string.sub,
        abs = math.abs, ceil = math.ceil, floor = math.floor,
        max = math.max, min = math.min, HUGE = math.huge, random = math.random,
        PI = math.pi, cos = math.cos, sin = math.sin, atan2 = _atan2,
        log = math.log, pow = _pow, sqrt = math.sqrt, exp = math.exp,
        sort = table.sort,
        math = math, table = table, string = string,
        sb_stats = function()
            _sb_calls = _sb_calls + 1
            local dt_score = (os.clock() - _sb_t0) * 4000.0
            return (_sb_calls > dt_score) and _sb_calls or dt_score
        end,
    }
end

if type(_G["_sandbox_init"]) == "function" then
    _botwar_env = _make_botwar_sandbox()
    _G["_sandbox_init"](_botwar_env)
end

local loop_fn = nil
local reset_fn = _G[reset_fn_name]

if _botwar_env ~= nil then
    if loop_fn_name ~= "auto" and type(_botwar_env[loop_fn_name]) == "function" then
        loop_fn = _botwar_env[loop_fn_name]
        _botwar_mode = true
    elseif loop_fn_name == "auto" then
        for _, cand in ipairs({"bots_waffle3z_optimized", "bots_waffle3z", "botwar_act"}) do
            if type(_botwar_env[cand]) == "function" then
                loop_fn = _botwar_env[cand]
                _botwar_mode = true
                break
            end
        end
    end
end

if loop_fn == nil then
    if loop_fn_name ~= "auto" and type(_G[loop_fn_name]) == "function" then
        loop_fn = _G[loop_fn_name]
        if string.sub(loop_fn_name, 1, 5) == "bots_" or string.sub(loop_fn_name, 1, 7) == "botwar_" then
            _botwar_mode = true
        end
    else
        for _, cand in ipairs({"bot_sparky", "bot_vacuumbot", "bot_esparanbot", "bot_act", "bot_loop"}) do
            if type(_G[cand]) == "function" then
                loop_fn = _G[cand]
                break
            end
        end
        if loop_fn == nil then
            for _, cand in ipairs({"bots_waffle3z_optimized", "bots_waffle3z", "botwar_act"}) do
                if type(_G[cand]) == "function" then
                    loop_fn = _G[cand]
                    _botwar_mode = true
                    break
                end
            end
        end
    end
end

if reset_fn_name == "auto" or type(reset_fn) ~= "function" then
    for _, cand in ipairs({"reset_sparky", "reset_vacuumbot", "reset_esparanbot", "reset_bot"}) do
        if type(_G[cand]) == "function" then
            reset_fn = _G[cand]
            break
        end
    end
end
if type(loop_fn) ~= "function" then
    io.stderr:write("Lua bot function not found: " .. tostring(loop_fn_name) .. "\n")
    os.exit(2)
end

local function _build_botwar_data()
    local res = {}
    for i = 1, #_users do
        local u = _users[i]
        res[u.n] = {
            n = u.n,
            is_user = true,
            team = u.user_team_n,
            neutral = u.user_neutral,
        }
    end
    for i = 1, #_planets do
        local p = _planets[i]
        local u = _items[p.owner_n]
        res[p.n] = {
            n = p.n,
            is_planet = true,
            x = p.position_x,
            y = p.position_y,
            r = p.planet_r,
            ships = p.ships_value,
            production = p.ships_production,
            owner = p.owner_n,
            team = u and u.user_team_n or p.owner_n,
            neutral = u and u.user_neutral or false,
        }
    end
    for i = 1, #_fleets do
        local f = _fleets[i]
        local u = _items[f.owner_n]
        local sid = tostring(f.n)
        res[sid] = {
            _n = f.n,
            n = sid,
            is_fleet = true,
            x = f.position_x,
            y = f.position_y,
            r = f.fleet_r or 6.0,
            ships = f.fleet_ships,
            owner = f.owner_n,
            team = u and u.user_team_n or f.owner_n,
            target = f.fleet_target,
        }
    end
    return res
end

-- Protocol loop:
-- Input lines:
--   RESET
--   U <n> <is_neutral:0|1>
--   P <id> <owner> <ships> <x> <y> <prod> <rad>
--   F <id> <owner> <ships> <x> <y> <target> [rad]
--   ACT <player_id>
-- Output lines:
--   S <src> <dst> <pct>
--   R <fid> <dst>
--   END
while true do
    local line = io.read("*l")
    if not line then break end
    if line == "RESET" then
        _items = {}
        _users = {}
        _planets = {}
        _fleets = {}
        _orders = {}
        _pending_fleet_id = -1000
        _botwar_mem = {}
        if type(reset_fn) == "function" then reset_fn() end
        io.write("OK\n")
        io.flush()
    elseif line == "BEGIN" then
        -- Preserve persistent planet table identities across ticks (bots like Sparky cache planet refs!)
        _fleets = {}
        _orders = {}
        _pending_fleet_id = -1000
        for k, v in pairs(_items) do
            if v.has_fleet then _items[k] = nil end
        end
    else
        local tag = string.sub(line, 1, 1)
        if tag == "U" then
            local n, neu = string.match(line, "^U (%-?%d+) (%d+)$")
            n = tonumber(n)
            local is_neu = (tonumber(neu) == 1)
            local u = _items[n]
            if not u then
                u = make_item({
                    n = n,
                    has_user = true,
                    user_neutral = is_neu,
                    user_team_n = n,
                    ships_production_enabled = not is_neu,
                    title_value = is_neu and "neutral" or ("p" .. tostring(n)),
                })
                _items[n] = u
                _users[#_users + 1] = u
            end
        elseif tag == "P" then
            local id, own, sh, x, y, pr, r = string.match(line, "^P (%d+) (%d+) ([%-%d%.eE%+]+) ([%-%d%.eE%+]+) ([%-%d%.eE%+]+) ([%-%d%.eE%+]+) ([%-%d%.eE%+]+)$")
            id = tonumber(id)
            own = tonumber(own)
            local p = _items[id]
            if not p then
                p = make_item({
                    n = id,
                    has_planet = true,
                    owner_n = own,
                    ships_value = tonumber(sh),
                    position_x = tonumber(x),
                    position_y = tonumber(y),
                    ships_production = tonumber(pr),
                    ships_production_enabled = (own ~= 100000),
                    planet_r = tonumber(r),
                })
                _items[id] = p
                _planets[#_planets + 1] = p
            else
                p.owner_n = own
                p.ships_value = tonumber(sh)
                p.position_x = tonumber(x)
                p.position_y = tonumber(y)
                p.ships_production = tonumber(pr)
                p.ships_production_enabled = (own ~= 100000)
                p.planet_r = tonumber(r)
            end
        elseif tag == "F" then
            local id, own, sh, x, y, tgt, fr = string.match(line, "^F (%d+) (%d+) ([%-%d%.eE%+]+) ([%-%d%.eE%+]+) ([%-%d%.eE%+]+) (%d+)%s*([%-%d%.eE%+]*)$")
            id = tonumber(id)
            local fl = make_item({
                n = id,
                has_fleet = true,
                owner_n = tonumber(own),
                fleet_ships = tonumber(sh),
                position_x = tonumber(x),
                position_y = tonumber(y),
                fleet_target = tonumber(tgt),
                fleet_r = (fr and fr ~= "") and tonumber(fr) or 6.0,
            })
            _items[id] = fl
            _fleets[#_fleets + 1] = fl
        elseif string.sub(line, 1, 3) == "ACT" then
            local pid = tonumber(string.match(line, "^ACT (%d+)$"))
            local u = _items[pid]
            if u then
                if _botwar_mode then
                    if not _botwar_mem[pid] then _botwar_mem[pid] = {} end
                    _sb_calls = 0
                    _sb_t0 = os.clock()
                    local data = _build_botwar_data()
                    local ok, res = pcall(loop_fn, {
                        items = data,
                        user = pid,
                        options = nil,
                        memory = _botwar_mem[pid],
                    })
                    if not ok then
                        io.stderr:write("Lua botwar error: " .. tostring(res) .. "\n")
                    elseif type(res) == "table" and res.to ~= nil and res.from ~= nil then
                        local pct = tonumber(res.percent) or 50
                        pct = math.max(5, math.min(100, math.floor(pct / 5 + 0.5) * 5))
                        local from = type(res.from) == "table" and res.from or { res.from }
                        local dst = data[res.to]
                        if dst and dst.is_planet then
                            for _, fk in pairs(from) do
                                local src = data[fk]
                                if src then
                                    if src.is_planet and src.owner == pid then
                                        _orders[#_orders + 1] = string.format("S %d %d %d", src.n, dst.n, pct)
                                    elseif src.is_fleet and src.owner == pid and src.target ~= dst.n then
                                        _orders[#_orders + 1] = string.format("R %d %d", src._n, dst.n)
                                    end
                                end
                            end
                        end
                    end
                else
                    local ok, err = pcall(loop_fn, u)
                    if not ok then
                        io.stderr:write("Lua bot error: " .. tostring(err) .. "\n")
                    end
                end
            end
            for i = 1, #_orders do
                if _orders[i] ~= "" then
                    io.write(_orders[i] .. "\n")
                end
            end
            io.write("END\n")
            io.flush()
        end
    end
end
"""


class GalconLuaBotBridge:
    """Persistent ``lua5.3`` bridge to benchmark unmodified Galcon 2 Lua bots (``g2`` & ``botwar`` APIs) in ``GalconSim``."""

    def __init__(
        self,
        lua_bot_path: str | Path,
        *,
        loop_fn: str = "auto",
        reset_fn: str = "auto",
    ) -> None:
        raw_str = str(lua_bot_path)
        if "::" in raw_str and loop_fn == "auto":
            path_part, fn_part = raw_str.rsplit("::", 1)
            raw_str = path_part
            loop_fn = fn_part.strip() or "auto"
        self.lua_bot_path = Path(raw_str).resolve()
        self.loop_fn = loop_fn
        if not self.lua_bot_path.is_file():
            raise FileNotFoundError(f"Lua bot file not found: {self.lua_bot_path}")
        lua_bin = shutil.which("lua5.3") or shutil.which("lua")
        if lua_bin is None:
            raise RuntimeError("lua5.3 / lua interpreter not found on PATH.")

        self._tmp_harness = Path(tempfile.mkstemp(prefix="galcon_g2_bridge_", suffix=".lua")[1])
        self._tmp_harness.write_text(_LUA_HARNESS, encoding="utf-8")
        self._proc = subprocess.Popen(
            [lua_bin, str(self._tmp_harness), str(self.lua_bot_path), loop_fn, reset_fn],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=str(self.lua_bot_path.parent),
        )
        self._last_tick = 10**9

    def reset(self) -> None:
        assert self._proc.stdin is not None and self._proc.stdout is not None
        self._proc.stdin.write("RESET\n")
        self._proc.stdin.flush()
        resp = self._proc.stdout.readline().strip()
        if resp != "OK":
            err = self._proc.stderr.read() if self._proc.stderr else ""
            raise RuntimeError(f"Lua bridge reset failed: got {resp!r}, stderr={err!r}")

    def act(self, sim: GalconSim, player: int) -> List[Order]:
        if sim.tick <= self._last_tick:
            self.reset()
        self._last_tick = sim.tick

        assert self._proc.stdin is not None and self._proc.stdout is not None
        uid_off = 100000
        lines = [
            "BEGIN",
            f"U {uid_off} 1",
            f"U {uid_off + 1} 0",
            f"U {uid_off + 2} 0",
        ]
        for p in sorted(sim.planets.values(), key=lambda x: x.id):
            lines.append(
                f"P {p.id} {uid_off + p.owner} {p.ships:.4f} {p.x:.4f} {p.y:.4f} {p.production:.4f} {p.radius:.4f}"
            )
        for fl in sorted(sim.fleets.values(), key=lambda x: x.id):
            lines.append(
                f"F {fl.id} {uid_off + fl.owner} {fl.ships:.4f} {fl.x:.4f} {fl.y:.4f} {fl.target} {fl.radius:.4f}"
            )
        lines.append(f"ACT {uid_off + player}")
        self._proc.stdin.write("\n".join(lines) + "\n")
        self._proc.stdin.flush()

        orders: List[Order] = []
        while True:
            out = self._proc.stdout.readline()
            if not out:
                break
            out = out.strip()
            if out == "END":
                break
            parts = out.split()
            if len(parts) == 4 and parts[0] == "S":
                orders.append(SendOrder(int(parts[1]), int(parts[2]), int(parts[3])))
            elif len(parts) == 3 and parts[0] == "R":
                orders.append(RedirOrder(int(parts[1]), int(parts[2])))
        return orders

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
                self._proc.terminate()
                self._proc.wait(timeout=2)
            except Exception:
                self._proc.kill()
        if self._tmp_harness.exists():
            self._tmp_harness.unlink(missing_ok=True)

    def __del__(self) -> None:
        self.close()
