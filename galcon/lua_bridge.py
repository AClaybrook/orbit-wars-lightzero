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
package.path = "./?.lua;" .. package.path
local _orig_require = require
function require(mod)
    local clean = string.gsub(mod, "%.lua$", "")
    return _orig_require(clean)
end

local _items = {}
local _users = {}
local _planets = {}
local _fleets = {}
local _orders = {}

local ItemMT = {}
ItemMT.__index = ItemMT
ItemMT.__tostring = function(self) return tostring(self.n) end
ItemMT.__concat = function(a, b) return tostring(a) .. tostring(b) end

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
    local p = math.floor(tonumber(perc) + 0.5)
    if p < 1 then return end
    if p > 100 then p = 100 end
    local send_s = math.floor(self.ships_value * (p / 100.0) + 1e-9)
    if send_s < 1 then return end
    self.ships_value = math.max(0.0, self.ships_value - send_s)
    _orders[#_orders + 1] = string.format("S %d %d %d", self.n, target.n, p)
end

function ItemMT:fleet_redirect(target)
    if not self.has_fleet or not target or not target.has_planet then return end
    if self.fleet_target ~= target.n then
        self.fleet_target = target.n
        _orders[#_orders + 1] = string.format("R %d %d", self.n, target.n)
    end
end

local function make_item(tbl)
    return setmetatable(tbl, ItemMT)
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

local loop_fn = _G[loop_fn_name]
local reset_fn = _G[reset_fn_name]
if loop_fn_name == "auto" or type(loop_fn) ~= "function" then
    for _, cand in ipairs({"bot_sparky", "bot_vacuumbot", "bot_esparanbot", "bot_act", "bot_loop"}) do
        if type(_G[cand]) == "function" then
            loop_fn = _G[cand]
            break
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

-- Protocol loop:
-- Input lines:
--   RESET
--   U <n> <is_neutral:0|1>
--   P <id> <owner> <ships> <x> <y> <prod> <rad>
--   F <id> <owner> <ships> <x> <y> <target>
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
        if type(reset_fn) == "function" then reset_fn() end
        io.write("OK\n")
        io.flush()
    elseif line == "BEGIN" then
        -- Preserve persistent planet table identities across ticks (bots like Sparky cache planet refs!)
        _fleets = {}
        _orders = {}
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
            local id, own, sh, x, y, tgt = string.match(line, "^F (%d+) (%d+) ([%-%d%.eE%+]+) ([%-%d%.eE%+]+) ([%-%d%.eE%+]+) (%d+)$")
            id = tonumber(id)
            local fl = make_item({
                n = id,
                has_fleet = true,
                owner_n = tonumber(own),
                fleet_ships = tonumber(sh),
                position_x = tonumber(x),
                position_y = tonumber(y),
                fleet_target = tonumber(tgt),
            })
            _items[id] = fl
            _fleets[#_fleets + 1] = fl
        elseif string.sub(line, 1, 3) == "ACT" then
            local pid = tonumber(string.match(line, "^ACT (%d+)$"))
            local u = _items[pid]
            if u then
                local ok, err = pcall(loop_fn, u)
                if not ok then
                    io.stderr:write("Lua bot error: " .. tostring(err) .. "\n")
                end
            end
            for i = 1, #_orders do
                io.write(_orders[i] .. "\n")
            end
            io.write("END\n")
            io.flush()
        end
    end
end
"""


class GalconLuaBotBridge:
    """Persistent ``lua5.3`` bridge to benchmark unmodified Galcon 2 Lua bots in ``GalconSim``."""

    def __init__(
        self,
        lua_bot_path: str | Path,
        *,
        loop_fn: str = "auto",
        reset_fn: str = "auto",
    ) -> None:
        self.lua_bot_path = Path(lua_bot_path).resolve()
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
            lines.append(f"F {fl.id} {uid_off + fl.owner} {fl.ships:.4f} {fl.x:.4f} {fl.y:.4f} {fl.target}")
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
