from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any

from playwright.sync_api import Browser, BrowserContext, Page, Playwright, sync_playwright


GAME_URL = "https://html-classic.itch.zone/html/18868512/index.html?v=1787145254"

# The game simulates with a fixed 120 Hz timestep (see its core/loop.js).
SIM_RATE = 120


# The game keeps a persistent save in localStorage (unlocked heroes, eclipse
# shards, Sanctuary upgrades). Left alone it accumulates across the episodes of
# a same session, so late episodes would run on a stronger character than early
# ones — and than the random baseline. Wiping it at document start gives every
# episode the exact same starting conditions.
CLEAR_SAVE_JS = """
try {
    localStorage.clear();
} catch (e) { /* storage unavailable: nothing to clear */ }
"""


# Dungeons are not comparable across two policies unless they are the same
# dungeons. The game already seeds its whole generation from one number
# (`startRun(seed)` in js/game/game.js: layout, waves, loot) and keeps
# combat on Math.random. Both are seeded here from the URL fragment, before
# any module runs, so two runs with the same seed and the same inputs play
# out identically — which is what makes paired comparisons possible.
SEED_SETUP_JS = """
(() => {
    const match = /eos-seed=([0-9]+)/.exec(location.hash || '');
    if (!match) return;
    const seed = Number(match[1]) >>> 0;
    window.__eosRunSeed = seed;
    // Same mulberry32 as the game's own js/core/rng.js, on a distinct stream.
    let a = (seed ^ 0x5bd1e995) >>> 0;
    Math.random = () => {
        a |= 0;
        a = (a + 0x6D2B79F5) | 0;
        let t = Math.imul(a ^ (a >>> 15), 1 | a);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
})();
"""


ACTIONS: dict[str, list[str]] = {
    "noop": [],
    "up": ["KeyW"],
    "down": ["KeyS"],
    "left": ["KeyA"],
    "right": ["KeyD"],
    "up_left": ["KeyW", "KeyA"],
    "up_right": ["KeyW", "KeyD"],
    "down_left": ["KeyS", "KeyA"],
    "down_right": ["KeyS", "KeyD"],
    "shoot_up": ["ArrowUp"],
    "shoot_down": ["ArrowDown"],
    "shoot_left": ["ArrowLeft"],
    "shoot_right": ["ArrowRight"],
    "shoot_up_left": ["ArrowUp", "ArrowLeft"],
    "shoot_up_right": ["ArrowUp", "ArrowRight"],
    "shoot_down_left": ["ArrowDown", "ArrowLeft"],
    "shoot_down_right": ["ArrowDown", "ArrowRight"],
    "dash": ["Space"],
    "dash_up": ["KeyW", "Space"],
    "dash_down": ["KeyS", "Space"],
    "dash_left": ["KeyA", "Space"],
    "dash_right": ["KeyD", "Space"],
    "dash_up_left": ["KeyW", "KeyA", "Space"],
    "dash_up_right": ["KeyW", "KeyD", "Space"],
    "dash_down_left": ["KeyS", "KeyA", "Space"],
    "dash_down_right": ["KeyS", "KeyD", "Space"],
    "interact": ["KeyE"],
    # Hero skill (Shift), a whole mechanic the agent had no way to reach:
    # Solen's soul volley fires a ring of 12 shards every 7 s. The aimed
    # variants also hold a fire key, because volley and flask follow the aim.
    "skill": ["ShiftLeft"],
    "skill_up": ["ShiftLeft", "ArrowUp"],
    "skill_down": ["ShiftLeft", "ArrowDown"],
    "skill_left": ["ShiftLeft", "ArrowLeft"],
    "skill_right": ["ShiftLeft", "ArrowRight"],
    "up_shoot_up": ["KeyW", "ArrowUp"],
    "up_shoot_down": ["KeyW", "ArrowDown"],
    "up_shoot_left": ["KeyW", "ArrowLeft"],
    "up_shoot_right": ["KeyW", "ArrowRight"],
    "down_shoot_up": ["KeyS", "ArrowUp"],
    "down_shoot_down": ["KeyS", "ArrowDown"],
    "down_shoot_left": ["KeyS", "ArrowLeft"],
    "down_shoot_right": ["KeyS", "ArrowRight"],
    "left_shoot_up": ["KeyA", "ArrowUp"],
    "left_shoot_down": ["KeyA", "ArrowDown"],
    "left_shoot_left": ["KeyA", "ArrowLeft"],
    "left_shoot_right": ["KeyA", "ArrowRight"],
    "right_shoot_up": ["KeyD", "ArrowUp"],
    "right_shoot_down": ["KeyD", "ArrowDown"],
    "right_shoot_left": ["KeyD", "ArrowLeft"],
    "right_shoot_right": ["KeyD", "ArrowRight"],
    "up_shoot_up_left": ["KeyW", "ArrowUp", "ArrowLeft"],
    "up_shoot_up_right": ["KeyW", "ArrowUp", "ArrowRight"],
    "up_shoot_down_left": ["KeyW", "ArrowDown", "ArrowLeft"],
    "up_shoot_down_right": ["KeyW", "ArrowDown", "ArrowRight"],
    "down_shoot_up_left": ["KeyS", "ArrowUp", "ArrowLeft"],
    "down_shoot_up_right": ["KeyS", "ArrowUp", "ArrowRight"],
    "down_shoot_down_left": ["KeyS", "ArrowDown", "ArrowLeft"],
    "down_shoot_down_right": ["KeyS", "ArrowDown", "ArrowRight"],
    "left_shoot_up_left": ["KeyA", "ArrowUp", "ArrowLeft"],
    "left_shoot_up_right": ["KeyA", "ArrowUp", "ArrowRight"],
    "left_shoot_down_left": ["KeyA", "ArrowDown", "ArrowLeft"],
    "left_shoot_down_right": ["KeyA", "ArrowDown", "ArrowRight"],
    "right_shoot_up_left": ["KeyD", "ArrowUp", "ArrowLeft"],
    "right_shoot_up_right": ["KeyD", "ArrowUp", "ArrowRight"],
    "right_shoot_down_left": ["KeyD", "ArrowDown", "ArrowLeft"],
    "right_shoot_down_right": ["KeyD", "ArrowDown", "ArrowRight"],
    # Diagonal move while firing. `getMove` and `getShoot` (js/core/input.js)
    # resolve two independent axes, so holding W+A+ArrowUp is a legal input
    # the action set simply never offered. Without these, half the escape
    # directions cost the shot, which is why the policy stood still to fire
    # while enemies moving at 125 px/s closed on a player who moves at 290.
    "up_left_shoot_up": ["KeyW", "KeyA", "ArrowUp"],
    "up_left_shoot_down": ["KeyW", "KeyA", "ArrowDown"],
    "up_left_shoot_left": ["KeyW", "KeyA", "ArrowLeft"],
    "up_left_shoot_right": ["KeyW", "KeyA", "ArrowRight"],
    "up_left_shoot_up_left": ["KeyW", "KeyA", "ArrowUp", "ArrowLeft"],
    "up_left_shoot_up_right": ["KeyW", "KeyA", "ArrowUp", "ArrowRight"],
    "up_left_shoot_down_left": ["KeyW", "KeyA", "ArrowDown", "ArrowLeft"],
    "up_left_shoot_down_right": ["KeyW", "KeyA", "ArrowDown", "ArrowRight"],
    "up_right_shoot_up": ["KeyW", "KeyD", "ArrowUp"],
    "up_right_shoot_down": ["KeyW", "KeyD", "ArrowDown"],
    "up_right_shoot_left": ["KeyW", "KeyD", "ArrowLeft"],
    "up_right_shoot_right": ["KeyW", "KeyD", "ArrowRight"],
    "up_right_shoot_up_left": ["KeyW", "KeyD", "ArrowUp", "ArrowLeft"],
    "up_right_shoot_up_right": ["KeyW", "KeyD", "ArrowUp", "ArrowRight"],
    "up_right_shoot_down_left": ["KeyW", "KeyD", "ArrowDown", "ArrowLeft"],
    "up_right_shoot_down_right": ["KeyW", "KeyD", "ArrowDown", "ArrowRight"],
    "down_left_shoot_up": ["KeyS", "KeyA", "ArrowUp"],
    "down_left_shoot_down": ["KeyS", "KeyA", "ArrowDown"],
    "down_left_shoot_left": ["KeyS", "KeyA", "ArrowLeft"],
    "down_left_shoot_right": ["KeyS", "KeyA", "ArrowRight"],
    "down_left_shoot_up_left": ["KeyS", "KeyA", "ArrowUp", "ArrowLeft"],
    "down_left_shoot_up_right": ["KeyS", "KeyA", "ArrowUp", "ArrowRight"],
    "down_left_shoot_down_left": ["KeyS", "KeyA", "ArrowDown", "ArrowLeft"],
    "down_left_shoot_down_right": ["KeyS", "KeyA", "ArrowDown", "ArrowRight"],
    "down_right_shoot_up": ["KeyS", "KeyD", "ArrowUp"],
    "down_right_shoot_down": ["KeyS", "KeyD", "ArrowDown"],
    "down_right_shoot_left": ["KeyS", "KeyD", "ArrowLeft"],
    "down_right_shoot_right": ["KeyS", "KeyD", "ArrowRight"],
    "down_right_shoot_up_left": ["KeyS", "KeyD", "ArrowUp", "ArrowLeft"],
    "down_right_shoot_up_right": ["KeyS", "KeyD", "ArrowUp", "ArrowRight"],
    "down_right_shoot_down_left": ["KeyS", "KeyD", "ArrowDown", "ArrowLeft"],
    "down_right_shoot_down_right": ["KeyS", "KeyD", "ArrowDown", "ArrowRight"],
}


TERMINAL_STATES = {"gameover", "victory"}


# The actions that leave the aim to the mouse channel of `step()`: every
# entry that holds no arrow key. Movement, dash, interact and the hero skill
# (twenty of them) are what a policy decides once aiming is a separate,
# continuous input.
MOVE_ACTIONS = [
    name for name, codes in ACTIONS.items()
    if not any(code.startswith("Arrow") for code in codes)
]


RANDOM_BASELINE_ACTIONS = [
    "noop",
    "up",
    "down",
    "left",
    "right",
    "shoot_up",
    "shoot_down",
    "shoot_left",
    "shoot_right",
    "dash",
    "up_shoot_up",
    "down_shoot_down",
    "left_shoot_left",
    "right_shoot_right",
]


# Installed at document start so every page load exposes the same one-call
# state snapshot, shared by the real-time and turbo stepping modes.
OBSERVE_SETUP_JS = """
window.__eosObserve = () => {
    const g = window.__eosGame;
    const dbg = window.__eosDebug || {};
    const p = g.player || {};
    const s = g.stats || {};
    const room = g.room || null;
    const node = g.node || null;

    const dist = (x, y) => Math.hypot((p.x ?? 0) - x, (p.y ?? 0) - y);

    // Pillars. Combat rooms from depth 2 onwards are seeded with them, and
    // the Library biome fills whole rows of shelves. They stop movement AND
    // projectiles (Room.projectileBlocked), so an enemy standing behind one
    // is simply invulnerable to a policy that keeps firing at it.
    const obstacles = (room?.obstacles || []).map((o) => ({
        x: o.x, y: o.y, r: o.r, distance: dist(o.x, o.y),
    }));
    obstacles.sort((a, b) => a.distance - b.distance);

    // Is the straight line from the player to a point clear of every pillar?
    const hasLineOfSight = (tx, ty) => {
        const px = p.x ?? 0;
        const py = p.y ?? 0;
        const sx = tx - px;
        const sy = ty - py;
        const lenSq = sx * sx + sy * sy;
        for (const o of obstacles) {
            let t = lenSq > 1e-6 ? ((o.x - px) * sx + (o.y - py) * sy) / lenSq : 0;
            t = Math.min(Math.max(t, 0), 1);
            const cx = px + sx * t - o.x;
            const cy = py + sy * t - o.y;
            if (cx * cx + cy * cy < o.r * o.r) return false;
        }
        return true;
    };

    const activeEnemies = [];
    const enemies = dbg.enemies;
    if (enemies) {
        for (let i = 0; i < enemies.count; i++) {
            const e = enemies.items[i];
            if (!e || e.hp <= 0 || e.spawnTimer > 0) continue;
            activeEnemies.push({
                x: e.x ?? 0,
                y: e.y ?? 0,
                hp: e.hp ?? 0,
                max_hp: e.maxHp ?? 0,
                type: e.type?.id || e.type?.name || null,
                state: e.state || null,
                elite: !!e.elite,
                boss: !!e.type?.boss,
                attack: e.attackName || null,
                attack_step: e.atkStep ?? 0,
                attack_time: e.atkT ?? 0,
                phase: e.phaseIdx ?? 0,
                radius: e.radius ?? 14,
                // An orb takes distance / shotSpeed to arrive; over that
                // flight a walking enemy leaves its own hitbox behind, so
                // aiming at where it stands is aiming behind it.
                vx: e.vx ?? 0,
                vy: e.vy ?? 0,
                distance: dist(e.x ?? 0, e.y ?? 0),
                los: hasLineOfSight(e.x ?? 0, e.y ?? 0),
                // What the game knows about this body and announces on
                // screen: the elite's crown (affix), the glass guardian's
                // shield and its facing, the spectral phase, how hard it
                // hits, how far along its telegraph it is (stateT) and how
                // long before its next attack (shootCd). A boss also
                // carries the direction it locked for a charge (auxX/Y)
                // and where its echo or reflection stands (aux2X/Y).
                affix: e.affix?.id || null,
                ghosting: !!e.ghosting,
                front_shield: !!e.type?.frontShield,
                shield_dir: [e.auxX ?? 0, e.auxY ?? 0],
                contact_damage: e.contactDamage ?? e.type?.contactDamage ?? 1,
                shot_dmg: e.shotDmg ?? 1,
                shot_speed_mult: e.shotSpeedMult ?? 1,
                speed: e.speed ?? e.type?.speed ?? 0,
                shoot_cd: e.shootCd ?? 0,
                state_t: e.stateT ?? 0,
                lock_dir: [e.auxX ?? 0, e.auxY ?? 0],
                aux2: [e.aux2X ?? 0, e.aux2Y ?? 0],
                duo: !!(e.duoPartner && e.duoPartner.hp > 0),
                arena_on: !!e.arenaOn,
                arena_kind: e.arenaKind || null,
                arena_t: e.arenaT ?? 0,
                tier: e.tier ?? 0,
            });
        }
    }
    activeEnemies.sort((a, b) => a.distance - b.distance);

    // What a relic actually does, not just how rare it is. The bank holds
    // 159 items whose effects differ by a factor of four: +0.3 damage on a
    // base of 1 is worth thirty times a range bonus, and +4 max life on a
    // bar of 6 is worth more than either.
    const itemMods = (it) => {
        if (!it) return null;
        const add = it.mods?.add || {};
        const mult = it.mods?.mult || {};
        return {
            dmg_add: add.damage ?? 0,
            hp_add: add.maxHp ?? 0,
            dmg_mult: mult.damage ?? 1,
            fire_mult: mult.fireDelay ?? 1,
            move_mult: mult.moveSpeed ?? 1,
            // the whole modifier set, as the game composes it
            // (player.recompute): additive bonuses, multipliers, and the
            // behaviour counters behind orbs, homing, piercing and so on
            id: it.id || null,
            add: { ...add },
            mult: { ...mult },
            flags: { ...(it.mods?.flags || {}) },
            cursed: !!it.cursed,
            unique: !!it.unique,
        };
    };

    const activePickups = [];
    const pickups = dbg.pickups;
    if (pickups) {
        for (let i = 0; i < pickups.count; i++) {
            const item = pickups.items[i];
            if (!item) continue;
            activePickups.push({
                x: item.x ?? 0,
                y: item.y ?? 0,
                type: item.type || null,
                value: item.value ?? 0,
                choice: !!item.choice,
                item_name: item.item?.name || null,
                item_rarity: item.item?.rarity || null,
                item_desc: item.item?.desc || null,
                item_mods: itemMods(item.item),
                distance: dist(item.x ?? 0, item.y ?? 0),
            });
        }
    }
    activePickups.sort((a, b) => a.distance - b.distance);

    let availableDoors = [];
    let nearestDoor = null;
    if (room?.node?.doors && room.doorsOpen) {
        const centers = {
            up: { x: room.w / 2, y: 0 },
            down: { x: room.w / 2, y: room.h },
            left: { x: 0, y: room.h / 2 },
            right: { x: room.w, y: room.h / 2 },
        };
        availableDoors = Object.entries(room.node.doors)
            .filter(([, open]) => !!open)
            .map(([dir]) => ({
                dir,
                x: centers[dir].x,
                y: centers[dir].y,
                distance: dist(centers[dir].x, centers[dir].y),
            }))
            .sort((a, b) => a.distance - b.distance);
        nearestDoor = availableDoors[0] || null;
    }

    // Door leading toward the best unvisited room (BFS over the dungeon
    // graph, i.e. the minimap). Rooms are ranked by type so the run gears
    // up (treasure, shop) before facing the guardian, which is always
    // routed last.
    let targetDoor = nearestDoor;
    let targetRoomType = null;
    let targetRoomHops = 0;
    if (availableDoors.length && g.dungeon?.nodes && node) {
        const dirs = { up: [0, -1], right: [1, 0], down: [0, 1], left: [-1, 0] };
        const key = (x, y) => x + ',' + y;
        const seen = new Set([key(node.gx, node.gy)]);
        const queue = [[node.gx, node.gy, null, 0]];
        const PRIORITY = { treasure: 5, shop: 4, miniboss: 2, boss: 1 };
        let best = null;
        while (queue.length) {
            const [cx, cy, firstDir, hops] = queue.shift();
            const cur = g.dungeon.nodes.get(key(cx, cy));
            if (!cur) continue;
            if (!cur.visited && firstDir) {
                const prio = PRIORITY[cur.type] ?? 3;
                const score = prio * 100 - hops;
                if (!best || score > best.score) {
                    best = { score, firstDir, type: cur.type || null, hops };
                }
                continue;
            }
            for (const [dir, [dx, dy]] of Object.entries(dirs)) {
                if (!cur.doors || !cur.doors[dir]) continue;
                const nk = key(cx + dx, cy + dy);
                if (seen.has(nk)) continue;
                seen.add(nk);
                queue.push([cx + dx, cy + dy, firstDir || dir, hops + 1]);
            }
        }
        if (best) {
            targetDoor = availableDoors.find((d) => d.dir === best.firstDir) || nearestDoor;
            targetRoomType = best.type;
            targetRoomHops = best.hops;
        }
    }

    // Enemies still materialising (spawn window): invisible to the active
    // list, yet the right moment to pre-position instead of freezing.
    const spawningEnemies = [];
    if (enemies) {
        for (let i = 0; i < enemies.count; i++) {
            const e = enemies.items[i];
            if (!e || e.hp <= 0 || !(e.spawnTimer > 0)) continue;
            spawningEnemies.push({
                x: e.x ?? 0,
                y: e.y ?? 0,
                type: e.type?.id || null,
                spawn_timer: e.spawnTimer ?? 0,
                distance: dist(e.x ?? 0, e.y ?? 0),
            });
        }
    }
    spawningEnemies.sort((a, b) => a.distance - b.distance);

    let nearestShopItem = null;
    let shopItems = [];
    if (node?.type === 'shop' && node.shopStock && room?.shopSlots) {
        const slots = room.shopSlots();
        shopItems = node.shopStock
            .map((slot, i) => {
                if (!slot || slot.sold || !slots[i]) return null;
                return {
                    x: slots[i].x,
                    y: slots[i].y,
                    kind: slot.kind || null,
                    price: slot.price ?? 0,
                    item_name: slot.item?.name || null,
                    item_rarity: slot.item?.rarity || null,
                    item_desc: slot.item?.desc || null,
                    item_mods: itemMods(slot.item),
                    distance: dist(slots[i].x, slots[i].y),
                };
            })
            .filter(Boolean)
            .sort((a, b) => a.distance - b.distance);
        nearestShopItem = shopItems[0] || null;
    }

    const nearChoice = g.nearChoice ? {
        x: g.nearChoice.x ?? 0,
        y: g.nearChoice.y ?? 0,
        type: g.nearChoice.type || null,
        value: g.nearChoice.value ?? 0,
        choice: !!g.nearChoice.choice,
        item_name: g.nearChoice.item?.name || null,
        item_rarity: g.nearChoice.item?.rarity || null,
        item_desc: g.nearChoice.item?.desc || null,
        item_mods: itemMods(g.nearChoice.item),
        distance: dist(g.nearChoice.x ?? 0, g.nearChoice.y ?? 0),
    } : null;

    // The room's own optional interaction, which is NOT a choice pedestal:
    // the sacrifice altar, the gambler and the challenge brazier all live at
    // the same spot and answer to E. Each has its own price, so the policy
    // has to be able to tell them apart and decide.
    let roomFocus = null;
    if (room && node) {
        const available = {
            altar: !node.altarUsed,
            gambler: (node.gamblesLeft === undefined ? 3 : node.gamblesLeft) > 0,
            defi: !node.defiDone,
        }[node.type];
        if (available) {
            const fx = room.w / 2;
            const fy = room.h * 0.42;
            roomFocus = {
                kind: node.type, x: fx, y: fy, distance: dist(fx, fy),
            };
        }
    }

    const portal = room?.portalActive ? {
        x: room.w / 2,
        y: room.h / 2,
        distance: dist(room.w / 2, room.h / 2),
    } : null;

    // Enemy projectiles and telegraphed blast zones: both are dodgeable
    // by movement, provided the agent can actually see them.
    const activeShots = [];
    const shots = dbg.enemyShots;
    if (shots) {
        for (let i = 0; i < shots.count; i++) {
            const sh = shots.items[i];
            if (!sh) continue;
            activeShots.push({
                x: sh.x ?? 0,
                y: sh.y ?? 0,
                vx: sh.vx ?? 0,
                vy: sh.vy ?? 0,
                radius: sh.radius ?? 6,
                dmg: sh.dmg ?? 1,
                slow: sh.slow ?? 0,
                distance: dist(sh.x ?? 0, sh.y ?? 0),
            });
        }
    }
    activeShots.sort((a, b) => a.distance - b.distance);

    const activeHazards = [];
    if (dbg.hazards) {
        for (const hz of dbg.hazards) {
            activeHazards.push({
                x: hz.x,
                y: hz.y,
                r: hz.r,
                remaining: Math.max(0, (hz.fuse ?? 0) - (hz.t ?? 0)),
                dmg: hz.dmg ?? 1,
                distance: dist(hz.x, hz.y),
            });
        }
    }
    activeHazards.sort((a, b) => a.distance - b.distance);

    // ——— danger field ———
    // Truncated entity lists cannot describe the attacks that actually kill:
    // aftershock drops 19 blast circles at once, frostNova 16, tidalPulse
    // fires 27 orbs. So the danger is also summarised as a fixed-size field:
    // how lethal each of 16 directions looks at three travel distances,
    // computed over EVERY hazard, orb and enemy on screen.
    const roomW = room?.w ?? 1280;
    const roomH = room?.h ?? 720;
    const roomWt = room?.wt ?? 46;
    const dangerAt = (px, py, travel) => {
        let danger = 0;
        const wallClear = Math.min(
            px - roomWt, roomW - roomWt - px, py - roomWt, roomH - roomWt - py,
        );
        if (wallClear < 0) danger += 1.0;
        else if (wallClear < 40) danger += 0.5;
        else if (wallClear < 90) danger += 0.15;

        // A pillar is not lethal, it is simply not walkable: heading into
        // one wastes the step exactly as a wall does.
        for (const o of obstacles) {
            const clear = Math.hypot(px - o.x, py - o.y) - o.r;
            if (clear < 18) danger += 0.8;
            else if (clear < 45) danger += 0.2;
        }

        for (const hz of activeHazards) {
            const clear = Math.hypot(px - hz.x, py - hz.y) - hz.r;
            // Standing in a circle is only fatal if it resolves while we are
            // still there — a long fuse can be crossed, a short one cannot.
            if (clear < 0) danger += hz.remaining <= travel + 0.25 ? 1.0 : 0.55;
            else if (clear < 40) danger += 0.25;
        }
        for (const sh of activeShots) {
            const speedSq = sh.vx * sh.vx + sh.vy * sh.vy;
            if (speedSq <= 1e-6) continue;
            let t = ((px - sh.x) * sh.vx + (py - sh.y) * sh.vy) / speedSq;
            t = Math.min(Math.max(t, 0), 0.9);
            const d = Math.hypot(px - sh.x - sh.vx * t, py - sh.y - sh.vy * t) - sh.radius;
            if (d < 22) danger += 0.8;
            else if (d < 50) danger += 0.25;
        }
        for (const e of activeEnemies) {
            const d = Math.hypot(px - e.x, py - e.y) - e.radius;
            if (d < 42) danger += 0.7;
            else if (d < 100) danger += 0.25;
        }
        return Math.min(1, danger);
    };

    const DANGER_RADII = [60, 130, 220];
    const PLAYER_SPEED = 290;
    const dangerField = [];
    for (let i = 0; i < 16; i++) {
        const a = (i / 16) * Math.PI * 2;
        const ux = Math.cos(a);
        const uy = Math.sin(a);
        for (const radius of DANGER_RADII) {
            dangerField.push(dangerAt(
                (p.x ?? 0) + ux * radius,
                (p.y ?? 0) + uy * radius,
                radius / PLAYER_SPEED,
            ));
        }
    }

    const shotSectors = new Array(16).fill(0);
    for (const sh of activeShots) {
        const speedSq = sh.vx * sh.vx + sh.vy * sh.vy;
        if (speedSq < 1) continue;
        const px = p.x ?? 0;
        const py = p.y ?? 0;
        let t = ((px - sh.x) * sh.vx + (py - sh.y) * sh.vy) / speedSq;
        if (t < 0 || t > 1.0) continue;
        const cx = sh.x + sh.vx * t - px;
        const cy = sh.y + sh.vy * t - py;
        if (Math.hypot(cx, cy) > 40 + sh.radius) continue;
        // sector of the direction the orb comes FROM
        const a = Math.atan2(-sh.vy, -sh.vx);
        const k = ((Math.round(a / (Math.PI / 8)) % 16) + 16) % 16;
        shotSectors[k] = Math.max(shotSectors[k], 1 - t);
    }

    return {
        state: g.state,
        hp: p.hp ?? 0,
        max_hp: p.maxHp ?? 0,
        x: p.x ?? 0,
        y: p.y ?? 0,
        floor: g.floorNum ?? 0,
        kills: s.kills ?? 0,
        rooms: s.rooms ?? 0,
        floors: s.floors ?? 0,
        time: s.time ?? 0,
        hp_missing: Math.max(0, (p.maxHp ?? 0) - (p.hp ?? 0)),
        gold: p.gold ?? 0,
        item_count: p.items?.length ?? 0,
        room_type: node?.type || null,
        room_cleared: !!node?.cleared,
        abyss_gate: !!node?.abyssGate,
        doors_open: !!room?.doorsOpen,
        portal_active: !!room?.portalActive,
        enemy_count: activeEnemies.length,
        pickup_count: activePickups.length,
        nearest_enemy: activeEnemies[0] || null,
        nearest_pickup: activePickups[0] || null,
        nearest_shop_item: nearestShopItem,
        shop_items: shopItems,
        nearest_door: nearestDoor,
        target_door: targetDoor,
        near_choice: nearChoice,
        room_focus: roomFocus,
        portal,
        available_doors: availableDoors,
        pickups: activePickups.slice(0, 8),
        enemies: activeEnemies.slice(0, 8),
        shots: activeShots.slice(0, 16),
        hazards: activeHazards.slice(0, 20),
        spawning: spawningEnemies.slice(0, 3),
        danger_field: dangerField,
        obstacles: obstacles.slice(0, 8),
        door_los: targetDoor ? hasLineOfSight(targetDoor.x, targetDoor.y) : true,
        room_w: roomW,
        room_h: roomH,
        room_wt: roomWt,
        // Player capabilities the policy can actually act on: the hero skill
        // (Shift) is a full mechanic, the dash carries i-frames, and the
        // Guardian's shield eats hits before life does.
        skill_ready: (p.skillCd ?? 0) <= 0,
        skill_cd: p.skillCd ?? 0,
        shield_hits: p.shieldHits ?? 0,
        dash_ready: (p.dashCd ?? 0) <= 0 && (p.dashT ?? 0) <= 0,
        dash_cd: p.dashCd ?? 0,
        i_frames: p.iFrames ?? 0,
        damage: p.stats?.damage ?? 1,
        fire_delay: p.stats?.fireDelay ?? 0.21,
        move_speed: p.stats?.moveSpeed ?? 290,
        // Shots dissipate after this distance (PLAYER_CFG.shotRange, 470 px
        // before relics). Anything fired from further away never arrives.
        shot_range: p.stats?.range ?? 470,
        shot_speed: p.stats?.shotSpeed ?? 580,
        target_room_type: targetRoomType,
        target_room_hops: targetRoomHops,
        room_key: node ? `${g.floorNum}:${node.gx},${node.gy}` : null,
        // Updated on every hit, not only on the fatal one: this is what the
        // death audit reads to attribute each lost heart to a culprit.
        killer: g.killerName || null,
        killer_was_boss: !!g.killerWasBoss,
        // — added with the mouse aim —
        // Time before the next orb can leave, the hero's own velocity,
        // the dash in progress and the swamp's slow: what a shot or a
        // dodge is worth right now.
        fire_cd: Math.max(0, p.fireT ?? 0),
        player_vx: p.vx ?? 0,
        player_vy: p.vy ?? 0,
        dash_t: p.dashT ?? 0,
        slow_t: p.slowT ?? 0,
        // The build: every relic carried, and the synergies it lit.
        items: (p.items || []).map((it) => ({
            id: it.id || null, rarity: it.rarity || null, mods: itemMods(it),
        })),
        synergies: p.syn ? Array.from(p.syn) : [],
        mods: p.mods ? { ...p.mods } : {},
        // The room's remaining decisions and their state.
        second_wave: !!node?.secondWave && !node?.secondWaveDone,
        gambles_left: node?.gamblesLeft === undefined ? 3 : node.gamblesLeft,
        defi_done: !!node?.defiDone,
        defi_active: !!g.defiActive,
        defi_t: g.defiT ?? 0,
        in_transition: !!g.transition,
        // The biome's own hazards: slowing patches (brambles, mud) and the
        // ruins' gust, read from the strata module.
        strata_zones: (window.__eosStrates?.zones || []).map((z) => ({
            x: z.x, y: z.y, r: z.r, distance: dist(z.x, z.y),
        })),
        wind: window.__eosStrates?.wind ? {
            dir: [window.__eosStrates.wind.dirX, window.__eosStrates.wind.dirY],
            t: window.__eosStrates.wind.t,
            windup: window.__eosStrates.wind.windup,
            dur: window.__eosStrates.wind.dur,
            push: window.__eosStrates.wind.push,
        } : null,
        // Orbs by bearing: for each of 16 sectors around the player, how
        // soon the most pressing orb heading this way arrives (1 = now,
        // 0 = none within a second), over EVERY orb on screen.
        shot_sectors: shotSectors,
    };
};
"""


# One synchronous env step: press the keys, advance the fixed-step
# simulation, release, optionally draw, and read the new state — a single
# browser round-trip instead of key events plus a real-time sleep.
STEP_TURBO_JS = """({ codes, frames, render, aim }) => {
    // Mouse aim: the game fires toward the cursor while the button is held
    // and no arrow key is down (player.js). A far point keeps the direction
    // stable over the 43 px the hero can cover during one step. Only in
    // play: the title menu reacts to the cursor position.
    const m = window.__eosDebug.mouse;
    const g = window.__eosGame;
    if (aim && g.state === 'play') {
        m.x = g.player.x + aim[0] * 5000;
        m.y = g.player.y + aim[1] * 5000;
        m.down = true;
    } else {
        m.down = false;
    }
    for (const code of codes) window.dispatchEvent(new KeyboardEvent('keydown', { code }));
    window.__eosAdvance(frames, 1 / 120);
    for (const code of codes) window.dispatchEvent(new KeyboardEvent('keyup', { code }));
    m.down = false;
    if (render) window.__eosRender();
    return window.__eosObserve();
}"""


# Training-only curriculum: how equipped a run typically is when it reaches a
# given floor, so a jumped-in episode fights with a realistic build instead of
# teaching the network to survive naked against a tier-4 guardian.
# Floor 2 is the median measured by scripts/audit_deaths.py (2 items, 40 gold,
# half life); floors 3 to 5 are extrapolated from it and must be re-calibrated
# once the teacher reaches them often enough to measure.
CURRICULUM_ITEMS = {1: 0, 2: 2, 3: 4, 4: 6, 5: 8}
CURRICULUM_GOLD = {1: 0, 2: 40, 3: 45, 4: 50, 5: 55}
# Runs do not arrive at a new floor healed up — attrition is the whole point of
# the genre, and the measured arrival on floor 2 is 3 life out of 6.
CURRICULUM_HP_RATIO = 0.6

# Below this displacement in one step, the player is considered not to have
# moved: a move key was held and something absorbed it (a wall, a prop).
STUCK_TRAVEL = 8.0


# Jump straight to a floor with a plausible build, using the game's own item
# roller so the loot stays inside the rarity cap of that depth.
JUMP_TO_FLOOR_JS = """({ floor, items, gold, hpRatio }) => {
    const g = window.__eosGame;
    const dbg = window.__eosDebug;
    g.startFloor(floor);
    const owned = new Set(g.player.items.map((it) => it.id));
    for (let i = 0; i < items; i++) {
        const it = dbg.rollItem(g.rng, owned, {
            maxRarity: g.rarityCap(),
            noCursed: true,
        });
        owned.add(it.id);
        g.player.applyItem(it);
    }
    g.player.gold = gold;
    g.player.hp = Math.max(1, Math.round(g.player.maxHp * hpRatio));
}"""


@dataclass(frozen=True)
class Observation:
    state: str
    hp: float
    max_hp: float
    x: float
    y: float
    floor: int
    kills: int
    rooms: int
    floors: int
    time: float
    hp_missing: float
    gold: int
    item_count: int
    room_type: str | None
    room_cleared: bool
    abyss_gate: bool
    doors_open: bool
    portal_active: bool
    enemy_count: int
    pickup_count: int
    nearest_enemy: dict[str, Any] | None
    nearest_pickup: dict[str, Any] | None
    nearest_shop_item: dict[str, Any] | None
    shop_items: list[dict[str, Any]]
    nearest_door: dict[str, Any] | None
    target_door: dict[str, Any] | None
    near_choice: dict[str, Any] | None
    room_focus: dict[str, Any] | None
    portal: dict[str, Any] | None
    available_doors: list[dict[str, Any]]
    pickups: list[dict[str, Any]]
    enemies: list[dict[str, Any]]
    shots: list[dict[str, Any]]
    hazards: list[dict[str, Any]]
    spawning: list[dict[str, Any]]
    target_room_type: str | None
    target_room_hops: int
    killer: str | None
    killer_was_boss: bool
    danger_field: list[float]
    obstacles: list[dict[str, Any]]
    door_los: bool
    room_w: float
    room_h: float
    room_wt: float
    skill_ready: bool
    skill_cd: float
    shield_hits: int
    dash_ready: bool
    dash_cd: float
    i_frames: float
    damage: float
    fire_delay: float
    move_speed: float
    shot_range: float
    shot_speed: float
    room_key: str | None
    # Filled in by the environment, which is the only place that sees the
    # sequence of observations: consecutive steps spent without moving, and
    # steps spent in the current room. A run that stops progressing burns its
    # budget exactly as surely as one that dies, and nothing in a single
    # snapshot can show it.
    stuck_steps: int = 0
    room_steps: int = 0
    # Read since the mouse aim (see __eosObserve): the hero's own timers
    # and velocity, the build, the room's pending decisions, the biome's
    # strata and the orbs by bearing.
    fire_cd: float = 0.0
    player_vx: float = 0.0
    player_vy: float = 0.0
    dash_t: float = 0.0
    slow_t: float = 0.0
    items: list[dict[str, Any]] = field(default_factory=list)
    synergies: list[str] = field(default_factory=list)
    mods: dict[str, Any] = field(default_factory=dict)
    second_wave: bool = False
    gambles_left: int = 3
    defi_done: bool = False
    defi_active: bool = False
    defi_t: float = 0.0
    in_transition: bool = False
    strata_zones: list[dict[str, Any]] = field(default_factory=list)
    wind: dict[str, Any] | None = None
    shot_sectors: list[float] = field(default_factory=list)


class EclipseEnv:
    def __init__(
        self,
        *,
        headless: bool = True,
        slow_mo: int = 0,
        step_seconds: float = 0.15,
        turbo: bool = False,
    ) -> None:
        self.headless = headless
        self.slow_mo = slow_mo
        self.step_seconds = step_seconds
        # Turbo drives the game's fixed-step loop synchronously instead of
        # holding keys in real time: same simulated duration per step, but
        # the wall-clock cost drops to the simulation itself.
        self.turbo = turbo
        self._frames_per_step = max(1, round(step_seconds * SIM_RATE))
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self.page: Page | None = None
        self._last_obs: Observation | None = None
        self.seed: int | None = None

    def __enter__(self) -> "EclipseEnv":
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def start(self) -> None:
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            headless=self.headless,
            slow_mo=self.slow_mo,
        )
        # ignore_https_errors: tolerate TLS-inspecting proxies whose root
        # certificate Chromium does not know (blocks goto and route.fetch).
        self._context = self._browser.new_context(
            viewport={"width": 1280, "height": 720},
            ignore_https_errors=True,
        )
        if self.turbo:
            self._context.add_init_script("window.__eosTurbo = true;")
        self._context.add_init_script(CLEAR_SAVE_JS)
        self._context.add_init_script(SEED_SETUP_JS)
        self._context.add_init_script(OBSERVE_SETUP_JS)
        self.page = self._context.new_page()
        self.page.route("**/js/main.js", self._patch_main_js)
        self.page.route("**/js/game/strates.js", self._patch_strates_js)
        self._last_obs = None

    def close(self) -> None:
        # Tolerant teardown so a crashed browser can be restarted cleanly.
        for closer in (
            lambda: self._context.close() if self._context else None,
            lambda: self._browser.close() if self._browser else None,
            lambda: self._playwright.stop() if self._playwright else None,
        ):
            try:
                closer()
            except Exception:
                pass
        self._context = None
        self._browser = None
        self._playwright = None
        self.page = None
        self._last_obs = None

    def reset(
        self,
        *,
        start_floor: int = 1,
        start_items: int | None = None,
        start_gold: int | None = None,
        seed: int | None = None,
    ) -> Observation:
        """Start a fresh run.

        `start_floor` > 1 is a TRAINING-ONLY device: it drops the run straight
        into a deeper floor with a plausible build, so demonstrations can cover
        the guardians a normal run reaches once every few hours. Evaluation
        always calls reset() with no argument and asserts floor == 1.

        `seed` fixes the dungeon (layout, waves, loot) and the combat dice:
        two policies played on the same seeds face the same runs, which is
        the only way to compare them without drowning the difference in the
        variance between dungeons. None keeps the game's own clock seed.
        """
        page = self._require_page()
        self._last_obs = None
        self.seed = seed
        url = GAME_URL
        if seed is not None:
            # A fragment-only change is a same-document navigation for the
            # browser: pass through a blank page to force a real reload.
            page.goto("about:blank")
            url = f"{GAME_URL}#eos-seed={int(seed) & 0xFFFFFFFF}"
        page.goto(url, wait_until="domcontentloaded")
        self._wait_for_game()
        if self.turbo:
            self._advance_to_state(("heroSelect", "play"))
            self._advance_to_state(("play",))
            self._advance(int(0.3 * SIM_RATE))
        else:
            page.keyboard.press("Space")
            self._wait_for_state("heroSelect")
            page.keyboard.press("Space")
            self._wait_for_state("play")
            time.sleep(0.3)
        if start_floor > 1:
            self._jump_to_floor(start_floor, start_items, start_gold)
        return self.observe()

    def _jump_to_floor(
        self, floor: int, items: int | None, gold: int | None
    ) -> None:
        if items is None:
            items = CURRICULUM_ITEMS.get(floor, 0)
        if gold is None:
            gold = CURRICULUM_GOLD.get(floor, 0)
        self._require_page().evaluate(
            JUMP_TO_FLOOR_JS,
            {
                "floor": floor,
                "items": items,
                "gold": gold,
                "hpRatio": CURRICULUM_HP_RATIO,
            },
        )
        if self.turbo:
            self._advance(int(0.3 * SIM_RATE))
        else:
            time.sleep(0.3)

    def step(
        self, action: str, aim: tuple[float, float] | None = None
    ) -> tuple[Observation, float, bool, dict[str, Any]]:
        """Hold the keys of `action` for one step; `aim` fires at the mouse.

        `aim` is a unit vector in game coordinates (y downward). The game
        gives the arrow keys priority over the cursor, so an action that
        already shoots cannot take an aim: refusing it beats a silent no-op.
        """
        if action not in ACTIONS:
            raise ValueError(f"Unknown action {action!r}. Available actions: {sorted(ACTIONS)}")
        if aim is not None:
            if any(code.startswith("Arrow") for code in ACTIONS[action]):
                raise ValueError(f"{action!r} shoots with the arrows; it cannot take an aim.")
            norm = (aim[0] ** 2 + aim[1] ** 2) ** 0.5
            aim = None if norm < 1e-6 else (aim[0] / norm, aim[1] / norm)

        # Reuse the previous snapshot as "before": a single evaluate per step
        # roughly doubles the real-time step rate.
        before = self._last_obs if self._last_obs is not None else self.observe()
        if self.turbo:
            after = self._step_turbo(ACTIONS[action], aim)
        else:
            if aim is not None:
                self._aim_realtime(aim)
            self._hold_keys(ACTIONS[action], self.step_seconds)
            if aim is not None:
                self._require_page().mouse.up()
            after = self.observe()
        reward = self._reward(before, after)
        done = after.state in TERMINAL_STATES
        return after, reward, done, {"action": action, "aim": aim}

    def observe(self) -> Observation:
        data = self._eval_game("() => window.__eosObserve()")
        return self._track(Observation(**data))

    def _track(self, obs: Observation) -> Observation:
        """Attach the two counters only a sequence of observations can give."""
        previous = self._last_obs
        if previous is None or obs.room_key != previous.room_key:
            stuck, room_steps = 0, 0
        else:
            moved = ((obs.x - previous.x) ** 2 + (obs.y - previous.y) ** 2) ** 0.5
            stuck = previous.stuck_steps + 1 if moved < STUCK_TRAVEL else 0
            room_steps = previous.room_steps + 1
        obs = replace(obs, stuck_steps=stuck, room_steps=room_steps)
        self._last_obs = obs
        return obs

    @staticmethod
    def compute_score(obs: Observation) -> float:
        return obs.floors * 100 + obs.rooms * 10 + obs.kills + obs.time * 0.1

    def get_score(self) -> float:
        obs = self._last_obs if self._last_obs is not None else self.observe()
        return self.compute_score(obs)

    def is_done(self) -> bool:
        obs = self._last_obs if self._last_obs is not None else self.observe()
        return obs.state in TERMINAL_STATES

    def _step_turbo(
        self, codes: list[str], aim: tuple[float, float] | None = None
    ) -> Observation:
        data = self._require_page().evaluate(
            STEP_TURBO_JS,
            {
                "codes": codes,
                "frames": self._frames_per_step,
                "render": not self.headless,
                "aim": list(aim) if aim is not None else None,
            },
        )
        return self._track(Observation(**data))

    def _aim_realtime(self, aim: tuple[float, float]) -> None:
        """Real-time counterpart of the turbo aim: move the cursor, press.

        Playwright clips the cursor to the viewport, so the point is kept
        inside the canvas rather than far away (input.js maps client
        coordinates back to the 1280x720 game view).
        """
        page = self._require_page()
        rect = page.evaluate(
            "() => { const c = document.getElementById('game').getBoundingClientRect();"
            " const p = window.__eosGame.player;"
            " return [c.left, c.top, c.width, c.height, p.x, p.y]; }"
        )
        left, top, width, height, px, py = rect
        gx = min(max(px + aim[0] * 600, 1), 1279)
        gy = min(max(py + aim[1] * 600, 1), 719)
        page.mouse.move(left + gx / 1280 * width, top + gy / 720 * height)
        page.mouse.down()

    def _advance(self, frames: int) -> None:
        self._require_page().evaluate(
            "(frames) => window.__eosAdvance(frames, 1 / 120)", frames
        )

    def _advance_to_state(self, states: tuple[str, ...], *, max_frames: int = 2400) -> None:
        # Menus only react while the simulation runs, so tap the key and
        # advance in small chunks until one of the requested states shows up.
        page = self._require_page()
        advanced = 0
        while True:
            current = page.evaluate("() => window.__eosGame.state")
            if current in states:
                return
            if advanced >= max_frames:
                raise TimeoutError(f"Expected game state {states!r}, got {current!r}.")
            page.evaluate(
                """(code) => {
                    window.dispatchEvent(new KeyboardEvent('keydown', { code }));
                    window.__eosAdvance(2, 1 / 120);
                    window.dispatchEvent(new KeyboardEvent('keyup', { code }));
                    window.__eosAdvance(10, 1 / 120);
                }""",
                "Space",
            )
            advanced += 12

    def _hold_keys(self, keys: list[str], seconds: float) -> None:
        page = self._require_page()
        for key in keys:
            page.keyboard.down(key)
        time.sleep(seconds)
        for key in reversed(keys):
            page.keyboard.up(key)

    def _reward(self, before: Observation, after: Observation) -> float:
        return (
            (after.floors - before.floors) * 100
            + (after.rooms - before.rooms) * 10
            + (after.kills - before.kills)
            + max(0.0, after.time - before.time) * 0.1
            - max(0.0, before.hp - after.hp) * 2
        )

    def _eval_game(self, expression: str) -> Any:
        page = self._require_page()
        return page.evaluate(expression)

    def _wait_for_game(self, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._require_page().evaluate(
                "() => !!window.__eosGame && (!window.__eosTurbo || !!window.__eosAdvance)"
            ):
                return
            time.sleep(0.05)
        raise TimeoutError("Game instance was not exposed as window.__eosGame.")

    def _wait_for_state(self, state: str, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            current = self._require_page().evaluate("() => window.__eosGame && window.__eosGame.state")
            if current == state:
                return
            time.sleep(0.05)
        current = self._require_page().evaluate("() => window.__eosGame && window.__eosGame.state")
        raise TimeoutError(f"Expected game state {state!r}, got {current!r}.")

    def _require_page(self) -> Page:
        if self.page is None:
            raise RuntimeError("Environment is not started. Use EclipseEnv as a context manager.")
        return self.page

    @staticmethod
    def _patch_strates_js(route: Any) -> None:
        # The biome strata (slowing patches, the ruins' gust) live in a
        # module-private object; hand it to the observer.
        response = route.fetch()
        source = response.text()
        patched = source + "\nif (typeof window !== 'undefined') window.__eosStrates = state;\n"
        route.fulfill(
            response=response,
            body=patched,
            headers={**response.headers, "content-type": "application/javascript"},
        )

    @staticmethod
    def _patch_main_js(route: Any) -> None:
        response = route.fetch()
        source = response.text()
        patched = source.replace(
            "import { Game } from './game/game.js';",
            "import { Game } from './game/game.js';\n"
            "import { enemies } from './game/enemy.js';\n"
            "import { pickups } from './game/pickup.js';\n"
            "import { enemyShots } from './game/enemyshot.js';\n"
            "import { hazards } from './game/boss.js';\n"
            "import { rollItem } from './data/items.js';\n"
            "import { consommer } from './core/hitstop.js';\n"
            "import { getMouse } from './core/input.js';",
            1,
        )
        patched = patched.replace(
            "const game = new Game(view);",
            "const game = new Game(view); "
            "window.__eosGame = game; "
            "window.__eosDebug = { enemies, pickups, enemyShots, hazards, rollItem, mouse: getMouse() }; "
            # The hero-select screen starts the run without a seed: hand it
            # the one from the URL so the dungeon is the requested one.
            "game.startRun = ((orig) => function (seed) { "
            "return orig.call(this, seed ?? window.__eosRunSeed); "
            "})(game.startRun);",
            1,
        )
        # Audio adds startup cost and has no effect on the simulation.
        patched = patched.replace(
            "initInput(() => { initAudio(); startMusic(); });",
            "initInput(window.__eosTurbo ? null : () => { initAudio(); startMusic(); });",
            1,
        )
        # Expose the fixed-step loop so turbo mode can drive the simulation
        # synchronously (same hitstop behaviour as the real loop). Real-time
        # mode keeps the original requestAnimationFrame loop.
        patched = patched.replace(
            "  startLoop({\n"
            "    update: (dt) => { pollGamepad(); game.update(dt); },\n"
            "    frameEnd: () => endFrame(),\n"
            "    render: () => game.render(ctx),\n"
            "  });",
            "  const loopHooks = {\n"
            "    update: (dt) => { pollGamepad(); game.update(dt); },\n"
            "    frameEnd: () => endFrame(),\n"
            "    render: () => game.render(ctx),\n"
            "  };\n"
            "  window.__eosAdvance = (frames, dt) => {\n"
            "    for (let i = 0; i < frames; i += 1) {\n"
            "      if (!consommer()) loopHooks.update(dt);\n"
            "    }\n"
            "    loopHooks.frameEnd();\n"
            "  };\n"
            "  window.__eosRender = () => loopHooks.render(0);\n"
            "  if (!window.__eosTurbo) startLoop(loopHooks);",
            1,
        )
        route.fulfill(
            response=response,
            body=patched,
            headers={**response.headers, "content-type": "application/javascript"},
        )
