"""Feature vector of the phase 5 agent.

Rule: everything the teacher consults is a feature. The phase 4 vector
(features.py, 215 values) left out what the rules read most - the enemy's
family, the elite's affix, how far along a telegraph is, the guardian's
attack by name, the build carried - and the clone could only guess them
from a silhouette's speed. This vector names them. It also carries the aim
the shared module chose, so the network knows which enemy the orbs are
flying at while it decides where to step.

Normalisation uses the game's constants (config.js, enemies.js), never
statistics of a data set: the demonstrations are stored in float16.
"""

from __future__ import annotations

import math

from src import bestiary, economy
from src.eos_env import Observation
from src.features import DIAG, N_DANGER, ROOM_H, ROOM_TYPES, ROOM_W, STEP_TRAVEL


FEATURE_VERSION = "v2"

FAMILIES = ["chase", "charge", "kite", "bomb", "summon", "shield", "flee", "boss"]
AFFIXES = ["explosive", "vampirique", "glaciale", "invocatrice", "spectrale"]
BOSS_ATTACKS = [
    "heavyCharge", "tripleLunge", "ringVolley", "doubleRing", "spiralBurst", "fanVolley",
    "blinkAway", "summonWave", "slamQuake", "aftershock", "brambleRing", "sweepBeam",
    "magmaWake", "rockRain", "frostNova", "mirrorEcho", "shatterClone", "tidalPulse",
]
INVENTORY_FLAGS = ["orbs", "homing", "multishot", "pierce", "chain", "bounces", "ghost", "drones"]
TARGET_TYPES = ["treasure", "shop", "miniboss", "boss"]
N_ENEMIES = 5
N_SHOTS = 4
N_HAZARDS = 4
N_OBSTACLES = 4
N_PICKUPS = 2


def _clip(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _entity(obs: Observation, entity: dict | None, extra: list[float]) -> list[float]:
    if not entity:
        return [0.0] * (4 + len(extra))
    return [
        1.0,
        (entity["x"] - obs.x) / ROOM_W,
        (entity["y"] - obs.y) / ROOM_H,
        entity.get("distance", DIAG) / DIAG,
        *extra,
    ]


def _enemy_extra(obs: Observation, e: dict) -> list[float]:
    row = bestiary.info(e)
    kind = row["behaviour"]
    max_hp = e.get("max_hp") or 1.0
    state = e.get("state")
    attack = e.get("attack")
    vulnerable = attack == "heavyCharge" and e.get("attack_step", 0) >= 2
    windup = state in {"aim", "windup", "prime", "dash", "channel"} or (
        state == "attack" and attack in {"tripleLunge", "heavyCharge", "slamQuake"} and not vulnerable
    )
    dx, dy = e["x"] - obs.x, e["y"] - obs.y
    d = math.hypot(dx, dy) or 1.0
    # closing speed toward the player -> time before contact, clipped
    closing = -(e.get("vx", 0.0) * dx + e.get("vy", 0.0) * dy) / d
    reach = d - e.get("radius", row["radius"]) - 15.0
    contact_t = reach / closing if closing > 1.0 else 3.0
    out = [
        e.get("hp", 0.0) / max_hp,
        1.0 if e.get("elite") else 0.0,
        1.0 if e.get("boss") else 0.0,
        1.0 if windup else 0.0,
        1.0 if vulnerable else 0.0,
        1.0 if e.get("los", True) else 0.0,
        _clip(e.get("vx", 0.0) / 400.0),
        _clip(e.get("vy", 0.0) / 400.0),
    ]
    out.extend(1.0 if kind == f else 0.0 for f in FAMILIES)
    out.extend(1.0 if e.get("affix") == a else 0.0 for a in AFFIXES)
    out.extend([
        e.get("radius", row["radius"]) / 50.0,
        min(e.get("contact_damage", row["contact"]), 3) / 3.0,
        1.0 if row.get("death_shots") else 0.0,
        1.0 if row.get("death_hazard") else 0.0,
        1.0 if row.get("flies") else 0.0,
        min(e.get("state_t", 0.0), 1.2) / 1.2,
        min(max(e.get("shoot_cd", 0.0), 0.0), 5.0) / 5.0,
        _clip(contact_t / 3.0, 0.0, 1.0),
        1.0 if e.get("ghosting") else 0.0,
        1.0 if e.get("front_shield") else 0.0,
    ])
    return out


N_ENEMY_EXTRA = 8 + len(FAMILIES) + len(AFFIXES) + 10


def featurize(
    obs: Observation,
    previous: Observation | None = None,
    aim: tuple[float, float] | None = None,
    target: dict | None = None,
) -> list[float]:
    """Flatten an observation, plus the aim the shared module chose."""
    if obs.state != "play":
        return [0.0] * N_FEATURES

    f: list[float] = [
        obs.x / ROOM_W,
        obs.y / ROOM_H,
        obs.hp / 16.0,
        obs.max_hp / 16.0,
        obs.hp_missing / max(1.0, obs.max_hp),
        min(obs.gold, 200) / 200.0,
        min(obs.item_count, 12) / 12.0,
        1.0 if obs.doors_open else 0.0,
        1.0 if obs.portal_active else 0.0,
        1.0 if obs.room_cleared else 0.0,
        1.0 if obs.abyss_gate else 0.0,
        min(obs.enemy_count, 8) / 8.0,
        min(obs.pickup_count, 8) / 8.0,
        obs.floor / 5.0,
        1.0 if obs.skill_ready else 0.0,
        min(obs.skill_cd, 10.0) / 10.0,
        min(obs.shield_hits, 2) / 2.0,
        1.0 if obs.dash_ready else 0.0,
        min(obs.dash_cd, 1.2) / 1.2,
        min(obs.i_frames, 0.6) / 0.6,
        min(obs.damage, 6.0) / 6.0,
        min(obs.fire_delay, 0.4) / 0.4,
        min(obs.move_speed, 450.0) / 450.0,
        min(obs.shot_range, 900.0) / 900.0,
        min(obs.shot_speed, 1100.0) / 1100.0,
        min(obs.fire_cd, 0.21) / 0.21,
        _clip(obs.dash_t / 0.13, 0.0, 1.0),
        _clip(obs.slow_t / 1.2, 0.0, 1.0),
        _clip(obs.player_vx / 450.0),
        _clip(obs.player_vy / 450.0),
        aim[0] if aim else 0.0,
        aim[1] if aim else 0.0,
        1.0 if aim else 0.0,
        1.0 if obs.second_wave else 0.0,
    ]
    f.extend(1.0 if obs.room_type == kind else 0.0 for kind in ROOM_TYPES)

    # which enemy slot the aim module picked (none = out of combat)
    slot = -1
    if target is not None:
        for i, e in enumerate(obs.enemies[:N_ENEMIES]):
            if e is target or (e.get("x") == target.get("x") and e.get("y") == target.get("y")):
                slot = i
                break
    f.extend(1.0 if slot == i else 0.0 for i in range(N_ENEMIES))
    f.append(1.0 if slot < 0 else 0.0)

    for i in range(N_ENEMIES):
        e = obs.enemies[i] if i < len(obs.enemies) else None
        f.extend(_entity(obs, e, _enemy_extra(obs, e) if e else [0.0] * N_ENEMY_EXTRA))

    # the guardian on screen, by attack name
    boss = next((e for e in obs.enemies if e.get("boss")), None)
    attack = boss.get("attack") if boss else None
    f.extend(1.0 if attack == name else 0.0 for name in BOSS_ATTACKS)
    f.extend([
        min(boss.get("attack_time", 0.0), 2.0) / 2.0 if boss else 0.0,
        min(boss.get("attack_step", 0), 6) / 6.0 if boss else 0.0,
        min(boss.get("phase", 0), 3) / 3.0 if boss else 0.0,
        1.0 if boss and boss.get("arena_on") else 0.0,
    ])

    for i in range(N_PICKUPS):
        p = obs.pickups[i] if i < len(obs.pickups) else None
        if p:
            kind = p.get("type")
            extra = [
                1.0 if kind in {"heart", "heartHalf"} else 0.0,
                1.0 if kind == "item" else 0.0,
                1.0 if kind == "gold" else 0.0,
                _clip(economy.item_value(obs, p), -1.0, 4.0) / 4.0 if kind == "item" else 0.0,
            ]
        else:
            extra = [0.0] * 4
        f.extend(_entity(obs, p, extra))

    f.extend(_entity(obs, obs.target_door, [1.0 if obs.door_los else 0.0]))
    f.extend(_entity(obs, obs.portal, []))
    for i in range(N_OBSTACLES):
        o = obs.obstacles[i] if i < len(obs.obstacles) else None
        f.extend(_entity(obs, o, [o["r"] / 90.0] if o else [0.0]))

    for i in range(N_SHOTS):
        sh = obs.shots[i] if i < len(obs.shots) else None
        if sh:
            f.extend([
                1.0, (sh["x"] - obs.x) / ROOM_W, (sh["y"] - obs.y) / ROOM_H,
                sh.get("distance", DIAG) / DIAG, _clip(sh["vx"] / 320.0), _clip(sh["vy"] / 320.0),
            ])
        else:
            f.extend([0.0] * 6)
    sectors = obs.shot_sectors
    f.extend(sectors if len(sectors) == 16 else [0.0] * 16)

    for i in range(N_HAZARDS):
        hz = obs.hazards[i] if i < len(obs.hazards) else None
        if hz:
            f.extend([
                1.0, (hz["x"] - obs.x) / ROOM_W, (hz["y"] - obs.y) / ROOM_H,
                hz.get("distance", DIAG) / DIAG, min(hz["r"], 140.0) / 140.0,
                min(hz.get("remaining", 0.0), 1.3) / 1.3,
            ])
        else:
            f.extend([0.0] * 6)

    f.extend([1.0 if obs.near_choice else 0.0,
              obs.near_choice.get("distance", DIAG) / DIAG if obs.near_choice else 0.0])
    focus = obs.room_focus
    f.extend(_entity(obs, focus, [
        1.0 if focus and focus.get("kind") == k else 0.0 for k in ("altar", "gambler", "defi")
    ] if focus else [0.0] * 3))
    shop = obs.nearest_shop_item
    f.extend(_entity(obs, shop, [
        1.0 if shop.get("price", 0) <= obs.gold else 0.0,
        _clip(economy.item_value(obs, shop), -1.0, 4.0) / 4.0 if shop.get("kind") == "item" else 0.0,
    ] if shop else [0.0, 0.0]))
    spawn = obs.spawning[0] if obs.spawning else None
    f.extend([
        min(len(obs.spawning), 3) / 3.0,
        1.0 if spawn else 0.0,
        (spawn["x"] - obs.x) / ROOM_W if spawn else 0.0,
        (spawn["y"] - obs.y) / ROOM_H if spawn else 0.0,
    ])
    f.extend(1.0 if obs.target_room_type == t else 0.0 for t in TARGET_TYPES)
    f.append(min(obs.target_room_hops, 6) / 6.0)

    if previous is not None and previous.state == "play":
        travel = math.hypot(obs.x - previous.x, obs.y - previous.y)
        speed = min(travel / STEP_TRAVEL, 1.0)
        f.extend([speed, 1.0 if speed < 0.05 else 0.0])
    else:
        f.extend([0.0, 0.0])
    f.extend([min(obs.stuck_steps, 12) / 12.0, min(obs.room_steps, 900) / 900.0])

    mods = obs.mods or {}
    f.extend(min(mods.get(flag, 0), 3) / 3.0 for flag in INVENTORY_FLAGS)

    field = obs.danger_field
    f.extend(field if len(field) == N_DANGER else [0.0] * N_DANGER)
    return f


def _count() -> int:
    from src.features import _dummy_observation

    return len(featurize(_dummy_observation()))


N_FEATURES = 0
N_FEATURES = _count()
