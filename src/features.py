from __future__ import annotations

import math

from src.eos_env import Observation


ROOM_W = 1280.0
ROOM_H = 720.0
DIAG = (ROOM_W**2 + ROOM_H**2) ** 0.5

# Depth of the final guardian (js/data/difficulty.js).
FINAL_FLOOR = 5

ROOM_TYPES = [
    "start",
    "combat",
    "shop",
    "treasure",
    "boss",
    "altar",
    "defi",
    "scelle",
    "gambler",
]

# Distance covered in one step at full speed, used to normalise the motion
# feature (a value near zero while a move key is held means "stuck").
STEP_TRAVEL = 60.0

# 16 directions x 3 travel distances, produced by the environment.
N_DANGER = 48


# Keyboard aim is quantised to eight rays, 45 deg apart, and orbs travel at
# 580 px/s. Both facts are geometry of the observation rather than policy, and
# both the feature vector and the rule-based policy read them.
AIM_SECTOR = math.radians(45.0)
SHOT_RADIUS = 5.5


def aim_miss(obs: Observation, target: dict) -> float:
    """Lateral distance by which the closest firing ray passes the target.

    Measured on the target's current position, not on where an orb would
    meet it: leading the shot over the flight time was tried in V38 and cost
    0.09 life per room. Chasers accelerate at 850 px/s2 and wobble, so their
    instantaneous velocity predicts half a second badly enough to move the
    aim point more than it corrects it.
    """
    dx = target["x"] - obs.x
    dy = target["y"] - obs.y
    distance = math.hypot(dx, dy)
    if distance < 1e-6:
        return 0.0
    bearing = math.atan2(dy, dx)
    error = bearing - round(bearing / AIM_SECTOR) * AIM_SECTOR
    return abs(math.sin(error)) * distance


_RARITY_RANK = {
    None: 0,
    "common": 1,
    "rare": 2,
    "epic": 3,
    "legendary": 4,
}


# Life is the binding resource of a run: six half-hearts for forty-six rooms,
# and healing is rare and perishable. A relic that adds max life is therefore
# worth more than the same relative gain in damage, which only shortens the
# fights.
LIFE_WEIGHT = 1.6


def item_value(obs: Observation, entry: dict) -> float:
    """What a relic is worth to THIS run, rather than how rare it is.

    Rarity is a decent proxy and it was the only criterion until now, but the
    bank holds 159 items whose effects differ by a factor of four: +0.3 damage
    on a base of 1 is worth thirty times a range bonus, and the +4 max life of
    a single relic is worth two thirds of the whole life bar. Both are read
    from the game's own modifiers.
    """
    mods = entry.get("item_mods")
    if not mods:
        return 0.25 * _RARITY_RANK.get(entry.get("item_rarity"), 0)

    damage = max(0.1, obs.damage)
    after = (damage + mods.get("dmg_add", 0.0)) * mods.get("dmg_mult", 1.0)
    dps_gain = after / damage / max(0.2, mods.get("fire_mult", 1.0)) - 1.0
    life_gain = mods.get("hp_add", 0.0) / max(1, obs.max_hp)
    move_gain = mods.get("move_mult", 1.0) - 1.0
    # Everything else the bank offers (range, shot size, crit, on-kill effects)
    # has no direct term here; rarity stands in for it, at a weight small
    # enough that it never outranks a real stat.
    return (
        dps_gain
        + LIFE_WEIGHT * life_gain
        + 0.5 * move_gain
        + 0.05 * _RARITY_RANK.get(entry.get("item_rarity"), 0)
    )


def is_combat_state(obs: Observation) -> bool:
    """Is something threatening the player right now?

    Used to keep every fighting state in the demonstration sets while the
    highly redundant travel states get subsampled.
    """
    return bool(obs.enemy_count or obs.shots or obs.hazards or obs.spawning)


def _entity(obs: Observation, entity: dict | None, extra: list[float] | None = None) -> list[float]:
    """present flag + normalised offset and distance toward an entity."""
    if not entity:
        return [0.0, 0.0, 0.0, 0.0] + [0.0] * len(extra or [])
    values = [
        1.0,
        (entity["x"] - obs.x) / ROOM_W,
        (entity["y"] - obs.y) / ROOM_H,
        entity.get("distance", DIAG) / DIAG,
    ]
    return values + (extra or [])


def featurize(obs: Observation, previous: Observation | None = None) -> list[float]:
    """Flatten an observation into a normalised feature vector."""
    if obs.state != "play":
        return [0.0] * N_FEATURES

    features: list[float] = [
        obs.x / ROOM_W,
        obs.y / ROOM_H,
        obs.hp / obs.max_hp if obs.max_hp else 0.0,
        obs.hp_missing / obs.max_hp if obs.max_hp else 0.0,
        min(obs.gold, 200) / 200.0,
        min(obs.item_count, 12) / 12.0,
        float(obs.doors_open),
        float(obs.portal_active),
        float(obs.room_cleared),
        float(obs.abyss_gate),
        min(obs.enemy_count, 8) / 8.0,
        min(obs.pickup_count, 8) / 8.0,
        # Depth drives every difficulty curve of the game (enemy life x4.2,
        # shot damage, guardian tier), so the same situation calls for a
        # different answer on floor 1 and on floor 5.
        min(obs.floor, FINAL_FLOOR) / FINAL_FLOOR,
        # What the player can do right now, beyond moving and shooting.
        float(obs.skill_ready),
        min(obs.skill_cd, 10.0) / 10.0,
        min(obs.shield_hits, 2) / 2.0,
        float(obs.dash_ready),
        min(obs.dash_cd, 1.2) / 1.2,
        min(obs.i_frames, 0.6) / 0.6,
        min(obs.damage, 6.0) / 6.0,
        min(obs.fire_delay, 0.4) / 0.4,
        min(obs.move_speed, 450.0) / 450.0,
        min(obs.shot_range, 900.0) / 900.0,
    ]

    features.extend(1.0 if obs.room_type == kind else 0.0 for kind in ROOM_TYPES)

    for i in range(3):
        enemy = obs.enemies[i] if i < len(obs.enemies) else None
        extra = None
        if enemy:
            max_hp = enemy.get("max_hp") or 1
            state = enemy.get("state")
            attack = enemy.get("attack")
            vulnerable = attack == "heavyCharge" and enemy.get("attack_step", 0) >= 2
            windup = (
                state in {"aim", "windup", "prime", "dash"}
                or (
                    state == "attack"
                    and attack in {"tripleLunge", "heavyCharge", "slamQuake"}
                    and not vulnerable
                )
            )
            extra = [
                enemy.get("hp", 0) / max_hp,
                1.0 if enemy.get("elite") or enemy.get("boss") else 0.0,
                1.0 if windup else 0.0,
                1.0 if vulnerable else 0.0,
                1.0 if enemy.get("boss") else 0.0,
                # How far into the current attack: the same pattern calls for
                # a dodge at the start and a punish at the end.
                min(enemy.get("phase", 0), 3) / 3.0,
                min(enemy.get("attack_time", 0.0), 2.0) / 2.0,
                min(enemy.get("attack_step", 0), 6) / 6.0,
                # A pillar on the line means every shot is absorbed: the
                # answer is to move, not to keep firing.
                1.0 if enemy.get("los", True) else 0.0,
                # Where it is going, and how far off a firing ray it sits.
                # Keyboard aim has eight directions: hitting means circling
                # until the target is on one of them, which is a geometric
                # relation the network would otherwise have to rediscover.
                max(-1.0, min(1.0, enemy.get("vx", 0.0) / 400.0)),
                max(-1.0, min(1.0, enemy.get("vy", 0.0) / 400.0)),
                min(aim_miss(obs, enemy), 300.0) / 300.0,
            ]
        else:
            extra = [0.0] * 12
        features.extend(_entity(obs, enemy, extra))

    for i in range(2):
        pickup = obs.pickups[i] if i < len(obs.pickups) else None
        extra = None
        if pickup:
            kind = pickup.get("type")
            extra = [
                1.0 if kind in {"heart", "heartHalf"} else 0.0,
                1.0 if kind == "item" else 0.0,
                1.0 if kind == "gold" else 0.0,
                # What the relic is worth to this build. The teacher picks on
                # this number rather than on rarity, so without it the network
                # would see two identical pedestals and one arbitrary choice.
                max(-1.0, min(2.0, item_value(obs, pickup))) / 2.0
                if kind == "item"
                else 0.0,
            ]
        else:
            extra = [0.0, 0.0, 0.0, 0.0]
        features.extend(_entity(obs, pickup, extra))

    features.extend(_entity(obs, obs.target_door, [1.0 if obs.door_los else 0.0]))
    features.extend(_entity(obs, obs.portal))

    # Nearest two pillars: they block movement and projectiles alike.
    for i in range(2):
        obstacle = obs.obstacles[i] if i < len(obs.obstacles) else None
        extra = [obstacle["r"] / 90.0] if obstacle else [0.0]
        features.extend(_entity(obs, obstacle, extra))

    # Threats the dodging behaviour reacts to: without them the network
    # could never reproduce sidesteps and blast escapes.
    for i in range(2):
        shot = obs.shots[i] if i < len(obs.shots) else None
        if shot:
            features.extend(
                [
                    1.0,
                    (shot["x"] - obs.x) / ROOM_W,
                    (shot["y"] - obs.y) / ROOM_H,
                    shot.get("distance", DIAG) / DIAG,
                    shot.get("vx", 0.0) / 320.0,
                    shot.get("vy", 0.0) / 320.0,
                ]
            )
        else:
            features.extend([0.0] * 6)

    for i in range(2):
        hazard = obs.hazards[i] if i < len(obs.hazards) else None
        if hazard:
            features.extend(
                [
                    1.0,
                    (hazard["x"] - obs.x) / ROOM_W,
                    (hazard["y"] - obs.y) / ROOM_H,
                    hazard.get("distance", DIAG) / DIAG,
                    hazard.get("r", 0.0) / 140.0,
                    hazard.get("remaining", 0.0) / 1.3,
                ]
            )
        else:
            features.extend([0.0] * 6)

    choice = obs.near_choice
    features.extend([1.0 if choice else 0.0, choice["distance"] / DIAG if choice else 0.0])

    # The room's own optional interaction (altar, gambler, brazier): each has
    # its own price, so which one it is has to be part of the state.
    focus = obs.room_focus
    features.extend(
        _entity(
            obs,
            focus,
            [
                1.0 if focus and focus["kind"] == "altar" else 0.0,
                1.0 if focus and focus["kind"] == "gambler" else 0.0,
                1.0 if focus and focus["kind"] == "defi" else 0.0,
            ],
        )
    )

    shop = obs.nearest_shop_item
    affordable = 1.0 if shop and obs.gold >= shop.get("price", 0) else 0.0
    shop_worth = (
        max(-1.0, min(2.0, item_value(obs, shop))) / 2.0
        if shop and shop.get("kind") == "item"
        else 0.0
    )
    features.extend(_entity(obs, shop, [affordable, shop_worth]))

    # Spawn window: where enemies are about to appear.
    spawn = obs.spawning[0] if obs.spawning else None
    features.extend(
        [
            min(len(obs.spawning), 3) / 3.0,
            1.0 if spawn else 0.0,
            (spawn["x"] - obs.x) / ROOM_W if spawn else 0.0,
            (spawn["y"] - obs.y) / ROOM_H if spawn else 0.0,
        ]
    )

    # Minimap routing: what kind of room the target door leads to.
    target_type = obs.target_room_type
    features.extend(
        [
            1.0 if target_type == "treasure" else 0.0,
            1.0 if target_type == "shop" else 0.0,
            1.0 if target_type == "miniboss" else 0.0,
            1.0 if target_type == "boss" else 0.0,
            min(obs.target_room_hops, 6) / 6.0,
        ]
    )

    if previous is not None and previous.state == "play":
        travel = ((obs.x - previous.x) ** 2 + (obs.y - previous.y) ** 2) ** 0.5
        speed = min(travel / STEP_TRAVEL, 1.0)
        features.extend([speed, 1.0 if speed < 0.05 else 0.0])
    else:
        features.extend([0.0, 0.0])

    # Being wedged against a prop, and overstaying in a room, are both
    # invisible in a single snapshot yet cost as much budget as dying.
    features.extend(
        [
            min(obs.stuck_steps, 12) / 12.0,
            min(obs.room_steps, 900) / 900.0,
        ]
    )

    # Danger field: 16 directions x 3 travel distances, already in [0, 1] and
    # computed over every hazard, orb and enemy on screen (see eos_env.py).
    # This is what makes multi-zone patterns representable at all.
    field = obs.danger_field
    features.extend(field if len(field) == N_DANGER else [0.0] * N_DANGER)

    return features


def _dummy_observation() -> Observation:
    """An empty but complete observation, for counting a vector's length."""
    from dataclasses import fields

    dummy_values = {
        "state": "play",
        "hp": 6,
        "max_hp": 6,
        "x": 0.0,
        "y": 0.0,
        "floor": 1,
        "kills": 0,
        "rooms": 0,
        "floors": 0,
        "time": 0.0,
        "hp_missing": 0,
        "gold": 0,
        "item_count": 0,
        "room_type": "start",
        "room_cleared": False,
        "abyss_gate": False,
        "doors_open": False,
        "portal_active": False,
        "enemy_count": 0,
        "pickup_count": 0,
        "nearest_enemy": None,
        "nearest_pickup": None,
        "nearest_shop_item": None,
        "shop_items": [],
        "nearest_door": None,
        "target_door": None,
        "near_choice": None,
        "room_focus": None,
        "portal": None,
        "available_doors": [],
        "pickups": [],
        "enemies": [],
        "shots": [],
        "hazards": [],
        "spawning": [],
        "target_room_type": None,
        "target_room_hops": 0,
        "killer": None,
        "killer_was_boss": False,
        "danger_field": [0.0] * N_DANGER,
        "obstacles": [],
        "door_los": True,
        "room_w": ROOM_W,
        "room_h": ROOM_H,
        "room_wt": 46.0,
        "skill_ready": False,
        "skill_cd": 0.0,
        "shield_hits": 0,
        "dash_ready": True,
        "dash_cd": 0.0,
        "i_frames": 0.0,
        "damage": 1.0,
        "fire_delay": 0.21,
        "move_speed": 290.0,
        "shot_range": 470.0,
        "shot_speed": 580.0,
        "room_key": "1:0,0",
        "stuck_steps": 0,
        "room_steps": 0,
        "fire_cd": 0.0,
        "player_vx": 0.0,
        "player_vy": 0.0,
        "dash_t": 0.0,
        "slow_t": 0.0,
        "items": [],
        "synergies": [],
        "mods": {},
        "second_wave": False,
        "gambles_left": 3,
        "defi_done": False,
        "defi_active": False,
        "defi_t": 0.0,
        "in_transition": False,
        "strata_zones": [],
        "wind": None,
        "shot_sectors": [0.0] * 16,
    }
    assert {f.name for f in fields(Observation)} == set(dummy_values)
    return Observation(**dummy_values)


def _count_features() -> int:
    return len(featurize(_dummy_observation()))


N_FEATURES = 0
N_FEATURES = _count_features()
