from __future__ import annotations

import math

from src import aiming, bestiary, economy, mover
from src.eos_env import Observation
from src.features import AIM_SECTOR, SHOT_RADIUS, aim_miss, item_value


RARITY_RANK = {
    None: 0,
    "common": 1,
    "rare": 2,
    "epic": 3,
    "legendary": 4,
}

# Loot already lying on a pedestal is never a risk — the risk was in the
# fight that produced it, and that fight is already over. The old blanket ban
# on "risky rooms" also refused the epic relic of a sealed room AFTER the
# policy had beaten its two reinforced elites: the whole cost, none of the
# reward. Only the gambler stays out, because paying gold for a random draw
# is variance with no floor.
NO_LOOT_ROOM_TYPES = {"gambler"}

# The sacrifice altar trades 1 heart of MAX life for a guaranteed epic relic
# (legendary from floor 4). Over five floors that trade is strongly positive:
# a run dies to the damage it cannot avoid, not to the heart it never had.
# The floor is 3 hearts, so a bad draw can never leave the run one hit away
# from death.
ALTAR_MIN_MAX_HP = 6
# How far the policy will break off a fight to grab a heart. Kept short: a
# long detour is a long time not shooting, and the far side of a room is
# where props block the straight line the policy walks.
COMBAT_PICKUP_RANGE = 140.0
ROOM_WIDTH = 1280.0
ROOM_HEIGHT = 720.0
WALL_MARGIN = 120.0
# Below this distance a target is walked at directly, with a tight dead
# zone (see _move_towards).
CLOSE_APPROACH = 80.0

# The build rules: False = the phase 4 valuation (five modifiers, altar
# only, shop by value), True = economy.py (every modifier, brazier,
# gambler, gold reserve).
BUILD_V2 = False


def _value(obs: Observation, entry: dict) -> float:
    return economy.item_value(obs, entry) if BUILD_V2 else item_value(obs, entry)

# Dodging thresholds. Enemy shots are slow orbs meant to be dodged by
# movement; hazards are telegraphed blast circles that resolve when their
# fuse runs out.
SHOT_THREAT_HORIZON = 0.8
SHOT_THREAT_RADIUS = 30.0
HAZARD_MARGIN = 28.0
HAZARD_DASH_WINDOW = 0.35
TELEGRAPH_STATES = {"aim", "windup", "prime"}


def heuristic_action(obs: Observation) -> str:
    """Small rule-based policy used as a smarter target than pure random."""
    if obs.state != "play":
        return "noop"

    # Wedged against a prop or a wall: no objective is reachable by walking
    # straight at it, and the run silently burns its whole budget there.
    unstick = _unstick(obs)
    if unstick:
        return unstick

    # Survival first: leave telegraphed blast zones and sidestep incoming
    # projectiles before any other objective.
    escape = _hazard_escape(obs)
    if escape:
        return escape
    dodge = _shot_dodge(obs)
    if dodge:
        return dodge

    # Spawn window: enemies are materialising and nothing is shootable
    # yet — pre-position toward the room centre instead of freezing.
    if obs.enemy_count == 0 and obs.spawning:
        dx = ROOM_WIDTH / 2 - obs.x
        dy = ROOM_HEIGHT / 2 - obs.y
        nearest_spawn = obs.spawning[0]
        if nearest_spawn.get("distance", 9999) < 150:
            dx = obs.x - nearest_spawn["x"]
            dy = obs.y - nearest_spawn["y"]
        if abs(dx) > 40 or abs(dy) > 40:
            return _best_direction(obs, [(dx, dy)])

    skill = _skill_action(obs)
    if skill:
        return skill

    # Pedestals offer several relics and only one can be taken: walk to the
    # best of them rather than to the nearest.
    choice = _best_choice_item(obs)
    if choice and obs.room_type not in NO_LOOT_ROOM_TYPES:
        if choice["distance"] <= 55:
            return "interact"
        if obs.enemy_count == 0:
            return _travel(obs, choice)
    elif (
        obs.near_choice
        and obs.near_choice["distance"] <= 58
        and obs.room_type not in NO_LOOT_ROOM_TYPES
    ):
        return "interact"

    altar = _altar_action(obs)
    if altar:
        return altar

    shop = _best_shop_item(obs)
    if shop:
        if shop["distance"] <= 58:
            return "interact"
        return _travel(obs, shop)

    enemy = obs.nearest_enemy
    if enemy:
        pickup = _best_pickup(obs)
        if pickup:
            # Fetching a heart is no reason to stop firing.
            return _combine(_move_towards(obs, pickup), _shoot_towards(obs, enemy))
        return _combat_action(obs, enemy)

    # Loot the cleared room before leaving: gold, hearts and items stay on
    # the floor otherwise (the doors open as soon as the fight ends).
    pickup = _best_pickup(obs)
    if pickup:
        if pickup["type"] == "item" and pickup.get("choice") and pickup["distance"] <= 55:
            return "interact"
        if pickup["distance"] > 20:
            return _collect(obs, pickup)

    if obs.portal_active and obs.portal:
        if obs.portal["distance"] <= 65:
            return "interact"
        return _travel(obs, obs.portal)

    exit_door = obs.target_door or obs.nearest_door
    if obs.doors_open and exit_door:
        return _door_approach(obs, exit_door)

    return "noop"


STUCK_TRIGGER = 6
# Escapes tried in turn while stuck, so a blocked run works its way around
# an obstacle instead of pressing the same key into it forever. Deterministic
# on purpose: a clone can only imitate what is reproducible.
UNSTICK_TURNS = [90, -90, 150, -150, 180]


def _unstick(obs: Observation) -> str | None:
    """Walk around whatever is absorbing the movement.

    The heuristic has no pathfinding: it walks straight at its objective. A
    prop between the player and a dropped heart is enough to freeze a run for
    thousands of steps — measured at 7 380 steps, 82 %% of an episode budget.
    """
    if obs.stuck_steps < STUCK_TRIGGER:
        return None

    # Rotate the blocked heading a bit further every few steps.
    turn = UNSTICK_TURNS[
        ((obs.stuck_steps - STUCK_TRIGGER) // 4) % len(UNSTICK_TURNS)
    ]
    angle = math.radians(turn)
    target = obs.target_door or obs.nearest_door or obs.nearest_enemy
    if target:
        dx, dy = target["x"] - obs.x, target["y"] - obs.y
    else:
        dx, dy = ROOM_WIDTH / 2 - obs.x, ROOM_HEIGHT / 2 - obs.y
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    move = _best_direction(obs, [(dx * cos_a - dy * sin_a, dx * sin_a + dy * cos_a)])
    if obs.dash_ready and obs.stuck_steps >= STUCK_TRIGGER + 6:
        return "dash_" + move
    return move


SKILL_CROWD_RANGE = 300.0
SKILL_CROWD_COUNT = 3
SKILL_SAFE_GAP = 130.0


def _skill_action(obs: Observation) -> str | None:
    """Solen's soul volley: a ring of 12 shards around the player, free every
    7 s and never used until now.

    A lone target only catches one or two shards, so spending a step on it is
    a loss; a crowd catches most of the ring, and a stunned guardian at
    point-blank catches it twice over.
    """
    if not obs.skill_ready or obs.enemy_count == 0:
        return None

    close = [e for e in obs.enemies if e.get("distance", 9999) <= SKILL_CROWD_RANGE]
    # Firing the ring costs a step of standing still, and a crowd is exactly
    # where standing still is paid for in contact damage. Spend it only from
    # outside everyone's reach — the shards travel, the player does not.
    if len(close) >= SKILL_CROWD_COUNT and all(
        e.get("distance", 9999) > SKILL_SAFE_GAP for e in close
    ):
        return "skill"
    for enemy in close:
        stunned = (
            enemy.get("attack") == "heavyCharge" and enemy.get("attack_step", 0) >= 2
        )
        if enemy.get("boss") and stunned and enemy.get("distance", 9999) < 220:
            return "skill"
    return None


def should_delegate_action(obs: Observation, action: str) -> bool:
    if obs.state != "play":
        return False

    if obs.enemy_count == 0 and (obs.doors_open or obs.portal_active):
        return True

    if obs.nearest_enemy and (obs.enemy_count >= 3 or obs.hp * 2 <= obs.max_hp):
        return True

    if obs.nearest_enemy and obs.enemy_count >= 2 and _is_static_combat_action(action):
        return True

    return False


def resolve_intent(obs: Observation, intent: str) -> str:
    if intent in {
        "noop",
        "up",
        "down",
        "left",
        "right",
        "up_left",
        "up_right",
        "down_left",
        "down_right",
        "shoot_up",
        "shoot_down",
        "shoot_left",
        "shoot_right",
        "shoot_up_left",
        "shoot_up_right",
        "shoot_down_left",
        "shoot_down_right",
        "dash",
        "dash_up",
        "dash_down",
        "dash_left",
        "dash_right",
        "dash_up_left",
        "dash_up_right",
        "dash_down_left",
        "dash_down_right",
    } or "_shoot_" in intent:
        return intent

    if intent == "heuristic":
        return heuristic_action(obs)
    if intent == "fight":
        return _fight_intent(obs)
    if intent == "kite":
        return _kite_intent(obs)
    if intent == "dash_away":
        return _dash_away_intent(obs)
    if intent == "exit":
        return _exit_intent(obs)
    if intent == "loot":
        return _loot_intent(obs)
    if intent == "interact":
        return "interact"
    if intent == "wait":
        return "noop"
    return heuristic_action(obs)


def _is_static_combat_action(action: str) -> bool:
    return action == "noop" or action == "dash" or action.startswith("shoot_")


def _fight_intent(obs: Observation) -> str:
    enemy = obs.nearest_enemy
    if enemy:
        return _combine(_defensive_move(obs, enemy), _shoot_towards(obs, enemy))
    return _exit_intent(obs)


def _kite_intent(obs: Observation) -> str:
    enemy = obs.nearest_enemy
    if enemy:
        return _combine(_move_away(obs, enemy), _shoot_towards(obs, enemy))
    return _exit_intent(obs)


def _dash_away_intent(obs: Observation) -> str:
    enemy = obs.nearest_enemy
    if enemy:
        return _dash_away(obs, enemy)
    return _exit_intent(obs)


def _exit_intent(obs: Observation) -> str:
    if obs.portal_active and obs.portal:
        if obs.portal["distance"] <= 65:
            return "interact"
        return _move_towards(obs, obs.portal)

    door = obs.target_door or obs.nearest_door
    if obs.doors_open and door:
        return _move_towards(obs, door)
    return heuristic_action(obs)


def _loot_intent(obs: Observation) -> str:
    if (
        obs.near_choice
        and obs.near_choice["distance"] <= 58
        and obs.room_type not in NO_LOOT_ROOM_TYPES
    ):
        return "interact"

    shop = obs.nearest_shop_item
    if shop and _shop_item_is_useful(obs, shop):
        if shop["distance"] <= 58:
            return "interact"
        return _move_towards(obs, shop)

    pickup = _best_pickup(obs)
    if pickup:
        if pickup["type"] == "item" and pickup.get("choice") and pickup["distance"] <= 55:
            return "interact"
        return _move_towards(obs, pickup)
    return _exit_intent(obs)


def _best_choice_item(obs: Observation) -> dict | None:
    """The best relic on a choice pedestal — taking one dismisses the others,
    so the nearest is rarely the right one."""
    choices = [
        p for p in obs.pickups if p.get("choice") and p.get("type") == "item"
    ]
    if not choices:
        return None
    return max(
        choices,
        key=lambda p: (_value(obs, p), -p.get("distance", 9999)),
    )


def _altar_action(obs: Observation) -> str | None:
    """Seal the pact at the sacrifice altar: 1 heart of max life for a
    guaranteed epic relic. Refused while the run is already fragile."""
    focus = obs.room_focus
    if not focus:
        return None
    kind = focus.get("kind")
    if BUILD_V2:
        wanted = (
            kind == "altar" and economy.wants_altar(obs)
            or kind == "gambler" and economy.wants_gamble(obs)
            or kind == "defi" and economy.wants_challenge(obs)
        )
        if not wanted:
            return None
    else:
        if kind != "altar":
            return None
        if obs.max_hp <= ALTAR_MIN_MAX_HP or obs.enemy_count:
            return None
        # Pay from a healthy bar, never from the last hearts of a wounded run.
        if obs.hp < obs.max_hp - 1:
            return None
    if focus["distance"] <= 88:
        return "interact"
    return _travel(obs, focus)


def _best_pickup(obs: Observation) -> dict | None:
    if not obs.pickups:
        return None

    candidates = []
    for pickup in obs.pickups:
        kind = pickup.get("type")
        if obs.enemy_count > 0:
            if kind not in {"heart", "heartHalf"} or obs.hp_missing <= 0:
                continue
            if pickup.get("distance", 9999) > COMBAT_PICKUP_RANGE:
                continue
        if kind in {"heart", "heartHalf"} and obs.hp_missing <= 0:
            continue
        priority = {
            "heart": 100 if obs.hp_missing > 0 else 0,
            "heartHalf": 90 if obs.hp_missing > 0 else 0,
            "item": 70 + RARITY_RANK.get(pickup.get("item_rarity"), 0) * 5,
            "gold": 30,
        }.get(kind, 0)
        if priority:
            candidates.append((priority - pickup.get("distance", 9999) * 0.01, pickup))
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])[1]


def _shop_item_is_useful(obs: Observation, shop: dict) -> bool:
    price = shop.get("price", 0)
    if obs.gold < price:
        return False
    if shop.get("kind") in {"heart", "heartHalf"}:
        return obs.hp_missing > 0
    if shop.get("kind") == "item":
        # Life is the scarce resource of the late floors and healing is only
        # sold in shops. Below half life, keep the purse for the heart on the
        # same shelf rather than spending it on a relic.
        if obs.hp * 2 <= obs.max_hp and _heart_on_sale(obs):
            return False
        return True
    return False


def _best_shop_item(obs: Observation) -> dict | None:
    """What is worth buying on this shelf, not simply what is closest.

    Healing is only sold in shops, and life is the scarce resource of the
    late floors: a heart comes before a relic whenever one is missing.
    """
    affordable = [
        item for item in obs.shop_items if item.get("price", 0) <= obs.gold
    ]
    if not affordable:
        return None

    hearts = [i for i in affordable if i.get("kind") in {"heart", "heartHalf"}]
    if obs.hp_missing > 0 and hearts:
        return min(hearts, key=lambda i: i.get("distance", 9999))

    if obs.hp * 2 <= obs.max_hp and _heart_on_sale(obs):
        return None  # save the purse for the heart on the same shelf

    relics = [i for i in affordable if i.get("kind") == "item"]
    if BUILD_V2:
        # keep a heart's price back when hurt, and only buy what helps
        relics = [
            i for i in relics
            if obs.gold - i.get("price", 0) >= economy.shop_reserve(obs)
            and economy.item_value(obs, i) >= 0.6
        ]
    if relics:
        return max(
            relics,
            key=lambda i: (_value(obs, i), -i.get("distance", 9999)),
        )
    return None


def _heart_on_sale(obs: Observation) -> bool:
    return any(
        item.get("kind") in {"heart", "heartHalf"} and item.get("price", 0) <= obs.gold
        for item in obs.shop_items
    )


def _move_towards(obs: Observation, target: dict) -> str:
    dx = target["x"] - obs.x
    dy = target["y"] - obs.y
    # Close to the objective, the 24 px dead zone made the hero orbit it:
    # a coin at 36 px is inside the zone on one axis, outside on the other,
    # and four steps of 43 px draw a square around it that the coin's
    # magnet (about 220 px/s) never closes. Two runs out of eighty burned
    # their whole budget that way.
    close = math.hypot(dx, dy) < CLOSE_APPROACH
    return _move_from_delta(dx, dy, deadzone=6 if close else 24)


def _travel(obs: Observation, target: dict) -> str:
    # Out of combat, dashing toward a distant objective buys game time:
    # more rooms fit into the same step budget.
    move = _move_towards(obs, target)
    if move != "noop" and obs.enemy_count == 0 and target.get("distance", 0) > 260:
        return "dash_" + move
    return move


DOOR_ALIGN_TOLERANCE = 34.0


def _door_approach(obs: Observation, door: dict) -> str:
    # The opening is narrow: walking diagonally pins the player against
    # the wall beside the frame. Close to the door, align on its axis
    # first, then cross.
    direction = door.get("dir")
    if door.get("distance", 9999) < 260:
        if direction in {"left", "right"} and abs(door["y"] - obs.y) > DOOR_ALIGN_TOLERANCE:
            return _move_from_delta(0, (door["y"] - obs.y) * 10)
        if direction in {"up", "down"} and abs(door["x"] - obs.x) > DOOR_ALIGN_TOLERANCE:
            return _move_from_delta((door["x"] - obs.x) * 10, 0)
    return _travel(obs, door)


# Coins are drawn to the player from this far (pickup.js: 110 px, at up to
# ~220 px/s). A hero running at 290 px/s toward a coin overruns it and the
# coin trails behind, never inside the 28 px pickup radius: five runs out of
# eighty spent their whole budget circling one. Standing still collects it.
GOLD_MAGNET = 100.0


def _collect(obs: Observation, pickup: dict) -> str:
    if pickup.get("type") == "gold" and pickup.get("distance", 9999) < GOLD_MAGNET:
        return "noop"
    return _travel(obs, pickup)


def _move_away(obs: Observation, target: dict) -> str:
    return _move_from_delta(obs.x - target["x"], obs.y - target["y"])


_RAYS = [
    "shoot_right",
    "shoot_down_right",
    "shoot_down",
    "shoot_down_left",
    "shoot_left",
    "shoot_up_left",
    "shoot_up",
    "shoot_up_right",
]


def _shoot_towards(obs: Observation, target: dict) -> str:
    # Take the ray the bearing is actually closest to. The old rule chose a
    # diagonal as soon as both offsets passed 45 px, which sent a shot 45 deg
    # out at a target sitting 14 deg off the horizontal.
    dx = target["x"] - obs.x
    dy = target["y"] - obs.y
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return "shoot_right"
    return _RAYS[round(math.atan2(dy, dx) / AIM_SECTOR) % len(_RAYS)]


_DIRS = {
    "up": (0.0, -1.0),
    "down": (0.0, 1.0),
    "left": (-1.0, 0.0),
    "right": (1.0, 0.0),
    "up_left": (-0.707, -0.707),
    "up_right": (0.707, -0.707),
    "down_left": (-0.707, 0.707),
    "down_right": (0.707, 0.707),
}


# The environment's danger field is 16 directions x 3 travel distances, with
# index i covering the angle i * 22.5 deg (x = cos, y = sin, y pointing down).
# The eight named directions land exactly on the even indices.
_DIR_FIELD_INDEX = {
    "right": 0,
    "down_right": 2,
    "down": 4,
    "down_left": 6,
    "left": 8,
    "up_left": 10,
    "up": 12,
    "up_right": 14,
}
_RADIUS_WEIGHT = (1.0, 0.7, 0.4)
_RADIUS_WEIGHT_SUM = sum(_RADIUS_WEIGHT)


def direction_danger(obs: Observation, name: str) -> float:
    """How lethal a direction looks, in [0, 1], over the next ~0.8 s."""
    field = obs.danger_field
    base = _DIR_FIELD_INDEX[name] * len(_RADIUS_WEIGHT)
    if len(field) < base + len(_RADIUS_WEIGHT):
        return 0.0
    weighted = sum(
        field[base + i] * weight for i, weight in enumerate(_RADIUS_WEIGHT)
    )
    return weighted / _RADIUS_WEIGHT_SUM


def _best_direction(
    obs: Observation,
    preferred: list[tuple[float, float]],
    *,
    cardinal_only: bool = False,
) -> str:
    """Pick the escape direction that also stays clear of every known threat,
    instead of blindly following the geometric ideal into another danger.

    The threat side of the score comes from the environment's danger field,
    which is computed over EVERY hazard, orb and enemy on screen. Scoring it
    here from the observation lists could only ever see the first few.
    """
    units = []
    for pdx, pdy in preferred:
        norm = (pdx * pdx + pdy * pdy) ** 0.5
        if norm > 1e-6:
            units.append((pdx / norm, pdy / norm))

    def score(name: str) -> float:
        ux, uy = _DIRS[name]
        alignment = max((ux * px + uy * py for px, py in units), default=0.0)
        return 2.0 * alignment - 4.0 * direction_danger(obs, name)

    names = ["up", "down", "left", "right"] if cardinal_only else list(_DIRS)
    return max(names, key=score)


def _hazard_escape(obs: Observation) -> str | None:
    """Leave a blast circle before its fuse runs out.

    Whether walking is enough is not a matter of taste: it is a distance to
    cover against a time left. Biome eruptions drop the circle straight onto
    the player's feet, so the ground to cover is the full radius, and the
    dash is the only thing that covers it in time.
    """
    # The tightest deadline first, not the nearest circle: a wide zone about
    # to resolve matters more than a small one with a long fuse.
    threatening = [
        hz for hz in obs.hazards if hz["distance"] <= hz["r"] + HAZARD_MARGIN
    ]
    if not threatening:
        return None
    hazard = min(threatening, key=lambda hz: hz["remaining"])

    dx = obs.x - hazard["x"]
    dy = obs.y - hazard["y"]
    if abs(dx) < 1 and abs(dy) < 1:
        dx = ROOM_WIDTH / 2 - obs.x
        dy = ROOM_HEIGHT / 2 - obs.y
    move = _best_direction(obs, [(dx, dy)])

    # The dash is a scarce resource: 0.85 s of cooldown, and the guardians'
    # charges need it more than a blast circle does. Spend it only when the
    # fuse is genuinely too short to walk out of. (Sizing the dash on the
    # ground left to cover was measured in V27 and cost more than it saved:
    # every dash spent here was one missing under a heavyCharge.)
    if hazard["remaining"] < HAZARD_DASH_WINDOW:
        return "dash_" + move

    enemy = obs.nearest_enemy
    if enemy:
        return _combine(move, _shoot_towards(obs, enemy))
    return move


def _shot_dodge(obs: Observation) -> str | None:
    """Break the line of the first orb that is actually going to arrive.

    Sidestepping perpendicular is the textbook answer against one orb. In a
    ring volley there is no clear side, so backing off is offered as a third
    option and the danger field arbitrates between the three.

    Two richer versions were built and measured (V41, V42): computing, for
    each of the eight directions, how close every incoming orb would pass to
    a player committed to it. Maximising that clearance halves the projectile
    hits and costs 18 %% of the rooms; constraining it and letting the fight
    pick among the safe directions restores the rooms and loses the dodging.
    Time spent dodging is time not killing, and killing faster is itself a
    defence: this simpler rule sits at a better point on that trade-off than
    either extreme.
    """
    for shot in obs.shots:
        rx = obs.x - shot["x"]
        ry = obs.y - shot["y"]
        vx = shot.get("vx", 0.0)
        vy = shot.get("vy", 0.0)
        speed_sq = vx * vx + vy * vy
        if speed_sq <= 1e-6:
            continue
        t_close = (rx * vx + ry * vy) / speed_sq
        if t_close < 0 or t_close > SHOT_THREAT_HORIZON:
            continue
        cx = rx - vx * t_close
        cy = ry - vy * t_close
        if cx * cx + cy * cy > SHOT_THREAT_RADIUS * SHOT_THREAT_RADIUS:
            continue
        move = _best_direction(obs, [(-vy, vx), (vy, -vx), (rx, ry)])
        enemy = obs.nearest_enemy
        if enemy:
            return _combine(move, _shoot_towards(obs, enemy))
        return move
    return None


def _strafe_move(obs: Observation, enemy: dict) -> str:
    ex = enemy["x"] - obs.x
    ey = enemy["y"] - obs.y
    return _best_direction(obs, [(-ey, ex), (ey, -ex)])


def _aim_is_on(obs: Observation, enemy: dict) -> bool:
    return aim_miss(obs, enemy) <= enemy.get("radius", 14.0) + SHOT_RADIUS


def _align_vector(obs: Observation, enemy: dict) -> tuple[float, float]:
    """Direction to walk to bring the target onto a firing ray.

    Circling the target is what aiming means on a keyboard. Moving along
    (-sin, cos) of the bearing turns it downwards, so the sign that closes
    the error is the one to take.
    """
    bearing = math.atan2(enemy["y"] - obs.y, enemy["x"] - obs.x)
    error = bearing - round(bearing / AIM_SECTOR) * AIM_SECTOR
    sign = 1.0 if error > 0 else -1.0
    return -math.sin(bearing) * sign, math.cos(bearing) * sign


# Read from the game's attack registry (js/game/boss.js). The aim of the
# charging patterns tracks the player until this instant and is fixed after
# it: any dodge started earlier is simply followed.
AIM_LOCK_CHARGE = 0.5
# The bodyslam blast circle, and the outer ring of the Colossus' aftershock
# beyond which a wide safe band actually exists.
SLAM_RADIUS = 140.0
AFTERSHOCK_CLEAR = 300.0


def _dash_sideways(obs: Observation, enemy: dict) -> str:
    """Cross the attack line with the dash's i-frames, or strafe if the dash
    is still recharging."""
    move = _strafe_move(obs, enemy)
    if move == "noop":
        move = _move_towards_center_if_near_wall(obs)
        if move == "noop":
            move = "up" if obs.y > ROOM_HEIGHT / 2 else "down"
    return "dash_" + move if obs.dash_ready else move


def _sweep_escape(obs: Observation, enemy: dict, shot: str) -> str:
    """The Zenith's beam always sweeps the same way, from its lock angle
    minus 0.7 rad to plus 0.7 rad. Orbiting against the rotation crosses
    angles the beam has already left instead of racing the orbs."""
    dx = obs.x - enemy["x"]
    dy = obs.y - enemy["y"]
    move = _best_direction(obs, [(dy, -dx)])
    if obs.dash_ready and enemy.get("distance", 9999) < 420:
        return "dash_" + move
    return _combine(move, shot)


def _boss_attack_response(obs: Observation, enemy: dict, shot: str) -> str | None:
    """Pattern-specific answers to the guardians' signature attacks.

    Timings come from the game's own attack registry: `attack_time` is the
    seconds elapsed in the current attack, and every guardian telegraphs for
    a fixed duration whatever its tier. Returning None falls back to the
    generic combat behaviour, which the danger field already handles well
    for anything that resolves into blast circles or orbs.
    """
    if enemy.get("state") != "attack":
        return None
    attack = enemy.get("attack")
    distance = enemy.get("distance", 9999)
    step = enemy.get("attack_step", 0)
    atk_t = enemy.get("attack_time", 0.0)

    if attack == "heavyCharge":
        if step >= 2:
            # Stunned against a wall: the free-damage window. Close enough to
            # hit, never close enough to be body-checked when it comes round
            # — the stun runs out on its own clock, not on ours.
            if distance > 420:
                return _combine(_move_towards(obs, enemy), shot)
            if distance < 200:
                return _combine(_move_away(obs, enemy), shot)
            return shot
        # The maw tracks the player until 0.5 s, then locks and charges at
        # 720 px/s from 0.75. Dodging DURING the tracking phase is worse
        # than useless: the aim simply follows, and the dash is on its
        # 0.85 s cooldown when the charge actually comes. Wait for the
        # lock, then cross the line.
        if step >= 1 or atk_t >= AIM_LOCK_CHARGE:
            return _dash_sideways(obs, enemy)
        return _combine(_strafe_move(obs, enemy), shot)

    if attack == "tripleLunge":
        # Three lunges, each re-aimed: dodging once is not enough. Even
        # steps are the tracking phase, odd steps the leap itself — and a
        # leap is what there is to dodge.
        if step % 2 == 1:
            return _dash_sideways(obs, enemy)
        move = _strafe_move(obs, enemy)
        return _combine(move, shot)

    if attack == "slamQuake":
        # Blast radius 140 resolving instantly at 0.85 s: only distance works.
        if distance < SLAM_RADIUS + 80:
            if atk_t >= 0.45 or distance < SLAM_RADIUS:
                return _dash_away(obs, enemy)
            move = _move_away(obs, enemy)
            return _combine(move, shot)
        return shot

    if attack == "aftershock":
        # Two concentric rings of blast circles land at 0.7 s, at radius 120
        # and 235. Their gaps are a few pixels wide — the only wide safe
        # band is beyond the outer ring.
        if distance < AFTERSHOCK_CLEAR:
            if atk_t >= 0.35:
                return _dash_away(obs, enemy)
            return _combine(_move_away(obs, enemy), shot)
        return shot

    if attack == "magmaWake":
        # The Pyroclast embers its own wake: never follow it, cut across.
        # Same tracking phase as the charge — clear the line after the lock.
        if step >= 1 or atk_t >= AIM_LOCK_CHARGE:
            if distance < 260:
                return _dash_sideways(obs, enemy)
            return _combine(_strafe_move(obs, enemy), shot)
        return _combine(_strafe_move(obs, enemy), shot)

    if attack == "sweepBeam":
        if step >= 1:
            return _sweep_escape(obs, enemy, shot)
        return _combine(_strafe_move(obs, enemy), shot)

    if attack == "fanVolley":
        # A narrow five-shot fan aimed until 0.4 s: sidestepping before it
        # fires clears the whole spread at once.
        if atk_t < 0.5:
            move = _strafe_move(obs, enemy)
            return _combine(move, shot)
        return None

    if attack == "brambleRing" and step == 0:
        # The crown of thorns grows around where the player IS at 0.5 s.
        # Moving means it is centred behind us, gap or no gap.
        return _combine(_strafe_move(obs, enemy), shot)

    if attack == "mirrorEcho" and step >= 1 and atk_t < 1.2:
        # The Weaver leaves an echo on the spot it teleported from, and it
        # detonates in a ring at 1.15 s. The boss is elsewhere now.
        return _combine(_move_away(obs, enemy), shot)

    if attack in {"ringVolley", "doubleRing", "tidalPulse"}:
        # Walls of orbs with narrow gaps. Far out the rings thin and the
        # generic dodge threads them; close in, the dash's i-frames are the
        # only reliable way through.
        if distance < 280 and obs.dash_ready:
            return _dash_sideways(obs, enemy)
        return None

    return None


def _clear_line_of_fire(obs: Observation, enemy: dict) -> str | None:
    """Step out from behind a pillar.

    Combat rooms from depth 2 onwards are seeded with pillars, and they stop
    projectiles just like walls do (Room.projectileBlocked). Firing at an
    enemy behind one does literally nothing, which is how a run can spend
    8 000 steps in a single room.
    """
    if enemy.get("los", True):
        return None
    blocker = next(
        (o for o in obs.obstacles if o.get("distance", 9999) < 320), None
    )
    if blocker is None:
        # Something blocks the line but no pillar is close: circling around
        # the target is still the way to open it.
        return _strafe_move(obs, enemy)
    # Slide along the pillar, on whichever side is otherwise safe.
    bx = blocker["x"] - obs.x
    by = blocker["y"] - obs.y
    return _best_direction(obs, [(-by, bx), (by, -bx)])


def _combat_action(obs: Observation, enemy: dict) -> str:
    shot = _shoot_towards(obs, enemy)
    distance = enemy.get("distance", 9999)
    boss = bool(enemy.get("boss"))

    reposition = _clear_line_of_fire(obs, enemy)
    if reposition:
        return reposition

    if boss:
        response = _boss_attack_response(obs, enemy, shot)
        if response:
            return response

    # A telegraphed attack aims at the player's current position: strafing
    # sideways breaks the line before the enemy commits.
    if enemy.get("state") in TELEGRAPH_STATES and distance < 420:
        return _combine(_strafe_move(obs, enemy), shot)
    if enemy.get("state") == "dash" and distance < 260:
        move = _strafe_move(obs, enemy)
        return "dash_" + move if move != "noop" else _dash_away(obs, enemy)

    # Bosses hit harder on contact and lunge further: keep more distance.
    near, retreat, defend, engage = (150, 260, 420, 560) if boss else (105, 190, 330, 470)
    if distance < retreat:
        if distance < near:
            return _dash_away(obs, enemy)
        return _combine(_move_away(obs, enemy), shot)

    hp_low = obs.hp * 2 <= obs.max_hp
    crowded = obs.enemy_count >= 2
    if hp_low or crowded or distance < defend:
        return _combine(_defensive_move(obs, enemy), shot)

    if distance > engage:
        return _combine(_move_towards(obs, enemy), shot)

    # Comfortable range, one enemy, full life: the old answer was to stand
    # still and fire, and it is where the audit found most of the damage
    # taken -- 92 of 261 hits with a single enemy on screen. Standing still
    # neither dodges nor aims. Circle instead, until the target is on a ray.
    if not _aim_is_on(obs, enemy):
        return _combine(_best_direction(obs, [_align_vector(obs, enemy)]), shot)
    return shot


def _dash_away(obs: Observation, target: dict) -> str:
    move = _move_away(obs, target)
    if move == "noop":
        # Overlapping the enemy: the away-vector collapses to zero, so pick
        # any escape direction instead of dashing in place.
        move = _move_towards_center_if_near_wall(obs)
        if move == "noop":
            move = "up" if obs.y > ROOM_HEIGHT / 2 else "down"
    return "dash_" + move


def _defensive_move(obs: Observation, enemy: dict) -> str:
    # Strafe perpendicular to the enemy axis, scored against walls, hazards
    # and the other enemies. Cardinal on purpose, and no longer because a
    # diagonal used to cost the shot: what a charge has to be cleared by is
    # the ground covered ACROSS its line, and a diagonal only covers 0.707 of
    # it. Opening the eight directions here was measured in V35 and tripled
    # the life lost to Vorace's heavyCharge, from 0.55 to 1.6 per episode.
    dx = enemy["x"] - obs.x
    dy = enemy["y"] - obs.y
    # Strafing already circles the target; when the aim is off the ray, take
    # the side that also closes the angle instead of choosing at random.
    if _aim_is_on(obs, enemy):
        preferred = [(-dy, dx), (dy, -dx)]
    else:
        preferred = [_align_vector(obs, enemy)]
    return _best_direction(obs, preferred, cardinal_only=True)


def _move_towards_center_if_near_wall(obs: Observation) -> str:
    if obs.x < WALL_MARGIN:
        return "right"
    if obs.x > ROOM_WIDTH - WALL_MARGIN:
        return "left"
    if obs.y < WALL_MARGIN:
        return "down"
    if obs.y > ROOM_HEIGHT - WALL_MARGIN:
        return "up"
    return "noop"


def _combine(move: str, shoot: str) -> str:
    if move == "noop":
        return shoot
    return f"{move}_{shoot}"


def _move_from_delta(dx: float, dy: float, deadzone: float = 24) -> str:
    horizontal = "right" if dx > deadzone else "left" if dx < -deadzone else ""
    vertical = "down" if dy > deadzone else "up" if dy < -deadzone else ""
    if vertical and horizontal:
        return f"{vertical}_{horizontal}"
    return vertical or horizontal or "noop"


# ---------------------------------------------------------------------------
# Decisions with a continuous aim
# ---------------------------------------------------------------------------

# The keyboard shoots along eight rays, so a hit was only possible when the
# target happened to sit near one of them (41 % of the shots, measured), and
# the rules above walk around the target to line it up. The mouse aims at
# any angle. `heuristic_decision` keeps every rule of `heuristic_action`
# unchanged and only replaces the shot: the same target, aimed exactly.


_last_target: dict | None = None


def heuristic_decision(
    obs: Observation,
    *,
    lead: bool = False,
    priority: bool = False,
    always_fire: bool = False,
    predictive: bool = False,
    build: bool = False,
    bosses: bool = False,
) -> tuple[str, tuple[float, float] | None]:
    """The rule-based policy as (movement action, aim vector or None).

    `lead` meets a moving target where the orb will find it; `priority`
    picks the target by what it costs to leave alive instead of by distance;
    `always_fire` keeps shooting during the moves the keyboard rules left
    silent (dashes, pure repositioning), which the arrows could not combine.
    """
    global _last_target, BUILD_V2
    BUILD_V2 = build
    if predictive:
        return _decision_v2(obs, lead=lead, priority=priority, bosses=bosses)
    action = heuristic_action(obs)
    move, shoots = split_shot(action)
    if not shoots:
        if not (always_fire and obs.enemies and move not in ("skill", "interact")):
            return move, None
    if priority:
        target = aiming.pick_target(obs, _last_target)
        _last_target = target
    else:
        target = obs.nearest_enemy
    if target is None:
        return move, None
    if lead:
        return move, aiming.aim_vector(obs, target, lead=True)
    return move, aim_at(obs, target)


def split_shot(action: str) -> tuple[str, bool]:
    """Strip the arrow part of an action name: ('up_shoot_left') -> ('up', True)."""
    if action.startswith("shoot_"):
        return "noop", True
    if "_shoot_" in action:
        return action.split("_shoot_", 1)[0], True
    return action, False


def aim_at(obs: Observation, target: dict) -> tuple[float, float]:
    """Unit vector from the player to the target's current position."""
    dx = target["x"] - obs.x
    dy = target["y"] - obs.y
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return (1.0, 0.0)
    return (dx / norm, dy / norm)


# ---------------------------------------------------------------------------
# The predictive teacher
# ---------------------------------------------------------------------------

# Distance the fight is comfortable at, by family (see bestiary.py and the
# game's enemy.js): far enough that a charger's dash falls short and a
# bomber never arms, close enough to stay inside the orb's 470 px range.
STANDOFF = {
    "chase": (150.0, 320.0),
    "charge": (240.0, 350.0),
    "kite": (300.0, 440.0),
    "bomb": (170.0, 400.0),
    "summon": (250.0, 330.0),
    "shield": (150.0, 300.0),
    "flee": (0.0, 900.0),
    "boss": (200.0, 420.0),
}

_last_dir: tuple[float, float] | None = None

# Steps in one room after which a fight is judged stuck (90 s of game
# time; a room is cleared in 150-300 steps when things go well).
STALLED_FIGHT = 600


def _decision_v2(
    obs: Observation, *, lead: bool = True, priority: bool = True, bosses: bool = False
) -> tuple[str, tuple[float, float] | None]:
    """Same targets and economy as the keyboard rules, movement by simulation.

    Every step in a fight goes through mover.choose: the rules only say
    what they would like (keep the family's distance band, follow the boss
    response, walk to a heart), and the simulation refuses whatever gets
    hit in the next 0.6 s. The shot is aimed at the chosen target on every
    step, dashes included, which the arrow keys could never do.
    """
    global _last_target, _last_dir
    if obs.state != "play":
        return "noop", None

    unstick = _unstick(obs)
    if unstick:
        return unstick, None

    target = aiming.pick_target(obs, _last_target) if priority else obs.nearest_enemy
    _last_target = target
    aim = aiming.aim_vector(obs, target, lead=lead) if target else None

    # Out of combat, the keyboard rules already know what to do (loot,
    # altar, shop, doors, portal); only the way there is checked against
    # the arena's or the biome's zones.
    if not obs.enemies:
        action = heuristic_action(obs)
        move, _ = split_shot(action)
        if move in ("interact", "skill") or move.startswith("skill"):
            return move, None
        # Every step goes through the simulation, even a quiet corridor: a
        # pillar between the hero and the door pinned it for whole runs
        # (the keyboard rule walked straight at the door), and the
        # simulation charges a pinned candidate for the travel it loses.
        prefer = _direction_of(move)
        picked = mover.choose(obs, mover.Goal(prefer=prefer, hold=prefer is None, previous=_last_dir),
                              allow_dash=move.startswith("dash_"))
        _last_dir = picked.direction if picked.action != "noop" else _last_dir
        if picked.action == "noop" and move == "noop":
            return "noop", None
        return picked.action, None

    skill = _skill_action(obs)
    if skill:
        return "skill", aim

    goal = mover.Goal(previous=_last_dir, target=target, avoid_future=bosses)
    heart = _best_pickup(obs)
    if heart is not None:
        goal.prefer = _unit(heart["x"] - obs.x, heart["y"] - obs.y)
        goal.band = (0.0, 900.0)
    elif target is not None:
        row = bestiary.info(target)
        lo, hi = STANDOFF.get(row["behaviour"], (150.0, 320.0))
        if row["behaviour"] == "bomb":
            lo = row.get("trigger_dist", 125) + 40.0
        if target.get("elite"):
            lo, hi = lo + 40.0, hi + 40.0
        if obs.floor >= 3:
            lo, hi = lo * 1.15, hi * 1.15
        goal.band = (lo, hi)
        dx, dy = target["x"] - obs.x, target["y"] - obs.y
        distance = math.hypot(dx, dy) or 1.0
        ux, uy = dx / distance, dy / distance
        # A fight that has lasted this long is not being won from the
        # band (serpentine shots, a slow tank behind a pillar): close in,
        # the simulation still refuses whatever gets hit.
        if obs.room_steps > STALLED_FIGHT and not target.get("boss"):
            lo, hi = 0.0, 120.0
            goal.band = (lo, hi)
            goal.weight_band = 2.0
        response = _boss_attack_response(obs, target, "shoot_up") if target.get("boss") else None
        table = _boss_prefer_v2(obs, target) if bosses and target.get("boss") else None
        if table is not None:
            goal.prefer, goal.hold = table
        elif response and response != "noop":
            move, _ = split_shot(response)
            goal.prefer = _direction_of(move.replace("dash_", ""))
            goal.hold = goal.prefer is None
        elif distance < lo:
            goal.prefer = (-ux, -uy)
        elif distance > hi:
            goal.prefer = (ux, uy)
        else:
            # inside the band: keep circling, on the side already taken
            side = 1.0
            if _last_dir is not None and (-uy * _last_dir[0] + ux * _last_dir[1]) < 0:
                side = -1.0
            goal.prefer = (-uy * side, ux * side)
            goal.weight_band = 0.5
    picked = mover.choose(obs, goal)
    if picked.action != "noop":
        _last_dir = picked.direction
    return picked.action, aim


def last_target() -> dict | None:
    """The enemy the last decision aimed at (for recording demonstrations)."""
    return _last_target


def _direction_of(move: str) -> tuple[float, float] | None:
    name = move[5:] if move.startswith("dash_") else move
    return mover.DIRECTIONS.get(name)


def _unit(dx: float, dy: float) -> tuple[float, float]:
    d = math.hypot(dx, dy)
    return (dx / d, dy / d) if d > 1e-6 else (1.0, 0.0)


# ---------------------------------------------------------------------------
# The inventory reflex
# ---------------------------------------------------------------------------

def economy_action(obs: Observation) -> str | None:
    """The room's economic decision, when there is one and the room is quiet.

    Pedestal choice, purchase, altar, gambler, brazier, loot left on the
    floor, portal: about thirty decisions in a run, the same rules for the
    teacher and the learned agent. None while enemies are up, or when the
    room offers nothing: then the fight, or the navigation, is someone
    else's business.
    """
    if obs.state != "play" or obs.enemy_count or obs.spawning:
        return None
    choice = _best_choice_item(obs)
    if choice and obs.room_type not in NO_LOOT_ROOM_TYPES:
        return "interact" if choice["distance"] <= 55 else _travel(obs, choice)
    if (
        obs.near_choice
        and obs.near_choice["distance"] <= 58
        and obs.room_type not in NO_LOOT_ROOM_TYPES
    ):
        return "interact"
    altar = _altar_action(obs)
    if altar:
        return altar
    shop = _best_shop_item(obs)
    if shop:
        return "interact" if shop["distance"] <= 58 else _travel(obs, shop)
    pickup = _best_pickup(obs)
    if pickup and pickup["distance"] > 20:
        return _collect(obs, pickup)
    if obs.portal_active and obs.portal:
        return "interact" if obs.portal["distance"] <= 65 else _travel(obs, obs.portal)
    return None


def _boss_prefer_v2(obs: Observation, boss: dict) -> tuple[tuple[float, float] | None, bool] | None:
    """Where to head during the attacks the keyboard rules got wrong.

    Read from js/game/boss.js. The simulation (mover.py) already refuses
    whatever gets hit inside 0.6 s; this only says which way to lean when
    the hit is further away than that. None = no opinion (the distance
    band and the older responses decide).
    """
    if boss.get("state") != "attack":
        return None
    name = boss.get("attack")
    step = boss.get("attack_step", 0)
    t = boss.get("attack_time", 0.0)
    lock = boss.get("lock_dir", [0.0, 0.0])
    aux2 = boss.get("aux2", [0.0, 0.0])
    away = lambda x, y: _unit(obs.x - x, obs.y - y)
    if name == "mirrorEcho" and step >= 1:
        # the echo detonates in a ring where the boss WAS (auxX/Y)
        return away(lock[0], lock[1]), False
    if name == "shatterClone" and step >= 1:
        near = min(((lock[0], lock[1]), (aux2[0], aux2[1])),
                   key=lambda q: math.hypot(q[0] - obs.x, q[1] - obs.y))
        return away(*near), False
    if name == "brambleRing" and step >= 1:
        # seven zones at 130 px around the player at the cast, one gap:
        # the disc at the centre is free - stand still if still in it
        ring = [hz for hz in obs.hazards if abs(hz.get("r", 0) - 55) < 1]
        if len(ring) >= 5:
            cx = sum(hz["x"] for hz in ring) / len(ring)
            cy = sum(hz["y"] for hz in ring) / len(ring)
            if math.hypot(cx - obs.x, cy - obs.y) < 40:
                return None, True
        return None
    if name == "frostNova" and step >= 1:
        # four arms, straight or diagonal: the refuge is the other family
        arms = [hz for hz in obs.hazards if abs(hz.get("r", 0) - 52) < 1]
        if arms:
            a0 = math.atan2(arms[0]["y"] - boss["y"], arms[0]["x"] - boss["x"])
            diagonal = abs(((a0 % (math.pi / 2)) - math.pi / 4)) < math.pi / 8
            bearing = math.atan2(obs.y - boss["y"], obs.x - boss["x"])
            base = 0.0 if diagonal else math.pi / 4
            best = min((base + k * math.pi / 2 for k in range(4)),
                       key=lambda a: abs((a - bearing + math.pi) % (2 * math.pi) - math.pi))
            d = max(150.0, math.hypot(obs.x - boss["x"], obs.y - boss["y"]))
            gx, gy = boss["x"] + math.cos(best) * d, boss["y"] + math.sin(best) * d
            return _unit(gx - obs.x, gy - obs.y), False
        return None
    if name == "tripleLunge" and step < 6:
        # keep out of the lunge's reach (161-200 px); sidestep once locked
        d = math.hypot(boss["x"] - obs.x, boss["y"] - obs.y)
        if d >= 250:
            return None
        u = _unit(boss["x"] - obs.x, boss["y"] - obs.y)
        side = (-u[1], u[0])
        if _last_dir is not None and side[0] * _last_dir[0] + side[1] * _last_dir[1] < 0:
            side = (u[1], -u[0])
        return side, False
    if name == "aftershock" and step == 0:
        # be more than 300 px from the boss when the rings burst
        return away(boss["x"], boss["y"]), False
    if name == "slamQuake" and step == 0:
        return away(boss["x"], boss["y"]), False
    if name == "sweepBeam" and t >= 0.4:
        # the beam sweeps from base-0.7 toward base+0.7: orbit toward the
        # side it started on (behind the sweep)
        bearing = math.atan2(obs.y - boss["y"], obs.x - boss["x"])
        start = aux2[0] if step >= 1 else math.atan2(lock[1], lock[0]) - 0.7
        diff = (start - bearing + math.pi) % (2 * math.pi) - math.pi
        sign = 1.0 if diff > 0 else -1.0
        return (-math.sin(bearing) * sign, math.cos(bearing) * sign), False
    if name == "heavyCharge" and step >= 2:
        # stunned: stand in the band and shoot
        return None, False
    return None
