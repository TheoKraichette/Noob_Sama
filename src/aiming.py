"""Where to shoot: which enemy, and at what angle.

The keyboard aims along eight rays; the mouse aims anywhere. Once the shot
can go at any angle, two questions remain and this module answers both.
Which target: the nearest enemy is not always the one that costs the most
life to leave alive (a summoner keeps the room full, a bomber must die
before it arms). What angle: the orb flies at ~580 px/s, so a target that
keeps moving is met where it will be, not where it is - within limits,
because chasers wobble sideways and any enemy that is winding up an attack
is braking to a stop.
"""

from __future__ import annotations

import math

from src import bestiary
from src.eos_env import Observation


# How much of the target's velocity the lead trusts, by family and state.
# Radial motion (toward or away from the player) is steady; sideways motion
# is the wobble of enemy.js (a sine with a 1.2 s period) and is discounted.
LEAD_TANGENT = {"chase": 0.4, "kite": 0.7, "summon": 0.7, "boss": 0.6, "flee": 0.7}
# States in which the enemy is braking to a halt: no lead at all.
BRAKING_STATES = {"aim", "windup", "prime", "channel", "attack", "transition", "recover"}


def pick_target(obs: Observation, current: dict | None = None) -> dict | None:
    """The enemy worth shooting first, or None when nothing is on screen."""
    best = None
    best_score = -math.inf
    for enemy in obs.enemies:
        score = target_score(obs, enemy)
        if current is not None and _same(enemy, current):
            score += 0.4  # hysteresis: do not flip between two equals
        if score > best_score:
            best, best_score = enemy, score
    return best


def target_score(obs: Observation, enemy: dict) -> float:
    row = bestiary.info(enemy)
    kind = row["behaviour"]
    state = enemy.get("state")
    distance = enemy.get("distance", 0.0)
    score = 1.0
    if kind == "summon":
        score += 3.0 + (1.0 if state == "channel" else 0.0)
    elif kind == "bomb" and state != "prime" and distance < row.get("trigger_dist", 125) + 60:
        score += 2.5
    elif kind == "kite" and state == "windup":
        score += 1.5
    elif kind == "charge" and state == "aim":
        score += 1.0
    elif kind == "boss":
        if state == "transition":
            return -math.inf  # invulnerable while roaring
        if enemy.get("attack") == "heavyCharge" and enemy.get("attack_step", 0) >= 2:
            score += 2.0  # stunned against a wall
    elif kind == "flee":
        score *= 0.2
    max_hp = enemy.get("max_hp") or 1.0
    score += 0.8 * (1.0 - enemy.get("hp", max_hp) / max_hp)
    if enemy.get("hp", max_hp) <= obs.damage:
        score += 2.0  # one more orb finishes it: do not look away
    score -= distance / 470.0
    if distance > obs.shot_range:
        score *= 0.3
    if enemy.get("los") is False:
        score *= 0.15
    return score


def aim_vector(obs: Observation, target: dict, lead: bool = True) -> tuple[float, float]:
    """Unit vector to shoot along so the orb meets the target."""
    dx = target["x"] - obs.x
    dy = target["y"] - obs.y
    distance = math.hypot(dx, dy)
    if distance < 1e-6:
        return (1.0, 0.0)
    if not lead or target.get("state") in BRAKING_STATES:
        return (dx / distance, dy / distance)
    vx = target.get("vx", 0.0)
    vy = target.get("vy", 0.0)
    kind = bestiary.behaviour(target)
    # A charger in its dash flies straight: trust everything. Everyone else:
    # keep the radial part, damp the sideways part.
    if kind == "charge" and target.get("state") == "dash":
        px, py = vx, vy
    else:
        ux, uy = dx / distance, dy / distance
        radial = vx * ux + vy * uy
        tx, ty = vx - radial * ux, vy - radial * uy
        damp = LEAD_TANGENT.get(kind, 0.4)
        px, py = radial * ux + damp * tx, radial * uy + damp * ty
    speed = max(200.0, obs.shot_speed)
    t = min(intercept_time(dx, dy, px, py, speed), LEAD_MAX_TIME)
    mx, my = dx + px * t, dy + py * t
    norm = math.hypot(mx, my)
    if norm < 1e-6:
        return (dx / distance, dy / distance)
    return (mx / norm, my / norm)


def intercept_time(dx: float, dy: float, vx: float, vy: float, speed: float) -> float:
    """Flight time for an orb at `speed` to meet a point moving at (vx, vy).

    Solves |d + v t| = s t exactly: the smallest positive root of
    (v.v - s^2) t^2 + 2 (d.v) t + d.d = 0. Falls back to the straight
    flight time when the target outruns the orb.
    """
    a = vx * vx + vy * vy - speed * speed
    b = 2.0 * (dx * vx + dy * vy)
    c = dx * dx + dy * dy
    straight = math.sqrt(c) / speed
    if abs(a) < 1e-6:
        return -c / b if b < 0 else straight
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return straight
    root = math.sqrt(disc)
    candidates = [t for t in ((-b - root) / (2 * a), (-b + root) / (2 * a)) if t > 0]
    return min(candidates) if candidates else straight


# A dash lasts 0.34 s and an orb rarely flies for more than half a second
# before the target has changed its mind: beyond this the prediction is
# noise, not information.
LEAD_MAX_TIME = 0.6


def _same(a: dict, b: dict) -> bool:
    return a.get("type") == b.get("type") and abs(a["x"] - b["x"]) < 60 and abs(a["y"] - b["y"]) < 60
