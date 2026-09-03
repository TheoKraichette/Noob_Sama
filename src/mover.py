"""Where to step: every move the keyboard allows, scored on the next
half-second.

The keyboard moves along eight directions, walking or dashing, or not at
all: seventeen candidates. Each is simulated for 0.6 s at 20 Hz against
what the game will do in that time, all of it read from the game's own
rules (bestiary.py, js/game/enemy.js, js/game/boss.js): orbs keep flying,
telegraphed zones burst when their fuse runs out, a charger that has locked
its aim dashes along that line, a primed bomb blows up on schedule, a
shooter in its windup releases its orb, a guardian's announced attack
unfolds. A candidate that gets hit is out; among the rest, the cheapest
one for the fight's purpose (distance band, line of sight, direction the
rules asked for) wins. Safety is a constraint here, never an objective: a
policy that maximises clearance runs to the corners and stops shooting
(V41-V42 of the log lost 18 % of the rooms doing exactly that).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from src import bestiary
from src.eos_env import Observation


DT = 0.05
HORIZON = 0.6
STEPS = int(round(HORIZON / DT))
PLAYER_R = 15.0
DASH_SPEED = 950.0
DASH_TIME = 0.13
DASH_IFRAMES = 0.16
ORB_HIT = 6.0 + PLAYER_R - 4.0     # enemyshot.js: radius + player.radius - 4
CONTACT_SLACK = -3.0                # game.js: enemy.radius + player.radius - 3
HAZARD_PAD = PLAYER_R * 0.4         # boss.js: hz.r + p.radius * 0.4
BLAST_PAD = PLAYER_R * 0.5          # game.js: blast + p.radius * 0.5
MARGIN = 6.0                        # one simulation step of slack

SQ = math.sqrt(0.5)
DIRECTIONS: dict[str, tuple[float, float]] = {
    "up": (0.0, -1.0), "down": (0.0, 1.0), "left": (-1.0, 0.0), "right": (1.0, 0.0),
    "up_left": (-SQ, -SQ), "up_right": (SQ, -SQ), "down_left": (-SQ, SQ), "down_right": (SQ, SQ),
}


@dataclass
class Circle:
    """A thing that hurts if the player's centre comes within `r` of it."""

    x: float
    y: float
    r: float
    dmg: float = 1.0


@dataclass
class Candidate:
    action: str
    dash: bool
    direction: tuple[float, float]
    path: list[tuple[float, float]] = field(default_factory=list)   # position at k*DT, k=0..STEPS
    hit_t: float | None = None
    hit_dmg: float = 0.0
    clearance: float = math.inf
    pinned: float = 0.0       # how much of the intended travel a wall ate
    end: tuple[float, float] = (0.0, 0.0)
    cost: float = 0.0
    safe: bool = True


# ---------------------------------------------------------------------------
# What the next 0.6 s hold
# ---------------------------------------------------------------------------

def threats(obs: Observation) -> list[list[Circle]]:
    """Dangerous circles at each simulation instant k*DT."""
    frames: list[list[Circle]] = [[] for _ in range(STEPS + 1)]
    px, py = obs.x, obs.y

    def orb(x0: float, y0: float, vx: float, vy: float, t0: float, dmg: float) -> None:
        for k in range(STEPS + 1):
            t = k * DT
            if t < t0:
                continue
            frames[k].append(Circle(x0 + vx * (t - t0), y0 + vy * (t - t0), ORB_HIT + MARGIN, dmg))

    def burst(x: float, y: float, r: float, at: float, dmg: float) -> None:
        for k in range(STEPS + 1):
            if abs(k * DT - at) <= DT * 0.75:
                frames[k].append(Circle(x, y, r + MARGIN, dmg))

    def ring(x: float, y: float, speed: float, t0: float, dmg: float) -> None:
        # A ring whose angular offset is random: the whole annulus is
        # off-limits inside the horizon (the real orbs are dodged one by
        # one once they exist).
        for k in range(STEPS + 1):
            t = k * DT
            if t <= t0:
                continue
            frames[k].append(Circle(x, y, -(speed * (t - t0)), dmg))  # negative r = annulus, see hits()

    for sh in obs.shots:
        orb(sh["x"], sh["y"], sh["vx"], sh["vy"], 0.0, sh.get("dmg", 1))

    for hz in obs.hazards:
        remaining = hz.get("remaining", 0.0)
        if remaining <= HORIZON + DT:
            burst(hz["x"], hz["y"], hz["r"] + HAZARD_PAD, remaining, hz.get("dmg", 1))

    for e in obs.enemies:
        row = bestiary.info(e)
        kind = row["behaviour"]
        state = e.get("state")
        st = e.get("state_t", 0.0)
        ex, ey = e["x"], e["y"]
        radius = e.get("radius", row["radius"])
        contact_r = radius + PLAYER_R + CONTACT_SLACK + MARGIN
        contact = e.get("contact_damage", row["contact"])
        vx, vy = e.get("vx", 0.0), e.get("vy", 0.0)
        lock = e.get("lock_dir", [0.0, 0.0])
        if contact <= 0:
            continue

        if kind == "boss":
            _boss_threats(obs, e, frames, orb, burst, ring)
            continue

        if kind == "charge" and state == "aim":
            launch = max(0.0, 0.5 - st)
            speed = row.get("charge_speed", 560)
            for k in range(STEPS + 1):
                t = k * DT
                if t < launch:
                    frames[k].append(Circle(ex, ey, contact_r, contact))
                else:
                    flight = min(t - launch, 0.34)
                    frames[k].append(Circle(ex + lock[0] * speed * flight, ey + lock[1] * speed * flight,
                                            contact_r, row.get("charge_damage", 2)))
            continue
        if kind == "charge" and state == "dash":
            left = max(0.0, 0.34 - st)
            for k in range(STEPS + 1):
                t = min(k * DT, left)
                frames[k].append(Circle(ex + vx * t, ey + vy * t, contact_r, row.get("charge_damage", 2)))
            continue
        if kind == "bomb" and state == "prime":
            at = max(0.0, row.get("fuse", 1.0) - st)
            burst(ex, ey, row.get("blast_radius", 110) + BLAST_PAD, at, row.get("blast_damage", 2))
            for k in range(STEPS + 1):
                if k * DT < at:
                    frames[k].append(Circle(ex, ey, contact_r, contact))
            continue
        if kind == "kite" and state == "windup":
            at = max(0.0, 0.52 - st)
            speed = row.get("shot_speed", 250) * e.get("shot_speed_mult", 1.0)
            n = row.get("shots", 1)
            spread = row.get("spread", 0.0)
            base = math.atan2(lock[1], lock[0])
            for i in range(n):
                a = base + (i - (n - 1) / 2) * spread
                dx, dy = math.cos(a), math.sin(a)
                orb(ex + dx * (radius + 8), ey + dy * (radius + 8), dx * speed, dy * speed, at, e.get("shot_dmg", 1))
        # Every other body walks on at its current velocity, capped at its
        # family speed (the wobble is in the velocity already).
        for k in range(STEPS + 1):
            t = k * DT
            frames[k].append(Circle(ex + vx * t, ey + vy * t, contact_r, contact))

    return frames


def _boss_threats(obs: Observation, e: dict, frames, orb, burst, ring) -> None:
    """A guardian's body, and the attack it has announced (js/game/boss.js)."""
    ex, ey = e["x"], e["y"]
    radius = e.get("radius", 30)
    contact = e.get("contact_damage", 2)
    contact_r = radius + PLAYER_R + CONTACT_SLACK + MARGIN
    vx, vy = e.get("vx", 0.0), e.get("vy", 0.0)
    name = e.get("attack")
    t_atk = e.get("attack_time", 0.0)
    step = e.get("attack_step", 0)
    lock = e.get("lock_dir", [0.0, 0.0])
    aux2 = e.get("aux2", [0.0, 0.0])
    tier = e.get("tier", 0)
    shot_dmg = e.get("shot_dmg", 1)
    mult = e.get("shot_speed_mult", 1.0)
    ring_n = 8 + min(8, tier * 2)

    moving = True
    if e.get("state") == "attack" and name:
        if name == "heavyCharge" and step == 0:
            launch = max(0.0, 0.75 - t_atk)
            speed = 720 * (1 + min(0.25, tier * 0.02))
            for k in range(STEPS + 1):
                t = k * DT
                if t < launch:
                    frames[k].append(Circle(ex, ey, contact_r, contact))
                else:
                    f = t - launch
                    frames[k].append(Circle(ex + lock[0] * speed * f, ey + lock[1] * speed * f, contact_r + 10, contact))
            moving = False
        elif name == "tripleLunge" and step < 6:
            # even step: aiming, odd step: the lunge itself (0.26 s at 620)
            speed = 620 * (1 + min(0.25, tier * 0.02))
            step_t = t_atk - aux2[1]
            if step % 2 == 0:
                launch = max(0.0, 0.45 - step_t)
                for k in range(STEPS + 1):
                    t = k * DT
                    if t < launch:
                        frames[k].append(Circle(ex, ey, contact_r, contact))
                    else:
                        f = min(t - launch, 0.26)
                        frames[k].append(Circle(ex + lock[0] * speed * f, ey + lock[1] * speed * f, contact_r + 10, contact))
            else:
                left = max(0.0, 0.26 - step_t)
                for k in range(STEPS + 1):
                    f = min(k * DT, left)
                    frames[k].append(Circle(ex + vx * f, ey + vy * f, contact_r + 10, contact))
            moving = False
        elif name == "magmaWake" and step == 0:
            launch = max(0.0, 0.8 - t_atk)
            for k in range(STEPS + 1):
                t = k * DT
                f = max(0.0, t - launch)
                frames[k].append(Circle(ex + lock[0] * 420 * f, ey + lock[1] * 420 * f, contact_r + 10, contact))
            moving = False
        elif name == "fanVolley" and step == 0:
            at = max(0.0, 0.55 - t_atk)
            base = math.atan2(lock[1], lock[0])
            for i in range(-2, 3):
                a = base + i * 0.22
                dx, dy = math.cos(a), math.sin(a)
                orb(ex + dx * (radius + 6), ey + dy * (radius + 6), dx * 265 * mult, dy * 265 * mult, at, shot_dmg)
        elif name in ("ringVolley", "doubleRing") and step == 0:
            ring(ex, ey, 225 * mult, max(0.0, 0.6 - t_atk), shot_dmg)
            if name == "doubleRing":
                ring(ex, ey, 225 * mult, max(0.0, 0.95 - t_atk), shot_dmg)
        elif name == "doubleRing" and step == 1:
            ring(ex, ey, 225 * mult, max(0.0, 0.95 - t_atk), shot_dmg)
        elif name == "tidalPulse":
            for i, (at, speed) in enumerate(((0.7, 150), (1.05, 195), (1.4, 240))):
                if i >= step:
                    ring(ex, ey, speed * mult, max(0.0, at - t_atk), shot_dmg)
        elif name == "mirrorEcho" and step == 1:
            ring(lock[0], lock[1], 215 * mult, max(0.0, 1.15 - t_atk), shot_dmg)
        elif name == "shatterClone" and step == 1:
            ring(lock[0], lock[1], 210 * mult, max(0.0, 1.2 - t_atk), shot_dmg)
            ring(aux2[0], aux2[1], 210 * mult, max(0.0, 1.2 - t_atk), shot_dmg)
        elif name == "slamQuake" and step == 0:
            burst(ex, ey, 140 + HAZARD_PAD, max(0.0, 0.85 - t_atk), shot_dmg)
        elif name == "sweepBeam" and step == 1:
            # orbs every 0.055 s along an angle sweeping 1.4 rad in 1.1 s
            start = aux2[0]
            sweep_t = t_atk - 0.6
            k0 = math.ceil(sweep_t / 0.055)
            for i in range(k0, k0 + 12):
                fire_t = i * 0.055
                if fire_t > 1.1:
                    break
                a = start + (fire_t / 1.1) * 1.4
                dx, dy = math.cos(a), math.sin(a)
                orb(ex + dx * (radius + 6), ey + dy * (radius + 6), dx * 300 * mult, dy * 300 * mult,
                    max(0.0, fire_t - sweep_t), shot_dmg)
        elif name == "aftershock" and step == 0:
            at = max(0.0, 0.7 - t_atk)
            for r, n, fuse in ((120, 7, 0.9), (235, 12, 1.35)):
                for i in range(n):
                    a = (i / n) * math.tau + r * 0.1
                    burst(ex + math.cos(a) * r, ey + math.sin(a) * r, 56 + HAZARD_PAD, at + fuse, shot_dmg)
        if name in ("heavyCharge", "tripleLunge", "magmaWake"):
            return
        if e.get("state") == "attack":
            moving = False  # every other attack brakes the boss
    if e.get("state") == "transition":
        moving = False
    for k in range(STEPS + 1):
        t = k * DT if moving else 0.0
        frames[k].append(Circle(ex + vx * t, ey + vy * t, contact_r, contact))


# ---------------------------------------------------------------------------
# The candidates and their fate
# ---------------------------------------------------------------------------

def candidates(obs: Observation, allow_dash: bool = True) -> list[Candidate]:
    speed = obs.move_speed * (0.6 if obs.slow_t > 0 else 1.0)
    out = [Candidate("noop", False, (0.0, 0.0))]
    for name, d in DIRECTIONS.items():
        out.append(Candidate(name, False, d))
        if allow_dash and obs.dash_ready:
            out.append(Candidate("dash_" + name, True, d))
    walls = _walls(obs)
    frames = threats(obs)
    for c in out:
        _simulate(obs, c, speed, walls)
        _judge(c, frames)
    return out


def _walls(obs: Observation):
    wt = obs.room_wt or 46.0
    return (wt + PLAYER_R, obs.room_w - wt - PLAYER_R, wt + PLAYER_R, obs.room_h - wt - PLAYER_R,
            [(o["x"], o["y"], o["r"] + PLAYER_R) for o in obs.obstacles])


def _simulate(obs: Observation, c: Candidate, speed: float, walls) -> None:
    x0, x1, y0, y1, pillars = walls
    x, y = obs.x, obs.y
    dx, dy = c.direction
    path = [(x, y)]
    wanted = 0.0
    got = 0.0
    for k in range(1, STEPS + 1):
        t = k * DT
        if c.dash and t <= DASH_TIME + 1e-9:
            v = DASH_SPEED
        elif c.dash and t - DT < DASH_TIME:
            v = DASH_SPEED * (DASH_TIME - (t - DT)) / DT + speed * (t - DASH_TIME) / DT
        else:
            v = speed
        nx, ny = x + dx * v * DT, y + dy * v * DT
        wanted += v * DT
        # walls, then pillars (room.js collideCircle, doors ignored: a
        # candidate that leaves through a door is judged on the room only)
        nx = min(max(nx, x0), x1)
        ny = min(max(ny, y0), y1)
        for ox, oy, orr in pillars:
            ddx, ddy = nx - ox, ny - oy
            d2 = ddx * ddx + ddy * ddy
            if 1e-6 < d2 < orr * orr:
                d = math.sqrt(d2)
                nx, ny = ox + ddx / d * orr, oy + ddy / d * orr
        got += math.hypot(nx - x, ny - y)
        x, y = nx, ny
        path.append((x, y))
    c.path = path
    c.end = (x, y)
    c.pinned = 0.0 if wanted < 1e-6 else max(0.0, 1.0 - got / wanted)


def _judge(c: Candidate, frames: list[list[Circle]]) -> None:
    clearance = math.inf
    for k, (x, y) in enumerate(c.path):
        t = k * DT
        immune = c.dash and t <= DASH_IFRAMES
        for circle in frames[k]:
            if circle.r < 0:
                # annulus: a ring of orbs at radius -r around the centre
                d = abs(math.hypot(x - circle.x, y - circle.y) + circle.r)
                margin = d - (ORB_HIT + MARGIN)
            else:
                margin = math.hypot(x - circle.x, y - circle.y) - circle.r
            clearance = min(clearance, margin)
            if margin < 0 and not immune:
                if c.hit_t is None:
                    c.hit_t = t
                c.hit_dmg = max(c.hit_dmg, circle.dmg)
    c.clearance = clearance
    c.safe = c.hit_t is None


# ---------------------------------------------------------------------------
# Choosing
# ---------------------------------------------------------------------------

@dataclass
class Goal:
    """What the fight wants from this step, beyond not getting hit."""

    prefer: tuple[float, float] | None = None     # direction the rules asked for
    target: dict | None = None                    # enemy to keep in the band
    band: tuple[float, float] = (0.0, math.inf)   # wanted distance to the target
    keep_los: bool = True
    hold: bool = False                            # standing still is what the rules want
    previous: tuple[float, float] | None = None   # last direction taken (hysteresis)
    weight_band: float = 1.0
    avoid_future: bool = False                    # zones bursting beyond the horizon


def choose(obs: Observation, goal: Goal, allow_dash: bool = True) -> Candidate:
    cands = candidates(obs, allow_dash)
    walk_safe = any(c.safe and not c.dash for c in cands)
    for c in cands:
        cost = 0.0
        if not c.safe:
            cost += 1000.0 - 100.0 * c.hit_t + 200.0 * c.hit_dmg
        if c.clearance < 30.0:
            cost += 8.0 * (1.0 - max(c.clearance, 0.0) / 30.0)
        ex, ey = c.end
        if goal.target is not None:
            d = math.hypot(goal.target["x"] - ex, goal.target["y"] - ey)
            lo, hi = goal.band
            off = lo - d if d < lo else d - hi if d > hi else 0.0
            cost += goal.weight_band * off / 100.0
            if goal.keep_los and not _line_clear(obs, ex, ey, goal.target["x"], goal.target["y"]):
                cost += 1.5
        cost += 5.0 * c.pinned
        if goal.prefer is not None and (c.direction[0] or c.direction[1]):
            cos = c.direction[0] * goal.prefer[0] + c.direction[1] * goal.prefer[1]
            cost -= 2.0 * cos
        elif goal.prefer is not None and goal.hold:
            cost -= 2.0
        elif goal.hold and c.action == "noop":
            cost -= 2.0
        elif goal.prefer is None and not goal.hold and c.action == "noop":
            cost += 0.5
        if goal.previous is not None and (c.direction[0] or c.direction[1]):
            if c.direction[0] * goal.previous[0] + c.direction[1] * goal.previous[1] < 0:
                cost += 0.3
        for z in obs.strata_zones:
            if math.hypot(z["x"] - ex, z["y"] - ey) < z["r"] + PLAYER_R * 0.5:
                cost += 0.8
        if goal.avoid_future:
            # A zone that bursts after the horizon is not a hit yet, but
            # ending the step inside it means having to leave it later,
            # under whatever else is flying by then.
            for hz in obs.hazards:
                if hz.get("remaining", 0.0) > HORIZON + DT and \
                        math.hypot(hz["x"] - ex, hz["y"] - ey) < hz["r"] + HAZARD_PAD + 20.0:
                    cost += 3.0
        # The dash recharges in 0.85 s: spending it where a step suffices
        # leaves nothing for the charge that follows (V27 of the log).
        if c.dash and walk_safe:
            cost += 5.0
        c.cost = cost
    return min(cands, key=lambda c: c.cost)


def _line_clear(obs: Observation, x0: float, y0: float, x1: float, y1: float) -> bool:
    sx, sy = x1 - x0, y1 - y0
    len_sq = sx * sx + sy * sy
    for o in obs.obstacles:
        t = ((o["x"] - x0) * sx + (o["y"] - y0) * sy) / len_sq if len_sq > 1e-6 else 0.0
        t = min(max(t, 0.0), 1.0)
        cx, cy = x0 + sx * t - o["x"], y0 + sy * t - o["y"]
        if cx * cx + cy * cy < o["r"] * o["r"]:
            return False
    return True
