"""What a relic, a purchase or a pact is worth to this run.

The bank holds 159 relics (js/data/items.js) whose effects the game
composes from three channels: additive bonuses, multipliers and behaviour
counters (orbs, homing, piercing, chain lightning...). The value below is
expressed in half-hearts, the currency a run actually spends: a heart of
max life is a hit survived on floor 4, an orb that intercepts enemy shots
saves a share of the 36 % of life lost to projectiles, and damage shortens
every fight, which is where the hits come from. The caps of
player.recompute (additive damage above +6 and rate of fire above +60 %
count half) are applied the same way.
"""

from __future__ import annotations

import math

from src.eos_env import Observation


RARITY_RANK = {None: 0, "common": 1, "rare": 2, "epic": 3, "legendary": 4}

# Relics whose price is mechanical, not arithmetic (items.js, "maudits").
REFUSED = {"verre-brulant", "faim-noire"}

# Named synergies (js/data/synergies.js): two counters that together get
# a real mechanical bonus.
SYNERGIES = [
    ("poison", "explodeOnKill"), ("chain", "poison"), ("multishot", "bounces"),
    ("drones", "poison"), ("split", "chain"), ("homing", "ghost"),
    ("orbs", "flameRing"), ("pierce", "bounces"), ("blink", "dashDamage"),
    ("slowShots", "novaOnHurt"), ("explodeOnKill", "chain"), ("multishot", "split"),
]

FLAG_WORTH = {
    "orbs": 0.5, "flameRing": 0.4, "drones": 0.5, "slowShots": 0.5, "poison": 0.5,
    "pierce": 0.35, "chain": 0.4, "homing": 0.3, "split": 0.3, "ghost": 0.5,
    "bounces": 0.2, "explodeOnKill": 0.3, "explodeOnCrit": 0.3, "blink": 0.4,
    "dashDamage": 0.1, "thorns": 0.1, "novaOnHurt": 0.15, "slowAllOnHurt": 0.3,
    "backShot": 0.15, "sideShots": 0.25, "multishot": 0.0,  # multishot is in the dps term
    # serpentine shots swing +/-31 degrees (game.js): they miss from the
    # distance this policy keeps
    "wave": -0.5,
}

BASE_DAMAGE = 1.0
BASE_FIRE_DELAY = 0.21


def dps(damage: float, fire_delay: float, mods: dict | None = None) -> float:
    """Damage per second of the build, extra shots counted at 0.8."""
    m = mods or {}
    volleys = 1.0 + 0.8 * m.get("multishot", 0) + 0.85 * min(1, m.get("backShot", 0)) \
        + 1.4 * min(1, m.get("sideShots", 0))
    return damage * volleys / max(0.07, fire_delay)


def item_value(obs: Observation, entry: dict) -> float:
    """Worth of the relic in `entry` (a pickup or a shop slot), in half-hearts."""
    mods = entry.get("item_mods") or {}
    rarity = RARITY_RANK.get(entry.get("item_rarity"), 0)
    if mods.get("add") is None:
        # observation without the full modifier set: rarity alone
        return 0.25 * rarity
    if mods.get("id") in REFUSED:
        return -10.0
    add = mods.get("add", {})
    mult = mods.get("mult", {})
    flags = mods.get("flags", {})
    if add.get("maxHp", 0) < 0:
        return -10.0
    floor = obs.floor
    have = obs.mods or {}

    # --- damage output ---
    before = dps(obs.damage, obs.fire_delay, have)
    dmg_add = add.get("damage", 0.0)
    if obs.damage > 6.0:
        dmg_add *= 0.5
    dmg_mult = mult.get("damage", 1.0)
    if obs.damage > 2.0 and dmg_mult > 1.0:
        dmg_mult = 1.0 + (dmg_mult - 1.0) * 0.5
    fire_mult = mult.get("fireDelay", 1.0)
    if fire_mult < 1.0 and obs.fire_delay < BASE_FIRE_DELAY / 1.6:
        fire_mult = 1.0 - (1.0 - fire_mult) * 0.5
    crit = add.get("critChance", 0.0) * (1.0 + add.get("critMult", 0.0)) \
        + 0.03 * add.get("critMult", 0.0)
    after_mods = dict(have)
    for k, v in flags.items():
        after_mods[k] = after_mods.get(k, 0) + v
    after = dps((obs.damage + dmg_add) * dmg_mult * (1.0 + crit), obs.fire_delay * fire_mult, after_mods)
    value = math.log2(max(after, 0.1) / max(before, 0.1)) * (1.3 if floor >= 4 else 1.0)

    # --- life ---
    value += 0.9 * (add.get("maxHp", 0.0) / 2.0) * (1.5 if floor >= 3 else 1.0)
    value += 0.25 * min(add.get("healNow", 0.0), obs.hp_missing)
    value += 8.0 * add.get("healOnKill", 0.0)

    # --- behaviours ---
    for flag, count in flags.items():
        worth = FLAG_WORTH.get(flag, 0.1)
        if flag == "ghost" and obs.obstacles:
            worth *= 2.0
        value += worth * count
    for a, b in SYNERGIES:
        if (flags.get(a) and have.get(b)) or (flags.get(b) and have.get(a)):
            value += 0.3
            break

    # --- mobility and reach ---
    value += 2.5 * (1.0 / mult.get("dashCooldown", 1.0) - 1.0)
    value += 4.0 * (mult.get("moveSpeed", 1.0) - 1.0)
    value += 0.8 * (mult.get("range", 1.0) - 1.0)
    value += 0.8 * (mult.get("shotSpeed", 1.0) - 1.0)
    value += 0.5 * (mult.get("shotRadius", 1.0) - 1.0)
    value += 0.05 * rarity
    return value


# --- decisions on the room's offers (js/game/game.js, js/data/difficulty.js) ---

def gambler_price(floor: int) -> int:
    return round(18 * (1 + (floor - 1) * 0.12))


def wants_altar(obs: Observation) -> bool:
    """One heart of max life for an epic relic: only from a wide, full bar."""
    if obs.enemy_count or obs.hp < obs.max_hp - 1:
        return False
    return obs.max_hp >= (8 if obs.floor <= 2 else 10)


def wants_gamble(obs: Observation) -> bool:
    """10 % jackpot, 50 % a relic, 40 % nothing: only with a spare purse."""
    if obs.enemy_count or obs.hp < obs.max_hp - 1 or obs.gambles_left <= 0:
        return False
    return obs.gold >= gambler_price(obs.floor) + 35


def wants_challenge(obs: Observation) -> bool:
    """Light the brazier: a timed wave at +20 % life for two rare relics."""
    if obs.enemy_count or obs.defi_done or obs.defi_active:
        return False
    power = dps(obs.damage, obs.fire_delay, obs.mods) / dps(BASE_DAMAGE, BASE_FIRE_DELAY)
    if obs.floor <= 3:
        return obs.hp >= obs.max_hp - 2 and power >= 1.5
    return obs.hp >= obs.max_hp and power >= 3.0


def shop_reserve(obs: Observation) -> int:
    """Gold to keep back for a heart when the run is already hurt."""
    return 25 if obs.hp < obs.max_hp - 1 else 0
