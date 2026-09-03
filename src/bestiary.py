"""What the game's data files say about each enemy family.

Read from the game's source (js/data/enemies.js, js/data/bosses.js, version
4.1.0, the build served by itch): the behaviour state machine each family
runs, its speed, its size, and the numbers that decide how far away a fight
is safe - a charger's dash length, a bomber's trigger distance and blast
radius, the orbs a grimoire releases when it dies. The policies consult this
table instead of guessing from a silhouette; the observation only carries
the type id.
"""

from __future__ import annotations

# behaviour: chase (walks at the player), charge (aims 0.5 s then dashes in
# a straight line for 0.34 s), kite (keeps 260-430 px, winds up 0.52 s then
# shoots), bomb (runs in, arms at trigger_dist, blows up after fuse), summon
# (flees, channels 1.1 s, two minions), shield (front shield, contourn it),
# flee (the mimic), boss (its own state machine, see bosses.js).
ENEMIES: dict[str, dict] = {
    "wisp": {"behaviour": "chase", "speed": 125, "radius": 14, "contact": 1, "hp": 3},
    "lurcher": {"behaviour": "charge", "speed": 150, "radius": 13, "contact": 1, "hp": 4, "charge_speed": 560, "charge_damage": 2},
    "tank": {"behaviour": "chase", "speed": 66, "radius": 25, "contact": 2, "hp": 11},
    "shooter": {"behaviour": "kite", "speed": 95, "radius": 13, "contact": 1, "hp": 3, "shot_speed": 250},
    "bomber": {"behaviour": "bomb", "speed": 158, "radius": 12, "contact": 1, "hp": 2, "trigger_dist": 125, "fuse": 1.0, "blast_radius": 110, "blast_damage": 2},
    "summoner": {"behaviour": "summon", "speed": 85, "radius": 15, "contact": 1, "hp": 5, "summons": "wisp"},
    "flyer": {"behaviour": "chase", "speed": 165, "radius": 12, "contact": 1, "hp": 2, "flies": True},
    # forest
    "ronce": {"behaviour": "charge", "speed": 135, "radius": 14, "contact": 1, "hp": 5, "charge_speed": 500, "charge_damage": 2, "death_hazard": (60, 0.9)},
    "spore": {"behaviour": "bomb", "speed": 130, "radius": 14, "contact": 1, "hp": 3, "trigger_dist": 135, "fuse": 1.2, "blast_radius": 135, "blast_damage": 2, "death_hazard": (70, 0.8)},
    # temple
    "sentinelle": {"behaviour": "kite", "speed": 90, "radius": 14, "contact": 1, "hp": 4, "shot_speed": 295, "shots": 3, "spread": 0.3},
    # library
    "grimoire": {"behaviour": "chase", "speed": 175, "radius": 12, "contact": 1, "hp": 3, "flies": True, "death_shots": (4, 190)},
    "chuchoteur": {"behaviour": "summon", "speed": 85, "radius": 15, "contact": 1, "hp": 6, "summons": "grimoire"},
    # sky ruins
    "zephyr": {"behaviour": "charge", "speed": 205, "radius": 11, "contact": 1, "hp": 2, "charge_speed": 640, "charge_damage": 2, "flies": True},
    # volcano
    "brasier": {"behaviour": "chase", "speed": 72, "radius": 26, "contact": 2, "hp": 13, "trail": (26, 0.6, 1.2)},
    "escarbille": {"behaviour": "bomb", "speed": 185, "radius": 10, "contact": 1, "hp": 2, "trigger_dist": 110, "fuse": 0.8, "blast_radius": 95, "blast_damage": 2, "death_hazard": (55, 0.7)},
    # swamp
    "noye": {"behaviour": "chase", "speed": 62, "radius": 22, "contact": 2, "hp": 10},
    "feufollet": {"behaviour": "kite", "speed": 118, "radius": 12, "contact": 1, "hp": 3, "shot_speed": 235, "shot_slow": True},
    # frost necropolis
    "revenant": {"behaviour": "charge", "speed": 125, "radius": 13, "contact": 1, "hp": 5, "charge_speed": 540, "charge_damage": 2},
    "revenant-brise": {"behaviour": "charge", "speed": 150, "radius": 11, "contact": 1, "hp": 2, "charge_speed": 580, "charge_damage": 2},
    "pleureuse": {"behaviour": "summon", "speed": 80, "radius": 15, "contact": 1, "hp": 7, "summons": "revenant"},
    # abyss
    "ombre": {"behaviour": "chase", "speed": 150, "radius": 13, "contact": 1, "hp": 4},
    "ombrelette": {"behaviour": "chase", "speed": 210, "radius": 9, "contact": 1, "hp": 1},
    "gardienverre": {"behaviour": "shield", "speed": 88, "radius": 20, "contact": 2, "hp": 8},
    # crypt
    "exhume": {"behaviour": "chase", "speed": 88, "radius": 18, "contact": 1, "hp": 6},
    "ossement": {"behaviour": "chase", "speed": 190, "radius": 10, "contact": 1, "hp": 2},
    # heart of the eclipse
    "echo": {"behaviour": "chase", "speed": 175, "radius": 12, "contact": 1, "hp": 3, "flies": True, "death_shots": (6, 210)},
    "fragment": {"behaviour": "bomb", "speed": 175, "radius": 11, "contact": 1, "hp": 2, "trigger_dist": 130, "fuse": 0.9, "blast_radius": 105, "blast_damage": 2},
    # the golden mimic: harmless, flees, evaporates after 7 s
    "mimique": {"behaviour": "flee", "speed": 235, "radius": 12, "contact": 0, "hp": 6},
}

BOSS_IDS = {
    "eclipse", "vorace", "tisseuse", "sylve", "zenith", "pyroclaste",
    "colosse", "gorgone", "roigivre", "miroir", "annonciateur",
}

UNKNOWN = {"behaviour": "chase", "speed": 150, "radius": 14, "contact": 1, "hp": 3}


def info(enemy: dict) -> dict:
    """The data row of an observed enemy (a boss gets its own family)."""
    kind = enemy.get("type")
    if enemy.get("boss") or kind in BOSS_IDS:
        return {"behaviour": "boss", "speed": 90, "radius": enemy.get("radius", 30), "contact": 2, "hp": 70}
    return ENEMIES.get(kind, UNKNOWN)


def behaviour(enemy: dict) -> str:
    return info(enemy)["behaviour"]
