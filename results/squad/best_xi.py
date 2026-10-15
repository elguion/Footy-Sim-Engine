"""
The specific Best XI, built from scouting-note descriptions rather than
randomized stats. Formation is fixed at 4-3-3, which is the natural fit
for this squad: exactly 2 CB + 2 FB from the 5 defenders provided, a
CDM + 2 CM from the 6 midfielders, and 3 forwards from the 5 attackers.

Slot index -> role (from tactics.FORMATIONS["4-3-3"]):
  0 GK  1 LB  2 CB  3 CB  4 RB  5 CDM  6 CM  7 CM  8 LW  9 RW  10 ST

Selection reasoning (brief, since the numbers below are the real content):
  - Chuks over Japrike: better decision-making/1v1s and commands the box,
    which outweighs Japrike's narrower set-piece-shot-stopping edge, since
    Japrike's poor wall organization undercuts that advantage anyway.
  - Ibrahim + Shippu at CB: Ibrahim's pace/recovery covers Shippu's lack of
    it, Shippu's physicality covers what Ibrahim doesn't lack anyway but
    reinforces -- a fast ball-player next to a physical stopper.
  - Richard (RB) + Yimika (LB): Richard for defensive reliability and
    recovery pace, Yimika for his passing/dribbling to combine with the
    midfield three. Michael (also a fullback option) is left as a
    developing backup per his own "can be worked on" description.
  - Anthony (CDM) as the disciplined destroyer, Bayo as the controlling
    passer (his reluctance to shoot is a non-issue in a CM role), Dani
    Miller as the clear standout given the near-absence of real weaknesses
    in his profile -- an advanced box-to-box/creative #8.
  - Hafeez (LW) and Victor (RW) either side of Eric (ST): Hafeez cuts
    inside for cutbacks, Victor stretches the pitch in behind, Eric is
    explicitly "the ideal 9" -- holds the ball up and finishes.
  - Joseph, Adeyemi, Kingsley, Ademide, Winston, Michael, Japrike form a
    capable bench; Adeyemi in particular reads as a designed impact sub
    ("best faced against tired legs"), not a starter.

One honest modeling gap: GK aerial command and set-piece organizing
("Japrike handles saves better but can't organize the wall", "Chuks weak
on crosses due to height") aren't separately modeled -- the corner/cross
mechanic resolves generically against the defending team's outfield
pressure, not goalkeeper-specific. The numbers below reflect what the
engine actually uses; they don't silently assume a mechanic that isn't there.
"""

from footysim.core import Player, Abilities, Team, Role
from footysim.tactics import FORMATIONS
from footysim.team_builder import clinical_finisher, playmaker, speed_dribbler

FORMATION_NAME = "4-3-3"

# slot_index: (name, Abilities)
ROSTER = {
    0: ("Chuks", Abilities(
        pace=65, passing=45, shooting=35, dribbling=50, tackling=88,
        crossing=35, heading=50, stamina_rating=80, vision=85, strength=78)),
    1: ("Yimika", Abilities(
        pace=72, passing=82, shooting=45, dribbling=78, tackling=72,
        crossing=70, heading=55, stamina_rating=75, vision=80, strength=68)),
    2: ("Ibrahim", Abilities(
        pace=82, passing=78, shooting=50, dribbling=60, tackling=85,
        crossing=50, heading=85, stamina_rating=75, vision=80, strength=85)),
    3: ("Shippu", Abilities(
        pace=50, passing=45, shooting=35, dribbling=40, tackling=80,
        crossing=35, heading=75, stamina_rating=70, vision=60, strength=88)),
    4: ("Richard", Abilities(
        pace=85, passing=55, shooting=40, dribbling=55, tackling=78,
        crossing=55, heading=55, stamina_rating=78, vision=65, strength=78)),
    5: ("Anthony", Abilities(
        pace=60, passing=55, shooting=45, dribbling=70, tackling=88,
        crossing=40, heading=60, stamina_rating=78, vision=78, strength=85)),
    6: ("Bayo", Abilities(
        pace=65, passing=85, shooting=42, dribbling=80, tackling=70,
        crossing=65, heading=55, stamina_rating=75, vision=82, strength=65)),
    7: ("Dani Miller", Abilities(
        pace=85, passing=72, shooting=68, dribbling=88, tackling=72,
        crossing=65, heading=55, stamina_rating=90, vision=85, strength=68)),
    8: ("Hafeez", Abilities(
        pace=72, passing=75, shooting=55, dribbling=85, tackling=45,
        crossing=68, heading=50, stamina_rating=72, vision=70, strength=60)),
    9: ("Victor", Abilities(
        pace=88, passing=62, shooting=55, dribbling=80, tackling=40,
        crossing=65, heading=50, stamina_rating=78, vision=68, strength=60)),
    10: ("Eric", Abilities(
        pace=65, passing=68, shooting=82, dribbling=60, tackling=45,
        crossing=45, heading=85, stamina_rating=72, vision=72, strength=80)),
}

# Standout traits worth representing as conditional modifiers, not just raw stats
SPECIAL_ABILITIES = {
    2: [playmaker(1.15)],          # Ibrahim: good passing range / set-piece delivery threat
    7: [speed_dribbler(1.2)],      # Dani Miller: pace + dribbling + attacking runs
    10: [clinical_finisher(1.2)],  # Eric: "the ideal 9", good finisher
}


def build_best_xi(tactic, team_id: str, name: str, attacking_direction: int) -> Team:
    if tactic.formation != FORMATION_NAME:
        raise ValueError(f"best_xi is built for {FORMATION_NAME}, tactic uses {tactic.formation}")

    slots = FORMATIONS[FORMATION_NAME]
    players = []
    for i, (role, _, _) in enumerate(slots):
        player_name, abilities = ROSTER[i]
        player = Player(
            player_id=f"{team_id}_{i}",
            name=player_name,
            role=role,
            slot_index=i,
            abilities=abilities,
            special_abilities=SPECIAL_ABILITIES.get(i, []),
            pass_range=22 + abilities.passing * 0.18,
            shot_range=14 + abilities.shooting * 0.14,
            cross_range=18 + abilities.crossing * 0.14,
            team_id=team_id,
        )
        players.append(player)

    return Team(team_id=team_id, name=name, players=players, tactic=tactic,
                attacking_direction=attacking_direction)
