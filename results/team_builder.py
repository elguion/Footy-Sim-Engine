"""
Helpers to construct a full 11-player team from a formation + ability
profile, so you don't hand-write 11 Player() calls every time.
"""

from __future__ import annotations
import numpy as np
from footysim.core import Player, Abilities, Team, Role, SpecialAbility, ActionType
from footysim.tactics import Tactic, FORMATIONS


def build_team(team_id: str, name: str, tactic: Tactic, attacking_direction: int,
                ability_overrides: dict | None = None,
                special_abilities: dict | None = None,
                seed: int | None = None) -> Team:
    """
    ability_overrides: optional dict {player_index: {ability_name: value}}
        to give specific players specific stats. Any ability not specified
        is randomized around a base level with small variance so players
        aren't all identical.
    special_abilities: optional dict {player_index: [SpecialAbility, ...]}
    """
    rng = np.random.default_rng(seed)
    ability_overrides = ability_overrides or {}
    special_abilities = special_abilities or {}

    slots = FORMATIONS[tactic.formation]
    players = []
    for i, (role, _, _) in enumerate(slots):
        base = 68 + rng.normal(0, 4)
        ab_kwargs = {}
        for field_name in ["pace", "passing", "shooting", "dribbling", "tackling",
                            "crossing", "heading", "stamina_rating", "vision", "strength"]:
            ab_kwargs[field_name] = float(np.clip(base + rng.normal(0, 6), 30, 99))

        overrides = ability_overrides.get(i, {})
        ab_kwargs.update(overrides)
        abilities = Abilities(**ab_kwargs)

        player = Player(
            player_id=f"{team_id}_{i}",
            name=f"{name} #{i+1} ({role.name})",
            role=role,
            slot_index=i,
            abilities=abilities,
            special_abilities=special_abilities.get(i, []),
            pass_range=22 + ab_kwargs["passing"] * 0.18,
            shot_range=14 + ab_kwargs["shooting"] * 0.14,
            cross_range=18 + ab_kwargs["crossing"] * 0.14,
            team_id=team_id,
        )
        players.append(player)

    return Team(team_id=team_id, name=name, players=players, tactic=tactic,
                attacking_direction=attacking_direction)


def clinical_finisher(modifier: float = 1.25) -> SpecialAbility:
    def condition(context: dict) -> float:
        return 1.0 if context.get("in_box") else 0.0
    return SpecialAbility(name="Clinical Finisher", action=ActionType.SHOOT,
                           modifier=modifier, condition=condition)


def playmaker(modifier: float = 1.2) -> SpecialAbility:
    def condition(context: dict) -> float:
        # stronger effect the longer the pass (more "vision-dependent")
        max_range = context.get("max_range", 30)
        distance = context.get("distance", 0)
        return min(distance / max_range, 1.0) if max_range else 0.0
    return SpecialAbility(name="Playmaker", action=ActionType.PASS,
                           modifier=modifier, condition=condition)


def speed_dribbler(modifier: float = 1.3) -> SpecialAbility:
    def condition(context: dict) -> float:
        return 1.0
    return SpecialAbility(name="Speed Dribbler", action=ActionType.DRIBBLE,
                           modifier=modifier, condition=condition)
