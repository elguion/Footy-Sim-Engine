"""
Resolution mechanism: converts (player, action, context) into a success
probability via one shared function, then draws a stochastic outcome.

Every action type (pass, shot, dribble, cross, tackle) calls
success_probability() with different context, rather than having separate
hand-rolled logic per action. This keeps ability, stamina, special
abilities, pressure, distance, and angle combining the same way everywhere.
"""

from __future__ import annotations
import numpy as np
from footysim.core import ActionType, TackleType, Player, PITCH_WIDTH, GOAL_Y_CENTER

STANDING_TACKLE_RANGE = 1.8   # meters
SLIDE_TACKLE_RANGE = 4.2      # meters -- covers more ground, used as a last resort


def _logistic(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


def nearest_opponent_distance(player: Player, opponents: list) -> float:
    if not opponents:
        return 999.0
    return min(p.distance_to(player.position) for p in opponents)


def success_probability(
    player: Player,
    action: ActionType,
    distance: float,
    max_range: float,
    opponents: list,
    context_extra: dict | None = None,
) -> float:
    """
    Core success-probability function. Returns a value in (0, 1).

    distance:   distance in meters the action covers (pass length, shot
                distance, dribble step, etc.)
    max_range:  the player's effective range for this action; beyond this
                the action is not attemptable (caller should check first)
    opponents:  list of opposing players, used to compute pressure
    """
    context = {"distance": distance, "max_range": max_range}
    if context_extra:
        context.update(context_extra)

    ability = player.effective_ability(action, context)  # 0-100ish, stamina+special applied

    pressure_dist = nearest_opponent_distance(player, opponents)
    # pressure penalty: strongest within ~2m, fades out by ~8m
    pressure_penalty = max(0.0, (8.0 - pressure_dist)) * 2.2

    # distance penalty: scales with how close to max range the action is
    distance_ratio = min(distance / max_range, 1.0) if max_range > 0 else 1.0
    distance_penalty = (distance_ratio ** 2) * 28.0

    angle_penalty = context_extra.get("angle_penalty", 0.0) if context_extra else 0.0

    # combine into a logistic score, centered so ability ~70 with no
    # pressure/distance penalty gives roughly 80-85% success
    z = (ability - 35.0) / 11.0 - pressure_penalty / 11.0 - distance_penalty / 11.0 - angle_penalty / 11.0
    p = _logistic(z)
    return float(np.clip(p, 0.02, 0.98))


def shot_angle_penalty(shot_position: np.ndarray, goal_x: float) -> float:
    """
    Penalty that grows as the shooting angle to goal gets tighter
    (e.g. shooting from near the byline vs. straight in front of goal).
    """
    dx = abs(goal_x - shot_position[0])
    dy = abs(GOAL_Y_CENTER - shot_position[1])
    if dx < 0.5:
        dx = 0.5
    angle = np.arctan2(dy, dx)  # 0 = straight on, larger = more angled
    return float(angle * 14.0)  # scale to roughly same magnitude as other penalties


def resolve(rng: np.random.Generator, p_success: float) -> bool:
    return bool(rng.random() < p_success)


def tackle_success_probability(defender: Player, attacker: Player, tackle_type: TackleType,
                                distance: float) -> float:
    """
    A tackle is a duel, not a single-actor roll: the defender's tackling,
    strength, and closing pace work against the attacker's dribbling,
    strength (holding them off), and pace (getting away before the tackle
    lands). Sliding tackles can be attempted from further out but carry a
    lower baseline success rate and a steeper distance penalty, reflecting
    that they're a committed, harder-to-control lunge rather than a
    controlled challenge.
    """
    attacker_resistance = (attacker.abilities.dribbling * 0.5
                            + attacker.abilities.pace * 0.3
                            + attacker.abilities.strength * 0.2)
    defender_power = (defender.abilities.tackling * 0.55
                       + defender.abilities.strength * 0.30
                       + defender.abilities.pace * 0.15)

    if tackle_type == TackleType.STANDING:
        range_penalty = (distance / STANDING_TACKLE_RANGE) * 4.0
        baseline_adjust = 0.0
    else:
        range_penalty = (distance / SLIDE_TACKLE_RANGE) * 6.0
        baseline_adjust = -9.0  # a lunge is inherently riskier even up close

    z = (defender_power - attacker_resistance + baseline_adjust - range_penalty) / 11.0
    p_defender_wins = _logistic(z)
    return float(np.clip(p_defender_wins, 0.03, 0.9))


def resolve_contest(rng: np.random.Generator, attacker_strength: float, defender_strength: float) -> bool:
    """
    For 50/50s and tackles: returns True if the attacker (first arg) wins
    the contest, based on relative strength via a logistic comparison.
    """
    z = (attacker_strength - defender_strength) / 12.0
    p_attacker_wins = _logistic(z)
    return bool(rng.random() < p_attacker_wins)
