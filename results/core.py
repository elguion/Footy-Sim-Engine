"""
Core data structures for the tactics simulation.

Pitch coordinate system:
    x: 0 (left goal line, team A defends) -> 105 (right goal line, team B defends)
    y: 0 (bottom touchline) -> 68 (top touchline)
    These are standard full-size pitch dimensions in meters.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
GOAL_WIDTH = 7.32
GOAL_Y_CENTER = PITCH_WIDTH / 2


class Role(Enum):
    GK = auto()
    CB = auto()
    LB = auto()
    RB = auto()
    CDM = auto()
    CM = auto()
    CAM = auto()
    LM = auto()
    RM = auto()
    LW = auto()
    RW = auto()
    ST = auto()


class ActionType(Enum):
    MOVE = auto()
    PASS = auto()
    DRIBBLE = auto()
    SHOOT = auto()
    CROSS = auto()
    TACKLE = auto()
    HOLD = auto()  # do nothing meaningful this tick (off-ball positioning only)


class TackleType(Enum):
    STANDING = auto()
    SLIDING = auto()


@dataclass
class Abilities:
    """
    Per-action ability ratings, 0-100 scale, same convention as familiar
    football games. These are the base (unmodified) ratings for a player.
    """
    pace: float = 70.0
    passing: float = 70.0
    shooting: float = 70.0
    dribbling: float = 70.0
    tackling: float = 70.0
    crossing: float = 70.0
    heading: float = 70.0
    stamina_rating: float = 70.0  # how slowly stamina depletes / how well it recovers
    vision: float = 70.0          # influences decision quality, not used for success rolls directly
    strength: float = 70.0        # physical duels: ball-shielding, winning/resisting a tackle

    def get(self, action: ActionType) -> float:
        mapping = {
            ActionType.PASS: self.passing,
            ActionType.DRIBBLE: self.dribbling,
            ActionType.SHOOT: self.shooting,
            ActionType.CROSS: self.crossing,
            ActionType.TACKLE: self.tackling,
        }
        return mapping.get(action, 70.0)


@dataclass
class SpecialAbility:
    """
    A conditional multiplier on top of base ability. The condition is a
    callable that receives the match context and returns True/False or a
    0-1 weight; the modifier is applied multiplicatively to the relevant
    ability when the condition is active.

    Example: Player with 'clinical_finisher' gets shooting *= 1.25 when
    the shot is taken from inside the penalty box.
    """
    name: str
    action: ActionType
    modifier: float  # multiplicative, e.g. 1.25 = +25%
    condition: callable  # (context: dict) -> float in [0,1], gating strength


@dataclass
class Player:
    player_id: str
    name: str
    role: Role
    abilities: Abilities
    slot_index: int = 0  # index into this team's formation slot list -- how role instructions are targeted
    special_abilities: list = field(default_factory=list)  # list[SpecialAbility]

    # ranges, in meters, beyond which the action is not attemptable at all
    pass_range: float = 35.0
    shot_range: float = 25.0
    cross_range: float = 30.0

    # mutable match state
    position: np.ndarray = field(default_factory=lambda: np.zeros(2))
    target_position: np.ndarray = field(default_factory=lambda: np.zeros(2))
    stamina: float = 100.0          # 0-100, depletes with distance covered
    has_ball: bool = False
    team_id: Optional[str] = None

    def effective_ability(self, action: ActionType, context: dict) -> float:
        """Base ability, scaled by current stamina, with special abilities applied."""
        base = self.abilities.get(action)
        stamina_factor = 0.65 + 0.35 * (self.stamina / 100.0)  # at 0 stamina, still 65% effective
        value = base * stamina_factor
        for sa in self.special_abilities:
            if sa.action == action:
                weight = sa.condition(context)
                if weight > 0:
                    # blend toward full modifier based on condition weight
                    value *= (1.0 + (sa.modifier - 1.0) * weight)
        return value

    def distance_to(self, point: np.ndarray) -> float:
        return float(np.linalg.norm(self.position - point))


@dataclass
class Ball:
    """
    The ball is its own agent with position and velocity. While `carrier`
    is set, the ball's position is synced to that player each tick (it's
    "at their feet"). While `in_flight` is True, no player controls it,
    it travels on its own trajectory and can be intercepted or collected
    by whichever player reaches it.
    """
    position: np.ndarray = field(default_factory=lambda: np.zeros(2))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    in_flight: bool = False
    carrier: Optional["Player"] = None
    target: Optional[np.ndarray] = None
    intended_receiver: Optional["Player"] = None
    passing_team: Optional["Team"] = None
    pass_quality: float = 70.0   # passer's effective ability at launch, used for interception contests
    kind: str = "ground"         # "ground" | "through" | "cross" | "shot" -- for future differentiation


@dataclass
class TeamStats:
    goals: int = 0
    shots: int = 0
    shots_on_target: int = 0
    passes_attempted: int = 0
    passes_completed: int = 0
    crosses_attempted: int = 0
    crosses_completed: int = 0
    tackles_attempted: int = 0
    tackles_won: int = 0
    sliding_tackles_attempted: int = 0
    fouls_committed: int = 0
    interceptions: int = 0
    possession_ticks: int = 0
    dribbles_attempted: int = 0
    dribbles_completed: int = 0
    distance_covered_total: float = 0.0


@dataclass
class Team:
    team_id: str
    name: str
    players: list  # list[Player]
    tactic: "Tactic"  # forward reference, defined in tactics.py
    attacking_direction: int = 1  # +1 means attacks toward x=105, -1 means attacks toward x=0
    stats: TeamStats = field(default_factory=TeamStats)

    def goalkeeper(self) -> Player:
        for p in self.players:
            if p.role == Role.GK:
                return p
        raise ValueError(f"Team {self.name} has no goalkeeper")

    def outfield(self) -> list:
        return [p for p in self.players if p.role != Role.GK]

    def own_goal_x(self) -> float:
        return 0.0 if self.attacking_direction == 1 else PITCH_LENGTH

    def opp_goal_x(self) -> float:
        return PITCH_LENGTH if self.attacking_direction == 1 else 0.0
