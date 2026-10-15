"""
Tactics are parameter sets, not hardcoded behavior. Every tactic exposes
the same interface so the simulation engine never needs special-case logic
per tactic: swap the Tactic object, get different team behavior.

A Tactic bundles:
  1. A formation (relative slot offsets for each role)
  2. Shape parameters (line height, width, compactness) that deform the
     formation based on ball position
  3. Decision parameters (pressing intensity, tempo, directness, risk
     tolerance) that weight the on-ball decision policy
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from footysim.core import Role, PITCH_LENGTH, PITCH_WIDTH


# ---------------------------------------------------------------------------
# Formations: base relative positions, defined for a team attacking toward
# +x, normalized to a 0-1 grid (0 = own goal line, 1 = opponent goal line;
# y: 0 = bottom touchline, 1 = top touchline). These get scaled to the real
# pitch and then shifted by the shape function during the match.
# ---------------------------------------------------------------------------

FORMATIONS = {
    "4-4-2": [
        (Role.GK,  0.03, 0.50),
        (Role.LB,  0.18, 0.12),
        (Role.CB,  0.15, 0.37),
        (Role.CB,  0.15, 0.63),
        (Role.RB,  0.18, 0.88),
        (Role.LM,  0.45, 0.12),
        (Role.CM,  0.42, 0.37),
        (Role.CM,  0.42, 0.63),
        (Role.RM,  0.45, 0.88),
        (Role.ST,  0.75, 0.38),
        (Role.ST,  0.75, 0.62),
    ],
    "4-3-3": [
        (Role.GK,  0.03, 0.50),
        (Role.LB,  0.18, 0.12),
        (Role.CB,  0.15, 0.37),
        (Role.CB,  0.15, 0.63),
        (Role.RB,  0.18, 0.88),
        (Role.CDM, 0.38, 0.50),
        (Role.CM,  0.50, 0.30),
        (Role.CM,  0.50, 0.70),
        (Role.LW,  0.75, 0.15),
        (Role.RW,  0.75, 0.85),
        (Role.ST,  0.80, 0.50),
    ],
    "5-3-2": [
        (Role.GK,  0.03, 0.50),
        (Role.LB,  0.20, 0.10),
        (Role.CB,  0.14, 0.30),
        (Role.CB,  0.12, 0.50),
        (Role.CB,  0.14, 0.70),
        (Role.RB,  0.20, 0.90),
        (Role.CM,  0.45, 0.30),
        (Role.CDM, 0.40, 0.50),
        (Role.CM,  0.45, 0.70),
        (Role.ST,  0.75, 0.40),
        (Role.ST,  0.75, 0.60),
    ],
    "4-2-3-1": [
        (Role.GK,  0.03, 0.50),
        (Role.LB,  0.18, 0.12),
        (Role.CB,  0.15, 0.37),
        (Role.CB,  0.15, 0.63),
        (Role.RB,  0.18, 0.88),
        (Role.CDM, 0.35, 0.37),
        (Role.CDM, 0.35, 0.63),
        (Role.CAM, 0.58, 0.50),
        (Role.LW,  0.65, 0.15),
        (Role.RW,  0.65, 0.85),
        (Role.ST,  0.80, 0.50),
    ],
}


@dataclass
class RoleInstruction:
    """
    An override for a single formation slot, FM-style. Any field left as
    None falls back to the team's own Tactic value for that field --
    override only what you actually want to change for this player.
    `leash_radius` is special: it isn't a Tactic field at all, it's a hard
    cap (in meters) on how far this player's tick-by-tick target can be
    from their home formation slot, regardless of what the team shape or
    pressing logic would otherwise pull them toward. That's what makes
    "tell the CDM to stay back" actually stick even under a high-press,
    ball-following team tactic.
    """
    label: str = ""
    leash_radius: Optional[float] = None
    line_height: Optional[float] = None
    width: Optional[float] = None
    press_intensity: Optional[float] = None
    press_trigger_distance: Optional[float] = None
    tempo: Optional[float] = None
    directness: Optional[float] = None
    risk_tolerance: Optional[float] = None
    width_in_possession: Optional[float] = None
    shoot_eagerness: Optional[float] = None


@dataclass
class Tactic:
    """
    All knobs a user sets to define a team's instructions. Values are
    0-1 unless noted, acting as weights/intensities rather than hard rules.
    """
    name: str
    formation: str = "4-4-2"

    # --- shape parameters (how formation deforms relative to ball) ---
    line_height: float = 0.5       # 0 = very deep (low block), 1 = very high (high line)
    width: float = 0.5             # 0 = narrow/compact, 1 = very wide
    compactness: float = 0.5       # 0 = players spread far from formation slot, 1 = tight to slot
    ball_following: float = 0.4    # how much the whole shape shifts toward ball x-position

    # --- pressing / defensive parameters ---
    press_intensity: float = 0.5   # 0 = no pressing (drop off), 1 = aggressive pressing
    press_trigger_distance: float = 20.0  # meters: how far from the ball a player will press

    # --- on-ball decision parameters ---
    tempo: float = 0.5             # 0 = slow/patient buildup, 1 = fast/direct
    directness: float = 0.5        # 0 = prefer short/safe passes, 1 = prefer forward/risky passes
    risk_tolerance: float = 0.5    # 0 = only attempt high-probability actions, 1 = will gamble
    width_in_possession: float = 0.5  # 0 = keep play central, 1 = use the flanks heavily
    shoot_eagerness: float = 0.5   # 0 = reluctant to shoot unless clear chance, 1 = shoots on sight

    # --- per-role customization ---
    default_leash_radius: float = 20.0  # meters: default max wander from home slot, overridable per role
    role_instructions: dict = field(default_factory=dict)  # {slot_index: RoleInstruction}

    def formation_slots(self):
        return FORMATIONS[self.formation]

    def slots_for_role(self, role: Role) -> list:
        """e.g. tactic.slots_for_role(Role.CB) -> [1, 2] for a back four."""
        return [i for i, (r, _, _) in enumerate(self.formation_slots()) if r == role]

    def instruct(self, slot_index: int, **overrides) -> RoleInstruction:
        """
        Set or update the instruction for one formation slot. Only pass
        the fields you want to override, e.g.:
            tactic.instruct(tactic.slots_for_role(Role.CDM)[0],
                             label="Stay back", leash_radius=8,
                             directness=0.15, press_intensity=0.15)
        """
        instr = self.role_instructions.get(slot_index, RoleInstruction())
        for key, value in overrides.items():
            if not hasattr(instr, key):
                raise ValueError(f"RoleInstruction has no field '{key}'")
            setattr(instr, key, value)
        self.role_instructions[slot_index] = instr
        return instr

    def clear_instruction(self, slot_index: int):
        self.role_instructions.pop(slot_index, None)


# A few preset tactics for convenience / quick comparisons.
PRESET_TACTICS = {
    "possession": Tactic(
        name="Possession-based",
        formation="4-3-3",
        line_height=0.55, width=0.55, compactness=0.6, ball_following=0.3,
        press_intensity=0.4, press_trigger_distance=15.0,
        tempo=0.3, directness=0.25, risk_tolerance=0.3,
        width_in_possession=0.45, shoot_eagerness=0.35,
    ),
    "high_press": Tactic(
        name="High Press",
        formation="4-2-3-1",
        line_height=0.8, width=0.5, compactness=0.75, ball_following=0.6,
        press_intensity=0.9, press_trigger_distance=28.0,
        tempo=0.65, directness=0.55, risk_tolerance=0.55,
        width_in_possession=0.5, shoot_eagerness=0.6,
    ),
    "low_block_counter": Tactic(
        name="Low Block Counter",
        formation="5-3-2",
        line_height=0.2, width=0.4, compactness=0.8, ball_following=0.15,
        press_intensity=0.2, press_trigger_distance=10.0,
        tempo=0.8, directness=0.85, risk_tolerance=0.6,
        width_in_possession=0.6, shoot_eagerness=0.65,
    ),
    "direct_wide": Tactic(
        name="Direct & Wide",
        formation="4-4-2",
        line_height=0.5, width=0.75, compactness=0.45, ball_following=0.4,
        press_intensity=0.55, press_trigger_distance=18.0,
        tempo=0.7, directness=0.7, risk_tolerance=0.5,
        width_in_possession=0.8, shoot_eagerness=0.55,
    ),
}


def _formation_shape_position(role: Role, slot_x: float, slot_y: float,
                               line_height: float, width: float, ball_following: float,
                               ball_x_norm: float, attacking_direction: int) -> np.ndarray:
    """
    The core 'formation shape function': base slot, shifted by
    ball_following, stretched/compressed by line_height / width. Shared by
    the team-wide and per-slot (role-instructed) callers below so the
    geometry logic exists in exactly one place.
    """
    shift = (ball_x_norm - 0.5) * ball_following
    x = slot_x + shift

    defensive_roles = {Role.GK, Role.CB, Role.LB, Role.RB}
    height_weight = 0.35 if role in defensive_roles else 0.12
    x += (line_height - 0.5) * height_weight
    x = float(np.clip(x, 0.02, 0.98))

    center_y = 0.5
    y = center_y + (slot_y - center_y) * (0.5 + width)
    y = float(np.clip(y, 0.02, 0.98))

    if attacking_direction == 1:
        real_x = x * PITCH_LENGTH
    else:
        real_x = (1 - x) * PITCH_LENGTH
    real_y = y * PITCH_WIDTH

    return np.array([real_x, real_y])


def target_position_for(player_role_index: int, tactic: Tactic, ball_x_norm: float,
                         attacking_direction: int) -> np.ndarray:
    """Team-wide version: every slot uses the tactic's own shape parameters
    with no per-role overrides. Kept for callers that don't need per-role
    customization (kickoff/goal-kick/corner reset positioning)."""
    role, slot_x, slot_y = tactic.formation_slots()[player_role_index]
    return _formation_shape_position(role, slot_x, slot_y, tactic.line_height, tactic.width,
                                      tactic.ball_following, ball_x_norm, attacking_direction)


def effective_value(tactic: Tactic, slot_index: int, field_name: str):
    """Resolve a tactical field for one slot: the slot's own override if
    set, else the team-wide tactic default."""
    instr = tactic.role_instructions.get(slot_index)
    if instr is not None:
        val = getattr(instr, field_name, None)
        if val is not None:
            return val
    return getattr(tactic, field_name)


def effective_leash(tactic: Tactic, slot_index: int) -> float:
    instr = tactic.role_instructions.get(slot_index)
    if instr is not None and instr.leash_radius is not None:
        return instr.leash_radius
    return tactic.default_leash_radius


def target_position_for_slot(slot_index: int, tactic: Tactic, ball_x_norm: float,
                              attacking_direction: int) -> np.ndarray:
    """Per-role version: resolves this slot's own line_height/width (if
    overridden) before computing the shape position. ball_following stays
    team-wide, since shifting the whole block together is a collective
    shape decision, not an individual one."""
    role, slot_x, slot_y = tactic.formation_slots()[slot_index]
    line_height = effective_value(tactic, slot_index, "line_height")
    width = effective_value(tactic, slot_index, "width")
    return _formation_shape_position(role, slot_x, slot_y, line_height, width,
                                      tactic.ball_following, ball_x_norm, attacking_direction)


def base_slot_position(slot_index: int, tactic: Tactic, attacking_direction: int) -> np.ndarray:
    """A player's 'home' position: the raw formation slot with no
    tactical shifting at all (neutral ball position, no line-height push,
    no width stretch). This is the anchor a player's leash radius is
    measured from."""
    role, slot_x, slot_y = tactic.formation_slots()[slot_index]
    return _formation_shape_position(role, slot_x, slot_y, line_height=0.5, width=0.5,
                                      ball_following=0.0, ball_x_norm=0.5,
                                      attacking_direction=attacking_direction)


def clamp_to_leash(anchor: np.ndarray, target: np.ndarray, leash_radius: float) -> np.ndarray:
    """Pull `target` back toward `anchor` if it's further than leash_radius
    away, preserving direction. This is what makes a tight leash (e.g. a
    disciplined CDM) override any other pull -- shape, ball-following, or
    pressing -- once the player would otherwise wander too far from home."""
    delta = target - anchor
    dist = float(np.linalg.norm(delta))
    if dist <= leash_radius or dist < 1e-6:
        return target
    return anchor + delta / dist * leash_radius
