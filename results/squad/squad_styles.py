"""
Four style variants for the Best XI, all on 4-3-3 so personnel and
formation are held constant and only the tactical approach differs --
isolating style as the variable actually being tested.

Two role instructions are applied to every style, since they're facts
about the players, not tactical choices:
  - Bayo (slot 6): "would rather pass than shoot" -> low personal
    shoot_eagerness regardless of team style.
  - Shippu (slot 3): slow, needs to play it safe -> tighter leash than
    the team default, so he doesn't get caught upfield in transition
    even under a high-press or direct style.
"""

from footysim.core import Role
from footysim.tactics import Tactic

BASE_FORMATION = "4-3-3"


def _apply_fixed_player_instructions(tactic: Tactic) -> Tactic:
    tactic.instruct(6, label="Prefers passing over shooting", shoot_eagerness=0.15)
    tactic.instruct(3, label="Play it safe, stay disciplined (slow CB)", leash_radius=12.0)
    return tactic


def possession_control() -> Tactic:
    t = Tactic(
        name="Possession Control", formation=BASE_FORMATION,
        line_height=0.55, width=0.5, compactness=0.6, ball_following=0.3,
        press_intensity=0.4, press_trigger_distance=15.0,
        tempo=0.3, directness=0.25, risk_tolerance=0.3,
        width_in_possession=0.45, shoot_eagerness=0.4,
    )
    return _apply_fixed_player_instructions(t)


def high_press() -> Tactic:
    t = Tactic(
        name="High Press", formation=BASE_FORMATION,
        line_height=0.75, width=0.5, compactness=0.7, ball_following=0.55,
        press_intensity=0.85, press_trigger_distance=25.0,
        tempo=0.6, directness=0.5, risk_tolerance=0.5,
        width_in_possession=0.5, shoot_eagerness=0.55,
    )
    return _apply_fixed_player_instructions(t)


def low_block_counter() -> Tactic:
    t = Tactic(
        name="Low Block Counter", formation=BASE_FORMATION,
        line_height=0.2, width=0.45, compactness=0.8, ball_following=0.15,
        press_intensity=0.25, press_trigger_distance=11.0,
        tempo=0.8, directness=0.85, risk_tolerance=0.55,
        width_in_possession=0.6, shoot_eagerness=0.55,
    )
    return _apply_fixed_player_instructions(t)


def direct_wide_target() -> Tactic:
    t = Tactic(
        name="Direct Wide Target", formation=BASE_FORMATION,
        line_height=0.5, width=0.7, compactness=0.45, ball_following=0.4,
        press_intensity=0.5, press_trigger_distance=18.0,
        tempo=0.65, directness=0.7, risk_tolerance=0.45,
        width_in_possession=0.75, shoot_eagerness=0.55,
    )
    return _apply_fixed_player_instructions(t)


STYLES = {
    "Possession Control": possession_control,
    "High Press": high_press,
    "Low Block Counter": low_block_counter,
    "Direct Wide Target": direct_wide_target,
}
