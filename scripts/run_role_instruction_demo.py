import numpy as np
from footysim.core import Role
from footysim.tactics import PRESET_TACTICS, base_slot_position
from footysim.team_builder import build_team
from footysim.match_engine import Match

# build the high-press tactic's CDM slot to "stay back" explicitly
tactic = PRESET_TACTICS["high_press"]
cdm_slots = tactic.slots_for_role(Role.CDM)
print("CDM slots in 4-2-3-1:", cdm_slots)

# two CDMs in 4-2-3-1 (role list has 2 CDM entries) -- instruct the first
cdm_slot = cdm_slots[0]
tactic.instruct(cdm_slot, label="Stay back, anchor", leash_radius=7.0,
                directness=0.1, press_intensity=0.1, press_trigger_distance=8.0)

home = build_team("home", "High Press FC", tactic, attacking_direction=1, seed=1)
away = build_team("away", "Opponent FC", PRESET_TACTICS["direct_wide"], attacking_direction=-1, seed=2)

match = Match(home, away, seed=7)

# track the instructed CDM's distance from its home slot over the match
cdm_player = home.players[cdm_slot]
other_cdm = home.players[cdm_slots[1]]  # the un-instructed one, for comparison
cdm_anchor = base_slot_position(cdm_slot, tactic, home.attacking_direction)
other_anchor = base_slot_position(cdm_slots[1], tactic, home.attacking_direction)

max_dist_instructed = 0.0
max_dist_other = 0.0

orig_step = match._step
def tracking_step():
    global max_dist_instructed, max_dist_other
    orig_step()
    d1 = np.linalg.norm(cdm_player.position - cdm_anchor)
    d2 = np.linalg.norm(other_cdm.position - other_anchor)
    max_dist_instructed = max(max_dist_instructed, d1)
    max_dist_other = max(max_dist_other, d2)
match._step = tracking_step

result = match.run()

print(f"\nFINAL SCORE: {home.name} {result.home_score} - {result.away_score} {away.name}")
print(f"\nInstructed CDM (leash=7.0) max distance from home slot: {max_dist_instructed:.1f}m")
print(f"Un-instructed CDM (default leash={tactic.default_leash_radius}) max distance from home slot: {max_dist_other:.1f}m")
print("\n(instructed CDM's max distance should stay <= ~7m; the other CDM should roam further)")
