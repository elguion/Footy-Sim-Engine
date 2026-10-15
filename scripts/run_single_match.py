from footysim.tactics import PRESET_TACTICS
from footysim.team_builder import build_team
from footysim.match_engine import Match

home = build_team("home", "Possession FC", PRESET_TACTICS["possession"], attacking_direction=1, seed=1)
away = build_team("away", "High Press FC", PRESET_TACTICS["high_press"], attacking_direction=-1, seed=2)

match = Match(home, away, seed=42)
result = match.run()

print(f"FINAL SCORE: {home.name} {result.home_score} - {result.away_score} {away.name}\n")

for team, stats in [(home, result.home_stats), (away, result.away_stats)]:
    total_ticks = result.home_stats.possession_ticks + result.away_stats.possession_ticks
    poss_pct = 100 * stats.possession_ticks / total_ticks if total_ticks else 0
    print(f"--- {team.name} ---")
    print(f"  Possession: {poss_pct:.1f}%")
    print(f"  Shots: {stats.shots} (on target: {stats.shots_on_target})")
    print(f"  Passes: {stats.passes_completed}/{stats.passes_attempted} "
          f"({100*stats.passes_completed/max(stats.passes_attempted,1):.1f}%)")
    print(f"  Crosses: {stats.crosses_completed}/{stats.crosses_attempted}")
    print(f"  Dribbles: {stats.dribbles_completed}/{stats.dribbles_attempted}")
    print(f"  Tackles won: {stats.tackles_won}/{stats.tackles_attempted} "
          f"(sliding: {stats.sliding_tackles_attempted})")
    print(f"  Fouls committed: {stats.fouls_committed}")
    print(f"  Interceptions: {stats.interceptions}")
    print()

print("Events:")
for tick, desc in result.events:
    minute = tick * 2 / 60
    print(f"  {minute:.0f}' - {desc}")
