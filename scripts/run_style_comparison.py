import time
import json
from footysim.tactics import PRESET_TACTICS
from footysim.team_builder import build_team
from squads.best_xi import build_best_xi
from squads.squad_styles import STYLES
from footysim.match_engine import Match

N_SEEDS = 15  # match-random seeds per (style, opponent) combo
OPPONENT_CONFIGS = ["possession", "high_press"]  # benchmark opponent presets, fixed personnel per seed

results = {}  # style_name -> list of per-match dicts
t_start = time.time()

for style_name, style_factory in STYLES.items():
    results[style_name] = []
    for opp_key in OPPONENT_CONFIGS:
        for seed in range(N_SEEDS):
            tactic = style_factory()  # fresh Tactic each match (role_instructions dict is mutable)
            home = build_best_xi(tactic, "home", "Best XI", attacking_direction=1)

            # same opponent-ability seed across all 4 styles for a given (opp_key, seed)
            # combo, so every style faces an identical set of opposing squads
            away = build_team("away", f"Opponent({opp_key})", PRESET_TACTICS[opp_key],
                               attacking_direction=-1, seed=1000 + seed)

            match = Match(home, away, seed=5000 + seed)
            r = match.run()

            results[style_name].append({
                "opponent": opp_key,
                "seed": seed,
                "gf": r.home_score,
                "ga": r.away_score,
                "shots": r.home_stats.shots,
                "shots_on_target": r.home_stats.shots_on_target,
                "possession_ticks": r.home_stats.possession_ticks,
                "opp_possession_ticks": r.away_stats.possession_ticks,
                "passes_completed": r.home_stats.passes_completed,
                "passes_attempted": r.home_stats.passes_attempted,
                "tackles_won": r.home_stats.tackles_won,
                "fouls": r.home_stats.fouls_committed,
            })
    print(f"done: {style_name} ({time.time()-t_start:.0f}s elapsed)")

with open("results/style_results.json", "w") as f:
    json.dump(results, f)

print(f"\nTotal matches: {sum(len(v) for v in results.values())}, total time: {time.time()-t_start:.0f}s")
