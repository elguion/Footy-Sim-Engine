"""
The match engine: discrete-time tick loop, with the ball as a separate
agent.

Ball states:
  - carried: ball.carrier is a Player, ball.position == carrier.position
  - in flight: ball.in_flight is True, ball.carrier is None, the ball
    moves toward ball.target at its own velocity each tick, and can be
    intercepted by any opposing player it passes close to, or collected
    by whoever is nearest when it arrives (if the intended receiver isn't
    there, it's a loose ball and nearest-player-wins, same as real football).

Each tick:
  1. If the ball is in flight: advance it, check interception, check arrival
  2. Off-ball movement for every player not currently carrying the ball
     (this runs whether or not the ball is in flight, players keep moving)
  3. If a player now carries the ball: that player's team evaluates a
     decision policy and picks an action (pass/dribble/shoot/cross), which
     may launch the ball into flight rather than resolving instantly
  4. Stats/state bookkeeping, terminal check
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from footysim.core import (
    Player, Team, Ball, Role, ActionType, TackleType,
    PITCH_LENGTH, PITCH_WIDTH, GOAL_Y_CENTER, GOAL_WIDTH
)
from footysim.tactics import (
    target_position_for, target_position_for_slot, effective_value,
    effective_leash, base_slot_position, clamp_to_leash,
)
from footysim.resolution import (
    success_probability, shot_angle_penalty, resolve, resolve_contest,
    tackle_success_probability, STANDING_TACKLE_RANGE, SLIDE_TACKLE_RANGE,
    nearest_opponent_distance,
)

TICK_SECONDS = 2.0
MATCH_MINUTES = 90
TICKS_PER_MATCH = int(MATCH_MINUTES * 60 / TICK_SECONDS)

MAX_SPRINT_SPEED = 7.5     # m/s, upper bound on player closing speed at pace 100
INTERCEPTION_RADIUS = 2.2  # meters: how close an opponent must get to the flying ball to contest it
CAPTURE_RADIUS = 3.0       # meters: how close the intended receiver must be when the ball arrives
BEATEN_TOLERANCE = 1.5     # meters: how much closer to goal the attacker must be to count as "beaten" the defender
SLIDE_TACKLE_GATE_BASE = 0.15  # baseline chance a last-man even attempts a slide when the situation calls for it


@dataclass
class MatchResult:
    home_score: int
    away_score: int
    home_stats: "TeamStats"
    away_stats: "TeamStats"
    events: list


class Match:
    def __init__(self, home: Team, away: Team, seed: int | None = None):
        self.home = home
        self.away = away
        self.rng = np.random.default_rng(seed)
        self.ball = Ball(position=np.array([PITCH_LENGTH / 2, PITCH_WIDTH / 2]))
        self.events = []
        self.tick = 0
        self._place_kickoff()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _place_kickoff(self):
        for team in (self.home, self.away):
            for i, p in enumerate(team.players):
                target = self._slot_target(team, i, ball_x_norm=0.5)
                p.position = target.copy()
                p.target_position = target.copy()
                p.stamina = 100.0
                p.has_ball = False
        striker = self._first_striker(self.home)
        self._set_carrier(striker)
        self.ball.position = striker.position.copy()

    def _first_striker(self, team: Team) -> Player:
        for p in team.players:
            if p.role == Role.ST:
                return p
        return team.outfield()[0]

    def other_team(self, team: Team) -> Team:
        return self.away if team is self.home else self.home

    def team_of(self, player: Player) -> Team:
        return self.home if player.team_id == self.home.team_id else self.away

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> MatchResult:
        for self.tick in range(TICKS_PER_MATCH):
            self._step()
        return MatchResult(
            home_score=self.home.stats.goals,
            away_score=self.away.stats.goals,
            home_stats=self.home.stats,
            away_stats=self.away.stats,
            events=self.events,
        )

    def _step(self):
        if self.ball.in_flight:
            self._process_ball_flight()

        carrier = self.ball.carrier
        carrying_team = self.team_of(carrier) if carrier else None
        defending_team = self.other_team(carrying_team) if carrier else None

        if carrier:
            carrying_team.stats.possession_ticks += 1

        self._update_off_ball_positions(carrying_team, defending_team, carrier)

        if carrier and not self.ball.in_flight:
            self._decide_and_resolve(carrier, carrying_team, defending_team)

        if not self.ball.in_flight:
            self._check_ball_out_of_bounds()

    # ------------------------------------------------------------------
    # Ball flight
    # ------------------------------------------------------------------
    def _process_ball_flight(self):
        ball = self.ball
        step = ball.velocity * TICK_SECONDS
        remaining = ball.target - ball.position
        dist_remaining = float(np.linalg.norm(remaining))
        step_dist = float(np.linalg.norm(step))

        arrived = step_dist >= dist_remaining
        ball.position = ball.target.copy() if arrived else ball.position + step

        opponents = self.other_team(ball.passing_team).players
        for opp in opponents:
            if opp.distance_to(ball.position) < INTERCEPTION_RADIUS:
                interceptor_strength = (opp.effective_ability(ActionType.TACKLE, {}) * 0.6
                                         + opp.abilities.vision * 0.4)
                attacker_wins = resolve_contest(self.rng, ball.pass_quality, interceptor_strength)
                if not attacker_wins:
                    self._complete_flight(opp, intercepted=True)
                    return

        if arrived:
            if not self._in_bounds(ball.position):
                # a misplaced pass that sails past the touchline/byline --
                # must be caught here, before any carrier is assigned,
                # since assigning a carrier snaps the ball back onto the
                # pitch via their position and would hide the fact it went out
                self._ball_went_out_during_flight()
                return
            receiver = ball.intended_receiver
            if receiver.distance_to(ball.position) < CAPTURE_RADIUS:
                self._complete_flight(receiver, intercepted=False)
            else:
                all_players = self.home.players + self.away.players
                nearest = min(all_players, key=lambda p: p.distance_to(ball.position))
                is_turnover = nearest.team_id != ball.passing_team.team_id
                self._complete_flight(nearest, intercepted=is_turnover)

    def _in_bounds(self, pos: np.ndarray) -> bool:
        return 0.0 <= pos[0] <= PITCH_LENGTH and 0.0 <= pos[1] <= PITCH_WIDTH

    def _ball_went_out_during_flight(self):
        ball = self.ball
        passing_team = ball.passing_team
        ball.in_flight = False
        ball.target = None
        ball.intended_receiver = None
        ball.carrier = None
        # pass attempted but not completed -- no passes_completed credit
        self._corner_kick_restart(passing_team)

    def _complete_flight(self, new_carrier: Player, intercepted: bool):
        ball = self.ball
        passing_team = ball.passing_team
        if not intercepted:
            passing_team.stats.passes_completed += 1
        else:
            receiving_team = self.team_of(new_carrier)
            receiving_team.stats.interceptions += 1
        ball.in_flight = False
        ball.target = None
        ball.intended_receiver = None
        self._set_carrier(new_carrier)

    def _set_carrier(self, player: Player):
        if self.ball.carrier:
            self.ball.carrier.has_ball = False
        player.has_ball = True
        self.ball.carrier = player
        self.ball.position = player.position.copy()

    # ------------------------------------------------------------------
    # Off-ball movement
    # ------------------------------------------------------------------
    def _update_off_ball_positions(self, carrying_team: Team | None, defending_team: Team | None, carrier: Player | None):
        ball_x = self.ball.position[0]

        for team in (self.home, self.away):
            tactic = team.tactic
            ball_x_norm = ball_x / PITCH_LENGTH
            if team.attacking_direction == -1:
                ball_x_norm = 1 - ball_x_norm

            is_defending = (team is defending_team)

            for i, p in enumerate(team.players):
                if p is carrier:
                    continue

                target = self._slot_target(team, i, ball_x_norm)

                if is_defending and carrier is not None:
                    press_trigger = effective_value(tactic, i, "press_trigger_distance")
                    press_intensity = effective_value(tactic, i, "press_intensity")
                    dist_to_ball = p.distance_to(self.ball.position)
                    if dist_to_ball < press_trigger and press_intensity > self.rng.random():
                        # pressing target is also subject to this slot's
                        # leash -- a disciplined player won't join the press
                        # past their instructed radius even if triggered
                        anchor = base_slot_position(i, tactic, team.attacking_direction)
                        leash = effective_leash(tactic, i)
                        target = clamp_to_leash(anchor, self.ball.position.copy(), leash)

                self._move_toward(p, target)

    def _slot_target(self, team: Team, slot_index: int, ball_x_norm: float) -> np.ndarray:
        """The single source of truth for where a slot 'wants' to be: its
        own resolved shape position (line_height/width overrides applied),
        clamped to that slot's leash around its home anchor. Used both for
        tick-by-tick open-play movement and for every discrete restart
        (kickoff, goal kick, corner), so an instructed player's leash holds
        even when the rest of the team reorganizes its shape for a restart."""
        tactic = team.tactic
        target = target_position_for_slot(slot_index, tactic, ball_x_norm=ball_x_norm,
                                           attacking_direction=team.attacking_direction)
        anchor = base_slot_position(slot_index, tactic, team.attacking_direction)
        leash = effective_leash(tactic, slot_index)
        return clamp_to_leash(anchor, target, leash)

    def _move_toward(self, player: Player, target: np.ndarray):
        delta = target - player.position
        dist = float(np.linalg.norm(delta))
        if dist < 0.01:
            return
        pace_factor = player.abilities.pace / 100.0
        stamina_factor = 0.55 + 0.45 * (player.stamina / 100.0)
        max_step = MAX_SPRINT_SPEED * pace_factor * stamina_factor * TICK_SECONDS
        step = min(dist, max_step)
        direction = delta / dist
        player.position = player.position + direction * step

        depletion_rate = 0.09 * (1.2 - player.abilities.stamina_rating / 100.0 * 0.5)
        player.stamina = max(0.0, player.stamina - step * depletion_rate)

    # ------------------------------------------------------------------
    # On-ball decision + resolution
    # ------------------------------------------------------------------
    def _decide_and_resolve(self, carrier: Player, team: Team, opp_team: Team):
        tactic = team.tactic
        goal_x = team.opp_goal_x()
        dist_to_goal = abs(goal_x - carrier.position[0])

        if self._attempt_defensive_tackle(carrier, team, opp_team):
            return  # ball was won (or a failed slide still ends the carrier's turn this tick)

        if carrier.role == Role.GK:
            self._goalkeeper_decide(carrier, team, opp_team)
            return

        action = self._choose_action(carrier, team, opp_team, dist_to_goal)

        if action == ActionType.SHOOT:
            self._attempt_shot(carrier, team, opp_team)
        elif action == ActionType.CROSS:
            self._attempt_cross(carrier, team, opp_team)
        elif action == ActionType.PASS:
            self._launch_pass(carrier, team, opp_team)
        else:
            self._attempt_dribble(carrier, team, opp_team)

    def _is_beaten(self, defender: Player, attacker: Player, opp_team: Team) -> bool:
        """True if the attacker is already closer to the defending team's
        goal than the defender is -- i.e. the defender is goal-side no
        longer, so a standing challenge from here doesn't make sense."""
        own_goal_x = opp_team.own_goal_x()
        attacker_dist = abs(attacker.position[0] - own_goal_x)
        defender_dist = abs(defender.position[0] - own_goal_x)
        return (defender_dist - attacker_dist) > BEATEN_TOLERANCE

    def _is_last_man(self, defender: Player, opp_team: Team) -> bool:
        own_goal_x = opp_team.own_goal_x()
        outfielders = opp_team.outfield()
        if not outfielders:
            return False
        most_goal_side = min(outfielders, key=lambda p: abs(p.position[0] - own_goal_x))
        return most_goal_side is defender

    def _attempt_defensive_tackle(self, carrier: Player, team: Team, opp_team: Team) -> bool:
        """
        Looks for a defender in a sensible position to challenge for the
        ball right now. Returns True if a tackle was actually attempted
        this tick (whether won or lost), so the caller knows the carrier's
        turn is spent; False means no defender had a sensible tackle on,
        so the carrier proceeds to its normal decision untouched.
        """
        defending_tactic = opp_team.tactic
        candidates = []
        for d in opp_team.players:
            if d.role == Role.GK:
                continue
            dist = d.distance_to(carrier.position)
            if dist > max(STANDING_TACKLE_RANGE, SLIDE_TACKLE_RANGE):
                continue
            if dist > effective_value(defending_tactic, d.slot_index, "press_trigger_distance"):
                continue  # not engaging at all, out of this defender's press radius
            candidates.append((d, dist))
        if not candidates:
            return False

        candidates.sort(key=lambda t: t[1])
        defender, dist = candidates[0]
        beaten = self._is_beaten(defender, carrier, opp_team)
        last_man = self._is_last_man(defender, opp_team)

        if dist <= STANDING_TACKLE_RANGE and not beaten:
            tackle_type = TackleType.STANDING
        elif last_man and dist <= SLIDE_TACKLE_RANGE and (beaten or dist > STANDING_TACKLE_RANGE):
            # last-ditch effort: gated so it's attempted only a fraction of
            # the time even when the situation calls for one, matching how
            # rarely real defenders actually commit to a slide
            gate = SLIDE_TACKLE_GATE_BASE + 0.5 * effective_value(defending_tactic, defender.slot_index, "press_intensity")
            if self.rng.random() > gate:
                return False
            tackle_type = TackleType.SLIDING
        else:
            return False  # no sensible challenge: too far, already beaten, not last man -- let him run

        opp_team.stats.tackles_attempted += 1
        if tackle_type == TackleType.SLIDING:
            opp_team.stats.sliding_tackles_attempted += 1

        p_defender_wins = tackle_success_probability(defender, carrier, tackle_type, dist)
        defender_wins = resolve(self.rng, p_defender_wins)

        if defender_wins:
            opp_team.stats.tackles_won += 1
            self._turnover(defender, opp_team)
            return True

        if tackle_type == TackleType.SLIDING and self.rng.random() < 0.15:
            opp_team.stats.fouls_committed += 1
            self.events.append(
                (self.tick, f"FOUL by {defender.name} ({opp_team.name}) -> "
                            f"free kick to {team.name} (set pieces not simulated yet)")
            )
        return False  # tackle failed: carrier keeps the ball, proceeds to act normally this tick

    def _choose_action(self, carrier: Player, team: Team, opp_team: Team, dist_to_goal: float) -> ActionType:
        tactic = team.tactic
        slot = carrier.slot_index
        shoot_eagerness = effective_value(tactic, slot, "shoot_eagerness")
        width_in_possession = effective_value(tactic, slot, "width_in_possession")
        risk_tolerance = effective_value(tactic, slot, "risk_tolerance")
        tempo = effective_value(tactic, slot, "tempo")
        directness = effective_value(tactic, slot, "directness")

        in_box = dist_to_goal < 18 and abs(carrier.position[1] - GOAL_Y_CENTER) < 20
        in_shot_range = dist_to_goal <= carrier.shot_range

        weights = {}
        weights[ActionType.SHOOT] = 0.0
        if in_shot_range and dist_to_goal < carrier.shot_range * 0.65:
            base = 0.02 + 0.12 * shoot_eagerness
            if in_box:
                base += 0.10
            weights[ActionType.SHOOT] = base

        is_wide = carrier.position[1] < PITCH_WIDTH * 0.22 or carrier.position[1] > PITCH_WIDTH * 0.78
        weights[ActionType.CROSS] = 0.0
        if is_wide and dist_to_goal < carrier.cross_range and dist_to_goal > 10:
            weights[ActionType.CROSS] = 0.15 + 0.3 * width_in_possession

        weights[ActionType.DRIBBLE] = 0.12 + 0.22 * risk_tolerance
        weights[ActionType.PASS] = 1.0 + 0.3 * (1 - tempo) + 0.2 * (1 - directness)

        total = sum(weights.values())
        r = self.rng.random() * total
        cum = 0.0
        for action, w in weights.items():
            cum += w
            if r <= cum:
                return action
        return ActionType.PASS

    # ------------------------------------------------------------------
    # Pass launch (ball becomes a flying agent)
    # ------------------------------------------------------------------
    def _launch_pass(self, carrier: Player, team: Team, opp_team: Team):
        teammates = [p for p in team.players if p is not carrier]
        receiver = self._choose_pass_target(carrier, team, teammates)
        if receiver is None:
            self._attempt_dribble(carrier, team, opp_team)
            return
        directness = effective_value(team.tactic, carrier.slot_index, "directness")
        forward_progress = (receiver.position[0] - carrier.position[0]) * team.attacking_direction
        is_through_ball = forward_progress > 5 and directness > 0.5
        self._launch_pass_to(carrier, team, opp_team, receiver, is_through_ball)

    def _launch_pass_to(self, carrier: Player, team: Team, opp_team: Team,
                         receiver: Player, is_through_ball: bool):
        """Shared launch mechanics once a receiver and pass style have been
        chosen, whether by the general pass-target scorer or by a
        role-specific decision (e.g. the goalkeeper's short/long choice)."""
        directness = effective_value(team.tactic, carrier.slot_index, "directness")
        target = receiver.position.copy()
        kind = "ground"
        if is_through_ball:
            lead_distance = 6.0 + 10.0 * directness
            target[0] += team.attacking_direction * lead_distance
            target[0] = float(np.clip(target[0], 1, PITCH_LENGTH - 1))
            kind = "through"

        intended_dist = float(np.linalg.norm(target - carrier.position))
        team.stats.passes_attempted += 1

        context = {"distance": intended_dist, "max_range": carrier.pass_range}
        pass_quality = carrier.effective_ability(ActionType.PASS, context)

        effective_range = carrier.pass_range if not is_through_ball else carrier.pass_range * 1.4
        if intended_dist > effective_range:
            pass_quality *= effective_range / intended_dist

        direction = (target - carrier.position)
        dir_norm = float(np.linalg.norm(direction))
        direction = direction / dir_norm if dir_norm > 0.01 else np.array([team.attacking_direction, 0.0])

        # --- execution noise: a pass is rarely pinpoint. Error grows with
        # pressure on the passer, with how far the ball has to travel, and
        # shrinks with passing ability. Lateral error (perpendicular to the
        # intended line) is what actually sends a pass out of play or wide
        # of its target even with no defender touching it; a smaller
        # along-line component gives over/under-hit passes too.
        pressure_dist = nearest_opponent_distance(carrier, opp_team.players)
        pressure_factor = 1.0 + max(0.0, (6.0 - pressure_dist)) * 0.10
        distance_factor = 1.0 + (intended_dist / max(carrier.pass_range, 1.0)) * 0.35
        quality_norm = float(np.clip(pass_quality / 100.0, 0.15, 1.0))
        miss_std = 0.8 * pressure_factor * distance_factor * (1.5 - quality_norm)

        perp = np.array([-direction[1], direction[0]])
        lateral_error = self.rng.normal(0, miss_std)
        overshoot_error = self.rng.normal(0, miss_std * 0.5)
        final_target = target + perp * lateral_error + direction * overshoot_error

        final_direction = final_target - carrier.position
        fd_norm = float(np.linalg.norm(final_direction))
        final_direction = final_direction / fd_norm if fd_norm > 0.01 else direction

        speed = 13.0 + pass_quality * 0.11

        ball = self.ball
        ball.in_flight = True
        ball.carrier = None
        carrier.has_ball = False
        ball.position = carrier.position.copy()
        ball.velocity = final_direction * speed
        ball.target = final_target
        ball.intended_receiver = receiver
        ball.passing_team = team
        ball.pass_quality = pass_quality
        ball.kind = kind

    def _goalkeeper_decide(self, gk: Player, team: Team, opp_team: Team):
        """
        Goalkeepers don't use the general action-weighting policy (no
        shooting, crossing, or dribbling out of the box). They choose
        between a short pass to a nearby defender and a long ball launched
        toward the most advanced outfield player, weighted by the team's
        own directness/tempo and how intensely the opponent is pressing
        (harder press -> more likely to go long rather than risk a short
        pass being cut out close to goal).
        """
        directness = effective_value(team.tactic, gk.slot_index, "directness")
        opp_press = opp_team.tactic.press_intensity
        short_prob = 0.6 + 0.25 * (1 - directness) - 0.25 * opp_press
        short_prob = float(np.clip(short_prob, 0.12, 0.9))
        go_long = self.rng.random() > short_prob

        if go_long:
            forward_most = max(team.outfield(),
                                key=lambda p: p.position[0] * team.attacking_direction)
            self._launch_pass_to(gk, team, opp_team, forward_most, is_through_ball=True)
        else:
            backs = [p for p in team.players if p.role in (Role.CB, Role.LB, Role.RB)]
            candidates = backs if backs else team.outfield()
            nearest_back = min(candidates, key=lambda p: gk.distance_to(p.position))
            self._launch_pass_to(gk, team, opp_team, nearest_back, is_through_ball=False)

    def _choose_pass_target(self, carrier: Player, team: Team, teammates: list) -> Player | None:
        tactic = team.tactic
        directness = effective_value(tactic, carrier.slot_index, "directness")
        width_in_possession = effective_value(tactic, carrier.slot_index, "width_in_possession")
        best, best_score = None, -1e9
        for mate in teammates:
            dist = carrier.distance_to(mate.position)
            if dist > carrier.pass_range * 1.6 or dist < 1.5:
                continue
            forward_progress = (mate.position[0] - carrier.position[0]) * team.attacking_direction
            safety = -dist
            wing_bonus = 0.0
            is_wide_mate = mate.position[1] < PITCH_WIDTH * 0.25 or mate.position[1] > PITCH_WIDTH * 0.75
            if is_wide_mate:
                wing_bonus = width_in_possession * 18.0
            score = (directness * forward_progress * 1.5
                     + (1 - directness) * safety
                     + wing_bonus)
            score += self.rng.normal(0, 3)
            if score > best_score:
                best_score, best = score, mate
        return best

    # ------------------------------------------------------------------
    # Shot / cross / dribble (instant resolution, ball syncs to outcome)
    # ------------------------------------------------------------------
    def _attempt_shot(self, carrier: Player, team: Team, opp_team: Team):
        team.stats.shots += 1
        goal_x = team.opp_goal_x()
        dist = abs(goal_x - carrier.position[0])
        angle_pen = shot_angle_penalty(carrier.position, goal_x)
        p = success_probability(
            carrier, ActionType.SHOOT, distance=dist, max_range=carrier.shot_range,
            opponents=opp_team.players, context_extra={"angle_penalty": angle_pen, "in_box": dist < 18},
        )
        on_target = resolve(self.rng, p * 0.75)
        if on_target:
            team.stats.shots_on_target += 1
            gk = opp_team.goalkeeper()
            save_p = success_probability(
                gk, ActionType.TACKLE, distance=dist, max_range=40,
                opponents=[], context_extra={"angle_penalty": 0.0},
            )
            gk_saves = resolve(self.rng, min(save_p * 0.95, 0.9))
            if not gk_saves:
                team.stats.goals += 1
                self.events.append((self.tick, f"GOAL: {carrier.name} ({team.name})"))
                self._reset_to_kickoff(scoring_team=team)
                return
        self._goal_kick_restart(opp_team)

    def _attempt_cross(self, carrier: Player, team: Team, opp_team: Team):
        team.stats.crosses_attempted += 1
        target_x = team.opp_goal_x()
        target = np.array([target_x - 11 * team.attacking_direction, GOAL_Y_CENTER])
        dist = carrier.distance_to(target)
        p = success_probability(
            carrier, ActionType.CROSS, distance=dist, max_range=carrier.cross_range,
            opponents=opp_team.players,
        )
        if resolve(self.rng, p):
            team.stats.crosses_completed += 1
            receiver = self._nearest_teammate_to(team, target, exclude=carrier)
            if receiver:
                self._set_carrier(receiver)
                return
        nearest_def = self._nearest_player_to(opp_team, target)
        self._turnover(nearest_def, opp_team)

    def _attempt_dribble(self, carrier: Player, team: Team, opp_team: Team):
        team.stats.dribbles_attempted += 1
        goal_x = team.opp_goal_x()
        direction = np.sign(goal_x - carrier.position[0]) or 1
        step_dist = 4.0
        p = success_probability(
            carrier, ActionType.DRIBBLE, distance=step_dist, max_range=10.0,
            opponents=opp_team.players,
        )
        if resolve(self.rng, p):
            team.stats.dribbles_completed += 1
            new_pos = carrier.position + np.array([direction * step_dist, 0.0])
            new_pos[0] = float(np.clip(new_pos[0], 1, PITCH_LENGTH - 1))
            carrier.position = new_pos
            self.ball.position = new_pos.copy()
        else:
            nearest_def = min(opp_team.players, key=lambda p: p.distance_to(carrier.position))
            self._turnover(nearest_def, opp_team)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    def _nearest_teammate_to(self, team: Team, point: np.ndarray, exclude: Player) -> Player | None:
        candidates = [p for p in team.players if p is not exclude]
        if not candidates:
            return None
        return min(candidates, key=lambda p: p.distance_to(point))

    def _nearest_player_to(self, team: Team, point: np.ndarray) -> Player:
        return min(team.players, key=lambda p: p.distance_to(point))

    def _turnover(self, new_carrier: Player, new_team: Team):
        new_team.stats.interceptions += 1
        self._set_carrier(new_carrier)

    def _reset_to_kickoff(self, scoring_team: Team):
        conceding_team = self.other_team(scoring_team)
        for team in (self.home, self.away):
            for i, p in enumerate(team.players):
                target = self._slot_target(team, i, ball_x_norm=0.5)
                p.position = target.copy()
                p.has_ball = False
        striker = self._first_striker(conceding_team)
        self._set_carrier(striker)

    def _check_ball_out_of_bounds(self):
        x, y = self.ball.position
        if 0.0 <= x <= PITCH_LENGTH and 0.0 <= y <= PITCH_WIDTH:
            return
        # Simplified rule: any ball leaving the field of play restarts as a
        # corner for whichever team was last in possession (no throw-ins or
        # offside modeled yet -- that's the explicit scope cut for now).
        if self.ball.carrier:
            possessing_team = self.team_of(self.ball.carrier)
        elif self.ball.passing_team:
            possessing_team = self.ball.passing_team
        else:
            possessing_team = self.home
        self._corner_kick_restart(possessing_team)

    def _corner_kick_restart(self, attacking_team: Team):
        defending_team = self.other_team(attacking_team)
        self.events.append((self.tick, f"Out of play -> corner for {attacking_team.name}"))

        for team in (self.home, self.away):
            is_attacking = team is attacking_team
            ball_x_norm = 0.93 if is_attacking else 0.07
            for i, p in enumerate(team.players):
                target = self._slot_target(team, i, ball_x_norm)
                p.position = target.copy()
                p.has_ball = False

        wide_roles = (Role.LW, Role.RW, Role.LM, Role.RM)
        takers = [p for p in attacking_team.players if p.role in wide_roles]
        taker = max(takers, key=lambda p: p.abilities.crossing) if takers else attacking_team.outfield()[0]

        corner_y = 1.0 if self.rng.random() < 0.5 else PITCH_WIDTH - 1.0
        taker.position = np.array([attacking_team.opp_goal_x() - 1.0 * attacking_team.attacking_direction, corner_y])
        taker.position[0] = float(np.clip(taker.position[0], 1, PITCH_LENGTH - 1))
        self._set_carrier(taker)

        target_point = np.array([attacking_team.opp_goal_x() - 8.0 * attacking_team.attacking_direction, GOAL_Y_CENTER])
        attacking_team.stats.crosses_attempted += 1
        dist = taker.distance_to(target_point)
        p = success_probability(
            taker, ActionType.CROSS, distance=dist, max_range=max(taker.cross_range, dist),
            opponents=defending_team.players,
        )
        if resolve(self.rng, p):
            attacking_team.stats.crosses_completed += 1
            receiver = self._nearest_teammate_to(attacking_team, target_point, exclude=taker)
            if receiver:
                self._set_carrier(receiver)
                return
        nearest_def = self._nearest_player_to(defending_team, target_point)
        self._turnover(nearest_def, defending_team)

    def _goal_kick_restart(self, team: Team):
        opponents = self.other_team(team)
        for i, p in enumerate(team.players):
            target = self._slot_target(team, i, ball_x_norm=0.15)
            p.position = target.copy()
            p.has_ball = False
        for i, p in enumerate(opponents.players):
            target = self._slot_target(opponents, i, ball_x_norm=0.85)
            p.position = target.copy()
            p.has_ball = False

        gk = team.goalkeeper()
        gk.position = np.array([team.own_goal_x() + 5 * team.attacking_direction, PITCH_WIDTH / 2])
        gk.position[0] = float(np.clip(gk.position[0], 1, PITCH_LENGTH - 1))
        self._set_carrier(gk)
