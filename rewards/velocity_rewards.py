import numpy as np

from typing import List

from rlgym.utils import RewardFunction, math
from rlgym.utils.common_values import BLUE_TEAM, ORANGE_TEAM, ORANGE_GOAL_BACK, BLUE_GOAL_BACK, BALL_MAX_SPEED, CAR_MAX_SPEED
from rlgym.utils.gamestates import GameState, PlayerData

from utils import same_player, same_team


class VelocityBallToGoalReward(RewardFunction):
    def __init__(self, own_goal=False, use_scalar_projection=False):
        super().__init__()
        self.own_goal = own_goal
        self.use_scalar_projection = use_scalar_projection

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.team_num == BLUE_TEAM and not self.own_goal \
                or player.team_num == ORANGE_TEAM and self.own_goal:
            objective = np.array(ORANGE_GOAL_BACK)
        else:
            objective = np.array(BLUE_GOAL_BACK)

        vel = np.copy(state.ball.linear_velocity)
        pos_diff = objective - state.ball.position
        if self.use_scalar_projection:
            # Vector version of v=d/t <=> t=d/v <=> 1/t=v/d
            # Max value should be max_speed / ball_radius = 2300 / 94 = 24.5
            # Used to guide the agent towards the ball
            inv_t = math.scalar_projection(vel, pos_diff)
            return inv_t
        else:
            # Regular component velocity
            norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
            vel /= BALL_MAX_SPEED
            return float(np.dot(norm_pos_diff, vel))

class VelocityPlayerToBallReward(RewardFunction):
    def __init__(self, use_scalar_projection=False):
        super().__init__()
        self.use_scalar_projection = use_scalar_projection

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        vel = np.copy(player.car_data.linear_velocity)
        pos_diff = state.ball.position - player.car_data.position
        if self.use_scalar_projection:
            # Vector version of v=d/t <=> t=d/v <=> 1/t=v/d
            # Max value should be max_speed / ball_radius = 2300 / 92.75 = 24.8
            # Used to guide the agent towards the ball
            inv_t = math.scalar_projection(vel, pos_diff)
            return inv_t
        else:
            # Regular component velocity
            norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
            vel /= CAR_MAX_SPEED
            return float(np.dot(norm_pos_diff, vel))


class GoalVelocityReward(RewardFunction):
  """
  a goal reward that scales with the speed of the ball as it crosses into the goal
  """

  def __init__(self):
    super().__init__()

    # Need to keep track of last registered value to detect changes
    self._goals = {}
    self._state = None
    self._goal_speed = 0

  def reset(self, initial_state: GameState, optional_data=None):
    self._state = None
    self._next_tick(initial_state)

  def _next_tick(self, state: GameState):
    if self._state == state:
      return

    if self._state is not None:
      self._goal_speed = np.linalg.norm(self._state.ball.linear_velocity)
      self._goals = { player.car_id : player.match_goals for player in self._state.players }

    self._state = state

  def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray, optional_data=None):
    self._next_tick(state)

    scorer = self._who_scored(state.players)

    if scorer is None:
      return 0

    # you don't get rewarded for how fast your teammates send the ball into the goal
    if not same_player(scorer, player) and same_team(scorer, player):
      return 0

    reward_coeff = 1 if same_player(player, scorer) else -1
    
    reward = reward_coeff * self._goal_speed / BALL_MAX_SPEED

    return reward

  def _who_scored(self, players: List[PlayerData]):
    scorer = [ player for player in players if player.match_goals > self._goals[player.car_id]]
    return scorer[0] if scorer else None

