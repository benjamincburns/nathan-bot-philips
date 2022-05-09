import os, sys

from typing import Any
import numpy

from redis import Redis

from rlgym.envs import Match
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils.terminal_conditions.common_conditions import GoalScoredCondition, NoTouchTimeoutCondition
from rewards import reward_function
from rlgym.utils.state_setters.default_state import DefaultState
from rlgym.utils.obs_builders.advanced_obs import AdvancedObs
from rlgym.utils.action_parsers.discrete_act import DiscreteAction

from rocket_learn.rollout_generator.redis_rollout_generator import RedisRolloutWorker


# ROCKET-LEARN ALWAYS EXPECTS A BATCH DIMENSION IN THE BUILT OBSERVATION
class ExpandAdvancedObs(AdvancedObs):
    def build_obs(self, player: PlayerData, state: GameState, previous_action: numpy.ndarray) -> Any:
        obs = super(ExpandAdvancedObs, self).build_obs(player, state, previous_action)
        return numpy.expand_dims(obs, 0)


streamer = False

if __name__ == "__main__":
    """

    Starts up a rocket-learn worker process, which plays out a game, sends back game data to the 
    learner, and receives updated model parameters when available

    """

    if len(sys.argv) == 2 and sys.argv[-1] == "--streamer":
        streamer = True

    # BUILD THE ROCKET LEAGUE MATCH THAT WILL USED FOR TRAINING
    # -ENSURE OBSERVATION, REWARD, AND ACTION CHOICES ARE THE SAME IN THE WORKER
    tick_skip = 8
    match = Match(
        game_speed=100 if not streamer else 1,
        self_play=True,
        team_size=1,
        tick_skip=tick_skip,
        state_setter=DefaultState(),
        obs_builder=ExpandAdvancedObs(),
        action_parser=DiscreteAction(),
        terminal_conditions=[NoTouchTimeoutCondition(round(30 * 120 / tick_skip)),
                             GoalScoredCondition()],
        reward_function=reward_function()
    )

    # LINK TO THE REDIS SERVER YOU SHOULD HAVE RUNNING (USE THE SAME PASSWORD YOU SET IN THE REDIS
    # CONFIG)
    #redis = Redis(host="127.0.0.1", password="you_better_use_a_password")
    redis_host = os.environ.get("REDIS_HOST", "127.0.0.1")
    redis_port = int(os.environ.get("REDIS_PORT", 6379))
    redis_password = os.environ.get("REDIS_PASSWORD", None)

    redis = Redis(host=redis_host, port=redis_port, password=redis_password)


    # LAUNCH ROCKET LEAGUE AND BEGIN TRAINING
    # -past_version_prob SPECIFIES HOW OFTEN OLD VERSIONS WILL BE RANDOMLY SELECTED AND TRAINED AGAINST
    RedisRolloutWorker(
        redis,
        os.environ.get("WORKER_NAME", "nathan-bot-philips"),
        match,
        streamer_mode=streamer,
        past_version_prob=0.2 if not streamer else 0,
        evaluation_prob=0.01 if not streamer else 0,
        force_paging=True
    ).run()
