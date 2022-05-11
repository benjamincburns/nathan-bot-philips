from rewards.jump_touch_reward import JumpTouchReward
from rewards.velocity_rewards import VelocityBallToGoalReward, VelocityPlayerToBallReward, GoalVelocityReward
from rewards.kickoff_reward import KickoffReward
from rlgym.utils.reward_functions.common_rewards.misc_rewards import EventReward, SaveBoostReward
from rlgym.utils.reward_functions import CombinedReward

def reward_function():
    return CombinedReward(
        (
            SaveBoostReward(),
            VelocityPlayerToBallReward(),
            KickoffReward(),
            VelocityBallToGoalReward(),
            JumpTouchReward(),
            GoalVelocityReward(),
            EventReward(
                team_goal=100.0,
                concede=-100.0,
                shot=10.0,
                save=75.0,
                demo=25.0
            ),
        ),
        (0.01, 0.001, 0.01, 0.01, 0.01, 1.0, 0.01)
    )
