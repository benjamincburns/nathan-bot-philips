import os
import wandb
import numpy
from typing import Any

import torch.jit
from torch.nn import Linear, Sequential, Tanh

from redis import Redis

from rlgym.utils.obs_builders.advanced_obs import AdvancedObs
from rlgym.utils.gamestates import PlayerData, GameState
from rewards import reward_function
from rlgym.utils.action_parsers.discrete_act import DiscreteAction

from rocket_learn.agent.actor_critic_agent import ActorCriticAgent
from rocket_learn.agent.discrete_policy import DiscretePolicy
from rocket_learn.ppo import PPO
from rocket_learn.rollout_generator.redis_rollout_generator import RedisRolloutGenerator
from rocket_learn.utils.util import SplitLayer


# ROCKET-LEARN ALWAYS EXPECTS A BATCH DIMENSION IN THE BUILT OBSERVATION
class ExpandAdvancedObs(AdvancedObs):
    def build_obs(self, player: PlayerData, state: GameState, previous_action: numpy.ndarray) -> Any:
        obs = super(ExpandAdvancedObs, self).build_obs(player, state, previous_action)
        return numpy.expand_dims(obs, 0)


if __name__ == "__main__":
    """
    
    Starts up a rocket-learn learner process, which ingests incoming data, updates parameters
    based on results, and sends updated model parameters out to the workers
    
    """

    # ROCKET-LEARN USES WANDB WHICH REQUIRES A LOGIN TO USE. YOU CAN SET AN ENVIRONMENTAL VARIABLE
    # OR HARDCODE IT IF YOU ARE NOT SHARING YOUR SOURCE FILES
    wandb.login(key=os.environ["WANDB_API_KEY"])
    logger = wandb.init(project="nathan-bot-philips", entity="some_rando_rl")
    logger.name = "Nathan Bot Phillips"

    # LINK TO THE REDIS SERVER YOU SHOULD HAVE RUNNING (USE THE SAME PASSWORD YOU SET IN THE REDIS
    # CONFIG)
    #redis = Redis(password="you_better_use_a_password")
    redis_host = os.environ.get("REDIS_HOST", "127.0.0.1")
    redis_port = int(os.environ.get("REDIS_PORT", 6379))
    redis_password = os.environ.get("REDIS_PASSWORD", None)

    redis = Redis(host=redis_host, port=redis_port, password=redis_password)

    # ENSURE OBSERVATION, REWARD, AND ACTION CHOICES ARE THE SAME IN THE WORKER
    def obs():
        return ExpandAdvancedObs()

    def rew():
        return reward_function()

    def act():
        return DiscreteAction()


    # THE ROLLOUT GENERATOR CAPTURES INCOMING DATA THROUGH REDIS AND PASSES IT TO THE LEARNER.
    # -save_every SPECIFIES HOW OFTEN OLD VERSIONS ARE SAVED TO REDIS. THESE ARE USED FOR TRUESKILL
    # COMPARISON AND TRAINING AGAINST PREVIOUS VERSIONS
    rollout_gen = RedisRolloutGenerator(redis, obs, rew, act,
                                        logger=logger,
                                        save_every=50000)

    # ROCKET-LEARN EXPECTS A SET OF DISTRIBUTIONS FOR EACH ACTION FROM THE NETWORK, NOT
    # THE ACTIONS THEMSELVES. SEE network_setup.readme.txt FOR MORE INFORMATION
    split = (3, 3, 3, 3, 3, 2, 2, 2)
    total_output = sum(split)

    # TOTAL SIZE OF THE INPUT DATA
    state_dim = 107

    shared1 = Tanh()
    shared2 = Linear(512, 512)
    shared3 = Tanh()
    shared4 = Linear(512, 256)

    critic = Sequential(
        Linear(state_dim, 512),
        shared1,
        shared2,
        shared3,
        shared4,
        Tanh(),
        Linear(256, 256),
        Tanh(),
        Linear(256, 256),
        Tanh(),
        Linear(256, 256),
        Tanh(),
        Linear(256, 1)
    )

    actor_net = Sequential(
        Linear(state_dim, 512),
        shared1,
        shared2,
        shared3,
        shared4,
        Tanh(),
        Linear(256, 256),
        Tanh(),
        Linear(256, 256),
        Tanh(),
        Linear(256, 256),
        Tanh(),
        Linear(256, total_output),
        SplitLayer(splits=split)
    )

    actor = DiscretePolicy(actor_net, split)

    optim = torch.optim.Adam([
        {"params": actor_net[1:5].parameters(), "lr": 5e-5}, # shared layer
        {"params": actor_net[0].parameters(), "lr": 5e-5}, # actor input layer
        {"params": actor_net[5:].parameters(), "lr": 5e-5}, # actor remaining layers
        {"params": critic[0].parameters(), "lr": 5e-5}, # critic input layer
        {"params": critic[5:].parameters(), "lr": 5e-5},# critic remaining layers
    ])

    # PPO REQUIRES AN ACTOR/CRITIC AGENT
    agent = ActorCriticAgent(actor=actor, critic=critic, optimizer=optim)

    alg = PPO(
        rollout_gen,
        agent,
        ent_coef=0.01,
        n_steps=1_000_000,
        batch_size=100_000,
        minibatch_size=20_000,
        epochs=30,
        gamma=0.995,
        logger=logger,
    )

    # BEGIN TRAINING. IT WILL CONTINUE UNTIL MANUALLY STOPPED
    # -iterations_per_save SPECIFIES HOW OFTEN CHECKPOINTS ARE SAVED
    # -save_dir SPECIFIES WHERE
    alg.run(iterations_per_save=5, save_dir="models")
