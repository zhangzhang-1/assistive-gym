import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
import gym, sys, argparse
import assistive_gym
from mushroom_rl.core import Environment
from mushroom_rl.algorithms.value import SARSALambdaContinuous
from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.core import Core
from mushroom_rl.features import Features
from mushroom_rl.features.tiles import Tiles
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.callbacks import CollectDataset
from mushroom_rl.utils.parameters import Parameter
from mushroom_rl.environments import Gym

# MDP
env_name = 'FeedingTiago-v0'
mdp = Gym(name='assistive_gym:'+env_name, horizon=np.inf, gamma=1.)


# Policy
epsilon = Parameter(value=0.)
pi = EpsGreedy(epsilon=epsilon)

n_tilings = 25
tilings = Tiles.generate(n_tilings, [25, 25],
                         mdp.info.observation_space.low,
                         mdp.info.observation_space.high)
features = Features(tilings=tilings)

approximator_params = dict(input_shape=(features.size,),
                           output_shape=(mdp.info.action_space.n,),
                           n_actions=mdp.info.action_space.n)

learning_rate = Parameter(.1 / n_tilings)

agent = SARSALambdaContinuous(mdp.info, pi, LinearApproximator,
                              approximator_params=approximator_params,
                              learning_rate=learning_rate,
                              lambda_coeff=.9, features=features)

# Algorithm
collect_dataset = CollectDataset()
callbacks = [collect_dataset]
core = Core(agent, mdp, callbacks_fit=callbacks)

# Train
core.learn(n_episodes=100, n_steps_per_fit=1)

# Visualize the learned policy
core.evaluate(n_episodes=1, render=True)