import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
import gym, sys, argparse
import assistive_gym
from mushroom_rl.core import Environment
from mushroom_rl.algorithms.value import FQI
from mushroom_rl.core import Core
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.utils.parameters import Parameter

env_name = 'FeedingTiago-v0'
mdp = Environment.make('Gym','assistive_gym:'+env_name)

# Policy
epsilon = Parameter(value=1.)
pi = EpsGreedy(epsilon=epsilon)

# Approximator
approximator_params = dict(input_shape=mdp.info.observation_space.shape,
                           n_actions=7,
                           n_estimators=50,
                           min_samples_split=5,
                           min_samples_leaf=2)
approximator = ExtraTreesRegressor

# Agent
agent = FQI(mdp.info, pi, approximator, n_iterations=20,
            approximator_params=approximator_params)

core = Core(agent, mdp)

core.learn(n_episodes=1, n_episodes_per_fit=1)

pi.set_epsilon(Parameter(0.))
initial_state = np.array([[-.5, 0.]])
dataset = core.evaluate(initial_states=initial_state)

print(compute_J(dataset, gamma=mdp.info.gamma))

