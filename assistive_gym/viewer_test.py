import gym, sys, argparse
import numpy as np

import assistive_gym
from mushroom_rl.core import Environment
# import assistive_gym

def sample_action(env, coop):
    if coop:
        return {'robot': env.env.action_space_robot.sample(), 'human': env.env.action_space_human.sample()}
    return env.env.action_space.sample()

def viewer(env_name):
    coop = 'Human' in env_name
    env = Environment.make('Gym','assistive_gym:'+env_name)

    while True:
        done = False
        env.render()
        observation = env.reset()
        action = sample_action(env, coop)
        if coop:
            print('Robot observation size:', np.shape(observation['robot']), 'Human observation size:', np.shape(observation['human']), 'Robot action size:', np.shape(action['robot']), 'Human action size:', np.shape(action['human']))
        else:
            print('Observation size:', np.shape(observation), 'Action size:', np.shape(action))

        while not done:
            observation, reward, done, info = env.step(sample_action(env, coop))
            if coop:
                done = done['__all__']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='mushroom Environment Viewer')
    parser.add_argument('--env', default='FeedingTiago-v0')
    args = parser.parse_args()

    viewer(args.env)



