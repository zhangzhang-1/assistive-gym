
import assistive_gym
from mushroom_rl.core import Core
from mushroom_rl.environments.gym_env import Gym

from mushroom_rl.utils.parameters import Parameter

from network_test import CriticNetwork, ActorNetwork

savepath='/home/zhang/assistive-gym/assistive_gym/savedagents/SAC/agent-best.msh'
agent= Parameter.load(savepath)

env_name = 'ScratchTiago-v0'
mdp = Gym(name='assistive_gym:' + env_name, horizon=200, gamma=0.99)

core = Core(agent, mdp)
n_steps = 5
total_steps_counter = 0
current_steps_counter = 0
move_condition = total_steps_counter < n_steps
dataset = list()
#
# while move_condition:
#     if (current_steps_counter == 0):
#         _state = core.mdp.env.reset()
#     action = core.agent.draw_action(_state)
#     next_state, reward, absorbing, _ = core.mdp.step(action)
#     state = _state
#     next_state = core._preprocess(next_state.copy())
#
#     total_steps_counter += 1
#     current_steps_counter += 1

# core.agent.stop()
# core.mdp.stop()
initial_state = core.mdp.reset()
initial_action = core.agent.draw_action(initial_state)
state = initial_state
for _ in range(10):
    core.mdp.render()
    core.mdp.reset()
    for ii in range(200):
        action = core.agent.draw_action(state)
        next_state, reward, done, _ = core.mdp.step(action)
        next_state = core._preprocess(next_state.copy())
        state = next_state
        if done:
            print('Episode finished after {} timesteps'.format(ii + 1))
            break
core.mdp.close()

# core.evaluate(n_steps=5 ,render=True)
