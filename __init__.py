from gym.envs.registration import register

tasks = ['ScratchItch', 'BedBathing', 'Feeding', 'Drinking', 'Dressing', 'ArmManipulation']
robots = ['PR2', 'Jaco', 'Baxter', 'Sawyer', 'Stretch', 'Panda', 'TiagoDualhand']

for task in tasks:
    for robot in robots:
        register(
            id='%s%s-v1' % (task, robot),
            entry_point='assistive_gym.envs:%s%sEnv' % (task, robot),
            max_episode_steps=200,
        )

for task in ['ScratchItch', 'Feeding']:
    for robot in robots:
        register(
            id='%s%sMesh-v1' % (task, robot),
            entry_point='assistive_gym.envs:%s%sMeshEnv' % (task, robot),
            max_episode_steps=200,
        )

register(
    id='HumanTesting-v1',
    entry_point='assistive_gym.envs:HumanTestingEnv',
    max_episode_steps=200,
)

register(
    id='SMPLXTesting-v1',
    entry_point='assistive_gym.envs:SMPLXTestingEnv',
    max_episode_steps=200,
)
register(
    id='demo-v0',
    entry_point='assistive_gym.envs:Demo0Env'
)
register(
    id='demo-v1',
    entry_point='assistive_gym.envs:Demo1Env'
)
register(
    id='demo-v2',
    entry_point='assistive_gym.envs:Demo2Env'
)
register(
    id='ReachingPR2-v0',
    entry_point='assistive_gym.envs:ReachingPR2Env',
    max_episode_steps=200,
)
register(
    id='ReachingJaco-v0',
    entry_point='assistive_gym.envs:ReachingJacoEnv',
    max_episode_steps=200,
)
register(
    id='ReachingPR2Human-v0',
    entry_point='assistive_gym.envs:ReachingPR2HumanEnv',
    max_episode_steps=200,
)
register(
    id='ReachingJacoHuman-v0',
    entry_point='assistive_gym.envs:ReachingJacoHumanEnv',
    max_episode_steps=200,
)
register(
    id='ReachingTiago-v0',
    entry_point='assistive_gym.envs:ReachingTiagoEnv',
    max_episode_steps=200,
)
register(
    id='ReachingTiagoHuman-v0',
    entry_point='assistive_gym.envs:ReachingTiagoHumanEnv',
    max_episode_steps=200,
)
register(
    id='FetchingTiago-v0',
    entry_point='assistive_gym.envs:FetchingTiagoEnv',
    max_episode_steps=200,
)
register(
    id='FetchingPR2-v0',
    entry_point='assistive_gym.envs:FetchingPR2Env',
    max_episode_steps=200,
)
register(
    id='DrinkingTiago-v0',
    entry_point='assistive_gym.envs:DrinkingTiagoEnv',
    max_episode_steps=200,
)
register(
    id='FeedingTiago-v0',
    entry_point='assistive_gym.envs:FeedingTiagoEnv',
    max_episode_steps=200,
)
register(
    id='ScratchTiago-v0',
    entry_point='assistive_gym.envs:ScratchTiagoEnv',
    max_episode_steps=200,
)
register(
    id='DressingTiago-v0',
    entry_point='assistive_gym.envs:DressingTiagoEnv',
    max_episode_steps=200,
)
register(
    id='BedBathingTiago-v0',
    entry_point='assistive_gym.envs:BedBathingTiagoEnv',
    max_episode_steps=200,
)
register(
    id='ArmManipulationTiago-v0',
    entry_point='assistive_gym.envs:ArmManipulationTiagoEnv',
    max_episode_steps=200,
)