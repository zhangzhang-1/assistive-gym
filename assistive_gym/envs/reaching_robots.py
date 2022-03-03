from .reaching import ReachingEnv
from .demofetching import FetchingEnv
from .tiagodrinking import DrinkingNewEnv
from .tiagofeeding import FeedingNewEnv
from .tiagoscratch import ScratchItchNewEnv
from .tiagodressing import DressingNewEnv
from .tiagoArmManipulation import ArmManipulationNewEnv
from .tiagoBedBathing import BedBathingNewEnv
from .agents import pr2, tiago_dualhand, baxter, sawyer, jaco, stretch, panda, human
from .agents.pr2 import PR2
from .agents.tiago_dualhand import tiago_dualhand
from .agents.baxter import Baxter
from .agents.sawyer import Sawyer
from .agents.jaco import Jaco
from .agents.stretch import Stretch
from .agents.panda import Panda
from .agents.human import Human
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env

robot_arm = 'right'
human_controllable_joint_indices = human.head_joints
class FetchingPR2Env(FetchingEnv):
    def __init__(self):
        super(FetchingPR2Env, self).__init__(robot=PR2(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))
class FetchingTiagoEnv(FetchingEnv):
    def __init__(self):
        super(FetchingTiagoEnv, self).__init__(robot=tiago_dualhand(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))
class ReachingPR2Env(ReachingEnv):
    def __init__(self):
        super(ReachingPR2Env, self).__init__(robot=PR2(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))
class ReachingTiagoEnv(ReachingEnv):
    def __init__(self):
        super(ReachingTiagoEnv, self).__init__(robot=tiago_dualhand(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))
class ReachingPR2HumanEnv(ReachingEnv):
    def __init__(self):
        super(ReachingPR2HumanEnv, self).__init__(robot=PR2(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
class ReachingTiagoHumanEnv(ReachingEnv):
    def __init__(self):
        super(ReachingTiagoHumanEnv, self).__init__(robot=tiago_dualhand(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
class ReachingJacoEnv(ReachingEnv):
    def __init__(self):
        super(ReachingJacoEnv, self).__init__(robot=Jaco(robot_arm),
                                                     human=Human(human_controllable_joint_indices, controllable=False))
class ReachingJacoHumanEnv(ReachingEnv):
    def __init__(self):
        super(ReachingJacoHumanEnv, self).__init__(robot=Jaco(robot_arm),
                                                       human=Human(human_controllable_joint_indices,
                                                                   controllable=True))
class DrinkingTiagoEnv(DrinkingNewEnv):
    def __init__(self):
        super(DrinkingTiagoEnv, self).__init__(robot=tiago_dualhand(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class FeedingTiagoEnv(FeedingNewEnv):
    def __init__(self):
        super(FeedingTiagoEnv, self).__init__(robot=tiago_dualhand(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class ScratchTiagoEnv(ScratchItchNewEnv):
    def __init__(self):
        super(ScratchTiagoEnv, self).__init__(robot=tiago_dualhand('left'), human=Human(human_controllable_joint_indices, controllable=False))

class BedBathingTiagoEnv(BedBathingNewEnv):
    def __init__(self):
        super(BedBathingTiagoEnv, self).__init__(robot=tiago_dualhand('left'), human=Human(human_controllable_joint_indices, controllable=False))
class ArmManipulationTiagoEnv(ArmManipulationNewEnv):
    def __init__(self):
        super(ArmManipulationTiagoEnv, self).__init__(robot=tiago_dualhand(['left','right']), human=Human(human_controllable_joint_indices, controllable=False))
class DressingTiagoEnv(DressingNewEnv):
    def __init__(self):
        super(DressingTiagoEnv, self).__init__(robot=tiago_dualhand('left'), human=Human(human_controllable_joint_indices, controllable=False))
