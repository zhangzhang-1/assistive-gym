from .arm_manipulation import ArmManipulationEnv
from .agents import pr2, baxter, sawyer, jaco, human
from .agents.pr2 import PR2
from .agents.baxter import Baxter
from .agents.sawyer import Sawyer
from .agents.jaco import Jaco
from .agents.stretch import Stretch
from .agents.human import Human

robot_arm = 'both'
human_controllable_joint_indices = human.right_arm_joints
class ArmManipulationPR2Env(ArmManipulationEnv):
    def __init__(self):
        super(ArmManipulationPR2Env, self).__init__(robot=PR2(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class ArmManipulationBaxterEnv(ArmManipulationEnv):
    def __init__(self):
        super(ArmManipulationBaxterEnv, self).__init__(robot=Baxter(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class ArmManipulationSawyerEnv(ArmManipulationEnv):
    def __init__(self):
        super(ArmManipulationSawyerEnv, self).__init__(robot=Sawyer(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class ArmManipulationJacoEnv(ArmManipulationEnv):
    def __init__(self):
        super(ArmManipulationJacoEnv, self).__init__(robot=Jaco(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class ArmManipulationStretchEnv(ArmManipulationEnv):
    def __init__(self):
        super(ArmManipulationStretchEnv, self).__init__(robot=Stretch('wheel_'+robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class ArmManipulationPR2HumanEnv(ArmManipulationEnv):
    def __init__(self):
        super(ArmManipulationPR2HumanEnv, self).__init__(robot=PR2(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))

class ArmManipulationBaxterHumanEnv(ArmManipulationEnv):
    def __init__(self):
        super(ArmManipulationBaxterHumanEnv, self).__init__(robot=Baxter(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))

class ArmManipulationSawyerHumanEnv(ArmManipulationEnv):
    def __init__(self):
        super(ArmManipulationSawyerHumanEnv, self).__init__(robot=Sawyer(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))

class ArmManipulationJacoHumanEnv(ArmManipulationEnv):
    def __init__(self):
        super(ArmManipulationJacoHumanEnv, self).__init__(robot=Jaco(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))

class ArmManipulationStretchHumanEnv(ArmManipulationEnv):
    def __init__(self):
        super(ArmManipulationStretchHumanEnv, self).__init__(robot=Stretch('wheel_'+robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
