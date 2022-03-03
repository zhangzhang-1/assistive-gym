import pybullet as p
import pybullet_data
from time import sleep
from pprint import pprint
import os

p.connect(p.GUI)
datapath = pybullet_data.getDataPath()
p.setAdditionalSearchPath(datapath)
directory = os.path.join(os.path.dirname(os.path.realpath(__file__)),'envs', 'assets')
human = p.loadURDF(os.path.join(directory, 'tiago_dualhand', 'tiago_dual_modified.urdf'), #'baxter', 'baxter_custom.urdf'),
                   useFixedBase=True, basePosition=[-1, -1, 0])
#human = object[0]
joint_num = p.getNumJoints(human)

available_joints_indices = [i for i in range(joint_num) if p.getJointInfo(human, i)[2] != p.JOINT_FIXED]
#pprint(available_joints_indices)

print("Tiago has ", len(available_joints_indices), "Moveable joints:")

for ii in range(len(available_joints_indices)): # joint_num
    info_tuple = p.getJointInfo(human, available_joints_indices[ii]) # ii)
    print(f"Joint index：{info_tuple[0]}\n\
            Joint name：{info_tuple[1]}\n\
            Joint type：{info_tuple[2]}\n\
            Index of the first position variable：{info_tuple[3]}\n\
            Index of the first speed variable：{info_tuple[4]}\n\
            Flags：{info_tuple[5]}\n\
            Joint damping：{info_tuple[6]}\n\
            Joint friction：{info_tuple[7]}\n\
            Lower limit：{info_tuple[8]}\n\
            Upper limit：{info_tuple[9]}\n\
            Max Force：{info_tuple[10]}\n\
            Max Velocity：{info_tuple[11]}\n\
            Link name：{info_tuple[12]}\n\
            Joint axis in local frame：{info_tuple[13]}\n\
            Joint position in parent frame：{info_tuple[14]}\n\
            Joint orientation in parent frame：{info_tuple[15]}\n\
            Parent index，'-1' if 'Base'：{info_tuple[16]}\n\n")