import pybullet as p
import pybullet_data
import math
import numpy as np
from time import sleep
import random

p.connect(p.GUI)
p.resetSimulation()
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
p.setGravity(0, 0, -10)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeUid = p.loadURDF("plane/plane.urdf")
tiago = p.loadURDF("tiago_dualhand/tiago_dual_modified.urdf", basePosition=[0, 0, 0.35], useFixedBase=True)
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
targetPos = [1.2,1.1,0.5]

# fetch targetPos
numJoints = p.getNumJoints(tiago)
available_joints_indices = [i for i in range(numJoints) if p.getJointInfo(tiago, i)[2] != p.JOINT_FIXED]
right_arm_indices = [46, 47, 48, 49, 50, 51, 52]
lowerlimits = [-1.17, -1.17, -0.78, -0.39, -2.09, -1.41, -2.09]
upperlimits = [1.57, 1.57, 3.92, 2.35, 2.09, 1.41, 2.09]
torso_index = [21]  # torso lift prismatic joint
right_gripper_indices = [58,59]
right_tool_joint = 56
step = 1 / 240
reached = False
iter = 0
dist = 1e30
while (not reached and iter < 1000):
    p.stepSimulation()
    jointPoses = p.calculateInverseKinematics(tiago, right_tool_joint, targetPos)
    p.setJointMotorControlArray(bodyIndex=tiago, jointIndices=available_joints_indices,
                                                controlMode=p.POSITION_CONTROL, targetPositions=jointPoses,
                                                positionGains=[0.5]*len(jointPoses), forces=[500]*len(jointPoses))
    # for i in range(len(jointPoses)):
    #     p.resetJointState(tiago, i, jointPoses[i])
    ls = p.getLinkState(tiago, right_tool_joint)
    newPos = ls[4]
    diff = [targetPos[0] - newPos[0], targetPos[1] - newPos[1], targetPos[2] - newPos[2]]
    dist = np.linalg.norm(diff)
    reached = (dist < 0.001)
    iter = iter + 1
    sleep(step)

