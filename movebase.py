import gym
from gym import error,spaces,utils
from gym.utils import seeding
import os
import pybullet as p
import pybullet_data
import math
import numpy as np
from time import sleep
from pprint import pprint
import random

p.connect(p.GUI)
p.resetSimulation()
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
p.setGravity(0, 0, -10)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeUid = p.loadURDF("plane/plane.urdf")
directory = os.path.join(os.path.dirname(os.path.realpath(__file__)),'envs', 'assets')
tiago = p.loadURDF(os.path.join(directory, 'tiago_dualhand', 'tiago_dual_modified.urdf'), #'baxter', 'baxter_custom.urdf'),
                   useFixedBase=True, basePosition=[0, 0, 0.05])
#tiago = p.loadURDF("tiago_dualhand/tiago_dual_modified.urdf", basePosition=[0, 0, 0.05], useFixedBase=True)
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
# move base
numJoints = p.getNumJoints(tiago)
available_joints_indices = [i for i in range(numJoints) if p.getJointInfo(tiago, i)[2] != p.JOINT_FIXED]
right_arm_indices = [46, 47, 48, 49, 50, 51, 52]

# right_arm_indices = [42, 43, 44, 45, 46, 47, 48]
lowerlimits = [-1.17, -1.17, -0.78, -0.39, -2.09, -1.41, -2.09]
upperlimits = [1.57, 1.57, 3.92, 2.35, 2.09, 1.41, 2.09]
torso_index = [21]  # torso lift prismatic joint
# torso_index = [16]
right_gripper_indices = [58, 59]
right_tool_joint = 60
iter = 0
dist = 1e30
step= 1/240
reached0 = False
reached1 = False
reached2 = False
targetpos = [-0.2, -0.5, 1.1]
def measure(targetpos):
    arrived = 0
    if targetpos[0] > 0 and targetpos[1] > 0:
        theta = np.arctan(targetpos[1] / targetpos[0])
        distance = np.linalg.norm(targetpos)
    elif targetpos[0] < 0 and targetpos[1] > 0:
        theta = np.pi - np.arctan(-targetpos[1] / targetpos[0])
        distance = np.linalg.norm(targetpos)
    elif targetpos[0] < 0 and targetpos[1] < 0:
        theta = np.pi + np.arctan(targetpos[1] / targetpos[0])
        distance = np.linalg.norm(targetpos)
    elif targetpos[0] > 0 and targetpos[1] < 0:
        theta = 2 * np.pi - np.arctan(-targetpos[1] / targetpos[0])
        distance = np.linalg.norm(targetpos)
    elif targetpos[0] == 0 and targetpos[1] > 0:
        theta = 0.5 * np.pi
        distance = targetpos[1]
    elif targetpos[0] == 0 and targetpos[1] < 0:
        theta = 1.5 * np.pi
        distance = -targetpos[1]
    elif targetpos[0] > 0 and targetpos[1] == 0:
        theta = 0
        distance = targetpos[0]
    elif targetpos[0] < 0 and targetpos[1] == 0:
        theta = np.pi
        distance = -targetpos[0]
    elif targetpos[0] > 0 and targetpos[1] == 0:
        theta = 0
        distance = 0
        arrived = 1
    return theta, distance, arrived
while (not reached0 and not reached1 and not reached2):
    p.stepSimulation()

    state_wheel_wl = p.getLinkState(tiago, 11, computeLinkVelocity=1)
    state_wheel_wr = p.getLinkState(tiago, 9, computeLinkVelocity=1)
    state_torso = p.getLinkState(tiago, 21)
    angVelo_wheel = 10
    ang_wheel = 1
    p.resetBaseVelocity(tiago, angularVelocity=[0, 0, 1])
    if targetpos[1] < 0:
        orientation_robot = np.pi * 2 + p.getEulerFromQuaternion(state_torso[5])[2]
    else:
        orientation_robot = p.getEulerFromQuaternion(state_torso[5])[2]
    theta_target = measure(targetpos)[0]
    print('target is towards', theta_target if theta_target < 2 * np.pi else theta_target - 2 * np.pi)
    # print('robot is towards', np.arctan(position_robot[1]/position_robot[0]))
    print('robot is towards', orientation_robot if orientation_robot < 2 * np.pi else orientation_robot - 2 * np.pi)
    reached0 =(abs(orientation_robot - theta_target) < 0.003)
    sleep(step)

while (reached0 and not reached1 and not reached2):
    rdm = random.uniform(-0.3, 0.3)
    middlepos_x = -0.5#targetpos[0] + rdm
    middlepos_y = -0.5#targetpos[1] + rdm
    wheel = [9,11]
    radius = 0.0985
    cylinder = 0.04
    currentIndex = wheel

    p.resetBaseVelocity(tiago, angularVelocity=[0, 0, 0])
    p.resetBaseVelocity(tiago,linearVelocity=[-1, -middlepos_y/middlepos_x, 0] if targetpos[0]<0 else [1, middlepos_y/middlepos_x, 0])
    state_base = p.getLinkState(tiago, 20)
    position_robot = np.array(state_base[0])

    p.stepSimulation()
    reached1 = (abs(position_robot[0] - middlepos_x) < 0.1)
    sleep(step)

while (reached0 and reached1 and not reached2): # and iter < 1000):
    p.stepSimulation()
    p.resetBaseVelocity(tiago, linearVelocity=[0, 0, 0])
    jointPoses = p.calculateInverseKinematics(tiago, right_tool_joint, targetpos)
    p.setJointMotorControlArray(bodyIndex=tiago, jointIndices=available_joints_indices,
                                                controlMode=p.POSITION_CONTROL, targetPositions=jointPoses,
                                                positionGains=[1]*len(jointPoses), forces=[500]*len(jointPoses))
    ls = p.getLinkState(tiago, right_tool_joint)
    newPos = ls[4]
    diff = [targetpos[0] - newPos[0], targetpos[1] - newPos[1], targetpos[2] - newPos[2]]
    dist = np.linalg.norm(diff)
    reached = (dist < 0.001)
    iter = iter + 1
    sleep(step)
    print('robot torso pos:', position_robot)
    print('end effector pos', newPos)
    print('distance to target', dist)


