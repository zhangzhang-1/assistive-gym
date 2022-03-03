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
tiago = p.loadURDF("tiago_dualhand/tiago_dual_modified.urdf", basePosition=[0, 0, 0.65], useFixedBase=False)
joint_num = p.getNumJoints(tiago)
prismatic_joints_indices = [i for i in range(joint_num) if p.getJointInfo(tiago, i)[2] == p.JOINT_PRISMATIC]
revolute_joints_indices = [i for i in range(joint_num) if p.getJointInfo(tiago, i)[2] == p.JOINT_REVOLUTE]
fixed_joints_indices = [i for i in range(joint_num) if p.getJointInfo(tiago, i)[2] == p.JOINT_FIXED]
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
# move wheels
ii = 0
while 1:
    wheel = [11,9]
    radius = 0.0985
    cylinder = 0.04
    currentIndex = wheel
    rdm = [random.uniform(0, 1)] * 3
    # for ii in range(len(wheel)):
    # p.resetBaseVelocity(tiago,linearVelocity=[1,0,0])
    # p.changeConstraint()
    # p.resetBasePositionAndOrientation(tiago, (0+ii, 0+ii, 0.65), (0.0, 0.0, 0.0, 1.0))
    p.stepSimulation()

    state_wheel_l = p.getLinkState(tiago, 11, computeLinkVelocity=1)
    state_wheel_r = p.getLinkState(tiago, 9, computeLinkVelocity=1)
    position_wheel_left = np.array(state_wheel_l[0])
    position_wheel_right = np.array(state_wheel_r[0])
    linVelo_wheel_left = np.array(state_wheel_l[6])
    linVelo_wheel_right = np.array(state_wheel_r[6])
    angVelo_wheel_left = np.array(state_wheel_l[7])
    angVelo_wheel_right = np.array(state_wheel_r[7])
    wheel_axle_half = 0.5 * (np.linalg.norm(position_wheel_right - position_wheel_left) - cylinder)

    left_wheel_ang_vel = (linVelo_wheel_left - angVelo_wheel_left * wheel_axle_half) / radius
    right_wheel_ang_vel = (linVelo_wheel_right + angVelo_wheel_right * wheel_axle_half) / radius
    scaling = 0.53
    interval = 1/240
    p.setJointMotorControlArray(bodyIndex=tiago, jointIndices=currentIndex,
                                 controlMode=p.VELOCITY_CONTROL,
                                 targetVelocities=[scaling * np.linalg.norm(left_wheel_ang_vel),scaling * np.linalg.norm(right_wheel_ang_vel)],
                                 velocityGains=[2]*len(wheel), forces=[500]*len(wheel))
    ii += interval * 0.2
    print(currentIndex, 'is moving')
    # print('angular velocity is',p.getDynamicsInfo(tiago,9))
    sleep(interval)

# randomly move prismatic joints

# while 1:
#     pris=np.array([21, 43, 44, 58, 59])
#     for ii in range(len(pris)):
#         p.stepSimulation()
#         rdm = random.uniform(-2, 2)
#         currentIndex = pris[ii]
#         currentPose = p.getLinkState(tiago, currentIndex)
#         currentPosition = currentPose[4]
#         newPosition = [currentPosition[0] + rdm,
#                         currentPosition[1] + rdm, currentPosition[2] + rdm]
#
#         print(currentIndex, 'is moving')
#         p.setJointMotorControl2(bodyIndex=tiago, jointIndex=currentIndex,
#                                  controlMode=p.POSITION_CONTROL, targetPosition=rdm,
#                                  positionGain=0.5, force=500)
#     sleep(1/240)

# randomly move revolute joints

# while 1:
#     revo = np.array([22, 23, 31, 32, 33, 34, 35, 36, 37, 46, 47, 48, 49, 50, 51, 52])
#     for ii in range(len(revo)):
#         p.stepSimulation()
#         rdm = random.uniform(-0.2, 0.2)
#         currentIndex = revo[ii]
#         currentPose = p.getLinkState(tiago, currentIndex)
#         currentPosition = currentPose[4]
#         newPosition = [currentPosition[0] + rdm,
#                         currentPosition[1] + rdm, currentPosition[2] + rdm]
#
#         print(currentIndex, 'is moving')
#         p.setJointMotorControl2(bodyIndex=tiago, jointIndex=currentIndex,
#                                  controlMode=p.POSITION_CONTROL, targetPosition=rdm,
#                                  positionGain=0.5, force=500)
#     sleep(1/240)