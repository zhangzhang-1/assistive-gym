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
tiago = p.loadURDF("tiago_dualhand/tiago_dual_modified.urdf", basePosition=[0, 0, 0.35], useFixedBase=False)
# tiago = p.loadURDF("PR2/pr2_no_torso_lift_tall.urdf", basePosition=[0, 0, 0.35], useFixedBase=False)
joint_num = p.getNumJoints(tiago)
prismatic_joints_indices = [i for i in range(joint_num) if p.getJointInfo(tiago, i)[2] == p.JOINT_PRISMATIC]
revolute_joints_indices = [i for i in range(joint_num) if p.getJointInfo(tiago, i)[2] == p.JOINT_REVOLUTE]
fixed_joints_indices = [i for i in range(joint_num) if p.getJointInfo(tiago, i)[2] == p.JOINT_FIXED]
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
# move with two wheels
ii = 0
while 1:
    # wheel = [12,14,16,18] # control the turnings of side wheels
    # wheel = [13,15,17,19] # side wheels forward
    wheel = [9,11]
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
    linVelo_wheel = 5
    angVelo_wheel = 0
    # linVelo_wheel = (np.array(state_wheel_l[6]) + np.array(state_wheel_r[6])) / 2
    # angVelo_wheel = (np.array(state_wheel_l[7]) + np.array(state_wheel_r[7])) / 2
    wheel_axle_half = 0.5 * (np.linalg.norm(position_wheel_right - position_wheel_left) - cylinder)
    left_wheel_wish = (linVelo_wheel - angVelo_wheel * wheel_axle_half) / radius
    right_wheel_wish = (linVelo_wheel + angVelo_wheel * wheel_axle_half) / radius
    scaling = 1.113
    interval = 1/240
    p.setJointMotorControlArray(bodyIndex=tiago, jointIndices=currentIndex,
                                 controlMode=p.VELOCITY_CONTROL,
                                 targetVelocities= #[linVelo_wheel]*len(wheel),
                                [scaling * right_wheel_wish, scaling * left_wheel_wish],
                                 velocityGains=[1]*len(wheel), forces=[500]*len(wheel))
    # ii += interval * 0.2
    print(currentIndex, 'is moving')
    print('linear velocity is',np.array(state_wheel_l[6]))
    sleep(interval)