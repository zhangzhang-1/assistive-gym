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
tiago = p.loadURDF("tiago_dualhand/tiago_dual_modified.urdf", basePosition=[0, 0, 0.05], useFixedBase=False)
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
# move via all six wheels
steering_wheel = [12, 14, 16, 18]  # rf,lf,rb,lb | control the steering of side wheels
wheel = [9, 11, 13, 15, 17, 19]  # side wheels forward
step = 1 / 240
targetpos = [1.5,1]

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
while 1:
    p.stepSimulation()
    # state_wheel_lf = p.getLinkState(tiago, 15, computeLinkVelocity=1)
    # state_wheel_lb = p.getLinkState(tiago, 19, computeLinkVelocity=1)
    # state_wheel_rf = p.getLinkState(tiago, 13, computeLinkVelocity=1)
    # state_wheel_rb = p.getLinkState(tiago, 17, computeLinkVelocity=1)
    state_wheel_wl = p.getLinkState(tiago, 11, computeLinkVelocity=1)
    state_wheel_wr = p.getLinkState(tiago, 9, computeLinkVelocity=1)
    # position_wheel_lf = np.array(state_wheel_lf[0])
    # position_wheel_rf = np.array(state_wheel_rf[0])
    angVelo_wheel = 10
    ang_wheel = 1

    p.setJointMotorControlArray(bodyIndex=tiago, jointIndices=steering_wheel,
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=[0.2979, 0.4729, -0.3623, -0.5634],
                                positionGains=[1]*len(steering_wheel), forces=[500] * len(steering_wheel))
    p.setJointMotorControlArray(bodyIndex=tiago, jointIndices=wheel,
                                 controlMode=p.VELOCITY_CONTROL,
                                 targetVelocities= [0.604 * angVelo_wheel, 0.196 * angVelo_wheel,
                                                    0.525 * angVelo_wheel, 0.335 * angVelo_wheel,
                                                    0.537 * angVelo_wheel, 0.353 * angVelo_wheel] ,
                                 velocityGains=[2] * 6, forces=[500] * 6)
    position_robot = (np.array(state_wheel_wr[0]) + np.array(state_wheel_wl[0])) / 2
    theta_target = measure(targetpos)[0]
    if abs(np.pi / 2 - np.arctan(position_robot[0]/position_robot[1])-theta_target) < 0.0003:
         break
    print('robot is towards', np.pi / 2 - np.arctan(position_robot[0]/position_robot[1]))
    sleep(step)

while 1:
    p.stepSimulation()
    state_wheel_wl = p.getLinkState(tiago, 11, computeLinkVelocity=1)
    state_wheel_wr = p.getLinkState(tiago, 9, computeLinkVelocity=1)
    p.setJointMotorControlArray(bodyIndex=tiago, jointIndices=steering_wheel,
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=[0, 0, 0, 0],
                                positionGains=[1]*len(steering_wheel), forces=[500] * len(steering_wheel))
    p.setJointMotorControlArray(bodyIndex=tiago, jointIndices=wheel,
                                 controlMode=p.VELOCITY_CONTROL,
                                 targetVelocities= [10] * 6,
                                 velocityGains=[2] * 6, forces=[500] * 6)
    position_robot = (np.array(state_wheel_wr[0]) + np.array(state_wheel_wl[0])) / 2
    print(position_robot[0],position_robot[1])
    if abs(position_robot[0]-targetpos[0])<0.003:
        break
    sleep(step)