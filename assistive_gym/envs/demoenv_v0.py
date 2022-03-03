import os, time
import numpy as np
import pybullet as p
import gym
from gym import error,spaces,utils
from gym.utils import seeding

import pybullet_data
import math
import random
class Demo0Env(gym.Env):

    def __init__(self):
        p.connect(p.GUI)
        p.resetDebugVisualizerCamera(cameraDistance=3.0,cameraYaw=0,\
                                     cameraPitch=-20,cameraTargetPosition=[0.55,-0.35,0.2])
        # [x y z finger_pos]
        self.action_space=spaces.Box(np.array([-1]*4),np.array([1]*4))
        # [finger_1 finger_2 end_effector_x-y-z]
        self.observation_space=spaces.Box(np.array([-1]*5),np.array([1]*5))

    def step(self,action):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)

       # obs = self._get_obs()
        numJoints = p.getNumJoints(self.robotid)
        available_joints_indices = [i for i in range(numJoints) if p.getJointInfo(self.robotid, i)[2] != p.JOINT_FIXED]
        right_arm_indices = [46, 47, 48, 49, 50, 51, 52]
        lowerlimits = [-1.17, -1.17, -0.78, -0.39, -2.09, -1.41, -2.09]
        upperlimits = [1.57, 1.57, 3.92, 2.35, 2.09, 1.41, 2.09]
        torso_index = [21]  # torso lift prismatic joint
        right_gripper_indices = [58, 59]
        right_tool_joint = 56
        iter = 0
        dist = 1e30
        reached1 = False
        reached2 = False
        targetpos = [2.2, 2.1, 1.5]
        while not reached1:
            middlepos_x = targetpos[0] - 0.1
            middlepos_y = targetpos[1] - 0.1

            p.resetBaseVelocity(self.robotid, linearVelocity=[1, middlepos_y / middlepos_x, 0])
            state_base = p.getLinkState(self.robotid, 20)
            position_robot = np.array(state_base[0])

            p.stepSimulation()
            reached1 = (abs(position_robot[0] - middlepos_x) < 0.001)
            #time.sleep(1 / 240)

        while (reached1 and not reached2 and iter < 1000):
            p.stepSimulation()
            p.resetBaseVelocity(self.robotid, linearVelocity=[0, 0, 0])
            jointPoses = p.calculateInverseKinematics(self.robotid, right_tool_joint, targetpos)
            p.setJointMotorControlArray(bodyIndex=self.robotid, jointIndices=available_joints_indices,
                                        controlMode=p.POSITION_CONTROL, targetPositions=jointPoses,
                                        positionGains=[0.01] * len(jointPoses), forces=[500] * len(jointPoses))

            ls = p.getLinkState(self.robotid, right_tool_joint)
            newPos = ls[4]
            diff = [targetpos[0] - newPos[0], targetpos[1] - newPos[1], targetpos[2] - newPos[2]]
            dist = np.linalg.norm(diff)
            reached2 = (dist < 0.001)
            iter = iter + 1
            #time.sleep(1 / 240)

        state_robot=p.getLinkState(self.robotid,46)[0]

        if state_robot[2]>0.5:
                reward=1
                done=True
        else:
                reward=0
                done=False
        info=state_robot[2]
        observation=state_robot

        #time.sleep(1/240)
        return observation,reward,done,info

    def reset(self):
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
        p.setGravity(0,0,-10)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        plane=p.loadURDF("plane/plane.urdf")
        rest_poses=[0, -0.215 ,0 ,-0.2 , 0, 0.25, 0.26]
        right_arm_indices = [46, 47, 48, 49, 50, 51, 52]
        # self.robotid=p.loadURDF("tiago/tiago_dual.urdf", useFixedBase=True,
        #                         flags=p.URDF_USE_INERTIA_FROM_FILE)
        self.robotid = p.loadURDF("tiago_dualhand/tiago_dual_modified.urdf",
                                  basePosition=[0, 0, 0.05], useFixedBase=True)
        for i in range(7):
            p.resetJointState(self.robotid,right_arm_indices[i],rest_poses[i])

        state_robot=p.getLinkState(self.robotid,1)[0]
        observation=state_robot
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
        p.stepSimulation()
        return observation

    def render(self, mode='human'):
        return None

    def create_sphere(self, radius=0.01, mass=0.0, pos=[0, 0, 0], visual=True, collision=True,
                      rgba=[0, 1, 1, 1], maximal_coordinates=False, return_collision_visual=False):
        sphere_collision = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=radius, physicsClientId=self.id) if collision else -1
        sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=radius, rgbaColor=rgba, physicsClientId=self.id) if visual else -1
        if return_collision_visual:
            return sphere_collision, sphere_visual
        body = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual,
                                 basePosition=pos, useMaximalCoordinates=maximal_coordinates, physicsClientId=self.id)
        sphere = Agent()
        sphere.init(body, self.id, self.np_random, indices=-1)
        return sphere
    def generate_target(self):
        # Set target
        self.mouth_pos = [0, -0.11, 0.03]
        target_pos, target_orient = p.multiplyTransforms(head_pos, head_orient, self.mouth_pos, [0, 0, 0, 1], physicsClientId=self.id)
        self.target = self.create_sphere(radius=0.01, mass=0.0, pos=target_pos, collision=False, rgba=[0, 1, 0, 1])
        self.update_targets()

    def update_targets(self):
        # update_targets() is automatically called at each time step for updating any targets in the environment.
        # Move the target marker onto the person's mouth
        head_pos, head_orient = self.human.get_pos_orient(self.human.head)
        target_pos, target_orient = p.multiplyTransforms(head_pos, head_orient, self.mouth_pos, [0, 0, 0, 1], physicsClientId=self.id)
        self.target_pos = np.array(target_pos)
        self.target.set_base_pos_orient(self.target_pos, [0, 0, 0, 1])




