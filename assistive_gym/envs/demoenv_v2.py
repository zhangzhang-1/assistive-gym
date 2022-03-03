import os, time
import numpy as np
import pybullet as p
import gym
from gym import error,spaces,utils
from gym.utils import seeding

from .util import Util
import pybullet_data
import math

import random

class Demo2Env(gym.Env):
    def __init__(self):
        self.directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets')

        self.numJoints = p.getNumJoints(self.tiago)
        self.available_joints_indices = [i for i in range(numJoints) if p.getJointInfo(self.tiago, i)[2] != p.JOINT_FIXED]
        self.right_arm_indices = [46, 47, 48, 49, 50, 51, 52]
        self.lowerlimits = [-1.17, -1.17, -0.78, -0.39, -2.09, -1.41, -2.09]
        self.upperlimits = [1.57, 1.57, 3.92, 2.35, 2.09, 1.41, 2.09]
        self.torso_index = [21]  # torso lift prismatic joint
        self.right_gripper_indices = [58, 59]
        self.right_tool_joint = 56
        self.iter = 0
        self.iteration = 0
        self.dist = 1e30
        self.step = 1 / 240
        self.reached0 = False
        self.reached1 = False
        self.reached2 = False
        self.robot_forces = 1.0
        self.robot_gains = 0.05
        self.distance_weight = 1.0
        self.action_weight = 0.01
        self.task_success_threshold = 0.03
        self.targetpos = np.array([2.2, 2.1, 1.5])

        self.action_robot_len = len(self.available_joints_indices)
        self.action_human_len = 0
        self.action_space = spaces.Box(
            low=np.array([-1.0] * (self.action_robot_len + self.action_human_len), dtype=np.float32),
            high=np.array([1.0] * (self.action_robot_len + self.action_human_len), dtype=np.float32), dtype=np.float32)
        self.obs_robot_len = 18 + len(self.available_joints_indices)
        self.obs_human_len = 19
        self.observation_space = spaces.Box(low=np.array([-1000000000.0]*(self.obs_robot_len+self.obs_human_len), dtype=np.float32),
                                            high=np.array([1000000000.0]*(self.obs_robot_len+self.obs_human_len),
                                                                                          dtype=np.float32), dtype=np.float32)
        self.action_space_robot = spaces.Box(low=np.array([-1.0]*self.action_robot_len, dtype=np.float32), high=np.array([1.0]*self.action_robot_len, dtype=np.float32), dtype=np.float32)
        self.action_space_human = spaces.Box(low=np.array([-1.0]*self.action_human_len, dtype=np.float32), high=np.array([1.0]*self.action_human_len, dtype=np.float32), dtype=np.float32)
        self.observation_space_robot = spaces.Box(low=np.array([-1000000000.0]*self.obs_robot_len, dtype=np.float32), high=np.array([1000000000.0]*self.obs_robot_len, dtype=np.float32), dtype=np.float32)
        self.observation_space_human = spaces.Box(low=np.array([-1000000000.0]*self.obs_human_len, dtype=np.float32), high=np.array([1000000000.0]*self.obs_human_len, dtype=np.float32), dtype=np.float32)

    def step(self, action):

        obs = self._get_obs()


        # Get human preferences
        end_effector_velocity = np.linalg.norm(p.getLinkState(self.right_end_effector))

        ee_top_center_pos = [0,0,0]
        reward_distance_mouth = -np.linalg.norm(self.targetpos - np.array(ee_top_center_pos)) # Penalize distances between top of cup and mouth
        reward_action = -np.linalg.norm(action) # Penalize actions



        reward = self.config('distance_weight')*reward_distance_mouth + self.config('action_weight')*reward_action + preferences_score


        info = {'task_success': int(reward_distance_mouth <= self.task_success_threshold), 'action_robot_len': self.action_robot_len,
                'action_human_len': self.action_human_len,'obs_robot_len': self.obs_robot_len, 'obs_human_len': self.obs_human_len}
        done = self.iteration >= 200


        return obs, reward, done, info

    def _get_obs(self):

        robot_joint_angles = p.getJointStates(self.tiago, self.available_joints_indices)
        # Fix joint angles to be in [-pi, pi]
        robot_joint_angles = (np.array(robot_joint_angles) + np.pi) % (2 * np.pi) - np.pi
       # ee_tc_pos = np.array(p.getLinkState(self.robot, 54, computeForwardKinematics=True, physicsClientId=self.id)[0])


        robot_obs = np.concatenate(
            [ - self.targetpos, robot_joint_angles ]).ravel()

        return robot_obs

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_seed(self, seed=1000):
        self.np_random.seed(seed)

    def set_frictions(self, links, lateral_friction=None, spinning_friction=None, rolling_friction=None):
        if type(links) == int:
            links = [links]
        for link in links:
            if lateral_friction is not None:
                p.changeDynamics(self.body, link, lateralFriction=lateral_friction, physicsClientId=self.id)
            if spinning_friction is not None:
                p.changeDynamics(self.body, link, spinningFriction=spinning_friction, physicsClientId=self.id)
            if rolling_friction is not None:
                p.changeDynamics(self.body, link, rollingFriction=rolling_friction, physicsClientId=self.id)

    def build_assistive_env(self):
        # Build plane, furniture, robot, human, etc. (just like world creation)
        # Load the ground plane
        plane = p.loadURDF(os.path.join(self.directory, 'plane', 'plane.urdf'), physicsClientId=self.id)

        # Randomly set friction of the ground
        self.plane.set_frictions(self.plane, lateral_friction=self.np_random.uniform(0.025, 0.5),
                                 spinning_friction=0, rolling_friction=0)
        # Disable rendering during creation
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=self.id)
        # Create robot
        self.tiago = p.loadURDF(os.path.join(self.directory, 'tiago_dualhand', 'tiago_dual_modified.urdf'),
                                useFixedBase=True, basePosition=[-10, -10, 0])
    def reset(self):
        p.resetSimulation(physicsClientId=self.id)

        if not self.gui:
            # Reconnect the physics engine to forcefully clear memory when running long training scripts
            self.disconnect()
            self.id = p.connect(p.DIRECT)
            self.util = Util(self.id, self.np_random)
        if self.gpu:
            self.util.enable_gpu()
        # Configure camera position
        p.resetDebugVisualizerCamera(cameraDistance=1.75, cameraYaw=-25, cameraPitch=-45,
                                     cameraTargetPosition=[-0.2, 0, 0.4],
                                     physicsClientId=self.id)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0, physicsClientId=self.id)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.id)
        p.setTimeStep(1/240, physicsClientId=self.id)
        # Disable real time simulation so that the simulation only advances when we call stepSimulation
        p.setRealTimeSimulation(0, physicsClientId=self.id)
        p.setGravity(0, 0, -9.81, physicsClientId=self.id)

        self.last_sim_time = None
        self.iteration = 0
        self.forces = []
        self.task_success = 0
        self.build_assistive_env()

        # Update robot motor gains


        self.generate_target()

        p.resetDebugVisualizerCamera(cameraDistance=1.10, cameraYaw=55, cameraPitch=-45,
                                     cameraTargetPosition=[-0.2, 0, 0.75], physicsClientId=self.id)


        target_ee_pos = np.array([-0.2, -0.5, 1.1]) + self.np_random.uniform(-0.05, 0.05, size=3)
        target_ee_orient = self.get_quaternion(self.robot.toc_ee_orient_rpy[self.task])
        # self.init_robot_pose

        # Open gripper to hold the tool
        self.robot.set_gripper_open_position(self.robot.right_gripper_indices, self.robot.gripper_pos[self.task],
                                             set_instantly=True)



        p.setPhysicsEngineParameter(numSubSteps=4, numSolverIterations=10, physicsClientId=self.id)


        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)


        for _ in range(50):
            p.stepSimulation(physicsClientId=self.id)

        self.init_env_variables()
        return self._get_obs()

    def init_env_variables(self, reset=False):
        if len(self.action_space.low) <= 1 or reset:
            obs_len = len(self._get_obs())
            self.observation_space.__init__(low=-np.ones(obs_len, dtype=np.float32) * 1000000000,
                                            high=np.ones(obs_len, dtype=np.float32) * 1000000000, dtype=np.float32)

            # Define action/obs lengths
            self.action_robot_len = len(self.available_joints_indices)
            self.action_human_len = 0
            self.obs_robot_len = len(self._get_obs('robot'))
            self.obs_human_len = 19
            self.action_space_robot = spaces.Box(low=np.array([-1.0] * self.action_robot_len, dtype=np.float32),
                                                 high=np.array([1.0] * self.action_robot_len, dtype=np.float32),
                                                 dtype=np.float32)
            self.action_space_human = spaces.Box(low=np.array([-1.0] * self.action_human_len, dtype=np.float32),
                                                 high=np.array([1.0] * self.action_human_len, dtype=np.float32),
                                                 dtype=np.float32)
            self.observation_space_robot = spaces.Box(
                low=np.array([-1000000000.0] * self.obs_robot_len, dtype=np.float32),
                high=np.array([1000000000.0] * self.obs_robot_len, dtype=np.float32), dtype=np.float32)
            self.observation_space_human = spaces.Box(
                low=np.array([-1000000000.0] * self.obs_human_len, dtype=np.float32),
                high=np.array([1000000000.0] * self.obs_human_len, dtype=np.float32), dtype=np.float32)

    def generate_target(self):
        # Set target
        self.sphere = self.create_sphere(radius=0.01, mass=0.0, pos=self.targetpos, collision=False, rgba=[0, 1, 0, 1])
        self.update_targets()

    def update_targets(self):
        # update_targets() is automatically called at each time step for updating any targets in the environment.
        p.resetBasePositionAndOrientation(self.sphere, self.targetpos, [0, 0, 0, 1])

    def create_sphere(self, radius=0.01, mass=0.0, pos=[0, 0, 0], visual=True, collision=True, rgba=[0, 1, 1, 1], maximal_coordinates=False, return_collision_visual=False):
        sphere_collision = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=radius, physicsClientId=self.id) if collision else -1
        sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=radius, rgbaColor=rgba, physicsClientId=self.id) if visual else -1
        if return_collision_visual:
            return sphere_collision, sphere_visual
        sphere = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, basePosition=pos, useMaximalCoordinates=maximal_coordinates, physicsClientId=self.id)
        return sphere
    # def take_step(self, actions, gains=None, forces=None, action_multiplier=0.05, step_sim=True):
    #     if gains is None:
    #         gains = [a.motor_gains for a in self.agents]
    #     elif type(gains) not in (list, tuple):
    #         gains = [gains]*len(self.agents)
    #     if forces is None:
    #         forces = [a.motor_forces for a in self.agents]
    #     elif type(forces) not in (list, tuple):
    #         forces = [forces]*len(self.agents)
    #     if self.last_sim_time is None:
    #         self.last_sim_time = time.time()
    #     self.iteration += 1
    #     self.forces = []
    #     actions = np.clip(actions, a_min=self.action_space.low, a_max=self.action_space.high)
    #     actions *= action_multiplier
    #     action_index = 0
    #     for i, agent in enumerate(self.agents):
    #         needs_action = not isinstance(agent, Human) or agent.controllable
    #         if needs_action:
    #             agent_action_len = len(agent.controllable_joint_indices)
    #             action = np.copy(actions[action_index:action_index+agent_action_len])
    #             action_index += agent_action_len
    #             if isinstance(agent, Robot):
    #                 action *= agent.action_multiplier
    #             if len(action) != agent_action_len:
    #                 print('Received agent actions of length %d does not match expected action length of %d' % (len(action), agent_action_len))
    #                 exit()
    #         # Append the new action to the current measured joint angles
    #         agent_joint_angles = agent.get_joint_angles(agent.controllable_joint_indices)
    #         # Update the target robot/human joint angles based on the proposed action and joint limits
    #         for _ in range(self.frame_skip):
    #             if needs_action:
    #                 below_lower_limits = agent_joint_angles + action < agent.controllable_joint_lower_limits
    #                 above_upper_limits = agent_joint_angles + action > agent.controllable_joint_upper_limits
    #                 action[below_lower_limits] = 0
    #                 action[above_upper_limits] = 0
    #                 agent_joint_angles[below_lower_limits] = agent.controllable_joint_lower_limits[below_lower_limits]
    #                 agent_joint_angles[above_upper_limits] = agent.controllable_joint_upper_limits[above_upper_limits]
    #             if isinstance(agent, Human) and agent.impairment == 'tremor':
    #                 if needs_action:
    #                     agent.target_joint_angles += action
    #                 agent_joint_angles = agent.target_joint_angles + agent.tremors * (1 if self.iteration % 2 == 0 else -1)
    #             else:
    #                 agent_joint_angles += action
    #         if isinstance(agent, Robot) and agent.action_duplication is not None:
    #             agent_joint_angles = np.concatenate([[a]*d for a, d in zip(agent_joint_angles, self.robot.action_duplication)])
    #             agent.control(agent.all_controllable_joints, agent_joint_angles, agent.gains, agent.forces)
    #         else:
    #             agent.control(agent.controllable_joint_indices, agent_joint_angles, gains[i], forces[i])
    #     if step_sim:
    #         # Update all agent positions
    #         for _ in range(self.frame_skip):
    #             p.stepSimulation(physicsClientId=self.id)
    #             for agent in self.agents:
    #                 if isinstance(agent, Human):
    #                     agent.enforce_joint_limits()
    #                     if agent.controllable:
    #                         agent.enforce_realistic_joint_limits()
    #             self.update_targets()
    #             if self.gui:
    #                 # Slow down time so that the simulation matches real time
    #                 self.slow_time()