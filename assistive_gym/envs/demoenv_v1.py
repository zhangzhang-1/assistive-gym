from .util import Util
import pybullet_data
import random
import os, time, configparser
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from screeninfo import get_monitors
import pybullet as p
from keras.models import load_model
from .human_creation import HumanCreation
from .agents import agent, human, robot, panda, tool, furniture
from .agents.agent import Agent
from .agents.tiago_dualhand import tiago_dualhand
from .agents.human import Human
from .agents.robot import Robot
from .agents.tool import Tool
from .agents.furniture import Furniture

class Demo1Env(gym.Env):
    def __init__(self, robot=robot, human=human, task='demo', obs_robot_len=21, obs_human_len=19, time_step=0.02, frame_skip=5, render=False, gravity=-9.81, seed=1001):
        self.robot = tiago_dualhand
        self.human = Human(human.left_arm_joints, controllable=False)
        self.task = 'demo'
        self.obs_robot_len = obs_robot_len
        self.obs_human_len = obs_human_len
        self.time_step = time_step
        self.frame_skip = frame_skip
        self.gravity = gravity
        self.id = None
        self.gui = False
        self.gpu = False
        self.view_matrix = None
        self.seed(seed)
        if render:
            self.render()
        else:
            self.id = p.connect(p.DIRECT)
            self.util = Util(self.id, self.np_random)

        self.directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets')
        self.human_creation = HumanCreation(self.id, np_random=self.np_random, cloth=('dressing' in task))
        self.human_limits_model = load_model(os.path.join(self.directory, 'realistic_arm_limits_model.h5'))
        self.action_robot_len = 23
        self.action_human_len = 19
        self.action_space = spaces.Box(low=np.array([-1.0]*(self.action_robot_len+self.action_human_len), dtype=np.float32), high=np.array([1.0]*(self.action_robot_len+self.action_human_len), dtype=np.float32), dtype=np.float32)
        self.obs_robot_len = obs_robot_len
        self.obs_human_len = obs_human_len
        self.observation_space = spaces.Box(low=np.array([-1000000000.0]*(self.obs_robot_len+self.obs_human_len), dtype=np.float32), high=np.array([1000000000.0]*(self.obs_robot_len+self.obs_human_len), dtype=np.float32), dtype=np.float32)
        self.action_space_robot = spaces.Box(low=np.array([-1.0]*self.action_robot_len, dtype=np.float32), high=np.array([1.0]*self.action_robot_len, dtype=np.float32), dtype=np.float32)
        self.action_space_human = spaces.Box(low=np.array([-1.0]*self.action_human_len, dtype=np.float32), high=np.array([1.0]*self.action_human_len, dtype=np.float32), dtype=np.float32)
        self.observation_space_robot = spaces.Box(low=np.array([-1000000000.0]*self.obs_robot_len, dtype=np.float32), high=np.array([1000000000.0]*self.obs_robot_len, dtype=np.float32), dtype=np.float32)
        self.observation_space_human = spaces.Box(low=np.array([-1000000000.0]*self.obs_human_len, dtype=np.float32), high=np.array([1000000000.0]*self.obs_human_len, dtype=np.float32), dtype=np.float32)

        self.agents = []
        self.plane = Agent()
        self.robot = robot
        self.human = human
        self.tool = Tool()
        self.furniture = Furniture()

        self.configp = configparser.ConfigParser()
        self.configp.read(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'config.ini'))
        # Human preference weights
        self.C_v = self.config('velocity_weight', 'human_preferences')
        self.C_f = self.config('force_nontarget_weight', 'human_preferences')
        self.C_hf = self.config('high_forces_weight', 'human_preferences')
        self.C_fd = self.config('food_hit_weight', 'human_preferences')
        self.C_fdv = self.config('food_velocities_weight', 'human_preferences')
        self.C_d = self.config('dressing_force_weight', 'human_preferences')
        self.C_p = self.config('high_pressures_weight', 'human_preferences')


    def step(self,action):
        if self.human.controllable:
            action = np.concatenate([action['robot'], action['human']])
        self.take_step(action)

        obs = self._get_obs()

        #reward_water, water_mouth_velocities, water_hit_human_reward = self.get_water_rewards()

        # Get human preferences
        end_effector_velocity = np.linalg.norm(self.robot.get_velocity(self.robot.right_end_effector))
        preferences_score = self.human_preferences(end_effector_velocity=end_effector_velocity, total_force_on_human=self.total_force_on_human, tool_force_at_target=self.cup_force_on_human)

        tool_pos, tool_orient = self.tool.get_base_pos_orient()
        tool_pos, tool_orient = p.multiplyTransforms(tool_pos, tool_orient, [0, 0.06, 0], self.get_quaternion([np.pi/2.0, 0, 0]), physicsClientId=self.id)
        tool_top_center_pos, _ = p.multiplyTransforms(tool_pos, tool_orient, self.tool_top_center_offset, [0, 0, 0, 1], physicsClientId=self.id)
        reward_distance_mouth_target = -np.linalg.norm(self.target_pos - np.array(tool_top_center_pos)) # Penalize distances between top of cup and mouth
        reward_action = -np.linalg.norm(action) # Penalize actions

        # Encourage robot to have a tilted end effector / cup
        tool_euler = self.get_euler(tool_orient)
        reward_tilt = -abs(tool_euler[0] - np.pi/2)

        reward = self.config('distance_weight')*reward_distance_mouth_target + self.config('action_weight')*reward_action + self.config('cup_tilt_weight')*reward_tilt + self.config('drinking_reward_weight')*reward_action + preferences_score

        if self.gui and reward_action != 0:
            print('Task success:', self.task_success, 'action reward:', reward_action)


        revo = [22, 23, 31, 32, 33, 34, 35, 36, 37, 46, 47, 48, 49, 50, 51, 52]
        pris = np.array([21, 43, 44, 58, 59])
        np.random.shuffle(pris)
        currentIndex = pris[0]
        currentPose = p.getLinkState(self.robotid, currentIndex)
        currentPosition = currentPose[4]
        rdm = random.uniform(0, 5)
        newPosition = [currentPosition[0] + rdm,
                           currentPosition[1] + rdm,
                           currentPosition[2] + rdm]
        print(currentIndex, 'is moving')
        p.stepSimulation()
        p.setJointMotorControl2(bodyIndex=self.robotid, jointIndex=currentIndex,
                                                                      controlMode=p.POSITION_CONTROL, targetPosition=rdm,
                                                                      positionGain=0.5, force=500)
        state_robot=p.getLinkState(self.robotid,currentIndex)[0]
        observation = state_robot
        if self.gui and reward_action >= 0:
            print('Task success:', self.task_success, 'Water reward:', reward_action)

        info = {'total_force_on_human': self.total_force_on_human, 'task_success': int(self.task_success >= self.config('task_success_threshold')), 'action_robot_len': self.action_robot_len, 'action_human_len': self.action_human_len, 'obs_robot_len': self.obs_robot_len, 'obs_human_len': self.obs_human_len}
        done = self.iteration >= 200
        if not self.human.controllable:
            return obs, reward, done, info
        else:
            # Co-optimization with both human and robot controllable
            return obs, {'robot': reward, 'human': reward}, {'robot': done, 'human': done, '__all__': done}, {
                'robot': info, 'human': info}

    def get_joint_angles(self, indices):
        robot_joint_states = []
        for ii in range(len(indices)):
            robot_joint_states[ii] = p.getJointStates(self.robot, jointIndices=ii, physicsClientId=self.id)
            xx[ii] = robot_joint_states[ii][0]
        return np.array(xx)

    def convert_to_realworld(self, pos, orient=[0, 0, 0, 1]):
        base_pos, base_orient = self.get_base_pos_orient()
        base_pos_inv, base_orient_inv = p.invertTransform(base_pos, base_orient, physicsClientId=self.id)
        real_pos, real_orient = p.multiplyTransforms(base_pos_inv, base_orient_inv, pos, orient if len(orient) == 4 else self.get_quaternion(orient), physicsClientId=self.id)
        return np.array(real_pos), np.array(real_orient)

    def _get_obs(self, agent=None):
        controllable_joint_indices = np.array([22, 23, 31, 32, 33, 34, 35, 36, 37, 46, 47, 48, 49, 50, 51, 52])

        for ii in range(len(controllable_joint_indices)):
            robot_joint_states = p.getJointState(self.robot, controllable_joint_indices[ii])
            xx[ii] = robot_joint_states[0]
        robot_joint_angles = np.array(xx)

        # Fix joint angles to be in [-pi, pi]
        robot_joint_angles = (np.array(robot_joint_angles) + np.pi) % (2 * np.pi) - np.pi


        target_pos_real, _ = self.convert_to_realworld(self.target_pos)

        robot_obs = np.concatenate(
            [ - target_pos_real, robot_joint_angles, head_pos_real,
             head_orient_real]).ravel()
        if agent == 'robot':
            return robot_obs
        if self.human.controllable:
            human_joint_angles = self.human.get_joint_angles(self.human.controllable_joint_indices)
            #tool_pos_human, tool_orient_human = self.human.convert_to_realworld(tool_pos, tool_orient)
            #head_pos_human, head_orient_human = self.human.convert_to_realworld(head_pos, head_orient)
            target_pos_human, _ = self.human.convert_to_realworld(self.target_pos)
            human_obs = np.concatenate(
                [ - target_pos_human, human_joint_angles,
                 head_pos_human, head_orient_human, [self.robot_force_on_human]]).ravel()
            if agent == 'human':
                return human_obs
            # Co-optimization with both human and robot controllable
            return {'robot': robot_obs, 'human': human_obs}
        return robot_obs

    def config(self, tag, section=None):
        return float(self.configp[self.task if section is None else section][tag])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_seed(self, seed=1000):
        self.np_random.seed(seed)

    def enable_gpu_rendering(self):
        self.gpu = True

    def disconnect(self):
        p.disconnect(self.id)

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
                                     cameraTargetPosition=[-0.2, 0, 0.4], physicsClientId=self.id)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0, physicsClientId=self.id)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.id)
        p.setTimeStep(self.time_step, physicsClientId=self.id)
        # Disable real time simulation so that the simulation only advances when we call stepSimulation
        p.setRealTimeSimulation(0, physicsClientId=self.id)
        p.setGravity(0, 0, self.gravity, physicsClientId=self.id)
        self.agents = []
        self.last_sim_time = None
        self.iteration = 0
        self.forces = []
        self.task_success = 0

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        plane=p.loadURDF("plane.urdf",basePosition=[0,0,-0.65])
        rest_poses=[0,-0.215,0,-2.57,0,2.356,2.356,0.08,0.08]
        self.robotid =p.loadURDF("tiago_dualhand/tiago_dual.urdf",
                                useFixedBase=False, basePosition=[0, 0, 0])
        for i in range(7):
            p.resetJointState(self.robotid,i,rest_poses[i])
        tableUid=p.loadURDF("table/table.urdf",basePosition=[0.5,0,-0.65])

        # Initialize the tool in the robot's gripper
        #self.tool.init(self.robotid, self.task, self.directory, self.id, self.np_random, right=True)
        state_robot=p.getLinkState(self.robotid,1)[0]
        observation=state_robot
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
        return self._get_obs()

    def build_assistive_env(self, furniture_type=None, fixed_human_base=True, human_impairment='random', gender='random'):
        # Build plane, furniture, robot, human, etc. (just like world creation)
        # Load the ground plane
        plane = p.loadURDF(os.path.join(self.directory, 'plane', 'plane.urdf'), physicsClientId=self.id)
        self.plane.init(plane, self.id, self.np_random, indices=-1)
        # Randomly set friction of the ground
        self.plane.set_frictions(self.plane.base, lateral_friction=self.np_random.uniform(0.025, 0.5),
                                 spinning_friction=0, rolling_friction=0)
        # Disable rendering during creation
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=self.id)
        # Create robot
        if self.robot is not None:
            self.robot.init(self.directory, self.id, self.np_random, fixed_base=not self.robot.mobile)
            self.agents.append(self.robot)
        # Create human
        # if self.human is not None and isinstance(self.human, Human):
        #     self.human.init(self.human_creation, self.human_limits_model, fixed_human_base, human_impairment, gender, self.config, self.id, self.np_random)
        #     if self.human.controllable or self.human.impairment == 'tremor':
        #         self.agents.append(self.human)
        # Create furniture (wheelchair, bed, or table)
        if furniture_type is not None:
            self.furniture.init(furniture_type, self.directory, self.id, self.np_random, wheelchair_mounted=self.robot.wheelchair_mounted if self.robot is not None else False)

    def init_env_variables(self, reset=False):
        if len(self.action_space.low) <= 1 or reset:
            obs_len = len(self._get_obs())
            self.observation_space.__init__(low=-np.ones(obs_len, dtype=np.float32)*1000000000, high=np.ones(obs_len, dtype=np.float32)*1000000000, dtype=np.float32)
            self.update_action_space()
            # Define action/obs lengths
            self.action_robot_len = len(self.robot.controllable_joint_indices)
            self.action_human_len = len(self.human.controllable_joint_indices) if self.human.controllable else 0
            self.obs_robot_len = len(self._get_obs('robot'))
            self.obs_human_len = len(self._get_obs('human'))
            self.action_space_robot = spaces.Box(low=np.array([-1.0]*self.action_robot_len, dtype=np.float32), high=np.array([1.0]*self.action_robot_len, dtype=np.float32), dtype=np.float32)
            self.action_space_human = spaces.Box(low=np.array([-1.0]*self.action_human_len, dtype=np.float32), high=np.array([1.0]*self.action_human_len, dtype=np.float32), dtype=np.float32)
            self.observation_space_robot = spaces.Box(low=np.array([-1000000000.0]*self.obs_robot_len, dtype=np.float32), high=np.array([1000000000.0]*self.obs_robot_len, dtype=np.float32), dtype=np.float32)
            self.observation_space_human = spaces.Box(low=np.array([-1000000000.0]*self.obs_human_len, dtype=np.float32), high=np.array([1000000000.0]*self.obs_human_len, dtype=np.float32), dtype=np.float32)

    def update_action_space(self):
        action_len = np.sum([len(a.controllable_joint_indices) for a in self.agents if not isinstance(a, Human) or a.controllable])
        self.action_space.__init__(low=-np.ones(action_len, dtype=np.float32), high=np.ones(action_len, dtype=np.float32), dtype=np.float32)

    def create_human(self, controllable=False, controllable_joint_indices=[], fixed_base=False, human_impairment='random', gender='random', mass=None, radius_scale=1.0, height_scale=1.0):
        '''
        human_impairement in ['none', 'limits', 'weakness', 'tremor']
        gender in ['male', 'female']
        '''
        self.human = Human(controllable_joint_indices, controllable=controllable)
        self.human.init(self.human_creation, self.human_limits_model, fixed_base, human_impairment, gender, None, self.id, self.np_random, mass=mass, radius_scale=radius_scale, height_scale=height_scale)
        if controllable or self.human.impairment == 'tremor':
            self.agents.append(self.human)
            self.update_action_space()
        return self.human

    def create_robot(self, robot_class, controllable_joints='right', fixed_base=True):
        self.robot = robot_class(controllable_joints)
        self.robot.init(self.directory, self.id, self.np_random, fixed_base=fixed_base)
        self.agents.append(self.robot)
        self.update_action_space()
        return self.robot

    def take_step(self, actions, gains=None, forces=None, action_multiplier=0.05, step_sim=True):
        if gains is None:
            gains = [a.motor_gains for a in self.agents]
        elif type(gains) not in (list, tuple):
            gains = [gains]*len(self.agents)
        if forces is None:
            forces = [a.motor_forces for a in self.agents]
        elif type(forces) not in (list, tuple):
            forces = [forces]*len(self.agents)
        if self.last_sim_time is None:
            self.last_sim_time = time.time()
        self.iteration += 1
        self.forces = []
        actions = np.clip(actions, a_min=self.action_space.low, a_max=self.action_space.high)
        actions *= action_multiplier
        action_index = 0
        for i, agent in enumerate(self.agents):
            needs_action = not isinstance(agent, Human) or agent.controllable
            if needs_action:
                agent_action_len = len(agent.controllable_joint_indices)
                action = np.copy(actions[action_index:action_index+agent_action_len])
                action_index += agent_action_len
                if isinstance(agent, Robot):
                    action *= agent.action_multiplier
                if len(action) != agent_action_len:
                    print('Received agent actions of length %d does not match expected action length of %d' % (len(action), agent_action_len))
                    exit()
            # Append the new action to the current measured joint angles
            agent_joint_angles = agent.get_joint_angles(agent.controllable_joint_indices)
            # Update the target robot/human joint angles based on the proposed action and joint limits
            for _ in range(self.frame_skip):
                if needs_action:
                    below_lower_limits = agent_joint_angles + action < agent.controllable_joint_lower_limits
                    above_upper_limits = agent_joint_angles + action > agent.controllable_joint_upper_limits
                    action[below_lower_limits] = 0
                    action[above_upper_limits] = 0
                    agent_joint_angles[below_lower_limits] = agent.controllable_joint_lower_limits[below_lower_limits]
                    agent_joint_angles[above_upper_limits] = agent.controllable_joint_upper_limits[above_upper_limits]
                if isinstance(agent, Human) and agent.impairment == 'tremor':
                    if needs_action:
                        agent.target_joint_angles += action
                    agent_joint_angles = agent.target_joint_angles + agent.tremors * (1 if self.iteration % 2 == 0 else -1)
                else:
                    agent_joint_angles += action
            if isinstance(agent, Robot) and agent.action_duplication is not None:
                agent_joint_angles = np.concatenate([[a]*d for a, d in zip(agent_joint_angles, self.robot.action_duplication)])
                agent.control(agent.all_controllable_joints, agent_joint_angles, agent.gains, agent.forces)
            else:
                agent.control(agent.controllable_joint_indices, agent_joint_angles, gains[i], forces[i])
        if step_sim:
            # Update all agent positions
            for _ in range(self.frame_skip):
                p.stepSimulation(physicsClientId=self.id)
                for agent in self.agents:
                    if isinstance(agent, Human):
                        agent.enforce_joint_limits()
                        if agent.controllable:
                            agent.enforce_realistic_joint_limits()
                self.update_targets()
                if self.gui:
                    # Slow down time so that the simulation matches real time
                    self.slow_time()

    def human_preferences(self, end_effector_velocity=0, total_force_on_human=0, tool_force_at_target=0, food_hit_human_reward=0, food_mouth_velocities=[], dressing_forces=[[]], arm_manipulation_tool_forces_on_human=[0, 0], arm_manipulation_total_force_on_human=0):
        # Slow end effector velocities
        reward_velocity = -end_effector_velocity

        # < 10 N force at target
        reward_high_target_forces = 0 if tool_force_at_target < 10 else -tool_force_at_target

        # --- Scratching, Wiping ---
        # Any force away from target is low
        reward_force_nontarget = -(total_force_on_human - tool_force_at_target)

        # --- Scooping, Feeding, Drinking ---
        if self.task in ['feeding', 'drinking','demo']:
            # Penalty when robot's body applies force onto a person
            reward_force_nontarget = -total_force_on_human
        # Penalty when robot spills food on the person
        reward_food_hit_human = food_hit_human_reward
        # Human prefers food entering mouth at low velocities
        reward_food_velocities = 0 if len(food_mouth_velocities) == 0 else -np.sum(food_mouth_velocities)

        # --- Dressing ---
        # Penalty when cloth applies force onto a person
        reward_dressing_force = -np.sum(np.linalg.norm(dressing_forces, axis=-1))

        # --- Arm Manipulation ---
        # Penalty for applying large pressure to the person (high forces over small surface areas)
        # if self.task == 'arm_manipulation':
        #     tool_right_contact_points = len(self.tool_right.get_closest_points(self.human, distance=0.01)[-1])
        #     tool_left_contact_points = len(self.tool_left.get_closest_points(self.human, distance=0.01)[-1])
        #     tool_right_pressure = 0 if tool_right_contact_points <= 0 else (arm_manipulation_tool_forces_on_human[0] / tool_right_contact_points)
        #     tool_left_pressure = 0 if tool_left_contact_points <= 0 else (arm_manipulation_tool_forces_on_human[1] / tool_left_contact_points)
        #
        #     reward_arm_manipulation_tool_pressures = -(tool_right_pressure + tool_left_pressure)
        #     reward_force_nontarget = -(arm_manipulation_total_force_on_human - np.sum(arm_manipulation_tool_forces_on_human))
        # else:
        #     reward_arm_manipulation_tool_pressures = 0.0

        return self.C_v*reward_velocity + self.C_f*reward_force_nontarget + self.C_hf*reward_high_target_forces + self.C_fd*reward_food_hit_human + self.C_fdv*reward_food_velocities + self.C_d*reward_dressing_force + self.C_p*reward_arm_manipulation_tool_pressures

    def init_robot_pose(self, target_ee_pos, target_ee_orient, start_pos_orient, target_pos_orients, arm='right', tools=[], collision_objects=[], wheelchair_enabled=True, right_side=True, max_iterations=3):
        base_position = None
        if self.robot.skip_pose_optimization:
            return base_position
        # Continually resample initial robot pose until we find one where the robot isn't colliding with the person
        for _ in range(max_iterations):
            if self.robot.mobile:
                # Randomize robot base pose
                pos = np.array(self.robot.toc_base_pos_offset[self.task])
                pos[:2] += self.np_random.uniform(-0.1, 0.1, size=2)
                orient = np.array(self.robot.toc_ee_orient_rpy[self.task])
                if self.task != 'dressing':
                    orient[2] += self.np_random.uniform(-np.deg2rad(30), np.deg2rad(30))
                else:
                    orient = orient[0]
                self.robot.set_base_pos_orient(pos, orient)
                # Randomize starting joint angles
                self.robot.randomize_init_joint_angles(self.task)
            elif self.robot.wheelchair_mounted and wheelchair_enabled:
                # Use IK to find starting joint angles for mounted robots
                self.robot.ik_random_restarts(right=(arm == 'right'), target_pos=target_ee_pos, target_orient=target_ee_orient, max_iterations=1000, max_ik_random_restarts=1000, success_threshold=0.01, step_sim=False, check_env_collisions=False, randomize_limits=True, collision_objects=collision_objects)
            else:
                # Use TOC with JLWKI to find an optimal base position for the robot near the person
                base_position, _, _ = self.robot.position_robot_toc(self.task, arm, start_pos_orient, target_pos_orients, self.human, step_sim=False, check_env_collisions=False, max_ik_iterations=100, max_ik_random_restarts=1, randomize_limits=False, right_side=right_side, base_euler_orient=[0, 0, 0 if right_side else np.pi], attempts=50)
            # Check if the robot or tool is colliding with objects in the environment. If so, then continue sampling.
            dists_list = []
            for tool in tools:
                tool.reset_pos_orient()
                for obj in collision_objects:
                    dists_list.append(tool.get_closest_points(obj, distance=0)[-1])
            for obj in collision_objects:
                dists_list.append(self.robot.get_closest_points(obj, distance=0)[-1])
            if all(not d for d in dists_list):
                break
        return base_position

    def slow_time(self):
        # Slow down time so that the simulation matches real time
        t = time.time() - self.last_sim_time
        if t < self.time_step:
            time.sleep(self.time_step - t)
        self.last_sim_time = time.time()

    def update_targets(self):
        pass

    def render(self,mode='human'):
        if not self.gui:
            self.gui = True
            if self.id is not None:
                self.disconnect()
            try:
                self.width = get_monitors()[0].width
                self.height = get_monitors()[0].height
            except Exception as e:
                self.width = 1920
                self.height = 1080
            self.id = p.connect(p.GUI,
                                options='--background_color_red=0.8 --background_color_green=0.9 --background_color_blue=1.0 --width=%d --height=%d' % (
                                self.width, self.height))
            self.util = Util(self.id, self.np_random)

    def get_euler(self, quaternion):
        return np.array(p.getEulerFromQuaternion(np.array(quaternion), physicsClientId=self.id))

    def get_quaternion(self, euler):
        return np.array(p.getQuaternionFromEuler(np.array(euler), physicsClientId=self.id))

    def setup_camera(self, camera_eye=[0.5, -0.75, 1.5], camera_target=[-0.2, 0, 0.75], fov=60, camera_width=1920//4, camera_height=1080//4):
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.view_matrix = p.computeViewMatrix(camera_eye, camera_target, [0, 0, 1], physicsClientId=self.id)
        self.projection_matrix = p.computeProjectionMatrixFOV(fov, camera_width / camera_height, 0.01, 100, physicsClientId=self.id)

    def setup_camera_rpy(self, camera_target=[-0.2, 0, 0.75], distance=1.5, rpy=[0, -35, 40], fov=60, camera_width=1920//4, camera_height=1080//4):
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(camera_target, distance, rpy[2], rpy[1], rpy[0], 2, physicsClientId=self.id)
        self.projection_matrix = p.computeProjectionMatrixFOV(fov, camera_width / camera_height, 0.01, 100, physicsClientId=self.id)

    def get_camera_image_depth(self, light_pos=[0, -3, 1], shadow=False, ambient=0.8, diffuse=0.3, specular=0.1):
        assert self.view_matrix is not None, 'You must call env.setup_camera() or env.setup_camera_rpy() before getting a camera image'
        w, h, img, depth, _ = p.getCameraImage(self.camera_width, self.camera_height, self.view_matrix, self.projection_matrix, lightDirection=light_pos, shadow=shadow, lightAmbientCoeff=ambient, lightDiffuseCoeff=diffuse, lightSpecularCoeff=specular, physicsClientId=self.id)
        img = np.reshape(img, (h, w, 4))
        depth = np.reshape(depth, (h, w))
        return img, depth

    def create_sphere(self, radius=0.01, mass=0.0, pos=[0, 0, 0], visual=True, collision=True, rgba=[0, 1, 1, 1], maximal_coordinates=False, return_collision_visual=False):
        sphere_collision = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=radius, physicsClientId=self.id) if collision else -1
        sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=radius, rgbaColor=rgba, physicsClientId=self.id) if visual else -1
        if return_collision_visual:
            return sphere_collision, sphere_visual
        body = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, basePosition=pos, useMaximalCoordinates=maximal_coordinates, physicsClientId=self.id)
        sphere = Agent()
        sphere.init(body, self.id, self.np_random, indices=-1)
        return sphere

    def create_spheres(self, radius=0.01, mass=0.0, batch_positions=[[0, 0, 0]], visual=True, collision=True, rgba=[0, 1, 1, 1]):
        sphere_collision = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=radius, physicsClientId=self.id) if collision else -1
        sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=radius, rgbaColor=rgba, physicsClientId=self.id) if visual else -1
        last_sphere_id = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, basePosition=[0, 0, 0], useMaximalCoordinates=False, batchPositions=batch_positions, physicsClientId=self.id)
        spheres = []
        for body in list(range(last_sphere_id-len(batch_positions)+1, last_sphere_id+1)):
            sphere = Agent()
            sphere.init(body, self.id, self.np_random, indices=-1)
            spheres.append(sphere)
        return spheres

    def create_agent_from_obj(self, visual_filename, collision_filename, scale=1.0, mass=1.0, pos=[0, 0, 0], orient=[0, 0, 0, 1], rgba=[1, 1, 1, 1], maximal=False):
        visual_shape = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=visual_filename, meshScale=scale, rgbaColor=rgba, physicsClientId=self.id)
        collision_shape = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=collision_filename, meshScale=scale, physicsClientId=self.id)
        body = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=collision_shape, baseVisualShapeIndex=visual_shape, basePosition=pos, baseOrientation=orient, useMaximalCoordinates=maximal, physicsClientId=self.id)
        agent = Agent()
        agent.init(body, self.id, self.np_random, indices=-1)
        return agent