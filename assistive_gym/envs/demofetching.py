import os
from gym import spaces
import numpy as np
import pybullet as p
from screeninfo import get_monitors
from .env import AssistiveEnv

class FetchingEnv(AssistiveEnv):
    def __init__(self, robot, human):
        super(FetchingEnv, self).__init__(robot=robot, human=human, task='reaching', obs_robot_len=(18 + len(robot.controllable_joint_indices) - (len(robot.wheel_joint_indices) if robot.mobile else 0)), obs_human_len=(19 + len(human.controllable_joint_indices)))

    def step(self, action):

        self.take_step(action)

        obs = self._get_obs()


        # Get human preferences
        end_effector_velocity = np.linalg.norm(self.robot.get_velocity(self.robot.right_end_effector))
        preferences_score = self.human_preferences(end_effector_velocity=end_effector_velocity)

        ee_top_center_pos = [0,0,0]
        reward_distance_mouth = -np.linalg.norm(self.target_pos - np.array(ee_top_center_pos)) # Penalize distances between top of cup and mouth
        reward_action = -np.linalg.norm(action) # Penalize actions



        reward = self.config('distance_weight')*reward_distance_mouth + self.config('action_weight')*reward_action + preferences_score


        info = { 'task_success': int(reward_distance_mouth <= self.config(
            'task_success_threshold')), 'action_robot_len': self.action_robot_len, 'action_human_len': self.action_human_len,
                'obs_robot_len': self.obs_robot_len, 'obs_human_len': self.obs_human_len}
        done = self.iteration >= 200


        return obs, reward, done, info

    def get_total_force(self):
        robot_force_on_human = np.sum(self.robot.get_contact_points(self.human)[-1])

        return robot_force_on_human

    def _get_obs(self, agent=None):

        robot_joint_angles = self.robot.get_joint_angles(self.robot.controllable_joint_indices)
        # Fix joint angles to be in [-pi, pi]
        robot_joint_angles = (np.array(robot_joint_angles) + np.pi) % (2 * np.pi) - np.pi
       # ee_tc_pos = np.array(p.getLinkState(self.robot, 54, computeForwardKinematics=True, physicsClientId=self.id)[0])
        if self.robot.mobile:
            # Don't include joint angles for the wheels
            robot_joint_angles = robot_joint_angles[len(self.robot.wheel_joint_indices):]

        target_pos_real, _ = self.robot.convert_to_realworld(self.target_pos)

        robot_obs = np.concatenate(
            [ - target_pos_real, robot_joint_angles]).ravel()

        return robot_obs

    def reset(self):
        super(FetchingEnv, self).reset()
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
        # if self.robot.wheelchair_mounted:
        #     wheelchair_pos, wheelchair_orient = self.furniture.get_base_pos_orient()
            self.robot.set_base_pos_orient([0,0,0] ,[0, 0, -np.pi / 2.0])

        #p.resetBasePositionAndOrientation(self.robot, [-0.85, -0.4, 0], [0,0,0,1])
        # Update robot and human motor gains
        self.robot.motor_gains  = 0.005


        self.generate_target()

        p.resetDebugVisualizerCamera(cameraDistance=1.10, cameraYaw=55, cameraPitch=-45,
                                     cameraTargetPosition=[-0.2, 0, 0.75], physicsClientId=self.id)


        target_ee_pos = np.array([-0.2, -0.5, 1.1]) + self.np_random.uniform(-0.05, 0.05, size=3)
        target_ee_orient = self.get_quaternion(self.robot.toc_ee_orient_rpy[self.task])


        # Open gripper to hold the tool
        self.robot.set_gripper_open_position(self.robot.right_gripper_indices, self.robot.gripper_pos[self.task],
                                             set_instantly=True)

        if not self.robot.mobile:
            self.robot.set_gravity(0, 0, 0)



        p.setPhysicsEngineParameter(numSubSteps=4, numSolverIterations=10, physicsClientId=self.id)


        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)


        for _ in range(50):
            p.stepSimulation(physicsClientId=self.id)

        self.init_env_variables()
        return self._get_obs()

    def generate_target(self):
        # Set target on mouth
        self.target_pos = [0, -0.75, 0.75]
        target_orient = [0, 0, 0, 1]
        self.target = self.create_sphere(radius=0.01, mass=0.0, pos=self.target_pos, collision=False, rgba=[0, 1, 0, 1])
        self.update_targets()

    def update_targets(self):
        # update_targets() is automatically called at each time step for updating any targets in the environment.
        # Move the target marker onto the person's mouth
        target_pos, target_orient = self.target_pos, [0, 0, 0, 1]
        self.target_pos = np.array(target_pos)
        self.target.set_base_pos_orient(self.target_pos, [0, 0, 0, 1])