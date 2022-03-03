import os
import numpy as np
import pybullet as p
from .robot import Robot

class tiago_dualhand(Robot):
    def __init__(self, controllable_joints='right'):
        right_arm_joint_indices = [46, 47, 48, 49, 50, 51, 52] # Controllable arm joints
        # left_arm_joint_indices = right_arm_joint_indices
        left_arm_joint_indices = [31, 32, 33, 34, 35, 36, 37] # Controllable arm joints
        wheel_joint_indices = []
        # wheel_joint_indices = [9,11, 12, 13,14,15,16,17,18,19] # 12-19 are 'ignore variables'
        right_end_effector = 60 # Used to get the pose of the end effector
        # left_end_effector = right_end_effector
        left_end_effector = 45 # Used to get the pose of the end effector
        right_gripper_indices = [58, 59] # Gripper actuated joints
        # left_gripper_indices = right_gripper_indices # Gripper actuated joints
        left_gripper_indices = [43, 44] # Those joints are fixed
        right_tool_joint = 56 # Joint that tools are attached to
        # left_tool_joint = right_tool_joint  # Joint that tools are attached to
        left_tool_joint = 41 # Joint that tools are attached to
        right_gripper_collision_indices = list(range(57, 60))
        #right_gripper_collision_indices = list(range(54, 61)) # Used to disable collision between gripper and tools
        left_gripper_collision_indices = list(range(42, 45))
        #left_gripper_collision_indices = list(range(39, 46)) # Used to disable collision between gripper and tools
        gripper_pos = {'scratch_itch': [0.025, 0.026], # Gripper open position for holding tools
                       'feeding': [0.01, 0],
                       'drinking': [0.037, 0.037],
                       'reaching': [0 ,0],
                       'bed_bathing': [0.025, 0.026],
                       'dressing': [0]*2,
                       'arm_manipulation': [0.025, 0.026]}
        tool_pos_offset = {'scratch_itch': [0, 0, -0.2], # Position offset between tool and robot tool joint
                           'feeding': [0.01, -0.02, -0.28],
                           'drinking': [0, -0.03, -0.16],  # [-0.02, 0.01, -0.16]
                           'reaching': [0, 0, 0],
                           'bed_bathing': [0, 0, -0.2],
                           'arm_manipulation': [0, -0.08, -0.31]}
        tool_orient_offset = {'scratch_itch': [0, np.pi/2, 0], # RPY orientation offset between tool and robot tool joint
                              'feeding': [-0.2, 0, 0],
                              'drinking': [0, np.pi/2, 0], # vertical: [np.pi/2, np.pi/2, 0]
                              'reaching': [0, 0, 0],
                              'bed_bathing': [-np.pi/2, 0, 0],
                              'arm_manipulation': [0, np.pi/2, 0]}
        toc_base_pos_offset = {'scratch_itch': [0.1, 0, 0], # Robot base offset before TOC base pose optimization
                               'feeding': [0.1, 0.2, 0],
                               'drinking': [0.5, 0.05, 0],
                               'reaching': [0, 0, 0],
                               'bed_bathing': [-0.1, 0, 0],
                               'dressing': [0, 0, 0],
                               'arm_manipulation': [0, 0, 0]}
        toc_ee_orient_rpy = {'scratch_itch': [0, 0, 0], # Initial end effector orientation
                             'feeding': [np.pi/2.0, 0, 0],
                             'drinking': [0, 0.25, np.pi/2],
                             'reaching': [0, 0, 0],
                             'bed_bathing': [0, 0, 0],
                             'dressing': [[0, 0, np.pi], [0, 0, np.pi*3/2.0]],
                             'arm_manipulation': [0, 0, 0]}
        wheelchair_mounted = False

        super(tiago_dualhand, self).__init__(controllable_joints=controllable_joints, right_arm_joint_indices=right_arm_joint_indices,
                                             left_arm_joint_indices=left_arm_joint_indices, wheel_joint_indices=wheel_joint_indices,
                                             right_end_effector=right_end_effector, left_end_effector=left_end_effector,
                                             right_gripper_indices=right_gripper_indices, left_gripper_indices=left_gripper_indices, gripper_pos=gripper_pos,
                                             right_tool_joint=right_tool_joint, left_tool_joint=left_tool_joint, tool_pos_offset=tool_pos_offset,
                                             tool_orient_offset=tool_orient_offset, right_gripper_collision_indices=right_gripper_collision_indices,
                                             left_gripper_collision_indices=left_gripper_collision_indices, toc_base_pos_offset=toc_base_pos_offset,
                                             toc_ee_orient_rpy=toc_ee_orient_rpy, wheelchair_mounted=wheelchair_mounted, half_range=True)


    def init(self, directory, id, np_random, fixed_base=True):
        self.body = p.loadURDF(os.path.join(directory, 'tiago_dualhand', 'tiago_dual_modified.urdf'),
                               useFixedBase=fixed_base, basePosition=[-1, -1, 0],
                              # flags=p.URDF_USE_INERTIA_FROM_FILE,
                               physicsClientId=id)
        super(tiago_dualhand, self).init(self.body, id, np_random)


        # Recolor robot
        for i in [19, 42, 64]:
            p.changeVisualShape(self.body, i, rgbaColor=[1.0, 1.0, 1.0, 1.0], physicsClientId=id)
        for i in [45, 51, 67, 73]:
            p.changeVisualShape(self.body, i, rgbaColor=[0.7, 0.7, 0.7, 1.0], physicsClientId=id)
        p.changeVisualShape(self.body, 20, rgbaColor=[0.8, 0.8, 0.8, 1.0], physicsClientId=id)
        p.changeVisualShape(self.body, 40, rgbaColor=[0.6, 0.6, 0.6, 1.0], physicsClientId=id)

    def reset_joints(self):
        super(tiago_dualhand, self).reset_joints()
        # Position end effectors whith dual arm robots
        self.set_joint_angles(self.right_arm_joint_indices, [0.5, -1, 0.2, -0.2, 0.3, 0, -0.2])
        self.set_joint_angles(self.left_arm_joint_indices, [-0.5, 0, -0.2, -0.2, -0.3, 0, -0.75])