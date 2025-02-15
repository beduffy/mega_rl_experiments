import pybullet as p
import numpy as np
import torch


def set_joint_angles_instantly(robot, angle_dict_to_try):
    # Enable motor control for all joints
    num_joints = p.getNumJoints(robot)
    for joint_idx in range(num_joints):
        joint_info = p.getJointInfo(robot, joint_idx)
        # print(joint_info)
        joint_name = joint_info[1].decode('utf-8')
        joint_type = joint_info[2]  # Get joint type

        # Set 0 position for all joints. Only control revolute and prismatic joints.
        # if joint_type in [p.JOINT_REVOLUTE]:
        #     p.setJointMotorControl2(robot, joint_idx,
        #                            controlMode=p.POSITION_CONTROL,
        #                            targetPosition=0,  # Radians for revolute, meters for prismatic
        #                            force=100)  # Maximum force in Newtons

        # TODO make head tilt and head pan to move
        if joint_name in angle_dict_to_try and joint_type in [p.JOINT_REVOLUTE] and joint_name not in ['head_tilt', 'head_pan']:
            p.setJointMotorControl2(robot, joint_idx,
                                controlMode=p.POSITION_CONTROL,
                                positionGain=0.01,
                                targetPosition=angle_dict_to_try[joint_name],  # Radians for revolute, meters for prismatic
                                force=500)  # Maximum force in Newtons


def get_dummy_image():
    """Generate black dummy image matching training specs"""
    # Create dummy image tensor: [1, 1, 3, 120, 160] (batch, camera, channels, H, W)
    return torch.zeros(1, 1, 3, 120, 160, dtype=torch.float32)
