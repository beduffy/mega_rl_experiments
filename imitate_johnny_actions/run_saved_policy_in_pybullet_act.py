import argparse
import time
from datetime import datetime

import torch
import pybullet as p
import pybullet_data
import numpy as np

# from imitate_johnny_actions.imitate_johnny_action import SimplePolicy, JOINT_ORDER
from imitate_johnny_actions.imitate_johnny_action_act import ACTPolicy, JOINT_ORDER

# TODO see how slow CNN is. profile. Also check GPU and stuff or?
# TODO how to pytorch AMD?
# TODO are all 24 joints outputting?!?



def load_policy(checkpoint_path, device='cpu'):
    """Load trained ACT policy from checkpoint"""
    # Load config from checkpoint instead of hardcoding
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    policy_config = checkpoint['config']
    policy_config['device'] = device  # Update device

    policy = ACTPolicy(policy_config)
    policy.load_state_dict(checkpoint['model_state'])
    policy.eval()
    return policy


def get_camera_image(robot, width=240, height=240):
    """Get robot's camera view for policy input (preserved for later use)"""
    # Camera position on robot's head (adjust these values based on your URDF)
    head_link = [i for i in range(p.getNumJoints(robot))
                if p.getJointInfo(robot, i)[1].decode() == 'head_tilt'][0]

    # Get camera position and orientation
    head_pos = p.getLinkState(robot, head_link)[0]
    head_orn = p.getLinkState(robot, head_link)[1]
    rot_matrix = p.getMatrixFromQuaternion(head_orn)
    forward_vec = np.array(rot_matrix[6:9])
    up_vec = np.array(rot_matrix[3:6])

    # Camera parameters
    view_matrix = p.computeViewMatrix(
        head_pos + forward_vec * 0.1,  # Camera position (slightly in front of head)
        head_pos + forward_vec * 2.0,  # Look at point
        up_vec
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60, aspect=1.0, nearVal=0.1, farVal=10.0
    )

    # Render image
    _, _, rgb, _, _ = p.getCameraImage(
        width=width, height=height,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )

    # Convert to tensor and add camera dimension
    image = torch.from_numpy(rgb[..., :3].transpose(2, 0, 1)).float() / 255.0
    return image.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 3, H, W] (batch, camera, channels, H, W)


def get_dummy_image():
    """Generate black dummy image matching training specs"""
    # Create dummy image tensor: [1, 1, 3, 120, 160] (batch, camera, channels, H, W)
    return torch.zeros(1, 1, 3, 120, 160, dtype=torch.float32)


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to trained policy checkpoint')
    parser.add_argument('--device', type=str, default='cpu',
                      choices=['cpu', 'cuda'], help='Device to run policy on')
    args = parser.parse_args()

    # Load policy
    policy = load_policy(args.checkpoint, device=args.device).to(args.device)


    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    # p.setGravity(0, 0, 0)

    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, lightPosition=[1, 1, 1])
    planeId = p.loadURDF("plane.urdf")

    # Load robot URDF
    use_fixed_base = False
    # use_fixed_base = True
    urdf_path = "/home/ben/all_projects/ainex_private_ws/ainex_private/src/ainex_simulations/ainex_description/urdf/ainex.urdf"
    robot = p.loadURDF(urdf_path, [0, 0, 0.25], useFixedBase=use_fixed_base)  # Start above ground

    # Add action buffer for sequence prediction
    pred_steps = 3
    action_buffer = []
    control_interval = 0.1  # Time between policy queries (seconds)
    last_control_time = time.time()

    # Initialize joint targets with neutral position
    joint_targets = {name: 0.0 for name in JOINT_ORDER}

    while True:
        current_time = time.time()

        # Get observations - USING DUMMY IMAGE FOR NOW
        image = get_dummy_image().to(args.device)  # Switch to get_camera_image() when ready
        qpos = torch.zeros(1, 24).to(args.device)  # Replace with actual qpos if available

        # Run policy at control interval
        if current_time - last_control_time > control_interval:
            start_time = time.time()
            with torch.no_grad():
                action_output = policy(qpos, image)
                action_array = action_output.cpu().numpy().flatten()  # Force to 1D array
                print(f"Action output shape: {action_output.shape}")  # Debug

                # Ensure we have 24 elements (one per joint)
                if len(action_array) != 24:
                    raise ValueError(f"Expected 24 action values, got {len(action_array)}")

            # Directly map to joint targets
            joint_targets = {name: action_array[i] for i, name in enumerate(JOINT_ORDER)}
            print("Sample joint targets:", {k: f"{v:.3f}" for k,v in list(joint_targets.items())[:3]})

            inference_time = time.time() - start_time
            print(f"Policy inference took: {inference_time:.3f}s")
            last_control_time = current_time

        # Apply to simulation (now using initialized joint_targets)
        set_joint_angles_instantly(robot, joint_targets)
        p.stepSimulation()
        time.sleep(1./240.)
