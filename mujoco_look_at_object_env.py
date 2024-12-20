import time
import enum
from typing import Tuple, Optional

import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from camera_controller import CameraController

# TODO do I want action primitives or some form of continuous action e.g. directly set yaw to 85 degrees rather than lots of discrete actions
# TODO claude 3.5 did badly. I think I need a real VLA model.
# TODO does it get slower over time?
# TODO curriculum learning, first begin with left vs right,, then larger randomness in left vs right, then anywhere so it has to hunt for it


class CameraAction(enum.IntEnum):
    """Discrete actions for camera control"""
    YAW_LEFT = 0
    YAW_RIGHT = 1
    MOVE_FORWARD = 2
    MOVE_BACKWARD = 3
    # TODO removing some for now to make problem easier. 
    # TODO could do experiments with many actions that do nothing. credit assignment lol
    # PITCH_UP = 2
    # PITCH_DOWN = 3
    # STRAFE_LEFT = 6
    # STRAFE_RIGHT = 7
    # NO_OP = 8


class LookAtObjectEnv(gym.Env):
    """Environment for training an agent to look at a target object"""
    
    def __init__(self, render_mode="human"):
        super().__init__()
        self.render_mode = render_mode
        
        # Load MuJoCo model with modified camera setup and proper inertial properties
        self.model = mujoco.MjModel.from_xml_string("""
        <mujoco>
            <default>
                <joint limited="false"/>
            </default>
            <worldbody>
                <light pos="0 0 3" dir="0 0 -1" castshadow="false"/>
                <geom type="plane" size="10 10 0.1" rgba=".9 .9 .9 1"/>
                <body name="target_sphere" pos="3 2.5 2">
                    <joint type="free"/>
                    <geom type="sphere" size="0.2" rgba="1 0 0 1"/>
                    <inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/>
                </body>
                <body name="camera_body" pos="0 0 2">
                    <joint type="free"/>
                    <inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/>
                    <geom type="box" size="0.05 0.05 0.05" rgba="0 0 1 0.2"/>
                    <camera name="rgb_camera" mode="fixed"/>
                </body>
            </worldbody>
        </mujoco>
        """)
        self.data = mujoco.MjData(self.model)
        
        # Camera settings
        self.image_width = 128
        self.image_height = 96
        self.camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "rgb_camera")
        self.camera_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "camera_body")
        self.sphere_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_sphere")
        
        # Print debug info
        print(f"Camera body ID: {self.camera_body_id}")
        print(f"qpos size: {self.data.qpos.shape}")
        
        # Define action and observation spaces with new image size
        self.action_space = spaces.Discrete(len(CameraAction))
        print(f"Action space: {self.action_space}")
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.image_height, self.image_width, 3),
            dtype=np.uint8
        )
        
        # Movement parameters
        self.yaw_delta = 2.0
        self.pitch_delta = 2.0
        self.translation_delta = 0.1
        
        # Reward parameters
        self.max_steps = 200
        self.current_step = 0
        self.target_distance_threshold = 10.0
        self.previous_distance = None
        
        # Target position
        self.target_position = [0.0, 0.0, 2.0]
    

    def close(self):
        """Clean up environment resources"""
        pass  # MuJoCo doesn't require explicit cleanup


    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        """Reset the environment"""
        super().reset(seed=seed)
        
        # Reset camera position and orientation
        mujoco.mj_resetData(self.model, self.data)
        
        # Set initial camera pose (7 values per free joint: 3 for position, 4 for quaternion)
        camera_qpos_start = 7 * self.sphere_body_id  # First body's joint
        self.data.qpos[camera_qpos_start:camera_qpos_start+3] = [0, 0, 2]  # Position
        self.data.qpos[camera_qpos_start+3:camera_qpos_start+7] = [1, 0, 0, 0]  # Quaternion (w,x,y,z)
        
        # Randomize sphere position
        sphere_qpos_start = 0  # Second body's joint
        positions = [(3.0, 2.5, 2.0), (3.0, -2.5, 2.0)]
        sphere_pos = positions[np.random.choice(2)]
        self.data.qpos[sphere_qpos_start:sphere_qpos_start+3] = sphere_pos
        self.data.qpos[sphere_qpos_start+3:sphere_qpos_start+7] = [1, 0, 0, 0]
        
        mujoco.mj_forward(self.model, self.data)
        rgb, distance = self.get_observation()
        self.previous_distance = distance
        
        info = {"distance": distance}
        return rgb, info
    

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Take a step in the environment
        
        Args:
            action (int): Action from CameraAction enum
        
        Returns:
            observation (np.ndarray): RGB image
            reward (float): Reward for the action
            terminated (bool): Whether episode is done
            truncated (bool): Whether episode was truncated
            info (dict): Additional information
        """
        self.current_step += 1
        
        # Get current camera pose indices
        camera_qpos_start = 7 * self.sphere_body_id
        pos_start = camera_qpos_start
        quat_start = camera_qpos_start + 3
        
        # Apply action
        if action == CameraAction.YAW_LEFT:
            rot = mujoco.rotations.euler2quat([0, 0, np.radians(self.yaw_delta)])
            current_quat = self.data.qpos[quat_start:quat_start+4]
            new_quat = mujoco.rotations.quat_mul(current_quat, rot)
            self.data.qpos[quat_start:quat_start+4] = new_quat
        elif action == CameraAction.YAW_RIGHT:
            rot = mujoco.rotations.euler2quat([0, 0, -np.radians(self.yaw_delta)])
            current_quat = self.data.qpos[quat_start:quat_start+4]
            new_quat = mujoco.rotations.quat_mul(current_quat, rot)
            self.data.qpos[quat_start:quat_start+4] = new_quat
        elif action == CameraAction.MOVE_FORWARD:
            forward = self.get_camera_forward()
            self.data.qpos[pos_start:pos_start+3] += forward * self.translation_delta
        elif action == CameraAction.MOVE_BACKWARD:
            forward = self.get_camera_forward()
            self.data.qpos[pos_start:pos_start+3] -= forward * self.translation_delta
        
        mujoco.mj_forward(self.model, self.data)
        
        # Get new observation
        rgb, distance = self.get_observation()
        
        # Calculate reward
        reward = self._calculate_reward(distance)
        
        # Check if done
        terminated = False
        if distance is not None and distance < self.target_distance_threshold:
            print("Target centered in view!!!!\n\n")
            terminated = True  # Success - target centered in view
            reward += 10.0  # Bonus reward for success
        
        truncated = self.current_step >= self.max_steps
        
        # Update previous distance
        self.previous_distance = distance
        
        info = {
            "distance": distance,
            "step": self.current_step,
        }
        
        return rgb, reward, terminated, truncated, info
    

    def _calculate_reward(self, distance: Optional[float]) -> float:
        """Calculate reward based on current observation
        
        Args:
            distance (float): Distance from image center to target center
        
        Returns:
            float: Reward value
        """
        if distance is None:
            return -1.0  # Penalty for losing sight of target
        
        if self.previous_distance is None:
            return 0.0
        
        # Reward for moving closer to target
        reward = self.previous_distance - distance
        
        # Small step penalty to encourage efficiency
        reward -= 0.1
        
        return reward
    

    def get_observation(self) -> Tuple[np.ndarray, Optional[float]]:
        """Get current observation from environment"""
        # Render camera view
        rgb = mujoco.mj_render(self.model, self.data, 
                              width=self.image_width,
                              height=self.image_height,
                              camera=self.camera_id)
        
        # Process red mask
        red_mask = (rgb[:,:,0] > 200) & (rgb[:,:,1] < 50) & (rgb[:,:,2] < 50)
        sphere_pixels = np.where(red_mask)
        
        if len(sphere_pixels[0]) == 0:
            return rgb, None
        
        # Calculate sphere center in image
        sphere_center_y = np.mean(sphere_pixels[0])
        sphere_center_x = np.mean(sphere_pixels[1])
        
        # Calculate image center
        image_center_y = rgb.shape[0] / 2
        image_center_x = rgb.shape[1] / 2
        
        # Calculate Euclidean distance from image center to sphere center
        distance = np.sqrt(
            (sphere_center_x - image_center_x)**2 + 
            (sphere_center_y - image_center_y)**2
        )
        
        return rgb, distance
    

    def get_camera_forward(self):
        """Calculate camera forward vector based on orientation"""
        quat_start = 7 * self.sphere_body_id + 3
        quat = self.data.qpos[quat_start:quat_start+4]
        rot_mat = mujoco.rotations.quat2mat(quat)
        return rot_mat[:, 0]  # First column is forward direction


def human_control_main():
    """Test environment with keyboard controls"""
    import pygame
    
    pygame.init()
    screen = pygame.display.set_mode((128, 96))
    env = LookAtObjectEnv()
    
    print("\nCamera Controls:")
    print("- A/D: Rotate camera left/right")
    print("- W/S: Move forward/backward")
    print("- R: Reset environment")
    print("- Esc: Quit")
    
    print_interval = 0.1
    last_print_time = time.time()
    
    rgb, info = env.reset()
    running = True
    
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
            keys = pygame.key.get_pressed()
            action = None
            
            if keys[pygame.K_a]:
                action = CameraAction.YAW_LEFT
            elif keys[pygame.K_d]:
                action = CameraAction.YAW_RIGHT
            elif keys[pygame.K_w]:
                action = CameraAction.MOVE_FORWARD
            elif keys[pygame.K_s]:
                action = CameraAction.MOVE_BACKWARD
            elif keys[pygame.K_r]:
                rgb, info = env.reset()
            elif keys[pygame.K_ESCAPE]:
                running = False
            
            if action is not None:
                rgb, reward, terminated, truncated, info = env.step(action)
                
                if time.time() - last_print_time >= print_interval:
                    distance_str = f"{info['distance']:.2f}" if info['distance'] is not None else "None"
                    print(f"Step {env.current_step}: reward={reward:.2f}, distance={distance_str}")
                    last_print_time = time.time()
                
                if terminated or truncated:
                    print("Episode finished!")
                    rgb, info = env.reset()
            
            # Display the camera view
            surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
            screen.blit(surface, (0, 0))
            pygame.display.flip()
            
            time.sleep(1./60.)
    
    finally:
        env.close()
        pygame.quit()


if __name__ == "__main__":
    human_control_main()