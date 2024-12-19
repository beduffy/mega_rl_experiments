import time
import enum
from typing import Tuple, Optional

import pybullet as p
import pybullet_data
import numpy as np


# TODO do I want action primitives or some form of continuous action e.g. directly set yaw to 85 degrees rather than lots of discrete actions
# TODO claude 3.5 did badly. I think I need a real VLA model.

class CameraAction(enum.IntEnum):
    """Discrete actions for camera control"""
    YAW_LEFT = 0
    YAW_RIGHT = 1
    PITCH_UP = 2
    PITCH_DOWN = 3
    MOVE_FORWARD = 4
    MOVE_BACKWARD = 5
    STRAFE_LEFT = 6
    STRAFE_RIGHT = 7
    NO_OP = 8


class CameraController:
    def __init__(self, distance=2.0, yaw=0.0, pitch=0.0):
        # Camera position in world space
        self.camera_position = [0.0, -3.0, 2.0]  # Starting position
        self.yaw = yaw
        self.pitch = pitch
        self.up_axis_index = 2
        self.initial_position = self.camera_position.copy()
        self.initial_yaw = yaw
        self.initial_pitch = pitch
        
        # Enable mouse picking and keyboard control
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 1)
        
        self.update_camera()
    

    def reset(self):
        """Reset camera to initial position and orientation"""
        self.camera_position = self.initial_position.copy()
        self.yaw = self.initial_yaw
        self.pitch = self.initial_pitch
        self.update_camera()
    

    def update_camera(self):
        """Update camera view based on current position and rotation"""
        # Calculate look-at point based on camera rotation
        yaw_rad = np.radians(self.yaw)
        pitch_rad = np.radians(self.pitch)
        
        # Calculate direction vector
        look_x = np.cos(yaw_rad) * np.cos(pitch_rad)
        look_y = np.sin(yaw_rad) * np.cos(pitch_rad)
        look_z = np.sin(pitch_rad)
        
        # Calculate target position 1 unit ahead of camera
        target_position = [
            self.camera_position[0] + look_x,
            self.camera_position[1] + look_y,
            self.camera_position[2] + look_z
        ]
        
        # Update debug visualizer camera with a single call
        p.resetDebugVisualizerCamera(
            cameraDistance=0.1,
            cameraYaw=self.yaw - 90,
            cameraPitch=self.pitch,
            cameraTargetPosition=target_position
        )
    

    def move_camera(self, forward=0, right=0, up=0):
        """Move camera in its local coordinate system"""
        yaw_rad = np.radians(self.yaw)
        
        # Forward/backward movement - along the look direction
        self.camera_position[0] += -forward * np.sin(yaw_rad)
        self.camera_position[1] += forward * np.cos(yaw_rad)
        
        # Left/right movement - perpendicular to look direction
        self.camera_position[0] += right * np.cos(yaw_rad)
        self.camera_position[1] += right * np.sin(yaw_rad)
        
        # Up/down movement in world space
        self.camera_position[2] += up
        
        self.update_camera()
    

    def get_camera_image(self, width=640, height=480):
        """Get RGB and depth image from current camera position"""
        # Calculate look-at point based on camera rotation
        yaw_rad = np.radians(self.yaw)
        pitch_rad = np.radians(self.pitch)
        
        # Calculate direction vector
        look_x = np.cos(yaw_rad) * np.cos(pitch_rad)
        look_y = np.sin(yaw_rad) * np.cos(pitch_rad)
        look_z = np.sin(pitch_rad)
        
        # Calculate target position 1 unit ahead of camera
        target_position = [
            self.camera_position[0] + look_x,
            self.camera_position[1] + look_y,
            self.camera_position[2] + look_z
        ]
        
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.camera_position,
            cameraTargetPosition=target_position,
            cameraUpVector=[0, 0, 1]
        )
        
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60.0,
            aspect=width / height,
            nearVal=0.1,
            farVal=100.0
        )
        
        # Get camera image
        (_, _, px, depth, _) = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = rgb_array[:, :, :3]
        
        return rgb_array, depth
    

    def print_current_camera_params(self):
        """Print current camera parameters"""
        params = p.getDebugVisualizerCamera()
        print("\nCamera Parameters:")
        print(f"Distance: {params[10]:.2f}")
        print(f"Yaw: {params[8]:.2f}")
        print(f"Pitch: {params[9]:.2f}")
        print(f"Target Position: {params[11]}")


class LookAtObjectEnv:
    """Environment for training an agent to look at a target object"""
    
    def __init__(self, render_mode="human"):
        self.render_mode = render_mode
        self.connection_mode = p.GUI if render_mode == "human" else p.DIRECT
        
        # Action and observation spaces
        self.action_space = len(CameraAction)
        self.observation_space_shape = (480, 640, 3)  # RGB image
        
        # Movement parameters
        self.yaw_delta = 2.0
        self.pitch_delta = 2.0
        self.translation_delta = 0.1
        
        # Reward parameters
        self.max_steps = 200
        self.current_step = 0
        self.target_distance_threshold = 10.0  # pixels from center
        self.previous_distance = None
        
        # Target position
        self.target_position = [0.0, 0.0, 2.0]
        
        self.reset()
    

    def reset(self) -> Tuple[np.ndarray, dict]:
        """Reset the environment
        
        Returns:
            observation (np.ndarray): RGB image
            info (dict): Additional information
        """
        # Connect to PyBullet if not already connected
        if not p.isConnected():
            p.connect(self.connection_mode)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
            # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        
        # Load ground plane for reference
        p.loadURDF("plane.urdf")
        
        # Reset camera
        if not hasattr(self, 'camera'):
            self.camera = CameraController()
        else:
            self.camera.reset()
        
        # Create or reset target sphere
        self.sphere_pos = self.target_position.copy()
        radius = 0.2  # Small radius for better visibility
        if not hasattr(self, 'sphere_id'):
            sphere_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
            sphere_visual = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=[1, 0, 0, 1])
            self.sphere_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=sphere_collision,
                baseVisualShapeIndex=sphere_visual,
                basePosition=self.sphere_pos
            )
        else:
            p.resetBasePositionAndOrientation(self.sphere_id, self.sphere_pos, [0, 0, 0, 1])
        
        # Reset episode variables
        self.current_step = 0
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
        
        # Apply action
        if action == CameraAction.YAW_LEFT:
            self.camera.yaw += self.yaw_delta
        elif action == CameraAction.YAW_RIGHT:
            self.camera.yaw -= self.yaw_delta
        elif action == CameraAction.PITCH_UP:
            self.camera.pitch = min(89, self.camera.pitch + self.pitch_delta)
        elif action == CameraAction.PITCH_DOWN:
            self.camera.pitch = max(-89, self.camera.pitch - self.pitch_delta)
        elif action == CameraAction.MOVE_FORWARD:
            self.camera.move_camera(forward=self.translation_delta)
        elif action == CameraAction.MOVE_BACKWARD:
            self.camera.move_camera(forward=-self.translation_delta)
        elif action == CameraAction.STRAFE_LEFT:
            self.camera.move_camera(right=-self.translation_delta)
        elif action == CameraAction.STRAFE_RIGHT:
            self.camera.move_camera(right=self.translation_delta)
        elif action == CameraAction.NO_OP:
            pass
        
        self.camera.update_camera()
        
        # Get new observation
        rgb, distance = self.get_observation()
        
        # Calculate reward
        reward = self._calculate_reward(distance)
        
        # Check if done
        terminated = False
        if distance is not None and distance < self.target_distance_threshold:
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
        """Get current observation from environment
        
        Returns:
            rgb (np.ndarray): RGB image
            distance (float): Distance from image center to target center
        """
        rgb, depth = self.camera.get_camera_image()
        
        # Find red sphere in image (simple color thresholding)
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


def human_control_main():
    """Test environment with keyboard controls"""
    env = LookAtObjectEnv()
    
    print("\nCamera Controls:")
    print("- A/D: Rotate camera left/right")
    print("- W/S: Move forward/backward")
    print("- Q/E: Strafe left/right")
    print("- Up/Down: Look up/down")
    print("- R: Reset environment")
    print("- P: Print camera parameters")
    print("- Esc: Quit")
    
    print_interval = 0.1
    last_print_time = time.time()
    
    rgb, info = env.reset()
    
    try:
        while True:
            keys = p.getKeyboardEvents()
            
            for key, state in keys.items():
                if state & p.KEY_WAS_TRIGGERED or state & p.KEY_IS_DOWN:
                    action = None
                    
                    if key == ord('a'):
                        action = CameraAction.YAW_LEFT
                    elif key == ord('d'):
                        action = CameraAction.YAW_RIGHT
                    elif key == ord('w'):
                        action = CameraAction.MOVE_FORWARD
                    elif key == ord('s'):
                        action = CameraAction.MOVE_BACKWARD
                    elif key == ord('q'):
                        action = CameraAction.STRAFE_LEFT
                    elif key == ord('e'):
                        action = CameraAction.STRAFE_RIGHT
                    elif key == p.B3G_UP_ARROW:
                        action = CameraAction.PITCH_UP
                    elif key == p.B3G_DOWN_ARROW:
                        action = CameraAction.PITCH_DOWN
                    elif key == ord('r'):
                        rgb, info = env.reset()
                    elif key == ord('p'):
                        env.camera.print_current_camera_params()
                    
                    if action is not None:
                        rgb, reward, terminated, truncated, info = env.step(action)
                        
                        if time.time() - last_print_time >= print_interval:
                            distance_str = f"{info['distance']:.2f}" if info['distance'] is not None else "None"
                            print(f"Step {env.current_step}: reward={reward:.2f}, distance={distance_str}")
                            last_print_time = time.time()
                        
                        if terminated or truncated:
                            print("Episode finished!")
                            rgb, info = env.reset()
            
            p.stepSimulation()
            time.sleep(1./240.)
    
    finally:
        env.close()


if __name__ == "__main__":
    human_control_main()